from __future__ import annotations

import atexit
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import (
    CORE_COLLECTIONS,
    DB_PATH,
    DENSE_CANDIDATES,
    EMBED_FALLBACK_MODEL,
    EMBED_MODEL,
    KNOWLEDGE_DIR,
    LEXICAL_CANDIDATES,
    PAIR_BOOST,
    QDRANT_PATH,
    TOP_K,
)
from app.extractors import SUPPORTED_EXTENSIONS, build_document_units, load_teacher_notes
from app.ollama_client import OllamaClient
from app.schemas import DocumentUnit, EvalCase, FileIngestStatus, IngestResponse, ProblemPairRecord, RetrievalHit


class KnowledgeStore:
    def __init__(
        self,
        db_path: Path = DB_PATH,
        knowledge_dir: Path = KNOWLEDGE_DIR,
        *,
        enable_vector_store: bool = True,
    ) -> None:
        self.db_path = db_path
        self.knowledge_dir = knowledge_dir
        self.ollama = OllamaClient()
        self.qdrant = QdrantClient(path=str(QDRANT_PATH)) if enable_vector_store else None
        self._ensure_schema()
        atexit.register(self.close)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            if self._is_legacy_documents_schema(conn):
                conn.execute("DROP TABLE IF EXISTS units")
                conn.execute("DROP TABLE IF EXISTS documents")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_family TEXT NOT NULL,
                    chapter TEXT,
                    embed_model TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS units (
                    point_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    chapter TEXT,
                    problem_id TEXT,
                    pair_key TEXT,
                    title TEXT,
                    source_quality TEXT NOT NULL,
                    page_or_slide INTEGER,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    excerpt TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(file_path) REFERENCES documents(file_path) ON DELETE CASCADE
                );
                """
            )

    def _is_legacy_documents_schema(self, conn: sqlite3.Connection) -> bool:
        rows = conn.execute("PRAGMA table_info(documents)").fetchall()
        if not rows:
            return False
        column_names = {row["name"] for row in rows}
        return "source_family" not in column_names

    def ingest_directory(
        self,
        *,
        force_full_rebuild: bool = False,
        rebuild_scope: str = "core",
        include_silver: bool = True,
    ) -> IngestResponse:
        include_finals = rebuild_scope == "all"
        files = sorted(path for path in self.knowledge_dir.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS)
        statuses: list[FileIngestStatus] = []

        if rebuild_scope != "silver_only":
            for path in files:
                try:
                    status = self.ingest_file(
                        path,
                        force_full_rebuild=force_full_rebuild,
                        include_finals=include_finals,
                    )
                except Exception as exc:
                    status = FileIngestStatus(
                        file_path=str(path.resolve()),
                        status="failed",
                        message=str(exc),
                    )
                if status.status != "skipped" or include_finals or "final" not in path.name.lower():
                    statuses.append(status)

        if include_silver:
            statuses.extend(self.ingest_teacher_notes(force_full_rebuild=force_full_rebuild))

        indexed_files = sum(1 for item in statuses if item.status == "indexed")
        skipped_files = sum(1 for item in statuses if item.status == "skipped")
        failed_files = sum(1 for item in statuses if item.status == "failed")
        return IngestResponse(
            processed_files=len(statuses),
            indexed_files=indexed_files,
            skipped_files=skipped_files,
            failed_files=failed_files,
            total_units=self.count_units(),
            by_collection=self.collection_counts(),
            statuses=statuses,
        )

    def ingest_file(
        self,
        path: Path,
        *,
        force_full_rebuild: bool = False,
        include_finals: bool = False,
    ) -> FileIngestStatus:
        source_family = self._collection_for_path(path)
        file_path = str(path.resolve())
        file_hash = self._hash_file(path)

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT file_hash, embed_model FROM documents WHERE file_path = ?",
                (file_path,),
            ).fetchone()

        if (
            existing
            and existing["file_hash"] == file_hash
            and existing["embed_model"] == EMBED_MODEL
            and not force_full_rebuild
        ):
            return FileIngestStatus(
                file_path=file_path,
                status="skipped",
                collection=source_family,
                chapter=self._chapter_for_path(path),
                message="No file changes detected.",
            )

        units = build_document_units(path, include_finals=include_finals)
        if not units:
            return FileIngestStatus(
                file_path=file_path,
                status="skipped",
                collection=source_family,
                chapter=self._chapter_for_path(path),
                message="No indexable units produced for this file.",
            )

        embeddings = self._embed_texts([unit.text for unit in units])
        self._replace_units(file_path, source_family, units, embeddings, file_hash)
        sample = units[0]
        return FileIngestStatus(
            file_path=file_path,
            status="indexed",
            collection=source_family,
            chapter=sample.chapter,
            units=len(units),
            message="Indexed successfully.",
        )

    def ingest_teacher_notes(self, *, force_full_rebuild: bool = False) -> list[FileIngestStatus]:
        statuses: list[FileIngestStatus] = []
        notes = load_teacher_notes(include_silver=True)
        if not notes:
            return statuses

        grouped: dict[str, list[DocumentUnit]] = defaultdict(list)
        for note in notes:
            grouped[note.file_path].append(note)

        for file_path, units in grouped.items():
            path = Path(file_path)
            file_hash = self._hash_bytes(json.dumps([unit.model_dump() for unit in units], sort_keys=True).encode("utf-8"))

            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT file_hash, embed_model FROM documents WHERE file_path = ?",
                    (file_path,),
                ).fetchone()

            if (
                existing
                and existing["file_hash"] == file_hash
                and existing["embed_model"] == EMBED_MODEL
                and not force_full_rebuild
            ):
                collection_name = units[0].source_family
                statuses.append(
                    FileIngestStatus(
                        file_path=file_path,
                        status="skipped",
                        collection=collection_name,
                        chapter=units[0].chapter,
                        message=f"No {collection_name} changes detected.",
                    )
                )
                continue

            embeddings = self._embed_texts([unit.text for unit in units])
            collection_name = units[0].source_family
            self._replace_units(file_path, collection_name, units, embeddings, file_hash, source_type="json")
            statuses.append(
                FileIngestStatus(
                    file_path=file_path,
                    status="indexed",
                    collection=collection_name,
                    chapter=units[0].chapter,
                    units=len(units),
                    message=f"Indexed {collection_name}.",
                )
            )
        return statuses

    def search(
        self,
        query: str,
        *,
        top_k: int = TOP_K,
        source_families: list[str] | None = None,
        chapter: str | None = None,
        problem_id: str | None = None,
        prefer_pair_keys: list[str] | None = None,
    ) -> list[RetrievalHit]:
        source_families = source_families or CORE_COLLECTIONS
        prefer_pair_keys = prefer_pair_keys or []
        query_embedding = self._embed_texts([query])[0]
        dense_hits = self._dense_candidates(query_embedding, source_families, chapter=chapter, problem_id=problem_id)
        lexical_hits = self._lexical_candidates(query, source_families, chapter=chapter, problem_id=problem_id)

        merged: dict[str, RetrievalHit] = {}
        for point_id, dense_score in dense_hits.items():
            unit = self.get_unit(point_id)
            if unit is None:
                continue
            merged[point_id] = RetrievalHit(unit=unit, score=dense_score, dense_score=dense_score)

        for point_id, lexical_score in lexical_hits.items():
            unit = merged.get(point_id).unit if point_id in merged else self.get_unit(point_id)
            if unit is None:
                continue
            hit = merged.setdefault(point_id, RetrievalHit(unit=unit, score=0.0))
            hit.lexical_score = lexical_score

        results: list[RetrievalHit] = []
        for hit in merged.values():
            quality_weight = 1.0 if hit.unit.source_quality == "gold" else 0.94
            family_weight = {
                "theory_gold": 1.08,
                "problem_gold": 1.05,
                "solution_gold": 0.98,
                "notes_silver": 0.96,
                "final_solution_silver": 1.06,
            }.get(hit.unit.source_family, 1.0)
            pair_bonus = PAIR_BOOST if hit.unit.pair_key and hit.unit.pair_key in prefer_pair_keys else 0.0
            hit.score = (0.78 * hit.dense_score) + (0.22 * hit.lexical_score)
            hit.score = round(hit.score * quality_weight * family_weight + pair_bonus, 6)
            results.append(hit)

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def get_unit(self, point_id: str) -> DocumentUnit | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT point_id, file_path, collection_name, chapter, problem_id, pair_key, title,
                       source_quality, page_or_slide, chunk_index, text, excerpt, metadata_json
                FROM units
                WHERE point_id = ?
                """,
                (point_id,),
            ).fetchone()
        return self._row_to_unit(row) if row else None

    def sample_eval_cases(self, *, chapter: str | None = None, limit_per_chapter: int = 3) -> list[EvalCase]:
        query = """
            SELECT point_id, chapter, text, pair_key
            FROM units
            WHERE collection_name = 'problem_gold'
        """
        params: list[object] = []
        if chapter:
            query += " AND chapter = ?"
            params.append(chapter)
        query += " ORDER BY chapter, problem_id, chunk_index"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        counts: Counter[str] = Counter()
        cases: list[EvalCase] = []
        for row in rows:
            chapter_value = row["chapter"] or "unknown"
            if counts[chapter_value] >= limit_per_chapter:
                continue
            counts[chapter_value] += 1
            cases.append(
                EvalCase(
                    case_id=row["point_id"],
                    question=row["text"][:2000],
                    chapter=row["chapter"],
                    expected_pair_key=row["pair_key"],
                )
            )
        return cases

    def get_problem_pair_records(
        self,
        *,
        chapter: str,
        limit: int | None = None,
        problem_ids: list[str] | None = None,
    ) -> list[ProblemPairRecord]:
        problem_ids = problem_ids or []
        with self._connect() as conn:
            problem_rows = conn.execute(
                """
                SELECT problem_id, pair_key, title, text, file_path, page_or_slide, chunk_index
                FROM units
                WHERE chapter = ? AND collection_name = 'problem_gold'
                ORDER BY problem_id, page_or_slide, chunk_index
                """,
                (chapter,),
            ).fetchall()
            solution_rows = conn.execute(
                """
                SELECT problem_id, pair_key, title, text, file_path, page_or_slide, chunk_index
                FROM units
                WHERE chapter = ? AND collection_name = 'solution_gold'
                ORDER BY problem_id, page_or_slide, chunk_index
                """,
                (chapter,),
            ).fetchall()

        problem_map = collapse_problem_units(problem_rows)
        solution_map = collapse_problem_units(solution_rows)
        candidate_ids = sorted(set(problem_map) & set(solution_map), key=natural_problem_sort_key)
        if problem_ids:
            allowed = set(problem_ids)
            candidate_ids = [problem_id for problem_id in candidate_ids if problem_id in allowed]
        if limit is not None:
            candidate_ids = candidate_ids[:limit]

        records: list[ProblemPairRecord] = []
        for problem_id in candidate_ids:
            problem_row = problem_map[problem_id]
            solution_row = solution_map[problem_id]
            records.append(
                ProblemPairRecord(
                    chapter=chapter,
                    problem_id=problem_id,
                    pair_key=problem_row["pair_key"] or solution_row["pair_key"],
                    problem_title=problem_row["title"],
                    problem_text=problem_row["text"],
                    official_solution_text=solution_row["text"],
                    problem_source_path=problem_row["file_path"],
                    solution_source_path=solution_row["file_path"],
                    problem_page_or_slide=problem_row["page_or_slide"],
                    solution_page_or_slide=solution_row["page_or_slide"],
                )
            )
        return records

    def count_units(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM units").fetchone()
        return int(row["count"]) if row else 0

    def collection_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT collection_name, COUNT(*) AS count FROM units GROUP BY collection_name"
            ).fetchall()
        return {row["collection_name"]: int(row["count"]) for row in rows}

    def indexed_collections(self) -> list[str]:
        return sorted(self.collection_counts())

    def qdrant_reachable(self) -> bool:
        if self.qdrant is None:
            return False
        try:
            self.qdrant.get_collections()
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self.qdrant is None:
            return
        try:
            self.qdrant.close()
        except Exception:
            pass

    def _replace_units(
        self,
        file_path: str,
        collection_name: str,
        units: list[DocumentUnit],
        embeddings: list[list[float]],
        file_hash: str,
        *,
        source_type: str | None = None,
    ) -> None:
        if self.qdrant is None:
            raise RuntimeError("Vector store is disabled for this KnowledgeStore instance.")
        self._delete_existing_points(file_path)
        self._ensure_collection(collection_name, len(embeddings[0]))

        points = []
        for unit, embedding in zip(units, embeddings):
            payload = {
                "file_path": unit.file_path,
                "source_family": unit.source_family,
                "source_quality": unit.source_quality,
                "chapter": unit.chapter,
                "problem_id": unit.problem_id,
                "pair_key": unit.pair_key,
                "title": unit.title,
                "page_or_slide": unit.page_or_slide,
                "chunk_index": unit.chunk_index,
                "text": unit.text,
                "excerpt": unit.excerpt,
                "metadata": unit.metadata,
                "is_answer_like": unit.is_answer_like,
            }
            points.append(qmodels.PointStruct(id=unit.point_id, vector=embedding, payload=payload))

        self.qdrant.upsert(collection_name=collection_name, points=points)

        with self._connect() as conn:
            conn.execute("DELETE FROM units WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM documents WHERE file_path = ?", (file_path,))
            sample = units[0]
            conn.execute(
                """
                INSERT INTO documents (file_path, file_hash, source_type, source_family, chapter, embed_model, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    file_path,
                    file_hash,
                    source_type or sample.source_type,
                    collection_name,
                    sample.chapter,
                    EMBED_MODEL,
                ),
            )
            conn.executemany(
                """
                INSERT INTO units (
                    point_id, file_path, collection_name, chapter, problem_id, pair_key, title, source_quality,
                    page_or_slide, chunk_index, text, excerpt, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        unit.point_id,
                        unit.file_path,
                        collection_name,
                        unit.chapter,
                        unit.problem_id,
                        unit.pair_key,
                        unit.title,
                        unit.source_quality,
                        unit.page_or_slide,
                        unit.chunk_index,
                        unit.text,
                        unit.excerpt,
                        json.dumps(unit.metadata),
                    )
                    for unit in units
                ],
            )

    def _delete_existing_points(self, file_path: str) -> None:
        if self.qdrant is None:
            raise RuntimeError("Vector store is disabled for this KnowledgeStore instance.")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT point_id, collection_name FROM units WHERE file_path = ?",
                (file_path,),
            ).fetchall()
        ids_by_collection: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            ids_by_collection[row["collection_name"]].append(row["point_id"])
        for collection_name, point_ids in ids_by_collection.items():
            if not point_ids:
                continue
            self.qdrant.delete(
                collection_name=collection_name,
                points_selector=qmodels.PointIdsList(points=point_ids),
            )

    def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if self.qdrant is None:
            raise RuntimeError("Vector store is disabled for this KnowledgeStore instance.")
        collections = {item.name for item in self.qdrant.get_collections().collections}
        if collection_name in collections:
            current = self.qdrant.get_collection(collection_name=collection_name)
            vectors = current.config.params.vectors
            current_size = getattr(vectors, "size", None)
            if current_size == vector_size:
                return
            self.qdrant.delete_collection(collection_name=collection_name)
        self.qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

    def _dense_candidates(
        self,
        query_embedding: list[float],
        source_families: list[str],
        *,
        chapter: str | None = None,
        problem_id: str | None = None,
    ) -> dict[str, float]:
        if self.qdrant is None:
            raise RuntimeError("Vector store is disabled for this KnowledgeStore instance.")
        scores: dict[str, float] = {}
        for collection_name in source_families:
            if collection_name not in self.indexed_collections():
                continue
            query_filter = self._build_filter(chapter=chapter, problem_id=problem_id)
            response = self.qdrant.query_points(
                collection_name=collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=DENSE_CANDIDATES,
                with_payload=False,
            )
            for hit in response.points:
                scores[str(hit.id)] = max(scores.get(str(hit.id), 0.0), float(hit.score))
        return scores

    def _lexical_candidates(
        self,
        query: str,
        source_families: list[str],
        *,
        chapter: str | None = None,
        problem_id: str | None = None,
    ) -> dict[str, float]:
        tokens = tokenize(query)
        if not tokens:
            return {}
        placeholders = ",".join("?" for _ in source_families)
        sql = f"""
            SELECT point_id, text, title
            FROM units
            WHERE collection_name IN ({placeholders})
        """
        params: list[object] = list(source_families)
        if chapter:
            sql += " AND chapter = ?"
            params.append(chapter)
        if problem_id:
            sql += " AND problem_id = ?"
            params.append(problem_id)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored: list[tuple[str, float]] = []
        for row in rows:
            haystack = f"{row['title'] or ''} {row['text']}".lower()
            overlap = keyword_overlap(tokens, haystack)
            if overlap > 0:
                scored.append((row["point_id"], overlap))
        scored.sort(key=lambda item: item[1], reverse=True)
        return {point_id: score for point_id, score in scored[:LEXICAL_CANDIDATES]}

    def _build_filter(self, *, chapter: str | None = None, problem_id: str | None = None) -> qmodels.Filter | None:
        conditions: list[qmodels.FieldCondition] = []
        if chapter:
            conditions.append(qmodels.FieldCondition(key="chapter", match=qmodels.MatchValue(value=chapter)))
        if problem_id:
            conditions.append(qmodels.FieldCondition(key="problem_id", match=qmodels.MatchValue(value=problem_id)))
        if not conditions:
            return None
        return qmodels.Filter(must=conditions)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.ollama.embeddings(texts, model=EMBED_MODEL)
        except Exception:
            return self.ollama.embeddings(texts, model=EMBED_FALLBACK_MODEL)

    @staticmethod
    def _row_to_unit(row: sqlite3.Row) -> DocumentUnit:
        metadata = json.loads(row["metadata_json"])
        return DocumentUnit(
            point_id=row["point_id"],
            file_path=row["file_path"],
            source_type=Path(row["file_path"]).suffix.lower().lstrip(".") or "txt",
            source_family=row["collection_name"],
            source_quality=row["source_quality"],
            chapter=row["chapter"],
            problem_id=row["problem_id"],
            pair_key=row["pair_key"],
            title=row["title"],
            page_or_slide=row["page_or_slide"],
            chunk_index=row["chunk_index"],
            is_answer_like=row["collection_name"] in {"solution_gold", "final_solution_silver"},
            text=row["text"],
            excerpt=row["excerpt"],
            metadata=metadata,
        )

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _hash_bytes(payload: bytes) -> str:
        return hashlib.sha1(payload).hexdigest()

    @staticmethod
    def _chapter_for_path(path: Path) -> str | None:
        stem = path.stem.lower()
        for token in stem.split():
            if token.startswith("chapter") and token != "chapter":
                return token
        return None

    @staticmethod
    def _collection_for_path(path: Path) -> str:
        name = path.name.lower()
        if "solution" in name:
            return "solution_gold"
        if "problem" in name:
            return "problem_gold"
        if "final" in name:
            return "final_exam"
        return "theory_gold"


def tokenize(text: str) -> list[str]:
    return [token for token in re_split(text.lower()) if len(token) > 2]


def keyword_overlap(query_tokens: list[str], haystack: str) -> float:
    if not query_tokens or not haystack:
        return 0.0
    haystack_tokens = set(tokenize(haystack))
    overlap = len(set(query_tokens) & haystack_tokens)
    return overlap / max(len(set(query_tokens)), 1)


def re_split(text: str) -> list[str]:
    token = []
    tokens: list[str] = []
    for char in text:
        if char.isalnum() or char in {"_", "."}:
            token.append(char)
            continue
        if token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return tokens


def natural_problem_sort_key(problem_id: str) -> tuple[int, int]:
    try:
        left, right = problem_id.split(".", maxsplit=1)
        return int(left), int(right)
    except Exception:
        return (10**9, 10**9)


def collapse_problem_units(rows: list[sqlite3.Row]) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        problem_id = row["problem_id"]
        if not problem_id:
            continue
        item = grouped.setdefault(
            problem_id,
            {
                "pair_key": row["pair_key"],
                "title": row["title"],
                "file_path": row["file_path"],
                "page_or_slide": row["page_or_slide"],
                "parts": [],
            },
        )
        item["parts"].append(row["text"])
    collapsed: dict[str, dict[str, object]] = {}
    for problem_id, item in grouped.items():
        collapsed[problem_id] = {
            "pair_key": item["pair_key"],
            "title": item["title"],
            "file_path": item["file_path"],
            "page_or_slide": item["page_or_slide"],
            "text": "\n".join(part.strip() for part in item["parts"] if str(part).strip()),
        }
    return collapsed
