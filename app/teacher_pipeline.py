from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import EXCLUDED_FINAL_EXAMS, FINAL_BATCHES_DIR, KNOWLEDGE_DIR, SILVER_NOTES_DIR, TEACHER_BATCHES_DIR
from app.extractors import extract_pdf_pages
from app.knowledge_store import KnowledgeStore
from app.schemas import (
    FinalExamBatchBundle,
    FinalExamPage,
    FinalSolutionRecord,
    ProblemPairRecord,
    TeacherBatchBundle,
    TeacherNoteRecord,
)

PRIMARY_TEACHER_MODEL = "gpt-5.4"
REVIEW_TEACHER_MODEL = "claude-opus-4.1"
ARBITER_TEACHER_MODEL = "gemini-2.5-pro"
REQUIRED_NOTE_METADATA_KEYS = {"problem_id", "source_family", "derived_from", "tags"}
REQUIRED_FINAL_METADATA_KEYS = {"source_family", "derived_from_pages", "tags"}


def build_primary_teacher_prompt(records: list[ProblemPairRecord], *, teacher_model: str = PRIMARY_TEACHER_MODEL) -> str:
    batch = TeacherBatchBundle(
        chapter=records[0].chapter if records else "unknown",
        exported_problem_ids=[record.problem_id for record in records],
        records=records,
    )
    return f"""You are converting official control-systems problem-and-solution pairs into retrieval-friendly study cards.
Do not invent any new solution content.
Return JSON only.

Use the exact schema below for every note:
- note_id
- chapter
- pair_key
- note_type (method_card | formula_card | pitfall_card)
- title
- text
- teacher_model
- verification_status
- metadata

Rules:
- Only use the provided official problem text and official solution text.
- Output English only.
- For each problem, generate exactly 3 notes:
  1. one method_card
  2. one formula_card
  3. one pitfall_card
- method_card: summarize the problem type, the standard solving procedure, and the applicability conditions.
- formula_card: include only the key formulas actually used, define variables, and state conditions.
- pitfall_card: include 2 to 4 mistakes strongly tied to this problem or its solution path.
- Do not restate the full official solution.
- Keep each note concise and retrieval-friendly.
- Set teacher_model to "{teacher_model}".
- Set verification_status to "unverified".
- metadata must include:
  - problem_id
  - source_family
  - derived_from
  - tags

Input bundle:
{batch.model_dump_json(indent=2)}
"""


def build_final_teacher_prompt(bundle: FinalExamBatchBundle, *, teacher_model: str = PRIMARY_TEACHER_MODEL) -> str:
    return f"""You are converting control-systems final exam questions into complete worked-solution records.
Do not invent missing question text when the exam image is unclear. If needed, say that a symbol or clause is unclear.
Return JSON only.

Use the exact schema below for every solved question:
- solution_id
- exam_id
- question_id
- title
- question_text
- full_solution
- final_answer
- key_formulas
- method_tags
- teacher_model
- verification_status
- metadata

Rules:
- Output English only.
- Use the attached exam page images when available. They are the primary source if extracted text is noisy.
- Create one record per main question or clearly labeled sub-question if the exam separates them meaningfully.
- Use concise IDs such as Q1, Q2, Q3-a, Q4-c.
- full_solution must include the full derivation or reasoning steps.
- final_answer must be short and direct.
- key_formulas must be an array of the most important formulas actually used.
- method_tags must be a short array such as ["root_locus", "routh", "phase_margin"].
- Set teacher_model to "{teacher_model}".
- Set verification_status to "unverified".
- metadata must include:
  - source_family: "final_solution_silver"
  - derived_from_pages: array of page numbers used
  - tags: array of relevant topic tags
- Do not output markdown fences or commentary.

Exam source bundle:
{bundle.model_dump_json(indent=2)}
"""


def chapter_batch_dir(chapter: str) -> Path:
    target = TEACHER_BATCHES_DIR / chapter
    target.mkdir(parents=True, exist_ok=True)
    return target


def final_batch_dir(exam_id: str) -> Path:
    target = FINAL_BATCHES_DIR / exam_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def silver_note_path(chapter: str | None, stem: str) -> Path:
    safe_chapter = (chapter or "general").replace("/", "_")
    safe_stem = stem.replace("/", "_").replace("\\", "_")
    return SILVER_NOTES_DIR / f"{safe_chapter}_{safe_stem}.json"


def final_solution_path(exam_id: str) -> Path:
    safe_exam = sanitize_identifier(exam_id)
    return SILVER_NOTES_DIR / f"final_{safe_exam}_solutions.json"


def export_problem_pair_batch(
    *,
    chapter: str,
    limit: int | None = None,
    problem_ids: list[str] | None = None,
    teacher_model: str = PRIMARY_TEACHER_MODEL,
) -> dict[str, Path]:
    store = KnowledgeStore(enable_vector_store=False)
    try:
        records = store.get_problem_pair_records(chapter=chapter, limit=limit, problem_ids=problem_ids)
    finally:
        store.close()
    if not records:
        raise RuntimeError(f"No matched problem-solution pairs found for {chapter}.")

    batch_dir = chapter_batch_dir(chapter)
    source_path = batch_dir / f"{chapter}_problem_cards_source.json"
    prompt_path = batch_dir / f"{chapter}_chatgpt_prompt.txt"
    instructions_path = batch_dir / f"{chapter}_how_to_use.txt"

    bundle = TeacherBatchBundle(
        chapter=chapter,
        exported_problem_ids=[record.problem_id for record in records],
        records=records,
    )
    source_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    prompt_path.write_text(build_primary_teacher_prompt(records, teacher_model=teacher_model), encoding="utf-8")
    instructions_path.write_text(build_operator_instructions(chapter, records), encoding="utf-8")
    return {
        "source": source_path,
        "prompt": prompt_path,
        "instructions": instructions_path,
    }


def export_final_exam_batch(*, exam_id: str, teacher_model: str = PRIMARY_TEACHER_MODEL) -> dict[str, Path]:
    exam_path = resolve_final_exam_path(exam_id)
    if exam_path.stem in EXCLUDED_FINAL_EXAMS:
        raise RuntimeError(f"{exam_path.name} is explicitly excluded from the final-solution workflow.")

    batch_dir = final_batch_dir(exam_path.stem)
    source_path = batch_dir / f"{exam_path.stem}_source.json"
    prompt_path = batch_dir / f"{exam_path.stem}_chatgpt_prompt.txt"
    instructions_path = batch_dir / f"{exam_path.stem}_how_to_use.txt"
    bundle = build_final_exam_bundle(exam_path, batch_dir)

    source_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    prompt_path.write_text(build_final_teacher_prompt(bundle, teacher_model=teacher_model), encoding="utf-8")
    instructions_path.write_text(build_final_operator_instructions(bundle), encoding="utf-8")
    return {
        "source": source_path,
        "prompt": prompt_path,
        "instructions": instructions_path,
    }


def resolve_final_exam_path(exam_id: str) -> Path:
    exam_id = exam_id.strip()
    candidates = sorted(KNOWLEDGE_DIR.glob("*_final_test.pdf"))
    for path in candidates:
        if path.stem == exam_id or path.name == exam_id:
            return path
    available = ", ".join(path.stem for path in candidates)
    raise RuntimeError(f"Unknown final exam '{exam_id}'. Available exams: {available}")


def build_final_exam_bundle(exam_path: Path, batch_dir: Path) -> FinalExamBatchBundle:
    extracted_pages = {page_number: text for page_number, text in extract_pdf_pages(exam_path)}
    images_dir = batch_dir / "page_assets"
    images_dir.mkdir(parents=True, exist_ok=True)
    page_images = export_pdf_page_images(exam_path, images_dir)
    all_page_numbers = sorted(set(extracted_pages) | set(page_images))
    if not all_page_numbers:
        all_page_numbers = [1]
    pages = [
        FinalExamPage(
            exam_id=exam_path.stem,
            page_number=page_number,
            extracted_text=extracted_pages.get(page_number, ""),
            image_paths=page_images.get(page_number, []),
        )
        for page_number in all_page_numbers
    ]
    return FinalExamBatchBundle(
        exam_id=exam_path.stem,
        source_path=str(exam_path.resolve()),
        pages=pages,
    )


def export_pdf_page_images(exam_path: Path, output_dir: Path) -> dict[int, list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return {}

    reader = PdfReader(str(exam_path))
    exported: dict[int, list[str]] = {}
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            images = list(page.images)
        except Exception:
            continue
        if not images:
            continue
        try:
            largest = max(images, key=lambda item: len(item.data))
            extension = Path(largest.name).suffix or ".png"
            image_path = output_dir / f"page_{page_number}{extension}"
            image_path.write_bytes(largest.data)
            exported[page_number] = [str(image_path.resolve())]
        except Exception:
            continue
    return exported


def load_notes_payload(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig").strip()
    if raw.startswith("```"):
        stripped = raw.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        raw = stripped
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise RuntimeError("Teacher payload must be a JSON array or object.")
    return payload


def detect_payload_kind(payload: list[dict[str, Any]]) -> str:
    if not payload:
        raise RuntimeError("Teacher payload is empty.")
    first = payload[0]
    if {"question_id", "full_solution", "final_answer", "exam_id"}.issubset(first):
        return "final_solution"
    if {"note_type", "text", "title"}.issubset(first):
        return "note"
    raise RuntimeError("Unsupported payload schema.")


def normalize_teacher_notes(
    payload: list[dict[str, Any]],
    *,
    chapter: str | None = None,
    teacher_model: str | None = None,
) -> list[TeacherNoteRecord]:
    normalized: list[TeacherNoteRecord] = []
    for item in payload:
        record = dict(item)
        if chapter and not record.get("chapter"):
            record["chapter"] = chapter
        if teacher_model and not record.get("teacher_model"):
            record["teacher_model"] = teacher_model
        if not record.get("verification_status"):
            record["verification_status"] = "unverified"
        metadata = dict(record.get("metadata") or {})
        problem_id = metadata.get("problem_id") or extract_problem_id_from_pair_key(record.get("pair_key"))
        if problem_id:
            metadata.setdefault("problem_id", problem_id)
        metadata.setdefault("source_family", "notes_silver")
        if record.get("pair_key"):
            metadata.setdefault("derived_from", [record["pair_key"]])
        metadata.setdefault("tags", [])
        record["metadata"] = metadata
        normalized.append(TeacherNoteRecord.model_validate(record))
    return normalized


def normalize_final_solutions(
    payload: list[dict[str, Any]],
    *,
    exam_id: str | None = None,
    teacher_model: str | None = None,
) -> list[FinalSolutionRecord]:
    normalized: list[FinalSolutionRecord] = []
    for item in payload:
        record = dict(item)
        if exam_id and not record.get("exam_id"):
            record["exam_id"] = exam_id
        if teacher_model and not record.get("teacher_model"):
            record["teacher_model"] = teacher_model
        if not record.get("verification_status"):
            record["verification_status"] = "unverified"
        exam_value = record.get("exam_id")
        question_id = str(record.get("question_id") or "").strip()
        if not record.get("solution_id") and exam_value and question_id:
            record["solution_id"] = f"{sanitize_identifier(str(exam_value))}_{sanitize_identifier(question_id)}"
        record["key_formulas"] = ensure_string_list(record.get("key_formulas"))
        record["method_tags"] = ensure_string_list(record.get("method_tags"))
        metadata = dict(record.get("metadata") or {})
        metadata.setdefault("source_family", "final_solution_silver")
        metadata.setdefault("derived_from_pages", [])
        metadata.setdefault("tags", [])
        record["metadata"] = metadata
        normalized.append(FinalSolutionRecord.model_validate(record))
    return normalized


def validate_teacher_notes_file(
    path: Path,
    *,
    chapter: str | None = None,
    teacher_model: str | None = None,
) -> list[TeacherNoteRecord]:
    payload = load_notes_payload(path)
    records = normalize_teacher_notes(payload, chapter=chapter, teacher_model=teacher_model)
    note_ids: set[str] = set()
    for record in records:
        if record.note_id in note_ids:
            raise RuntimeError(f"Duplicate note_id detected: {record.note_id}")
        note_ids.add(record.note_id)
        if chapter and record.chapter != chapter:
            raise RuntimeError(f"Record {record.note_id} has chapter={record.chapter}, expected {chapter}")
        missing_keys = REQUIRED_NOTE_METADATA_KEYS - set(record.metadata)
        if missing_keys:
            raise RuntimeError(f"Record {record.note_id} is missing metadata keys: {sorted(missing_keys)}")
        if not isinstance(record.metadata.get("tags"), list):
            raise RuntimeError(f"Record {record.note_id} metadata.tags must be a list.")
    return records


def validate_final_solutions_file(
    path: Path,
    *,
    exam_id: str | None = None,
    teacher_model: str | None = None,
) -> list[FinalSolutionRecord]:
    payload = load_notes_payload(path)
    records = normalize_final_solutions(payload, exam_id=exam_id, teacher_model=teacher_model)
    solution_ids: set[str] = set()
    for record in records:
        if record.solution_id in solution_ids:
            raise RuntimeError(f"Duplicate solution_id detected: {record.solution_id}")
        solution_ids.add(record.solution_id)
        if exam_id and record.exam_id != exam_id:
            raise RuntimeError(f"Record {record.solution_id} has exam_id={record.exam_id}, expected {exam_id}")
        missing_keys = REQUIRED_FINAL_METADATA_KEYS - set(record.metadata)
        if missing_keys:
            raise RuntimeError(f"Record {record.solution_id} is missing metadata keys: {sorted(missing_keys)}")
        if not isinstance(record.metadata.get("derived_from_pages"), list):
            raise RuntimeError(f"Record {record.solution_id} metadata.derived_from_pages must be a list.")
        if not isinstance(record.metadata.get("tags"), list):
            raise RuntimeError(f"Record {record.solution_id} metadata.tags must be a list.")
    return records


def merge_teacher_notes_file(
    input_path: Path,
    *,
    chapter: str,
    teacher_model: str | None = None,
) -> Path:
    records = validate_teacher_notes_file(input_path, chapter=chapter, teacher_model=teacher_model)
    output_path = silver_note_path(chapter, "problem_cards")
    existing: list[TeacherNoteRecord] = []
    if output_path.exists():
        try:
            existing_payload = load_notes_payload(output_path)
            existing = normalize_teacher_notes(existing_payload, chapter=chapter)
        except Exception:
            existing = []

    merged = {record.note_id: record for record in existing}
    for record in records:
        merged[record.note_id] = record

    sorted_records = sorted(
        merged.values(),
        key=lambda item: (item.pair_key or "", note_type_rank(item.note_type), item.note_id),
    )
    output_path.write_text(
        json.dumps([record.model_dump() for record in sorted_records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def merge_final_solutions_file(
    input_path: Path,
    *,
    exam_id: str,
    teacher_model: str | None = None,
) -> Path:
    records = validate_final_solutions_file(input_path, exam_id=exam_id, teacher_model=teacher_model)
    output_path = final_solution_path(exam_id)
    existing: list[FinalSolutionRecord] = []
    if output_path.exists():
        try:
            existing_payload = load_notes_payload(output_path)
            existing = normalize_final_solutions(existing_payload, exam_id=exam_id)
        except Exception:
            existing = []

    merged = {record.solution_id: record for record in existing}
    for record in records:
        merged[record.solution_id] = record

    sorted_records = sorted(
        merged.values(),
        key=lambda item: (natural_question_sort_key(item.question_id), item.solution_id),
    )
    output_path.write_text(
        json.dumps([record.model_dump() for record in sorted_records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def build_operator_instructions(chapter: str, records: list[ProblemPairRecord]) -> str:
    ids = ", ".join(record.problem_id for record in records)
    return f"""Chapter: {chapter}
Problems in this batch: {ids}

How to use:
1. Open the prompt file in the same folder.
2. Copy the full prompt into ChatGPT.
3. Keep the response as raw JSON only. Do not ask for markdown or commentary.
4. Save the ChatGPT response to a local JSON file, for example:
   {chapter}_chatgpt_output.json
5. Validate it locally:
   python -m app.teacher_pipeline validate --input "<path to json>" --chapter {chapter}
6. Merge it into notes_silver:
   python -m app.teacher_pipeline merge --input "<path to json>" --chapter {chapter}
7. Rebuild the local index:
   python -m app.teacher_pipeline reingest
"""


def build_final_operator_instructions(bundle: FinalExamBatchBundle) -> str:
    page_lines = []
    for page in bundle.pages:
        image_hint = ", ".join(page.image_paths) if page.image_paths else "no image exported"
        page_lines.append(f"- Page {page.page_number}: {image_hint}")
    joined_pages = "\n".join(page_lines)
    return f"""Exam: {bundle.exam_id}
Source PDF: {bundle.source_path}

How to use:
1. Open the prompt file in the same folder.
2. Attach the exported page images listed below to ChatGPT when possible.
3. Paste the full prompt into ChatGPT and keep the response as raw JSON only.
4. Save the response to a local JSON file, for example:
   {bundle.exam_id}_chatgpt_output.json
5. Validate it locally:
   python -m app.teacher_pipeline validate --input "<path to json>" --exam {bundle.exam_id}
6. Merge it into final_solution_silver:
   python -m app.teacher_pipeline merge --input "<path to json>" --exam {bundle.exam_id}
7. Rebuild the local index:
   python -m app.teacher_pipeline reingest

Exported page images:
{joined_pages}
"""


def extract_problem_id_from_pair_key(pair_key: str | None) -> str | None:
    if not pair_key or "_problem_" not in pair_key:
        return None
    return pair_key.split("_problem_", maxsplit=1)[1]


def note_type_rank(note_type: str) -> int:
    return {
        "method_card": 0,
        "formula_card": 1,
        "pitfall_card": 2,
        "concept_card": 3,
    }.get(note_type, 9)


def natural_question_sort_key(question_id: str) -> tuple[int, str]:
    match = re.match(r"[Qq](\d+)(?:[-_]?([A-Za-z0-9]+))?", question_id.strip())
    if not match:
        return (10**9, question_id)
    suffix = match.group(2) or ""
    return (int(match.group(1)), suffix.lower())


def sanitize_identifier(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")


def ensure_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()]


def run_validate(path: Path, *, chapter: str | None = None, exam: str | None = None, teacher_model: str | None = None) -> str:
    payload = load_notes_payload(path)
    kind = detect_payload_kind(payload)
    if kind == "note":
        records = validate_teacher_notes_file(path, chapter=chapter, teacher_model=teacher_model)
        return f"Validated {len(records)} note record(s)."
    records = validate_final_solutions_file(path, exam_id=exam, teacher_model=teacher_model)
    return f"Validated {len(records)} final solution record(s)."


def run_merge(path: Path, *, chapter: str | None = None, exam: str | None = None, teacher_model: str | None = None) -> Path:
    payload = load_notes_payload(path)
    kind = detect_payload_kind(payload)
    if kind == "note":
        if not chapter:
            raise RuntimeError("Merging note payloads requires --chapter.")
        return merge_teacher_notes_file(path, chapter=chapter, teacher_model=teacher_model)
    if not exam:
        raise RuntimeError("Merging final solution payloads requires --exam.")
    return merge_final_solutions_file(path, exam_id=exam, teacher_model=teacher_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Utilities for ChatGPT-generated silver note workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export problem-solution source material and a ChatGPT prompt.")
    export_parser.add_argument("--chapter", required=True)
    export_parser.add_argument("--limit", type=int, default=None)
    export_parser.add_argument("--problem-ids", default="")
    export_parser.add_argument("--teacher-model", default=PRIMARY_TEACHER_MODEL)

    export_finals_parser = subparsers.add_parser("export-finals", help="Export a final exam batch and page assets.")
    export_finals_parser.add_argument("--exam", required=True)
    export_finals_parser.add_argument("--teacher-model", default=PRIMARY_TEACHER_MODEL)

    validate_parser = subparsers.add_parser("validate", help="Validate a ChatGPT JSON response before merge.")
    validate_parser.add_argument("--input", required=True)
    validate_parser.add_argument("--chapter", required=False)
    validate_parser.add_argument("--exam", required=False)
    validate_parser.add_argument("--teacher-model", default=None)

    merge_parser = subparsers.add_parser("merge", help="Merge validated ChatGPT JSON into data/silver_notes.")
    merge_parser.add_argument("--input", required=True)
    merge_parser.add_argument("--chapter", required=False)
    merge_parser.add_argument("--exam", required=False)
    merge_parser.add_argument("--teacher-model", default=None)

    reingest_parser = subparsers.add_parser("reingest", help="Re-index notes_silver and core sources locally.")
    reingest_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.command == "export":
        problem_ids = [item.strip() for item in args.problem_ids.split(",") if item.strip()]
        outputs = export_problem_pair_batch(
            chapter=args.chapter,
            limit=args.limit,
            problem_ids=problem_ids,
            teacher_model=args.teacher_model,
        )
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return

    if args.command == "export-finals":
        outputs = export_final_exam_batch(exam_id=args.exam, teacher_model=args.teacher_model)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return

    if args.command == "validate":
        try:
            message = run_validate(
                Path(args.input),
                chapter=args.chapter,
                exam=args.exam,
                teacher_model=args.teacher_model,
            )
        except (RuntimeError, ValidationError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Validation failed: {exc}") from exc
        print(message)
        return

    if args.command == "merge":
        try:
            output_path = run_merge(
                Path(args.input),
                chapter=args.chapter,
                exam=args.exam,
                teacher_model=args.teacher_model,
            )
        except (RuntimeError, ValidationError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Merge failed: {exc}") from exc
        print(str(output_path))
        return

    if args.command == "reingest":
        store = KnowledgeStore()
        result = store.ingest_directory(force_full_rebuild=args.force, rebuild_scope="core", include_silver=True)
        print(result.model_dump_json(indent=2))
        store.close()


if __name__ == "__main__":
    main()
