from __future__ import annotations

import json
from pathlib import Path

from app.config import SILVER_NOTES_DIR
from app.schemas import DocumentUnit, TeacherNoteRecord


PRIMARY_TEACHER_MODEL = "gpt-5.2"
REVIEW_TEACHER_MODEL = "claude-opus-4.1"
ARBITER_TEACHER_MODEL = "gemini-2.5-pro"


def build_primary_teacher_prompt(unit: DocumentUnit) -> str:
    return f"""
You are converting official control-systems course material into structured study notes.
Create only derived study support, not a new authoritative solution.

Source family: {unit.source_family}
Chapter: {unit.chapter or "unknown"}
Pair key: {unit.pair_key or "none"}
Title: {unit.title or "untitled"}

Source text:
{unit.text}

Return JSON only as an array of note records with these fields:
- note_id
- chapter
- pair_key
- note_type (concept_card | formula_card | method_card | pitfall_card)
- title
- text
- teacher_model
- verification_status
- metadata
"""


def build_review_teacher_prompt(source_unit: DocumentUnit, draft_notes: list[TeacherNoteRecord]) -> str:
    return f"""
Review the draft silver notes against the original official material.
Mark each note as verified only if it is faithful to the source and does not introduce unsupported claims.

Original source:
{source_unit.text}

Draft notes:
{json.dumps([note.model_dump() for note in draft_notes], ensure_ascii=False, indent=2)}

Return JSON only as the corrected array of note records.
"""


def silver_note_path(chapter: str | None, stem: str) -> Path:
    safe_chapter = (chapter or "general").replace("/", "_")
    safe_stem = stem.replace("/", "_").replace("\\", "_")
    return SILVER_NOTES_DIR / f"{safe_chapter}_{safe_stem}.json"
