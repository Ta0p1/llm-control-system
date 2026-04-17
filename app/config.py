from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
STATIC_DIR = ROOT_DIR / "static"
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "knowledge.db"
QDRANT_PATH = DATA_DIR / "qdrant"
SILVER_NOTES_DIR = DATA_DIR / "silver_notes"
TEACHER_BATCHES_DIR = DATA_DIR / "teacher_batches"
FINAL_BATCHES_DIR = TEACHER_BATCHES_DIR / "finals"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))

RUNTIME_MODEL = os.getenv("RUNTIME_MODEL", "qwen3.5:9b")
DEV_CHAT_MODEL = os.getenv("DEV_CHAT_MODEL", RUNTIME_MODEL)
TARGET_CHAT_MODEL = os.getenv("TARGET_CHAT_MODEL", RUNTIME_MODEL)
FALLBACK_CHAT_MODEL = os.getenv("FALLBACK_CHAT_MODEL", RUNTIME_MODEL)
VISION_MODEL = os.getenv("VISION_MODEL", RUNTIME_MODEL)

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:4b")
EMBED_FALLBACK_MODEL = os.getenv("EMBED_FALLBACK_MODEL", "bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "")
ENABLE_DOCLING = os.getenv("ENABLE_DOCLING", "0").lower() in {"1", "true", "yes"}

CHAT_MODEL_CANDIDATES = [RUNTIME_MODEL]
CORE_COLLECTIONS = ["theory_gold", "problem_gold", "solution_gold", "notes_silver", "final_solution_silver"]
PRIMARY_COLLECTIONS = ["theory_gold", "problem_gold", "notes_silver", "final_solution_silver"]
VERIFICATION_COLLECTIONS = ["solution_gold"]
FINAL_EXAM_FAMILY = "final_exam"
EXCLUDED_FINAL_EXAMS = {"AER372W_2025_final_test"}

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
TOP_K = int(os.getenv("TOP_K", "8"))
DENSE_CANDIDATES = int(os.getenv("DENSE_CANDIDATES", "24"))
LEXICAL_CANDIDATES = int(os.getenv("LEXICAL_CANDIDATES", "24"))
PAIR_BOOST = float(os.getenv("PAIR_BOOST", "0.08"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_PATH.mkdir(parents=True, exist_ok=True)
SILVER_NOTES_DIR.mkdir(parents=True, exist_ok=True)
TEACHER_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
FINAL_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
