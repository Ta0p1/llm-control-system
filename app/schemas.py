from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RuntimeProfile(BaseModel):
    gpu_name: str = "unknown"
    total_vram_mb: int = 0
    installed_models: list[str] = Field(default_factory=list)
    recommended_model: str
    ollama_reachable: bool = False


class HealthResponse(BaseModel):
    status: str
    knowledge_dir: str
    ollama_reachable: bool
    qdrant_reachable: bool
    installed_models: list[str]
    recommended_model: str
    indexed_collections: list[str] = Field(default_factory=list)
    total_units: int = 0
    runtime_profile: dict[str, Any]


class IngestRequest(BaseModel):
    force_full_rebuild: bool = False
    rebuild_scope: Literal["core", "all", "silver_only"] = "core"
    stages: list[str] = Field(default_factory=lambda: ["parse", "index"])
    include_silver: bool = True


class FileIngestStatus(BaseModel):
    file_path: str
    status: Literal["indexed", "skipped", "failed"]
    collection: str = ""
    chapter: str | None = None
    stage: str = "index"
    units: int = 0
    message: str = ""


class IngestResponse(BaseModel):
    processed_files: int
    indexed_files: int
    skipped_files: int
    failed_files: int
    total_units: int = 0
    by_collection: dict[str, int] = Field(default_factory=dict)
    statuses: list[FileIngestStatus]


class Citation(BaseModel):
    source_path: str
    source_family: str
    chapter: str | None = None
    problem_id: str | None = None
    page_or_slide: int | None = None
    score: float
    excerpt: str = ""


class RetrievalPlan(BaseModel):
    mode: str = "learning"
    query_text: str
    primary_collections: list[str] = Field(default_factory=list)
    verification_collections: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    stages: list[str] = Field(default_factory=list)


class EvidenceGroup(BaseModel):
    label: str
    source_family: str
    hit_count: int


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = "default"
    preferred_language: str = "english"
    mode: Literal["learning", "practice", "concept"] = "learning"
    images: list[str] = Field(default_factory=list)
    image_names: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    steps: list[str]
    citations: list[Citation]
    used_tools: list[str]
    confidence: float
    model_name: str
    retrieval_plan: RetrievalPlan
    verification_used: bool = False
    evidence_groups: list[EvidenceGroup] = Field(default_factory=list)


class DocumentUnit(BaseModel):
    point_id: str
    file_path: str
    source_type: str
    source_family: str
    source_quality: Literal["gold", "silver"] = "gold"
    chapter: str | None = None
    problem_id: str | None = None
    pair_key: str | None = None
    title: str | None = None
    unit_kind: str = "text"
    page_or_slide: int | None = None
    chunk_index: int = 0
    is_answer_like: bool = False
    text: str
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    unit: DocumentUnit
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


class SolvePlan(BaseModel):
    mode: Literal["concept", "solve"] = "solve"
    problem_restatement: str = ""
    knowns: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    method: str = ""
    solution_outline: list[str] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)
    tool_requests: list[dict[str, Any]] = Field(default_factory=list)


class VerificationResult(BaseModel):
    used: bool = False
    summary: str = ""
    supporting_pair_keys: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    case_id: str
    question: str
    chapter: str | None = None
    expected_pair_key: str | None = None


class EvalRequest(BaseModel):
    chapter: str | None = None
    limit_per_chapter: int = 3
    use_solution_verification: bool = True


class EvalCaseResult(BaseModel):
    case_id: str
    chapter: str | None = None
    confidence: float
    verification_used: bool
    top_citation: str | None = None
    answer_preview: str


class EvalResponse(BaseModel):
    total_cases: int
    average_confidence: float
    cases: list[EvalCaseResult] = Field(default_factory=list)


class TeacherNoteRecord(BaseModel):
    note_id: str
    chapter: str | None = None
    pair_key: str | None = None
    note_type: Literal["concept_card", "formula_card", "method_card", "pitfall_card"] = "concept_card"
    title: str
    text: str
    teacher_model: str
    verification_status: Literal["unverified", "verified"] = "unverified"
    metadata: dict[str, Any] = Field(default_factory=dict)
