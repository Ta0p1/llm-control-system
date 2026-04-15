from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import KNOWLEDGE_DIR, STATIC_DIR
from app.runtime import build_runtime_profile
from app.schemas import (
    ChatRequest,
    ChatResponse,
    EvalCaseResult,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from app.solver import ControlSystemAssistant

app = FastAPI(title="Offline Control System Assistant", version="0.2.0")

assistant = ControlSystemAssistant()
knowledge_store = assistant.store


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    runtime = build_runtime_profile()
    return HealthResponse(
        status="ok",
        knowledge_dir=str(KNOWLEDGE_DIR.resolve()),
        ollama_reachable=runtime.ollama_reachable,
        qdrant_reachable=knowledge_store.qdrant_reachable(),
        installed_models=runtime.installed_models,
        recommended_model=runtime.recommended_model,
        indexed_collections=knowledge_store.indexed_collections(),
        total_units=knowledge_store.count_units(),
        runtime_profile=runtime.model_dump(),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    if not KNOWLEDGE_DIR.exists():
        raise HTTPException(status_code=400, detail=f"Knowledge directory not found: {KNOWLEDGE_DIR}")
    try:
        return knowledge_store.ingest_directory(
            force_full_rebuild=request.force_full_rebuild,
            rebuild_scope=request.rebuild_scope,
            include_silver=request.include_silver,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if knowledge_store.count_units() == 0:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Run /ingest first.")
    try:
        return assistant.answer(
            request.message,
            request.session_id,
            preferred_language=request.preferred_language,
            mode=request.mode,
            images=request.images,
            image_names=request.image_names,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/eval/run", response_model=EvalResponse)
def run_eval(request: EvalRequest) -> EvalResponse:
    if knowledge_store.count_units() == 0:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Run /ingest first.")

    cases = knowledge_store.sample_eval_cases(chapter=request.chapter, limit_per_chapter=request.limit_per_chapter)
    results: list[EvalCaseResult] = []
    for case in cases:
        response = assistant.answer(
            case.question,
            session_id=f"eval:{case.case_id}",
            preferred_language="english",
            mode="learning" if request.use_solution_verification else "practice",
        )
        top_citation = response.citations[0].source_path if response.citations else None
        results.append(
            EvalCaseResult(
                case_id=case.case_id,
                chapter=case.chapter,
                confidence=response.confidence,
                verification_used=response.verification_used,
                top_citation=top_citation,
                answer_preview=response.answer[:300],
            )
        )

    average_confidence = round(
        sum(result.confidence for result in results) / len(results),
        4,
    ) if results else 0.0
    return EvalResponse(total_cases=len(results), average_confidence=average_confidence, cases=results)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


static_path = Path(STATIC_DIR)
app.mount("/static", StaticFiles(directory=static_path), name="static")
