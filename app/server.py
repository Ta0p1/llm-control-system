from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import KNOWLEDGE_DIR, OPENAI_MODEL, STATIC_DIR
from app.openai_client import OpenAIClient
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
    SessionHistoryResponse,
)
from app.session_store import SessionStore
from app.solver import ControlSystemAssistant, PipelineExecutionError

app = FastAPI(title="Offline Control System Assistant", version="0.3.0")

assistant = ControlSystemAssistant()
knowledge_store = assistant.store
session_store = SessionStore()
openai_client = OpenAIClient()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    runtime = build_runtime_profile(refresh=True)
    openai_configured, _ = openai_client.available()
    available_answer_modes = ["local"]
    if openai_configured:
        available_answer_modes.append("gpt")
    return HealthResponse(
        status="ok",
        knowledge_dir=str(KNOWLEDGE_DIR.resolve()),
        ollama_reachable=runtime.ollama_reachable,
        qdrant_reachable=knowledge_store.qdrant_reachable(),
        openai_configured=openai_configured,
        openai_model=OPENAI_MODEL if openai_configured else "",
        available_answer_modes=available_answer_modes,
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
        response = assistant.answer(
            request.message,
            request.session_id,
            preferred_language=request.preferred_language,
            mode=request.mode,
            answer_mode=request.answer_mode,
            images=request.images,
            image_names=request.image_names,
        )
        stage_summary = ", ".join(
            f"{name}={duration}ms" for name, duration in response.timing.stage_timings.items()
        )
        model_summary = ", ".join(
            f"{name}={duration}ms" for name, duration in response.timing.model_calls.items()
        )
        print(
            "[chat-timing] "
            f"session={request.session_id} "
            f"mode={request.mode} "
            f"answer_mode={response.answer_mode} "
            f"images={len(request.images)} "
            f"model={response.model_name} "
            f"total_ms={response.timing.total_duration_ms} "
            f"path={response.timing.metadata.get('path_type', 'unknown')} "
            f"verification={response.verification_used} "
            f"review={response.timing.metadata.get('review_used', False)} "
            f"primary_hits={response.timing.metadata.get('primary_hit_count', 0)} "
            f"verification_hits={response.timing.metadata.get('verification_hit_count', 0)} "
            f"tools={response.timing.metadata.get('tool_count', 0)} "
            f"context_chars={response.timing.metadata.get('compressed_context_chars', 0)} "
            f"stages=[{stage_summary}] "
            f"models=[{model_summary}]"
        )
        user_message_for_history = request.message
        if request.image_names:
            user_message_for_history = (
                f"{request.message}\n"
                f"[Attached images: {', '.join(request.image_names)}]"
            )
        session_store.append_exchange(
            request.session_id,
            user_message=user_message_for_history,
            assistant_message=response.answer,
        )
        return response
    except Exception as exc:
        if isinstance(exc, PipelineExecutionError):
            stage_summary = ", ".join(
                f"{name}={duration}ms" for name, duration in exc.stage_timings.items()
            )
            model_summary = ", ".join(
                f"{name}={duration}ms" for name, duration in exc.model_calls.items()
            )
            print(
                "[chat-timing-error] "
                f"session={request.session_id} "
                f"mode={request.mode} "
                f"answer_mode={request.answer_mode} "
                f"images={len(request.images)} "
                f"failed_stage={exc.timing_metadata.get('failed_stage', 'unknown')} "
                f"path={exc.timing_metadata.get('path_type', 'unknown')} "
                f"context_chars={exc.timing_metadata.get('compressed_context_chars', 0)} "
                f"stages=[{stage_summary}] "
                f"models=[{model_summary}] "
                f"error={exc}"
            )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/session/{session_id}/history", response_model=SessionHistoryResponse)
def session_history(session_id: str) -> SessionHistoryResponse:
    return session_store.load(session_id)


@app.post("/session/{session_id}/clear", response_model=SessionHistoryResponse)
def clear_session(session_id: str) -> SessionHistoryResponse:
    return session_store.clear(session_id)


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
            answer_mode="local",
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
