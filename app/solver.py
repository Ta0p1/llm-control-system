from __future__ import annotations

import base64
import io
import json
import math
import re
from time import perf_counter
from typing import Any, TypedDict

import sympy
from PIL import Image
from scipy import signal

try:
    import control as control_lib
except ImportError:
    control_lib = None

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - fallback stays available if dependency is missing.
    END = "__end__"
    StateGraph = None

from app.config import (
    ANSWER_NUM_PREDICT,
    PLAN_NUM_PREDICT,
    PRIMARY_COLLECTIONS,
    REVIEW_NUM_PREDICT,
    VERIFICATION_COLLECTIONS,
    VISION_MODEL,
)
from app.extractors import extract_problem_id
from app.knowledge_store import KnowledgeStore
from app.ollama_client import OllamaClient
from app.runtime import build_runtime_profile
from app.schemas import (
    ChatTiming,
    ChatResponse,
    Citation,
    EvidenceGroup,
    RetrievalHit,
    RetrievalPlan,
    SolvePlan,
    VerificationResult,
)

SYSTEM_PROMPT = """You are an offline control systems tutor.
Default to English.
If the user asks in another language, match the user's language.
Always ground the answer in retrieved context when available.
Never fabricate citations.
When a verification context is present, treat it as a checking layer rather than the only source of reasoning.
For solve questions, structure the answer with these sections:
Problem Restatement
Knowns / Unknowns
Theory Used
Step-by-Step Solution
Final Answer
Sources
If evidence is weak or missing, say so explicitly.
Before finalizing a solve answer, perform a global consistency check:
- stability claims must match pole real parts
- system type must match steady-state error conclusions
- time-response metrics must match the damping ratio and dominant poles used
- the final summary must not contradict earlier derivations
"""

PLANNING_SYSTEM_PROMPT = """You are an offline control systems planning assistant.
Default to English.
Return concise structured outputs only.
Do not explain at length.
Choose tools only when they clearly help verify a key quantity.
"""

IMAGE_PARSE_SYSTEM_PROMPT = """Extract control-systems problem content from images.
Return strict JSON only.
Do not guess unclear symbols; list them under unclear_items.
Keep every field concise.
"""

CONCEPT_ANSWER_PROMPT = """Write the final user-facing answer in {language} as clean Markdown.

Style rules:
- explain clearly, not like internal notes
- do not mention prompts, retrieval, verification layers, or tool metadata
- do not output JSON, YAML, or code fences unless the user explicitly asked for code
- avoid filler such as "Here is the answer" or "Based on the provided context"
- when a mathematical expression helps, write it using LaTeX delimiters like $...$ or $$...$$

Format rules:
- use `##` headings only when helpful
- keep paragraphs short
- use bullet points only when they improve readability
- include source markers like [1], [2] in the explanation when claims rely on evidence
- end with a short `## Sources` section that only lists the markers used
"""

SOLVE_ANSWER_PROMPT = """Write the final user-facing solution in {language} as clean Markdown.

Use exactly these sections in this order:
## Problem Restatement
## Knowns / Unknowns
## Theory Used
## Step-by-Step Solution
## Final Answer
## Sources

Output rules:
- this is a polished final solution, not internal reasoning notes
- keep the section order exactly as given
- be complete but not verbose
- use equations inline when short, and use $...$ or $$...$$ when a mathematical expression should be rendered clearly
- use short bullet lists when useful
- use source markers like [1], [2] where evidence supports a statement
- in `## Sources`, list only the markers actually used, one per bullet
- do not mention prompts, retrieval, verification layers, tool calls, or "provided context"
- do not output JSON, YAML, or code fences unless the user explicitly asked for code

Reasoning rules:
- use verification evidence only as a checking layer
- do not repeat the same evidence in multiple sections
- if information is insufficient, say so plainly in `## Final Answer`
"""


class PipelineExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage_timings: dict[str, float],
        model_calls: dict[str, float],
        timing_metadata: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.stage_timings = stage_timings
        self.model_calls = model_calls
        self.timing_metadata = timing_metadata


class WorkflowState(TypedDict, total=False):
    question: str
    preferred_language: str
    session_id: str
    chat_mode: str
    mode: str
    image_mode: str
    images: list[str]
    image_names: list[str]
    image_summary: str
    chapter_hint: str | None
    problem_id_hint: str | None
    primary_hits: list[RetrievalHit]
    verification_hits: list[RetrievalHit]
    retrieval_plan: RetrievalPlan
    solve_plan: SolvePlan
    tool_results: list[dict[str, Any]]
    used_tools: list[str]
    answer: str
    model_name: str
    confidence: float
    citations: list[Citation]
    evidence_groups: list[EvidenceGroup]
    verification_result: VerificationResult
    stage_timings: dict[str, float]
    model_calls: dict[str, float]
    timing_metadata: dict[str, Any]


class ControlSystemAssistant:
    def __init__(self) -> None:
        self.store = KnowledgeStore()
        self.ollama = OllamaClient()
        self.graph = self._build_graph()

    def answer(
        self,
        question: str,
        session_id: str,
        *,
        preferred_language: str = "english",
        mode: str = "learning",
        images: list[str] | None = None,
        image_names: list[str] | None = None,
    ) -> ChatResponse:
        runtime_profile = build_runtime_profile()
        initial_state: WorkflowState = {
            "question": question,
            "preferred_language": preferred_language,
            "session_id": session_id,
            "chat_mode": mode,
            "images": normalize_images(images or []),
            "image_names": image_names or [],
            "model_name": runtime_profile.recommended_model,
            "primary_hits": [],
            "verification_hits": [],
            "used_tools": [],
            "verification_result": VerificationResult(),
            "stage_timings": {},
            "model_calls": {},
            "timing_metadata": {
                "image_count": len(images or []),
                "verification_used": False,
                "review_used": False,
                "tool_count": 0,
                "primary_hit_count": 0,
                "verification_hit_count": 0,
            },
        }
        request_started = perf_counter()
        state = self._invoke_graph(initial_state)
        state["timing_metadata"]["total_duration_ms"] = round((perf_counter() - request_started) * 1000, 2)
        return ChatResponse(
            answer=state["answer"],
            steps=state["solve_plan"].solution_outline or default_steps(state["mode"], preferred_language),
            citations=state["citations"],
            used_tools=state["used_tools"],
            confidence=state["confidence"],
            model_name=state["model_name"],
            retrieval_plan=state["retrieval_plan"],
            verification_used=state["verification_result"].used,
            evidence_groups=state["evidence_groups"],
            timing=ChatTiming(
                total_duration_ms=state["timing_metadata"]["total_duration_ms"],
                stage_timings=state["stage_timings"],
                model_calls=state["model_calls"],
                metadata=state["timing_metadata"],
            ),
        )

    def _build_graph(self):
        if StateGraph is None:
            return None
        graph = StateGraph(WorkflowState)
        graph.add_node("classify", self._classify_node)
        graph.add_node("parse_images", self._parse_images_node)
        graph.add_node("retrieve_primary", self._retrieve_primary_node)
        graph.add_node("build_plan", self._build_plan_node)
        graph.add_node("run_tools", self._run_tools_node)
        graph.add_node("retrieve_verification", self._retrieve_verification_node)
        graph.add_node("compose_answer", self._compose_answer_node)
        graph.set_entry_point("classify")
        graph.add_edge("classify", "parse_images")
        graph.add_edge("parse_images", "retrieve_primary")
        graph.add_edge("retrieve_primary", "build_plan")
        graph.add_edge("build_plan", "run_tools")
        graph.add_edge("run_tools", "retrieve_verification")
        graph.add_edge("retrieve_verification", "compose_answer")
        graph.add_edge("compose_answer", END)
        return graph.compile()

    def _invoke_graph(self, initial_state: WorkflowState) -> WorkflowState:
        state = dict(initial_state)
        stage_sequence = [
            ("classify", self._classify_node),
            ("parse_images", self._parse_images_node),
            ("retrieve_primary", self._retrieve_primary_node),
        ]
        for stage_name, node in stage_sequence:
            started = perf_counter()
            try:
                state.update(node(state))
            except Exception as exc:
                state["stage_timings"][stage_name] = round((perf_counter() - started) * 1000, 2)
                state["timing_metadata"]["failed_stage"] = stage_name
                raise PipelineExecutionError(
                    str(exc),
                    stage_timings=dict(state["stage_timings"]),
                    model_calls=dict(state["model_calls"]),
                    timing_metadata=dict(state["timing_metadata"]),
                ) from exc
            state["stage_timings"][stage_name] = round((perf_counter() - started) * 1000, 2)
        if state.get("mode") == "solve":
            state["timing_metadata"]["path_type"] = "image_standard" if state.get("images") else "standard"
            remaining_stages = [
                ("build_plan", self._build_plan_node),
                ("run_tools", self._run_tools_node),
                ("retrieve_verification", self._retrieve_verification_node),
                ("compose_answer", self._compose_answer_node),
            ]
        else:
            state["timing_metadata"]["path_type"] = "fast"
            remaining_stages = [("compose_answer", self._compose_answer_node)]
        for stage_name, node in remaining_stages:
            started = perf_counter()
            try:
                state.update(node(state))
            except Exception as exc:
                state["stage_timings"][stage_name] = round((perf_counter() - started) * 1000, 2)
                state["timing_metadata"]["failed_stage"] = stage_name
                raise PipelineExecutionError(
                    str(exc),
                    stage_timings=dict(state["stage_timings"]),
                    model_calls=dict(state["model_calls"]),
                    timing_metadata=dict(state["timing_metadata"]),
                ) from exc
            state["stage_timings"][stage_name] = round((perf_counter() - started) * 1000, 2)
        state["timing_metadata"]["total_stage_count"] = len(stage_sequence) + len(remaining_stages)
        return state

    def _classify_node(self, state: WorkflowState) -> WorkflowState:
        question = state["question"]
        mode = classify_question(question)
        if state["chat_mode"] == "concept":
            mode = "concept"
        return {
            "mode": mode,
            "image_mode": classify_image_request(question, bool(state["images"])),
            "problem_id_hint": extract_problem_id(question),
            "chapter_hint": extract_chapter_hint(question),
            "solve_plan": SolvePlan(mode=mode),
            "retrieval_plan": RetrievalPlan(
                mode=state["chat_mode"],
                query_text=question,
                primary_collections=["theory_gold", "notes_silver"] if mode == "concept" else PRIMARY_COLLECTIONS,
                verification_collections=[] if mode == "concept" or state["chat_mode"] == "concept" else VERIFICATION_COLLECTIONS,
                filters={},
                stages=["classify"],
            ),
        }

    def _parse_images_node(self, state: WorkflowState) -> WorkflowState:
        if not state["images"]:
            return {"image_summary": ""}
        start = perf_counter()
        prompt = f"""
Return strict JSON with these fields only:
- question_text: string
- symbols: array of strings
- transfer_functions: array of strings
- diagram_clues: array of strings
- unclear_items: array of strings

Extract only visible control-systems information from the attached image(s).
User question: {state['question']}
"""
        image_message: dict[str, Any] = {"role": "user", "content": prompt, "images": state["images"]}
        try:
            raw_summary = self.ollama.chat(
                VISION_MODEL,
                [
                    {"role": "system", "content": IMAGE_PARSE_SYSTEM_PROMPT},
                    image_message,
                ],
                json_output=True,
            )
        finally:
            state["model_calls"]["parse_images_model_ms"] = round((perf_counter() - start) * 1000, 2)
        used_tools = sorted(set([*state["used_tools"], "vision"]))
        image_summary = compress_image_summary(raw_summary)
        state["timing_metadata"]["image_summary_chars"] = len(image_summary)
        return {"image_summary": image_summary, "used_tools": used_tools}

    def _retrieve_primary_node(self, state: WorkflowState) -> WorkflowState:
        retrieval_plan = state["retrieval_plan"]
        query_text = build_query_text(state["question"], state.get("image_summary", ""))
        primary_collections = retrieval_plan.primary_collections
        use_retrieval = not (state["images"] and state["image_mode"] == "visual")
        primary_hits = (
            self.store.search(
                query_text,
                source_families=primary_collections,
                chapter=state.get("chapter_hint"),
                problem_id=state.get("problem_id_hint"),
            )
            if use_retrieval
            else []
        )
        retrieval_plan.query_text = query_text
        retrieval_plan.filters = {
            "chapter": state.get("chapter_hint"),
            "problem_id": state.get("problem_id_hint"),
        }
        retrieval_plan.stages.append("retrieve_primary")
        state["timing_metadata"]["primary_hit_count"] = len(primary_hits)
        return {
            "primary_hits": primary_hits,
            "retrieval_plan": retrieval_plan,
            "confidence": compute_confidence(primary_hits, []),
        }

    def _build_plan_node(self, state: WorkflowState) -> WorkflowState:
        if state["mode"] != "solve":
            return {
                "solve_plan": SolvePlan(
                    mode="concept",
                    solution_outline=default_steps("concept", state["preferred_language"]),
                )
            }

        context = render_context(state["primary_hits"], limit=2, max_chars=280)
        planning_prompt = f"""
Return strict JSON only.

Language: {state['preferred_language']}.

Fields:
- mode: always solve
- problem_restatement: string
- knowns: array of strings
- targets: array of strings
- method: short string
- solution_outline: array of at most 4 short strings
- formulas: array of at most 4 short strings
- tool_requests: array using these tool types only:
  1. second_order_metrics with zeta, wn
  2. solve_equation with equation, variable, substitutions
  3. evaluate_expression with expression, substitutions
  4. transfer_function_analysis with numerator, denominator

User question:
{state['question']}

Image summary:
{state.get('image_summary', '') or 'No image summary.'}

Retrieved context:
{context}
"""
        model_started = perf_counter()
        try:
            plan_options: dict[str, Any] = {"temperature": 0}
            if PLAN_NUM_PREDICT is not None:
                plan_options["num_predict"] = PLAN_NUM_PREDICT
            raw = self.ollama.chat(
                state["model_name"],
                [
                    {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                    {"role": "user", "content": planning_prompt},
                ],
                json_output=True,
                options=plan_options,
            )
        finally:
            state["model_calls"]["build_plan_model_ms"] = round((perf_counter() - model_started) * 1000, 2)
        try:
            solve_plan = SolvePlan.model_validate(self.ollama.parse_json(raw))
        except Exception:
            solve_plan = SolvePlan(
                mode="solve",
                problem_restatement=state["question"],
                method="Use retrieved control-systems theory and matched worked examples to derive the solution.",
                solution_outline=default_steps("solve", state["preferred_language"]),
            )
        return {"solve_plan": solve_plan}

    def _run_tools_node(self, state: WorkflowState) -> WorkflowState:
        if state["mode"] != "solve":
            return {"tool_results": []}
        tool_results, tool_names = run_tool_requests(state["solve_plan"].tool_requests)
        used_tools = sorted(set([*state["used_tools"], *tool_names]))
        state["timing_metadata"]["tool_count"] = len(tool_results)
        return {"tool_results": tool_results, "used_tools": used_tools}

    def _retrieve_verification_node(self, state: WorkflowState) -> WorkflowState:
        retrieval_plan = state["retrieval_plan"]
        if state["mode"] != "solve" or state["chat_mode"] == "concept" or not retrieval_plan.verification_collections:
            retrieval_plan.stages.append("skip_verification")
            return {
                "verification_hits": [],
                "verification_result": VerificationResult(used=False, summary="Verification skipped."),
                "retrieval_plan": retrieval_plan,
                "confidence": compute_confidence(state["primary_hits"], []),
            }

        preferred_pairs = [hit.unit.pair_key for hit in state["primary_hits"][:6] if hit.unit.pair_key]
        verification_hits = self.store.search(
            state["retrieval_plan"].query_text,
            source_families=retrieval_plan.verification_collections,
            chapter=state.get("chapter_hint"),
            problem_id=state.get("problem_id_hint"),
            prefer_pair_keys=preferred_pairs,
        )
        retrieval_plan.stages.append("retrieve_verification")
        state["timing_metadata"]["verification_used"] = bool(verification_hits)
        state["timing_metadata"]["verification_hit_count"] = len(verification_hits)
        verification_result = VerificationResult(
            used=bool(verification_hits),
            summary="Verification used official worked solutions." if verification_hits else "No matching solution verification was retrieved.",
            supporting_pair_keys=[hit.unit.pair_key for hit in verification_hits if hit.unit.pair_key],
        )
        return {
            "verification_hits": verification_hits,
            "verification_result": verification_result,
            "retrieval_plan": retrieval_plan,
            "confidence": compute_confidence(state["primary_hits"], verification_hits),
        }

    def _compose_answer_node(self, state: WorkflowState) -> WorkflowState:
        if state["mode"] == "concept":
            primary_hits = [hit for hit in state["primary_hits"] if hit.unit.source_family in {"theory_gold", "notes_silver"}]
            primary_context = render_context(primary_hits or state["primary_hits"], limit=2, max_chars=180)
            verification_context = "Verification skipped for concept mode."
            prompt_header = CONCEPT_ANSWER_PROMPT.format(language=state["preferred_language"])
        else:
            primary_context = render_context(state["primary_hits"], limit=2, max_chars=180)
            verification_context = render_context(state["verification_hits"], limit=1, max_chars=120)
            prompt_header = SOLVE_ANSWER_PROMPT.format(language=state["preferred_language"])
        compact_plan = summarize_solve_plan(state["solve_plan"])
        tool_summary = summarize_tool_results(state.get("tool_results", []))
        compressed_context_chars = len(primary_context) + len(verification_context) + len(compact_plan) + len(tool_summary)
        state["timing_metadata"]["compressed_context_chars"] = compressed_context_chars
        final_prompt = f"""
{prompt_header}

Question mode: {state['mode']}
Learning mode: {state['chat_mode']}
User question: {state['question']}
Image summary: {state.get('image_summary', '') or 'No image summary.'}

Primary evidence:
{primary_context}

Verification evidence:
{verification_context}

Solve plan:
{compact_plan}

Tool results:
{tool_summary}

Hard consistency checks:
- if poles have negative real parts, do not call the system unstable
- if a system is type 2, step and ramp steady-state errors should be zero
- dominant-pole metrics must match the same pole pair used in the derivation
- the final summary must agree with the calculations above it
- if information is insufficient, say exactly what is missing
- do not output JSON, YAML, or code fences
- write only the final answer body, with no preface like "Here is the solution"
"""
        answer_options: dict[str, Any] = {"temperature": 0.1}
        if ANSWER_NUM_PREDICT is not None:
            answer_options["num_predict"] = ANSWER_NUM_PREDICT
        compose_started = perf_counter()
        try:
            raw_answer = self.ollama.chat(
                state["model_name"],
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": final_prompt},
                ],
                options=answer_options,
            )
        finally:
            state["model_calls"]["compose_answer_model_ms"] = round((perf_counter() - compose_started) * 1000, 2)
        state["timing_metadata"]["compose_answer_empty"] = not bool(raw_answer.strip())
        draft_answer = render_answer_output(
            raw_answer,
            mode=state["mode"],
            preferred_language=state["preferred_language"],
            question=state["question"],
            solve_plan=state["solve_plan"],
        )
        answer, review_used, review_ms = review_and_revise_answer(
            self.ollama,
            state["model_name"],
            preferred_language=state["preferred_language"],
            mode=state["mode"],
            question=state["question"],
            solve_plan=state["solve_plan"],
            draft_answer=draft_answer,
            tool_results=state.get("tool_results", []),
        )
        citations = build_citations([*state["primary_hits"], *state["verification_hits"]])
        evidence_groups = build_evidence_groups([*state["primary_hits"], *state["verification_hits"]])
        used_tools = state["used_tools"]
        if review_used:
            used_tools = sorted(set([*used_tools, "consistency-review"]))
        state["timing_metadata"]["review_used"] = review_used
        if review_ms > 0:
            state["model_calls"]["review_and_revise_model_ms"] = review_ms
        return {
            "answer": answer,
            "citations": citations,
            "evidence_groups": evidence_groups,
            "used_tools": used_tools,
        }


def classify_question(question: str) -> str:
    lowered = question.lower()
    patterns = [
        r"\bfind\b",
        r"\bdetermine\b",
        r"\bcompute\b",
        r"\bsolve\b",
        r"\bderive\b",
        r"\btransfer function\b",
        r"\bstability\b",
        r"\bsteady-state\b",
        r"\bovershoot\b",
        r"\bsettling time\b",
        r"\broot locus\b",
        r"\bstep response\b",
    ]
    if any(re.search(pattern, lowered) for pattern in patterns):
        return "solve"
    return "concept"


def classify_image_request(question: str, has_images: bool) -> str:
    if not has_images:
        return "none"
    lowered = question.lower()
    visual_patterns = [
        r"\bdescribe\b",
        r"\bwhat is in\b",
        r"\bwhat do you see\b",
        r"\bread the image\b",
        r"\bextract text\b",
        r"\btranscribe\b",
        r"\bsummarize the image\b",
    ]
    if any(re.search(pattern, lowered) for pattern in visual_patterns):
        return "visual"
    return "problem"


def extract_chapter_hint(question: str) -> str | None:
    match = re.search(r"\bchapter\s*(\d+)\b", question, flags=re.IGNORECASE)
    if match:
        return f"chapter{match.group(1)}"
    return None


def build_query_text(question: str, image_summary: str) -> str:
    if not image_summary.strip():
        return question
    return f"{question}\n\nImage-derived problem details:\n{image_summary}"


def render_context(hits: list[RetrievalHit], *, limit: int = 6, max_chars: int = 1400) -> str:
    if not hits:
        return "No retrieved evidence."

    lines: list[str] = []
    seen_signatures: set[tuple[str, str | None, int | None, str]] = set()
    for idx, hit in enumerate(hits[:limit], start=1):
        signature = (
            hit.unit.source_family,
            hit.unit.problem_id,
            hit.unit.page_or_slide,
            hit.unit.excerpt[:120],
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        page_or_slide = hit.unit.page_or_slide or "?"
        excerpt = compact_text(hit.unit.text, max_chars)
        lines.append(
            f"[{idx}] family={hit.unit.source_family} file={hit.unit.file_path} "
            f"page_or_slide={page_or_slide} chapter={hit.unit.chapter or '-'} "
            f"problem_id={hit.unit.problem_id or '-'} score={hit.score:.3f}\n{excerpt}"
        )
    return "\n\n".join(lines)


def build_citations(hits: list[RetrievalHit]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[tuple[str, str | None, int | None]] = set()
    for hit in hits[:8]:
        key = (hit.unit.file_path, hit.unit.problem_id, hit.unit.page_or_slide)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                source_path=hit.unit.file_path,
                source_family=hit.unit.source_family,
                chapter=hit.unit.chapter,
                problem_id=hit.unit.problem_id,
                page_or_slide=hit.unit.page_or_slide,
                score=round(hit.score, 4),
                excerpt=hit.unit.excerpt,
            )
        )
    return citations


def build_evidence_groups(hits: list[RetrievalHit]) -> list[EvidenceGroup]:
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.unit.source_family] = counts.get(hit.unit.source_family, 0) + 1
    return [
        EvidenceGroup(label=family.replace("_", " "), source_family=family, hit_count=count)
        for family, count in sorted(counts.items())
    ]


def compute_confidence(primary_hits: list[RetrievalHit], verification_hits: list[RetrievalHit]) -> float:
    scores = [hit.score for hit in primary_hits[:4]]
    if verification_hits:
        scores.extend(hit.score for hit in verification_hits[:2])
    if not scores:
        return 0.12
    mean_score = sum(scores) / len(scores)
    return max(0.12, min(0.97, round(mean_score, 4)))


def review_and_revise_answer(
    ollama: OllamaClient,
    model_name: str,
    *,
    preferred_language: str,
    mode: str,
    question: str,
    solve_plan: SolvePlan,
    draft_answer: str,
    tool_results: list[dict[str, Any]],
) -> tuple[str, bool, float]:
    if mode != "solve" or not draft_answer.strip():
        return draft_answer, False, 0.0

    patched_answer, patched = apply_deterministic_consistency_fixes(draft_answer, tool_results)
    heuristic_issues = detect_consistency_issues(patched_answer, tool_results)
    if not heuristic_issues:
        return patched_answer, patched, 0.0

    review_prompt = f"""
Return strict JSON only with these fields:
- verdict: "ok" or "revise"
- issues: array of strings
- revised_answer: string

Task:
Review the draft control-systems answer for internal consistency and alignment with the supplied evidence.
If the draft is already consistent, return verdict "ok" and copy the original answer into revised_answer.
If you find contradictions, fix only what is necessary and return verdict "revise".

Hard checks:
- stability labels must match the pole real parts shown in the answer or tool results
- system type must match the reported steady-state errors
- dominant-pole calculations and the reported rise time / overshoot / settling time must agree
- the final summary must not contradict earlier calculations
- do not add new citations or invent new evidence
- preserve the original section structure
- prefer minimal edits; keep every correct formula, value, and conclusion unchanged
- never replace a correct symbolic or numeric quantity with a placeholder such as "?"
- answer language: {preferred_language}

User question:
{question}

Solve plan:
{solve_plan.model_dump_json(indent=2)}

Tool results:
{json.dumps(tool_results, ensure_ascii=False, indent=2)}

Heuristic warnings:
{json.dumps(heuristic_issues, ensure_ascii=False, indent=2)}

Draft answer:
{patched_answer}
"""
    try:
        review_started = perf_counter()
        review_options: dict[str, Any] = {"temperature": 0}
        if REVIEW_NUM_PREDICT is not None:
            review_options["num_predict"] = REVIEW_NUM_PREDICT
        raw = ollama.chat(
            model_name,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": review_prompt},
            ],
            json_output=True,
            options=review_options,
        )
        review_ms = round((perf_counter() - review_started) * 1000, 2)
        parsed = ollama.parse_json(raw)
        verdict = str(parsed.get("verdict", "ok")).strip().lower()
        revised_answer = str(parsed.get("revised_answer", "")).strip()
        if verdict == "revise" and revised_answer and not revision_looks_degraded(patched_answer, revised_answer):
            return revised_answer, True, review_ms
        if revised_answer and not revision_looks_degraded(patched_answer, revised_answer):
            return revised_answer, False, review_ms
    except Exception:
        pass

    return patched_answer, patched, 0.0


def detect_consistency_issues(answer: str, tool_results: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    lowered = answer.lower()
    poles = extract_complex_values(answer)
    if poles:
        if all(value.real < 0 for value in poles):
            if re.search(r"\bunstable\b", lowered) and not re.search(r"\bnot unstable\b", lowered):
                issues.append("The answer lists poles with negative real parts but later calls the closed-loop system unstable.")
            if "positive real part" in lowered or "positive real parts" in lowered:
                issues.append("The answer says the poles have positive real parts even though the written poles have negative real parts.")
        if all(value.real > 0 for value in poles) and re.search(r"\bstable\b", lowered) and "unstable" not in lowered:
            issues.append("The answer labels the system stable even though the written poles have positive real parts.")

    for result in tool_results:
        if result.get("type") != "transfer_function_analysis":
            continue
        stable = result.get("stable")
        if stable is True and re.search(r"\bunstable\b", lowered) and not re.search(r"\bnot unstable\b", lowered):
            issues.append("The answer says the system is unstable, but the transfer-function tool result reports stable=true.")
        if stable is False and re.search(r"\bstable\b", lowered) and "unstable" not in lowered:
            issues.append("The answer says the system is stable, but the transfer-function tool result reports stable=false.")

    return issues


def apply_deterministic_consistency_fixes(answer: str, tool_results: list[dict[str, Any]]) -> tuple[str, bool]:
    fixed_answer = answer
    changed = False
    poles = extract_complex_values(answer)
    stable_hint: bool | None = None
    if poles:
        if all(value.real < 0 for value in poles):
            stable_hint = True
        elif all(value.real > 0 for value in poles):
            stable_hint = False

    for result in tool_results:
        if result.get("type") == "transfer_function_analysis" and isinstance(result.get("stable"), bool):
            stable_hint = bool(result["stable"])

    if stable_hint is True:
        replacement = "Note: The system is stable as the listed closed-loop poles all have negative real parts."
        updated = re.sub(
            r"(?im)^.*(?:positive real part|positive real parts|unstable).*$",
            replacement,
            fixed_answer,
        )
        if updated != fixed_answer:
            fixed_answer = updated
            changed = True
    elif stable_hint is False:
        replacement = "Note: The system is unstable because the relevant closed-loop poles have positive real parts."
        updated = re.sub(
            r"(?im)^.*(?:stable|negative real part|negative real parts).*$",
            replacement,
            fixed_answer,
        )
        if updated != fixed_answer:
            fixed_answer = updated
            changed = True

    return fixed_answer, changed


def extract_complex_values(text: str) -> list[complex]:
    pattern = re.compile(
        r"=\s*([+-]?\d+(?:\.\d+)?)\s*([+-])\s*j\s*(\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    values: list[complex] = []
    for match in pattern.finditer(text):
        real_part = float(match.group(1))
        imag_part = float(match.group(3))
        if match.group(2) == "-":
            imag_part *= -1
        values.append(complex(real_part, imag_part))
    return values


def revision_looks_degraded(original: str, revised: str) -> bool:
    original_q = original.count("?")
    revised_q = revised.count("?")
    if revised_q > original_q + 6:
        return True
    if len(revised.strip()) < max(120, int(len(original.strip()) * 0.55)):
        return True
    return False


def default_steps(mode: str, preferred_language: str) -> list[str]:
    if preferred_language.lower().startswith("en"):
        if mode == "solve":
            return [
                "Extract the known values, targets, and any image-derived symbols.",
                "Retrieve relevant theory and matched worked problems.",
                "Build a solution plan and verify key calculations with math tools.",
                "Check the derived answer against official worked solutions before finalizing.",
            ]
        return [
            "Retrieve the strongest local theory evidence.",
            "Summarize the concept in English and cite the supporting material.",
        ]
    if mode == "solve":
        return [
            "Extract the known values and the target quantity.",
            "Retrieve relevant theory and matched worked examples.",
            "Use math tools to verify the core calculations.",
            "Check the result against the official worked solutions.",
        ]
    return [
        "Retrieve the most relevant local theory material.",
        "Summarize the concept clearly with citations.",
    ]


def run_tool_requests(tool_requests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    used_tools: list[str] = []
    for request in tool_requests:
        tool_type = request.get("type")
        try:
            if tool_type == "evaluate_expression":
                result = evaluate_expression(request["expression"], request.get("substitutions", {}))
                used_tools.append("sympy")
            elif tool_type == "solve_equation":
                result = solve_equation(request["equation"], request["variable"], request.get("substitutions", {}))
                used_tools.append("sympy")
            elif tool_type == "second_order_metrics":
                result = second_order_metrics(float(request["zeta"]), float(request["wn"]))
                used_tools.append("scipy")
            elif tool_type == "transfer_function_analysis":
                result = transfer_function_analysis(request["numerator"], request["denominator"])
                used_tools.append("python-control" if control_lib is not None else "scipy.signal")
            else:
                continue
            result["type"] = tool_type
            results.append(result)
        except Exception as exc:
            results.append({"type": tool_type or "unknown", "error": str(exc)})
    return results, sorted(set(used_tools))


def evaluate_expression(expression: str, substitutions: dict[str, Any]) -> dict[str, Any]:
    parsed = sympy.sympify(expression)
    substituted = parsed.subs({sympy.Symbol(key): value for key, value in substitutions.items()})
    return {"expression": expression, "result": str(sympy.N(substituted))}


def solve_equation(equation: str, variable: str, substitutions: dict[str, Any]) -> dict[str, Any]:
    if "=" in equation:
        left, right = equation.split("=", maxsplit=1)
        expr = sympy.Eq(sympy.sympify(left), sympy.sympify(right))
    else:
        expr = sympy.Eq(sympy.sympify(equation), 0)
    expr = expr.subs({sympy.Symbol(key): value for key, value in substitutions.items()})
    symbol = sympy.Symbol(variable)
    solutions = sympy.solve(expr, symbol)
    return {"equation": equation, "variable": variable, "solutions": [str(item) for item in solutions]}


def second_order_metrics(zeta: float, wn: float) -> dict[str, Any]:
    if zeta <= 0 or wn <= 0:
        raise ValueError("zeta and wn must be positive.")
    mp = math.exp((-zeta * math.pi) / math.sqrt(max(1e-9, 1 - zeta**2))) if zeta < 1 else 0.0
    tp = math.pi / (wn * math.sqrt(max(1e-9, 1 - zeta**2))) if zeta < 1 else None
    ts_2 = 4.0 / (zeta * wn)
    ts_5 = 3.0 / (zeta * wn)
    return {
        "zeta": zeta,
        "wn": wn,
        "Mp": round(mp, 6),
        "Tp": None if tp is None else round(tp, 6),
        "Ts_2pct": round(ts_2, 6),
        "Ts_5pct": round(ts_5, 6),
    }


def transfer_function_analysis(numerator: list[float] | str, denominator: list[float] | str) -> dict[str, Any]:
    num = coerce_numeric_list(numerator)
    den = coerce_numeric_list(denominator)
    if control_lib is not None:
        system = control_lib.TransferFunction(num, den)
        poles = [complex(value) for value in control_lib.poles(system)]
        zeros = [complex(value) for value in control_lib.zeros(system)]
        _, response = control_lib.step_response(system)
    else:
        system = signal.TransferFunction(num, den)
        poles = [complex(value) for value in system.poles]
        zeros = [complex(value) for value in system.zeros]
        _, response = signal.step(system)
    stable = all(value.real < 0 for value in poles)
    return {
        "numerator": num,
        "denominator": den,
        "poles": [format_complex(value) for value in poles],
        "zeros": [format_complex(value) for value in zeros],
        "stable": stable,
        "step_final_value": round(float(response[-1]), 6) if len(response) else None,
        "engine": "python-control" if control_lib is not None else "scipy.signal",
    }


def coerce_numeric_list(value: list[float] | str) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            return [float(item) for item in json.loads(text)]
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    raise TypeError("Expected list or comma-separated string.")


def format_complex(value: complex) -> str:
    return f"{value.real:.6g}{value.imag:+.6g}j"


def normalize_images(images: list[str]) -> list[str]:
    normalized: list[str] = []
    for image_b64 in images:
        try:
            raw = base64.b64decode(image_b64)
            with Image.open(io.BytesIO(raw)) as img:
                img = img.convert("RGB")
                width, height = img.size
                min_size = 56
                max_size = 1280
                if width < min_size or height < min_size:
                    scale = max(min_size / max(width, 1), min_size / max(height, 1))
                    new_size = (
                        max(min_size, int(width * scale)),
                        max(min_size, int(height * scale)),
                    )
                    img = img.resize(new_size)
                    width, height = img.size
                if width > max_size or height > max_size:
                    img.thumbnail((max_size, max_size))
                output = io.BytesIO()
                img.save(output, format="PNG")
                normalized.append(base64.b64encode(output.getvalue()).decode())
        except Exception:
            normalized.append(image_b64)
    return normalized


def compact_text(text: str, max_chars: int) -> str:
    squashed = re.sub(r"\s+", " ", text).strip()
    if len(squashed) <= max_chars:
        return squashed
    return squashed[: max_chars - 3].rstrip() + "..."


def compress_image_summary(raw_summary: str) -> str:
    try:
        payload = OllamaClient.parse_json(raw_summary)
        question_text = compact_text(str(payload.get("question_text", "")).strip(), 320)
        symbols = [compact_text(str(item), 60) for item in payload.get("symbols", [])[:6]]
        transfer_functions = [compact_text(str(item), 90) for item in payload.get("transfer_functions", [])[:4]]
        diagram_clues = [compact_text(str(item), 90) for item in payload.get("diagram_clues", [])[:5]]
        unclear_items = [compact_text(str(item), 60) for item in payload.get("unclear_items", [])[:5]]
        parts = [
            f"question_text: {question_text or 'none'}",
            f"symbols: {', '.join(symbols) if symbols else 'none'}",
            f"transfer_functions: {', '.join(transfer_functions) if transfer_functions else 'none'}",
            f"diagram_clues: {', '.join(diagram_clues) if diagram_clues else 'none'}",
            f"unclear_items: {', '.join(unclear_items) if unclear_items else 'none'}",
        ]
        return "\n".join(parts)
    except Exception:
        return compact_text(raw_summary, 420)


def summarize_solve_plan(plan: SolvePlan) -> str:
    payload = {
        "mode": plan.mode,
        "problem_restatement": compact_text(plan.problem_restatement, 220),
        "knowns": [compact_text(item, 80) for item in plan.knowns[:6]],
        "targets": [compact_text(item, 80) for item in plan.targets[:4]],
        "method": compact_text(plan.method, 140),
        "solution_outline": [compact_text(item, 100) for item in plan.solution_outline[:4]],
        "formulas": [compact_text(item, 100) for item in plan.formulas[:4]],
        "tool_requests": plan.tool_requests[:3],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def summarize_tool_results(tool_results: list[dict[str, Any]]) -> str:
    if not tool_results:
        return "No tool results."
    compact_results: list[dict[str, Any]] = []
    for result in tool_results[:4]:
        compact_result: dict[str, Any] = {}
        for key, value in result.items():
            if key in {"numerator", "denominator"}:
                continue
            if isinstance(value, str):
                compact_result[key] = compact_text(value, 120)
            elif isinstance(value, list):
                compact_result[key] = [compact_text(str(item), 60) for item in value[:6]]
            else:
                compact_result[key] = value
        compact_results.append(compact_result)
    return json.dumps(compact_results, ensure_ascii=False, separators=(",", ":"))


def render_answer_output(
    raw_answer: str,
    *,
    mode: str,
    preferred_language: str,
    question: str,
    solve_plan: SolvePlan,
) -> str:
    if not raw_answer.strip():
        return build_fallback_answer(mode=mode, question=question, solve_plan=solve_plan)
    if not looks_like_structured_output(raw_answer):
        return clean_answer_output(normalize_plain_answer(raw_answer), mode=mode)
    try:
        payload = OllamaClient.parse_json(raw_answer)
    except Exception:
        salvaged = salvage_partial_answer(raw_answer)
        if salvaged:
            payload = salvaged
        else:
            fallback = compact_text(raw_answer, 2400)
            if looks_like_structured_output(raw_answer):
                return build_fallback_answer(mode=mode, question=question, solve_plan=solve_plan)
            return clean_answer_output(
                fallback or build_fallback_answer(mode=mode, question=question, solve_plan=solve_plan),
                mode=mode,
            )

    answer_summary = compact_text(str(payload.get("answer_summary", "")).strip(), 600)
    theory_used = [compact_text(str(item), 160) for item in payload.get("theory_used", [])[:4]]
    step_by_step = [compact_text(str(item), 220) for item in payload.get("step_by_step_solution", [])[:6]]
    final_answer = compact_text(str(payload.get("final_answer", "")).strip(), 500)
    missing_info = [compact_text(str(item), 140) for item in payload.get("missing_info", [])[:4]]

    if not any([answer_summary, theory_used, step_by_step, final_answer, missing_info]):
        return build_fallback_answer(mode=mode, question=question, solve_plan=solve_plan)

    if mode == "concept":
        lines: list[str] = []
        lines.append(answer_summary or compact_text(question, 300))
        if theory_used:
            lines.append("")
            lines.append("Theory Used")
            lines.extend(f"- {item}" for item in theory_used)
        if missing_info:
            lines.append("")
            lines.append("Missing Information")
            lines.extend(f"- {item}" for item in missing_info)
        lines.append("")
        lines.append("Sources")
        lines.append("- [1]")
        return clean_answer_output("\n".join(lines), mode=mode)

    problem_restatement = solve_plan.problem_restatement or question
    knowns = solve_plan.knowns[:6]
    targets = solve_plan.targets[:4]
    lines = [
        "Problem Restatement",
        compact_text(problem_restatement, 500),
        "",
        "Knowns / Unknowns",
    ]
    if knowns or targets:
        if knowns:
            lines.extend(f"- Known: {item}" for item in knowns)
        if targets:
            lines.extend(f"- Target: {item}" for item in targets)
    else:
        lines.append("- Not explicitly identified.")
    lines.extend(["", "Theory Used"])
    if theory_used:
        lines.extend(f"- {item}" for item in theory_used)
    else:
        lines.append("- Use retrieved control-systems evidence and matched examples.")
    lines.extend(["", "Step-by-Step Solution"])
    if step_by_step:
        lines.extend(f"{idx}. {item}" for idx, item in enumerate(step_by_step, start=1))
    else:
        lines.extend(f"{idx}. {item}" for idx, item in enumerate(solve_plan.solution_outline[:5], start=1))
    lines.extend(["", "Final Answer", final_answer or answer_summary or "The result could not be stated confidently."])
    if missing_info:
        lines.extend(["", "Missing Information"])
        lines.extend(f"- {item}" for item in missing_info)
    lines.extend(["", "Sources", "- [1]"])
    return clean_answer_output("\n".join(lines), mode=mode)


def build_fallback_answer(*, mode: str, question: str, solve_plan: SolvePlan) -> str:
    if mode == "concept":
        lines = [
            compact_text(solve_plan.problem_restatement or question, 320),
            "",
            "## Sources",
            "- [1]",
        ]
        return clean_answer_output("\n".join(lines), mode=mode)

    problem_restatement = compact_text(solve_plan.problem_restatement or question, 500)
    method = compact_text(solve_plan.method or "Use the retrieved evidence to derive the result.", 180)
    outline = solve_plan.solution_outline[:4] or default_steps("solve", "english")
    lines = [
        "## Problem Restatement",
        problem_restatement,
        "",
        "## Knowns / Unknowns",
    ]
    if solve_plan.knowns or solve_plan.targets:
        lines.extend(f"- Known: {item}" for item in solve_plan.knowns[:6])
        lines.extend(f"- Target: {item}" for item in solve_plan.targets[:4])
    else:
        lines.append("- Not explicitly identified from the generated answer.")
    lines.extend([
        "",
        "## Theory Used",
        f"- {method}",
        "",
        "## Step-by-Step Solution",
    ])
    lines.extend(f"{idx}. {compact_text(item, 180)}" for idx, item in enumerate(outline, start=1))
    lines.extend([
        "",
        "## Final Answer",
        "The model did not return a complete final statement, so this answer falls back to the extracted plan. Please use the cited sources below and, if needed, ask a narrower follow-up question.",
        "",
        "## Sources",
        "- [1]",
    ])
    return clean_answer_output("\n".join(lines), mode=mode)


def normalize_plain_answer(text: str, max_chars: int = 6000) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def clean_answer_output(text: str, *, mode: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"(?im)^\s*(here is (the )?(answer|solution)[:\-]?)\s*\n*", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(based on the (provided|retrieved) context[:,]?)\s*\n*", "", cleaned)
    cleaned = re.sub(r"(?s)^```[a-zA-Z0-9_-]*\n(.*?)\n```$", r"\1", cleaned).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = normalize_section_headings(cleaned, mode=mode)
    cleaned = normalize_sources_section(cleaned)
    return cleaned.strip()


def normalize_section_headings(text: str, *, mode: str) -> str:
    section_names = [
        "Problem Restatement",
        "Knowns / Unknowns",
        "Theory Used",
        "Step-by-Step Solution",
        "Final Answer",
        "Sources",
        "Missing Information",
    ]
    normalized = text
    for name in section_names:
        normalized = re.sub(rf"(?im)^\s*{re.escape(name)}\s*$", f"## {name}", normalized)
    if mode == "solve" and "## Problem Restatement" not in normalized:
        lines = normalized.splitlines()
        if lines and not lines[0].lstrip().startswith("#"):
            lines.insert(0, "## Problem Restatement")
            normalized = "\n".join(lines)
    return normalized


def normalize_sources_section(text: str) -> str:
    if "## Sources" not in text:
        return text
    before, after = text.split("## Sources", maxsplit=1)
    source_lines = [line.strip() for line in after.splitlines() if line.strip()]
    cleaned_lines: list[str] = []
    for line in source_lines:
        if re.search(r"Use the cited local evidence listed below", line, flags=re.IGNORECASE):
            continue
        if line.startswith("- [") or line.startswith("["):
            cleaned_lines.append(line if line.startswith("- ") else f"- {line}")
    if not cleaned_lines:
        cleaned_lines = ["- [1]"]
    return before.rstrip() + "\n\n## Sources\n" + "\n".join(cleaned_lines)


def looks_like_structured_output(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or '"answer_summary"' in stripped or '"final_answer"' in stripped


def salvage_partial_answer(text: str) -> dict[str, Any] | None:
    if not looks_like_structured_output(text):
        return None

    payload: dict[str, Any] = {}
    for field in ("answer_summary", "final_answer"):
        value = extract_partial_json_string(text, field)
        if value:
            payload[field] = value

    for field in ("theory_used", "step_by_step_solution", "missing_info"):
        items = extract_partial_json_array(text, field)
        if items:
            payload[field] = items

    return payload or None


def extract_partial_json_string(text: str, field: str) -> str:
    marker = f'"{field}"'
    start = text.find(marker)
    if start == -1:
        return ""
    start = text.find(":", start)
    if start == -1:
        return ""
    start = text.find('"', start)
    if start == -1:
        return ""
    start += 1
    chars: list[str] = []
    escaped = False
    for char in text[start:]:
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            break
        chars.append(char)
    return compact_text("".join(chars).strip(), 600)


def extract_partial_json_array(text: str, field: str) -> list[str]:
    marker = f'"{field}"'
    start = text.find(marker)
    if start == -1:
        return []
    start = text.find("[", start)
    if start == -1:
        return []
    end = text.find("]", start)
    if end == -1:
        segment = text[start + 1 :]
    else:
        segment = text[start + 1 : end]
    matches = re.findall(r'"([^"]+)', segment)
    cleaned = [compact_text(match.strip(), 220) for match in matches if match.strip()]
    return cleaned[:6]
