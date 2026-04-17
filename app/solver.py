from __future__ import annotations

import base64
import io
import json
import math
import re
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

from app.config import PRIMARY_COLLECTIONS, VERIFICATION_COLLECTIONS, VISION_MODEL
from app.extractors import extract_problem_id
from app.knowledge_store import KnowledgeStore
from app.ollama_client import OllamaClient
from app.runtime import build_runtime_profile
from app.schemas import (
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
            "used_tools": [],
            "verification_result": VerificationResult(),
        }
        state = self._invoke_graph(initial_state)
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
        if self.graph is not None:
            return self.graph.invoke(initial_state)
        state = dict(initial_state)
        for node in (
            self._classify_node,
            self._parse_images_node,
            self._retrieve_primary_node,
            self._build_plan_node,
            self._run_tools_node,
            self._retrieve_verification_node,
            self._compose_answer_node,
        ):
            state.update(node(state))
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
        prompt = f"""
Extract the control-systems problem information from the attached image(s).
Return plain English text only.
Capture visible question text, variables, labels, transfer functions, block connections, and any graph or table clues.
If a symbol is unclear, say it is unclear instead of guessing.
User question:
{state['question']}
"""
        image_message: dict[str, Any] = {"role": "user", "content": prompt, "images": state["images"]}
        image_summary = self.ollama.chat(
            VISION_MODEL,
            [
                {"role": "system", "content": "Read the image carefully and extract problem content in English."},
                image_message,
            ],
        )
        used_tools = sorted(set([*state["used_tools"], "vision"]))
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

        context = render_context(state["primary_hits"], limit=4, max_chars=850)
        planning_prompt = f"""
Return strict JSON only.

Language for all strings: {state['preferred_language']}.

Fields:
- mode: always solve
- problem_restatement: string
- knowns: array of strings
- targets: array of strings
- method: string
- solution_outline: array of strings
- formulas: array of strings
- tool_requests: array using these tool types only:
  1. second_order_metrics with zeta, wn
  2. solve_equation with equation, variable, substitutions
  3. evaluate_expression with expression, substitutions
  4. transfer_function_analysis with numerator, denominator

User question:
{state['question']}

Image summary:
{state.get('image_summary', '') or 'No image summary.'}

Retrieved theory/problem context:
{context}
"""
        raw = self.ollama.chat(
            state["model_name"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": planning_prompt},
            ],
            json_output=True,
        )
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
        primary_context = render_context(state["primary_hits"], limit=4, max_chars=900)
        verification_context = render_context(state["verification_hits"], limit=3, max_chars=750)
        tool_summary = json.dumps(state.get("tool_results", []), ensure_ascii=False, indent=2)
        final_prompt = f"""
Language target: {state['preferred_language']}
Question mode: {state['mode']}
Learning mode: {state['chat_mode']}

User question:
{state['question']}

Image summary:
{state.get('image_summary', '') or 'No image summary.'}

Primary evidence (theory, worked problems, silver notes, solved final references):
{primary_context}

Verification evidence (official solutions):
{verification_context}

Solve plan:
{state['solve_plan'].model_dump_json(indent=2)}

Tool results:
{tool_summary}

Requirements:
- Respond in {state['preferred_language']} unless the user explicitly asked otherwise.
- Use the verification evidence only as a checking layer, not as the sole source of reasoning.
- Clearly distinguish theory basis, problem similarity, recommended-solution verification, and final-solution references.
- Run a final consistency pass before you answer:
  - if poles have negative real parts, do not call the system unstable
  - if a system is identified as type 2, step and ramp steady-state errors should be zero and parabolic error should be finite
  - if you compute a dominant pole pair, the rise time, overshoot, and settling time must match that same pair
  - the closing summary must agree with the derivation above it
- If information is insufficient, state what is missing.
- Use source markers like [1], [2] and list them in the Sources section.
"""
        draft_answer = self.ollama.chat(
            state["model_name"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": final_prompt},
            ],
        )
        answer, review_used = review_and_revise_answer(
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
    for idx, hit in enumerate(hits[:limit], start=1):
        page_or_slide = hit.unit.page_or_slide or "?"
        lines.append(
            f"[{idx}] family={hit.unit.source_family} file={hit.unit.file_path} "
            f"page_or_slide={page_or_slide} chapter={hit.unit.chapter or '-'} "
            f"problem_id={hit.unit.problem_id or '-'} score={hit.score:.3f}\n{hit.unit.text[:max_chars]}"
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
) -> tuple[str, bool]:
    if mode != "solve" or not draft_answer.strip():
        return draft_answer, False

    patched_answer, patched = apply_deterministic_consistency_fixes(draft_answer, tool_results)
    heuristic_issues = detect_consistency_issues(patched_answer, tool_results)
    if patched and not heuristic_issues:
        return patched_answer, True

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
        raw = ollama.chat(
            model_name,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": review_prompt},
            ],
            json_output=True,
        )
        parsed = ollama.parse_json(raw)
        verdict = str(parsed.get("verdict", "ok")).strip().lower()
        revised_answer = str(parsed.get("revised_answer", "")).strip()
        if verdict == "revise" and revised_answer and not revision_looks_degraded(patched_answer, revised_answer):
            return revised_answer, True
        if revised_answer and not revision_looks_degraded(patched_answer, revised_answer):
            return revised_answer, False
    except Exception:
        pass

    if heuristic_issues:
        fallback_prompt = f"""
Revise the following control-systems answer so that it is internally consistent.
Keep the same structure and language ({preferred_language}).
Fix contradictions only. Do not add new citations or new evidence.
Prefer minimal edits, and do not replace correct values with placeholders.

Issues to fix:
{json.dumps(heuristic_issues, ensure_ascii=False, indent=2)}

Answer:
{patched_answer}
"""
        try:
            revised_answer = ollama.chat(
                model_name,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": fallback_prompt},
                ],
            ).strip()
            if revised_answer and not revision_looks_degraded(patched_answer, revised_answer):
                return revised_answer, True
        except Exception:
            pass

    return patched_answer, patched


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
                width, height = img.size
                min_size = 56
                if width < min_size or height < min_size:
                    scale = max(min_size / max(width, 1), min_size / max(height, 1))
                    new_size = (
                        max(min_size, int(width * scale)),
                        max(min_size, int(height * scale)),
                    )
                    img = img.resize(new_size)
                output = io.BytesIO()
                img.save(output, format="PNG")
                normalized.append(base64.b64encode(output.getvalue()).decode())
        except Exception:
            normalized.append(image_b64)
    return normalized
