from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.config import BENCHMARKS_DIR, SILVER_NOTES_DIR
from app.schemas import (
    BenchmarkProblemRecord,
    BenchmarkResponseRecord,
    BenchmarkRubric,
    BenchmarkRubricDimension,
    BenchmarkSummaryRecord,
    FinalSolutionRecord,
)

BENCHMARK_ID = "aer372_final_generation_baseline_v1"
BENCHMARK_TITLE = "AER372 Final Answer-Generation Lightweight Baseline"
BENCHMARK_DIR = BENCHMARKS_DIR / BENCHMARK_ID

GENERAL_DIMENSIONS = [
    BenchmarkRubricDimension(
        name="Correctness",
        max_points=4.0,
        description="Formulas, derivation logic, numerical values, and final conclusion are correct.",
    ),
    BenchmarkRubricDimension(
        name="Completeness",
        max_points=3.0,
        description="Covers the key steps, intermediate quantities, and required conclusions without major jumps.",
    ),
    BenchmarkRubricDimension(
        name="Method quality",
        max_points=2.0,
        description="Uses a standard control-theory solution path and chooses the right representation or approximation.",
    ),
    BenchmarkRubricDimension(
        name="Presentation",
        max_points=1.0,
        description="Answer is organized, readable, and symbol usage stays consistent.",
    ),
]

CURATED_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "AER372-B01",
        "source_solution_id": "AER372S_2023_final_test_Q1-a",
        "topic": "second_order_decay_ratio",
        "difficulty": "easy",
        "expected_capabilities": [
            "Map decay-ratio language to the standard logarithmic decrement formula.",
            "Solve symbolically for damping ratio zeta.",
            "Return a compact exact expression and a reasonable numeric value.",
        ],
        "required_elements": [
            "State the decay-ratio relation for a standard underdamped second-order system.",
            "Take logarithms and isolate zeta.",
            "Give the final exact expression for zeta and a numerical approximation.",
        ],
        "major_error_conditions": [
            "Uses percent overshoot formula instead of decay-ratio formula.",
            "Returns a damping ratio outside the physically valid 0 < zeta < 1 range.",
        ],
        "scoring_notes": [
            "A correct final value with missing derivation should lose completeness points.",
        ],
    },
    {
        "id": "AER372-B02",
        "source_solution_id": "AER372W_2024_final_test_Q2",
        "topic": "second_order_frequency_metrics",
        "difficulty": "easy",
        "expected_capabilities": [
            "Move between closed-loop T(s) and equivalent unity-feedback open-loop L(s).",
            "Derive bandwidth from the closed-loop magnitude condition.",
            "Derive crossover frequency and phase margin from the open-loop form.",
        ],
        "required_elements": [
            "Write T(s) in standard second-order form and derive or state a compatible open-loop L(s).",
            "Set the bandwidth condition |T(jw_bw)| = 1/sqrt(2).",
            "Set the crossover condition |L(jw_c)| = 1.",
            "Express phase margin using angle L(jw_c).",
        ],
        "major_error_conditions": [
            "Confuses bandwidth condition with the gain-crossover condition.",
            "Computes phase margin from the closed-loop transfer function instead of open-loop phase.",
        ],
        "scoring_notes": [
            "Equivalent final expressions are acceptable if algebraically consistent.",
        ],
    },
    {
        "id": "AER372-B03",
        "source_solution_id": "AER372S_2023_final_test_Q2",
        "topic": "system_type_tracking_regulation",
        "difficulty": "medium",
        "expected_capabilities": [
            "Derive Y/R and Y/W from the block diagram structure.",
            "Use low-frequency reasoning to enforce Type 1 tracking and disturbance regulation.",
            "State conditions on both Dc(s) and the feedback parameters a and b.",
        ],
        "required_elements": [
            "Express the plant and feedback blocks explicitly and derive the relevant closed-loop transfer functions.",
            "Impose the low-frequency unity-feedback condition on a and b.",
            "Identify the needed integrator structure in Dc(s).",
            "Explain why the same conditions make both tracking and regulation Type 1.",
        ],
        "major_error_conditions": [
            "Treats tracking and regulation as identical without deriving the disturbance path.",
            "Misses the low-frequency condition a = b or its equivalent H(0) = 1.",
            "Cancels or omits the required controller integrator.",
        ],
        "scoring_notes": [
            "If the answer states the right conditions but never justifies them with low-frequency behavior, cap completeness at 1 point.",
        ],
    },
    {
        "id": "AER372-B04",
        "source_solution_id": "AER372S_2023_final_test_Q4-a",
        "topic": "routh_stability",
        "difficulty": "medium",
        "expected_capabilities": [
            "Reconstruct the OLTF structure from Bode asymptote slopes and break frequencies.",
            "Form the closed-loop characteristic polynomial with proportional gain K.",
            "Apply Routh's criterion to find the stable gain interval.",
        ],
        "required_elements": [
            "Infer poles and zeros correctly from the slope changes.",
            "Use the break-point magnitudes to determine the OLTF scale factor.",
            "Write the characteristic polynomial after adding the proportional controller.",
            "Show the Routh first-column conditions and the final gain range.",
        ],
        "major_error_conditions": [
            "Infers the wrong number of poles or zeros from the asymptotes.",
            "Applies Routh directly to the open-loop transfer function instead of the closed-loop characteristic polynomial.",
            "Reports a stable gain interval with the wrong sign or wrong upper bound.",
        ],
        "scoring_notes": [
            "A correct final interval without the reconstructed OLTF should lose method-quality points.",
        ],
    },
    {
        "id": "AER372-B05",
        "source_solution_id": "AER372W_2024_final_test_Q3_b",
        "topic": "root_locus_crossover_gain",
        "difficulty": "medium",
        "expected_capabilities": [
            "Use the root-locus-derived plant model as a frequency-domain loop transfer function.",
            "Apply the gain-crossover magnitude condition at a prescribed frequency.",
            "Evaluate the plant magnitude at jw_c and solve for K_c.",
        ],
        "required_elements": [
            "Write L(s) = K G(s) or an equivalent proportional-control open-loop model.",
            "Use |L(jw_c)| = 1 at w_c = 3 rad/s.",
            "Compute or simplify G(j3) and its magnitude correctly.",
            "Return the numerical gain K_c.",
        ],
        "major_error_conditions": [
            "Uses the phase condition instead of the magnitude condition to determine K_c.",
            "Confuses closed-loop poles with open-loop poles when forming G(s).",
        ],
        "scoring_notes": [
            "Minor arithmetic mistakes with a clearly correct setup can still earn most correctness points.",
        ],
    },
    {
        "id": "AER372-B06",
        "source_solution_id": "AER372W_2024_final_test_Q3_c_ii",
        "topic": "root_locus_transient_estimation",
        "difficulty": "medium",
        "expected_capabilities": [
            "Identify the dominant pole pair among multiple closed-loop poles.",
            "Approximate the response as second order.",
            "Estimate rise time, overshoot, and settling time from sigma, omega_d, zeta, and omega_n.",
        ],
        "required_elements": [
            "Explain which pole pair is dominant and why.",
            "Compute zeta and omega_n from the given dominant poles.",
            "Use standard transient-response estimates for rise time, overshoot, and settling time.",
            "Comment briefly on what the numbers imply qualitatively.",
        ],
        "major_error_conditions": [
            "Chooses the faster pole pair as dominant even though it is farther left in the complex plane.",
            "Uses formulas for a non-oscillatory system despite complex dominant poles.",
        ],
        "scoring_notes": [
            "Reasonable engineering approximations are acceptable if the dominant-pole logic is sound.",
        ],
    },
    {
        "id": "AER372-B07",
        "source_solution_id": "AER372W_2024_final_test_Q1",
        "topic": "pid_ziegler_nichols",
        "difficulty": "hard",
        "expected_capabilities": [
            "Parameterize the plant from its zero and unstable pole pair.",
            "Derive the ultimate gain and oscillation frequency under proportional control.",
            "Convert ultimate values into PID gains using Ziegler-Nichols rules.",
        ],
        "required_elements": [
            "Write the plant in a real-coefficient form using the LHP zero and RHP pole pair.",
            "Form the proportional closed-loop characteristic equation and impose marginal stability.",
            "Solve for K_u and the ultimate frequency or period.",
            "Map K_u and T_u to PID gains in the requested parameterization.",
        ],
        "major_error_conditions": [
            "Treats the RHP pole pair as if it were stable or LHP.",
            "Applies Ziegler-Nichols tuning constants without first deriving K_u and T_u.",
            "Returns inconsistent PID parameter forms without defining the controller convention.",
        ],
        "design_gate_rules": [
            "If the controller structure is not PID or the Ziegler-Nichols mapping is wrong, cap Correctness at 1 point.",
        ],
        "scoring_notes": [
            "Full credit requires a complete symbolic dependency on the plant zero and pole parameters.",
        ],
    },
    {
        "id": "AER372-B08",
        "source_solution_id": "AER372H1_2022_final_test_Q4-d",
        "topic": "lag_compensation_disturbance_spec",
        "difficulty": "hard",
        "expected_capabilities": [
            "Start from a previously designed PD controller and augment low-frequency gain with a lag compensator.",
            "Translate the disturbance steady-state requirement into a DC-gain requirement.",
            "Choose lag break frequencies using the given upper-corner-frequency rule.",
        ],
        "required_elements": [
            "State or reuse the PD controller DC gain from the earlier part.",
            "Convert the disturbance error specification into a required low-frequency gain increase.",
            "Choose the lag zero at 10% of crossover frequency and compute the lag pole from the gain ratio.",
            "Write the final lag or combined controller explicitly.",
        ],
        "major_error_conditions": [
            "Places the lag pole above the lag zero, producing a lead instead of a lag network.",
            "Ignores the disturbance specification when choosing the lag gain ratio.",
            "Changes the controller structure without preserving the earlier PD design.",
        ],
        "design_gate_rules": [
            "If the added compensator is not a lag network, cap Correctness at 1 point.",
            "If the answer gives only a final controller without explaining the DC-gain requirement, cap Completeness at 1 point.",
        ],
        "scoring_notes": [
            "Small numeric variation is acceptable when the DC-gain target and pole-zero placement logic are correct.",
        ],
    },
]

PROMPT_TEMPLATE = """You are answering a closed-book control-systems final exam problem.

Requirements:
- Use English.
- Do not use external tools, code, calculators, or retrieval.
- Show the derivation steps, key formulas, necessary intermediate conclusions, and the final answer.
- You do not need LaTeX formatting, but your solution must be structured and readable.
- If you make an assumption, state it explicitly.

Output format:
1. Problem interpretation
2. Derivation
3. Final result
4. Brief sanity check

Problem:
{problem_text}
"""


def sanitize_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def load_solution_index() -> dict[str, FinalSolutionRecord]:
    records: dict[str, FinalSolutionRecord] = {}
    for path in sorted(SILVER_NOTES_DIR.glob("final_*_solutions.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for item in payload:
            record = FinalSolutionRecord.model_validate(item)
            records[record.solution_id] = record
    return records


def build_problem_records() -> list[BenchmarkProblemRecord]:
    solution_index = load_solution_index()
    problems: list[BenchmarkProblemRecord] = []
    for spec in CURATED_PROBLEMS:
        source_id = spec["source_solution_id"]
        if source_id not in solution_index:
            raise RuntimeError(f"Missing source solution record: {source_id}")
        source = solution_index[source_id]
        rubric = BenchmarkRubric(
            total_points=10.0,
            dimensions=GENERAL_DIMENSIONS,
            required_elements=spec["required_elements"],
            major_error_conditions=spec["major_error_conditions"],
            design_gate_rules=spec.get("design_gate_rules", []),
            scoring_notes=spec.get("scoring_notes", []),
        )
        problems.append(
            BenchmarkProblemRecord(
                id=spec["id"],
                source_exam=source.exam_id,
                topic=spec["topic"],
                difficulty=spec["difficulty"],
                problem_text=source.question_text.strip(),
                expected_capabilities=spec["expected_capabilities"],
                rubric=rubric,
            )
        )
    return problems


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def build_manifest(problems: list[BenchmarkProblemRecord]) -> dict[str, Any]:
    difficulty_counts = defaultdict(int)
    topic_counts = defaultdict(int)
    for problem in problems:
        difficulty_counts[problem.difficulty] += 1
        topic_counts[problem.topic] += 1

    return {
        "benchmark_id": BENCHMARK_ID,
        "title": BENCHMARK_TITLE,
        "task_type": "single_turn_text_only_answer_generation",
        "total_problems": len(problems),
        "total_points": int(sum(problem.rubric.total_points for problem in problems)),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "source_exams_considered": [
            "AER372H1_2022_final_test",
            "AER372S_2023_final_test",
            "AER372W_2024_final_test",
            "AER372W_2025_final_test",
        ],
        "selection_notes": [
            "2022 and 2023 were treated as one template family to avoid near-duplicate sampling.",
            "2024 and 2025 image-dependent Bode-plot design questions were not selected for the text-only v1 baseline because the numeric plant curves are not recoverable from plain text alone.",
            "The selected set preserves the planned coverage for second-order analysis, Routh stability, root-locus reasoning, controller design, and system type / steady-state error logic.",
        ],
    }


def build_rubric_markdown(problems: list[BenchmarkProblemRecord]) -> str:
    lines = [
        f"# {BENCHMARK_TITLE}",
        "",
        "## Shared scoring dimensions",
        "",
        "- Correctness: 4 points. Check formulas, derivation logic, numerical values, and the final conclusion.",
        "- Completeness: 3 points. Check whether the answer covers the key steps and intermediate quantities instead of jumping straight to the result.",
        "- Method quality: 2 points. Check whether the model chooses a standard control-theory route and the right approximation or representation.",
        "- Presentation: 1 point. Check whether the answer is readable, structured, and symbol usage is consistent.",
        "",
        "## Shared rules",
        "",
        "- If a design problem uses the wrong controller structure, cap Correctness at 1 point.",
        "- If an answer gives only a final result with little or no derivation, cap Completeness at 1 point.",
        "- If the answer reverses a stability direction, error type, or controller-network type, mark a major error.",
        "",
        "## Model quality bands",
        "",
        "- 70-80: Strong baseline",
        "- 55-69: Usable but inconsistent",
        "- 40-54: Partial capability",
        "- Below 40: Not reliable",
        "",
        "## Problem-specific review notes",
        "",
    ]

    for problem in problems:
        lines.append(f"### {problem.id} ({problem.topic}, {problem.difficulty})")
        lines.append("")
        lines.append("Required elements:")
        for item in problem.rubric.required_elements:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Major-error conditions:")
        for item in problem.rubric.major_error_conditions:
            lines.append(f"- {item}")
        if problem.rubric.design_gate_rules:
            lines.append("")
            lines.append("Design gate rules:")
            for item in problem.rubric.design_gate_rules:
                lines.append(f"- {item}")
        if problem.rubric.scoring_notes:
            lines.append("")
            lines.append("Scoring notes:")
            for item in problem.rubric.scoring_notes:
                lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def build_readme(problems: list[BenchmarkProblemRecord]) -> str:
    topic_lines = "\n".join(f"- `{problem.id}`: {problem.topic} ({problem.difficulty})" for problem in problems)
    return f"""# {BENCHMARK_TITLE}

This folder contains a lightweight benchmark for checking whether a model can read a control-systems final exam problem and generate a correct, complete, well-structured answer in a single text-only turn.

## What is included

- `manifest.json`: benchmark metadata and selection notes
- `problems.json`: the 8 curated benchmark problem records
- `prompt_template.txt`: the standard prompt wrapper
- `scoring_rubric.md`: the human-review rubric
- `response_template.json`: the blank response-record schema
- `score_sheet_template.csv`: the blank per-problem scoring sheet
- `summary_template.csv`: the blank run summary table

## Problem list

{topic_lines}

## Recommended workflow

1. Create a run folder:
   `python -m app.final_generation_baseline make-run --model "<model name>"`
2. Send each prompt under `runs/<model>/prompts/` to the model.
3. Paste the outputs into `runs/<model>/responses.json`.
4. Fill the scoring sheet using `scoring_rubric.md`.
5. Summarize the run:
   `python -m app.final_generation_baseline summarize --score-sheet "<path to score_sheet.csv>"`
"""


def write_score_sheet(path: Path, problems: list[BenchmarkProblemRecord], *, model_name: str) -> None:
    fieldnames = [
        "model_name",
        "problem_id",
        "topic",
        "difficulty",
        "correctness",
        "completeness",
        "method_quality",
        "presentation",
        "score_total",
        "major_error",
        "review_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for problem in problems:
            writer.writerow(
                {
                    "model_name": model_name,
                    "problem_id": problem.id,
                    "topic": problem.topic,
                    "difficulty": problem.difficulty,
                    "correctness": "",
                    "completeness": "",
                    "method_quality": "",
                    "presentation": "",
                    "score_total": "",
                    "major_error": "",
                    "review_notes": "",
                }
            )


def write_summary_template(path: Path) -> None:
    fieldnames = ["model_name", "avg_score", "topic_scores", "completeness_rate", "major_error_count"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def build_artifacts(output_dir: Path) -> dict[str, Path]:
    problems = build_problem_records()
    manifest = build_manifest(problems)

    manifest_path = output_dir / "manifest.json"
    problems_path = output_dir / "problems.json"
    prompt_path = output_dir / "prompt_template.txt"
    rubric_path = output_dir / "scoring_rubric.md"
    response_template_path = output_dir / "response_template.json"
    score_sheet_path = output_dir / "score_sheet_template.csv"
    summary_template_path = output_dir / "summary_template.csv"
    readme_path = output_dir / "README.md"

    write_json(manifest_path, manifest)
    write_json(problems_path, [problem.model_dump() for problem in problems])
    write_text(prompt_path, PROMPT_TEMPLATE)
    write_text(rubric_path, build_rubric_markdown(problems))
    write_json(
        response_template_path,
        [
            BenchmarkResponseRecord(
                model_name="<model_name>",
                problem_id=problem.id,
                raw_answer="",
                score_total=0.0,
                score_breakdown={
                    "Correctness": 0.0,
                    "Completeness": 0.0,
                    "Method quality": 0.0,
                    "Presentation": 0.0,
                },
                review_notes="",
            ).model_dump()
            for problem in problems
        ],
    )
    write_score_sheet(score_sheet_path, problems, model_name="<model_name>")
    write_summary_template(summary_template_path)
    write_text(readme_path, build_readme(problems))

    return {
        "manifest": manifest_path,
        "problems": problems_path,
        "prompt_template": prompt_path,
        "rubric": rubric_path,
        "response_template": response_template_path,
        "score_sheet_template": score_sheet_path,
        "summary_template": summary_template_path,
        "readme": readme_path,
    }


def build_prompt(problem_text: str) -> str:
    return PROMPT_TEMPLATE.format(problem_text=problem_text)


def make_run(model_name: str, output_dir: Path | None = None) -> dict[str, Path]:
    problems = build_problem_records()
    run_dir = output_dir or (BENCHMARK_DIR / "runs" / sanitize_name(model_name))
    prompts_dir = run_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for problem in problems:
        write_text(prompts_dir / f"{problem.id}.txt", build_prompt(problem.problem_text))

    response_path = run_dir / "responses.json"
    write_json(
        response_path,
        [
            BenchmarkResponseRecord(
                model_name=model_name,
                problem_id=problem.id,
                raw_answer="",
                score_total=0.0,
                score_breakdown={},
                review_notes="",
            ).model_dump()
            for problem in problems
        ],
    )
    write_score_sheet(run_dir / "score_sheet.csv", problems, model_name=model_name)
    write_summary_template(run_dir / "summary.csv")
    write_text(
        run_dir / "how_to_use.md",
        f"""# Run Guide

Model: {model_name}

1. Send each file under `prompts/` to the model in a fresh single turn.
2. Copy the raw model outputs into `responses.json`.
3. Fill in `score_sheet.csv` using the benchmark rubric.
4. Run:
   `python -m app.final_generation_baseline summarize --score-sheet "{(run_dir / "score_sheet.csv").resolve()}"`
""",
    )
    return {
        "run_dir": run_dir,
        "responses": response_path,
        "score_sheet": run_dir / "score_sheet.csv",
        "summary": run_dir / "summary.csv",
    }


def to_float(value: str) -> float:
    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return 0.0


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize_score_sheet(score_sheet_path: Path, output_path: Path | None = None) -> BenchmarkSummaryRecord:
    rows = list(csv.DictReader(score_sheet_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise RuntimeError(f"Score sheet has no rows: {score_sheet_path}")

    model_name = rows[0].get("model_name", "").strip() or "<unknown>"
    total_scores: list[float] = []
    completeness_scores: list[float] = []
    topic_scores: dict[str, list[float]] = defaultdict(list)
    major_error_count = 0

    for row in rows:
        correctness = to_float(row.get("correctness", ""))
        completeness = to_float(row.get("completeness", ""))
        method_quality = to_float(row.get("method_quality", ""))
        presentation = to_float(row.get("presentation", ""))
        score_total = to_float(row.get("score_total", ""))
        if score_total <= 0.0:
            score_total = correctness + completeness + method_quality + presentation
        total_scores.append(score_total)
        completeness_scores.append(completeness)
        topic_scores[row.get("topic", "unknown").strip() or "unknown"].append(score_total)
        if to_bool(row.get("major_error", "")):
            major_error_count += 1

    summary = BenchmarkSummaryRecord(
        model_name=model_name,
        avg_score=round(sum(total_scores) / len(total_scores), 2),
        topic_scores={topic: round(sum(values) / len(values), 2) for topic, values in sorted(topic_scores.items())},
        completeness_rate=round(sum(completeness_scores) / (3.0 * len(completeness_scores)), 4),
        major_error_count=major_error_count,
    )

    target = output_path or score_sheet_path.with_name("summary.json")
    write_json(target, summary.model_dump())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run the AER372 final answer-generation benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the benchmark artifact bundle.")
    build_parser.add_argument("--output-dir", default=str(BENCHMARK_DIR))

    run_parser = subparsers.add_parser("make-run", help="Create a model-specific run folder with prompts and blank sheets.")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--output-dir", default="")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize a completed scoring sheet.")
    summarize_parser.add_argument("--score-sheet", required=True)
    summarize_parser.add_argument("--output", default="")

    args = parser.parse_args()

    if args.command == "build":
        outputs = build_artifacts(Path(args.output_dir))
        print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))
        return

    if args.command == "make-run":
        output_dir = Path(args.output_dir) if args.output_dir else None
        outputs = make_run(args.model, output_dir=output_dir)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))
        return

    if args.command == "summarize":
        output_path = Path(args.output) if args.output else None
        summary = summarize_score_sheet(Path(args.score_sheet), output_path=output_path)
        print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
