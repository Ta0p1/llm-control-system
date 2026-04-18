# Offline Control System Assistant

An offline control-systems study and solving assistant built around `Qwen + Ollama + Qdrant + LangGraph + local math tools`.

## What This Version Does

- Defaults to English answers
- Supports image attachments for problem screenshots
- Indexes `knowledge/` into layered collections:
  - `theory_gold`
  - `problem_gold`
  - `solution_gold`
  - `notes_silver` from local JSON files in `data/silver_notes/`
  - `final_solution_silver` from local JSON files in `data/silver_notes/`
- Excludes the raw `AER372W_2025_final_test.pdf` from the silver-final workflow
- Uses a staged solve pipeline:
  - classify
  - parse image if present
  - retrieve theory/problems
  - build a solve plan
  - run local math tools
  - retrieve official solutions for verification
  - compose the final answer

## Local Stack

- Runtime model:
  - text reasoning: `qwen3:8b`
  - image parsing: `qwen2.5vl:7b`
- Embedding model:
  - preferred: `qwen3-embedding:4b`
  - fallback: `bge-m3`
- Retrieval:
  - local embedded `Qdrant`
- Workflow:
  - `LangGraph`
- Math tools:
  - `sympy`
  - `scipy`
  - `python-control`

## Important Defaults

- Raw final-test PDFs are not ingested directly into the main gold collections.
- Final worked solutions generated through the ChatGPT UI workflow are indexed into `final_solution_silver`.
- `Docling` support is installed but disabled by default.
- To enable Docling-based parsing explicitly:

```powershell
$env:ENABLE_DOCLING = "1"
```

## Install

1. Install [Ollama](https://ollama.com/)
2. Pull the local models:

```powershell
.\pull_models.ps1
```

3. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Start

```powershell
.\start.ps1
```

The app runs at [http://127.0.0.1:8000](http://127.0.0.1:8000).
`main` now defaults to `qwen3:8b` for text and `qwen2.5vl:7b` for image questions.
The `qwen3.5:9b` experiment is preserved on branch `codex/qwen35-9b-runtime-experiment`.

## Text-Only Diagram Input Guide

If your real usage environment cannot upload images and you must describe figures in text, use:

- [docs/TEXT_ONLY_IMAGE_DESCRIPTION_GUIDE.md](docs/TEXT_ONLY_IMAGE_DESCRIPTION_GUIDE.md)

It includes:

- a general template for describing missing images
- control-systems-specific templates for block diagrams, root locus, pole-zero maps, time-response plots, and frequency-response plots
- examples of good vs bad descriptions

## API

### `GET /health`

Returns:
- Ollama connectivity
- Qdrant availability
- installed models
- required runtime model
- indexed collections
- total indexed units

### `POST /ingest`

Example:

```json
{
  "force_full_rebuild": true,
  "rebuild_scope": "core",
  "include_silver": true
}
```

### `POST /chat`

Example:

```json
{
  "message": "For a standard second-order system, how do I estimate peak time and settling time from zeta and wn?",
  "session_id": "web",
  "preferred_language": "english",
  "mode": "learning"
}
```

### `POST /eval/run`

Example:

```json
{
  "chapter": "chapter3",
  "limit_per_chapter": 1,
  "use_solution_verification": true
}
```

## Final-Answer Baseline

This repo now includes a lightweight benchmark for evaluating whether a model can read an AER372 final-exam question and generate a correct, complete, well-structured answer in one text-only turn.

Build the benchmark bundle:

```powershell
python -m app.final_generation_baseline build
```

Create a run folder for a model:

```powershell
python -m app.final_generation_baseline make-run --model "<model name>"
```

After manual scoring, summarize the run:

```powershell
python -m app.final_generation_baseline summarize --score-sheet "<path to score_sheet.csv>"
```

## Silver Notes Workflow

This project now includes a local workflow for generating `notes_silver` with ChatGPT UI.

### 1. Export a chapter batch for ChatGPT

Example: export the first 5 problem-solution pairs from Chapter 2

```powershell
python -m app.teacher_pipeline export --chapter chapter2 --limit 5
```

This creates a local batch folder under:

```text
data/teacher_batches/chapter2/
```

with:

- `chapter2_problem_cards_source.json`
- `chapter2_chatgpt_prompt.txt`
- `chapter2_how_to_use.txt`

### 2. Use ChatGPT UI

- Open `chapter2_chatgpt_prompt.txt`
- Copy the full prompt into ChatGPT
- Ask for raw JSON only
- Save the ChatGPT reply as a local JSON file, for example:

```text
chapter2_chatgpt_output.json
```

### 3. Validate the returned JSON

```powershell
python -m app.teacher_pipeline validate --input "<path to chapter2_chatgpt_output.json>" --chapter chapter2
```

### 4. Merge into `notes_silver`

```powershell
python -m app.teacher_pipeline merge --input "<path to chapter2_chatgpt_output.json>" --chapter chapter2
```

This writes or updates:

```text
data/silver_notes/chapter2_problem_cards.json
```

### 5. Re-index locally

```powershell
python -m app.teacher_pipeline reingest
```

### Expected card types

For each problem pair, the prompt asks ChatGPT to generate:

- one `method_card`
- one `formula_card`
- one `pitfall_card`

### Review guidance

Before changing `verification_status` to `verified`, check:

- the note stays faithful to the official problem and solution
- no new unsupported conclusion was added
- formulas and variable meanings are correct
- `chapter`, `pair_key`, and `problem_id` are correct

## Final Solution Workflow

This project also supports a separate `final_solution_silver` workflow for all final tests except:

- `AER372W_2025_final_test.pdf`

### 1. Export a final exam batch

Example:

```powershell
python -m app.teacher_pipeline export-finals --exam AER372S_2023_final_test
```

This creates a local batch folder under:

```text
data/teacher_batches/finals/AER372S_2023_final_test/
```

with:

- `AER372S_2023_final_test_source.json`
- `AER372S_2023_final_test_chatgpt_prompt.txt`
- `AER372S_2023_final_test_how_to_use.txt`
- exported page images under `page_assets/` when available

### 2. Use ChatGPT UI for full worked solutions

- Open the prompt file in the exam batch folder
- Attach the exported page images when available
- Paste the prompt into ChatGPT
- Ask for raw JSON only
- Save the response as a local JSON file

### 3. Validate the returned final-solution JSON

```powershell
python -m app.teacher_pipeline validate --input "<path to final json>" --exam AER372S_2023_final_test
```

### 4. Merge into `final_solution_silver`

```powershell
python -m app.teacher_pipeline merge --input "<path to final json>" --exam AER372S_2023_final_test
```

This writes or updates:

```text
data/silver_notes/final_AER372S_2023_final_test_solutions.json
```

### 5. Re-index locally

```powershell
python -m app.teacher_pipeline reingest
```

### Expected final-solution fields

Each solved final question should include:

- `solution_id`
- `exam_id`
- `question_id`
- `title`
- `question_text`
- `full_solution`
- `final_answer`
- `key_formulas`
- `method_tags`
- `teacher_model`
- `verification_status`
- `metadata`
