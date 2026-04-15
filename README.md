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
- Excludes `final tests` from the active index for now
- Uses a staged solve pipeline:
  - classify
  - parse image if present
  - retrieve theory/problems
  - build a solve plan
  - run local math tools
  - retrieve official solutions for verification
  - compose the final answer

## Local Stack

- Chat model:
  - dev default: `qwen3:8b`
  - target preference: `qwen3:14b`
- Vision model:
  - `qwen2.5vl:3b`
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

- `final tests` are not indexed yet.
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

## API

### `GET /health`

Returns:
- Ollama connectivity
- Qdrant availability
- installed models
- recommended chat model
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

## Teacher Pipeline Scaffold

The project includes a local scaffold for future cloud-teacher note generation:

- primary teacher target: `gpt-5.2`
- review teacher target: `claude-opus-4.1`
- optional arbiter target: `gemini-2.5-pro`

Teacher-generated notes must be written as JSON into `data/silver_notes/`, then re-indexed with `/ingest`.
