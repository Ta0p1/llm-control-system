from __future__ import annotations

import json
import subprocess
from typing import Any

import httpx

from app.config import CHAT_MODEL_CANDIDATES, DEV_CHAT_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECONDS
from app.schemas import RuntimeProfile


def _run_nvidia_smi() -> tuple[str, int]:
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        line = completed.stdout.strip().splitlines()[0]
        name, memory_text = [part.strip() for part in line.split(",", maxsplit=1)]
        return name, int(memory_text)
    except Exception:
        return "unknown", 0


def fetch_installed_ollama_models() -> tuple[bool, list[str]]:
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
        payload = response.json()
        models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
        return True, models
    except Exception:
        return False, []


def recommend_chat_model(installed_models: list[str], total_vram_mb: int) -> str:
    preferred = CHAT_MODEL_CANDIDATES[0]
    fallback = DEV_CHAT_MODEL
    normalized = set(installed_models)

    if preferred in normalized and total_vram_mb >= 11000:
        return preferred
    if fallback in normalized:
        return fallback
    for candidate in CHAT_MODEL_CANDIDATES:
        if candidate in normalized:
            return candidate
    return fallback


def build_runtime_profile() -> RuntimeProfile:
    gpu_name, total_vram_mb = _run_nvidia_smi()
    ollama_reachable, installed_models = fetch_installed_ollama_models()
    recommended_model = recommend_chat_model(installed_models, total_vram_mb)
    return RuntimeProfile(
        gpu_name=gpu_name,
        total_vram_mb=total_vram_mb,
        installed_models=installed_models,
        recommended_model=recommended_model,
        ollama_reachable=ollama_reachable,
    )


def profile_as_dict() -> dict[str, Any]:
    return json.loads(build_runtime_profile().model_dump_json())
