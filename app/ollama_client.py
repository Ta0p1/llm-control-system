from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import EMBED_MODEL, OLLAMA_BASE_URL, OLLAMA_DISABLE_THINKING, OLLAMA_TIMEOUT_SECONDS


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=OLLAMA_TIMEOUT_SECONDS)

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        json_output: bool = False,
        options: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if OLLAMA_DISABLE_THINKING:
            payload["think"] = False
        if json_output:
            payload["format"] = "json"
        if options:
            payload["options"] = options

        with self._client() as client:
            response = client.post("/api/chat", json=payload)
            response.raise_for_status()
        data = response.json()
        message = data.get("message", {}) or {}
        content = str(message.get("content", "") or "").strip()
        if content:
            return content
        thinking = str(message.get("thinking", "") or "").strip()
        if thinking:
            return thinking
        return ""

    def embeddings(self, texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
        if not texts:
            return []

        payload = {"model": model, "input": texts}
        with self._client() as client:
            response = client.post("/api/embed", json=payload)
            if response.status_code == 404:
                return [self._legacy_embedding(text, model) for text in texts]
            response.raise_for_status()
        data = response.json()
        return data.get("embeddings", [])

    def _legacy_embedding(self, text: str, model: str) -> list[float]:
        with self._client() as client:
            response = client.post("/api/embeddings", json={"model": model, "prompt": text})
            response.raise_for_status()
        data = response.json()
        return [float(x) for x in data.get("embedding", [])]

    @staticmethod
    def strip_json_fence(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1)
        return cleaned.strip()

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        return json.loads(OllamaClient.strip_json_fence(text))
