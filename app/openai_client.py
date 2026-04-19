from __future__ import annotations

import json
from typing import Any

from app.config import (
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
    OPENAI_TIMEOUT_SECONDS,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency until installed.
    OpenAI = None


class OpenAIClient:
    def __init__(self) -> None:
        self.api_key = OPENAI_API_KEY
        self._client = None

    def available(self) -> tuple[bool, str]:
        if OpenAI is None:
            return False, "OpenAI SDK is not installed."
        if not self.api_key:
            return False, "OPENAI_API_KEY is not configured."
        return True, ""

    def _get_client(self):
        available, detail = self.available()
        if not available:
            raise RuntimeError(detail)
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, timeout=OPENAI_TIMEOUT_SECONDS)
        return self._client

    def response(
        self,
        *,
        instructions: str,
        input_text: str,
        images: list[str] | None = None,
        model: str = OPENAI_MODEL,
        reasoning_effort: str = OPENAI_REASONING_EFFORT,
        max_output_tokens: int | None = None,
    ) -> str:
        client = self._get_client()
        content: list[dict[str, Any]] = [{"type": "input_text", "text": input_text}]
        for image in images or []:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image}",
                }
            )
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
        }
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        response = client.responses.create(**kwargs)
        output_text = getattr(response, "output_text", "") or ""
        if output_text:
            return str(output_text).strip()
        return self._extract_text(response).strip()

    def embeddings(self, texts: list[str], *, model: str = OPENAI_EMBED_MODEL) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = client.embeddings.create(model=model, input=texts)
        return [list(item.embedding) for item in response.data]

    @staticmethod
    def _extract_text(response: Any) -> str:
        output = getattr(response, "output", []) or []
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", []) or []
            for chunk in content:
                text = getattr(chunk, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)

    @staticmethod
    def strip_json_fence(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1)
        return cleaned.strip()

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        return json.loads(OpenAIClient.strip_json_fence(text))
