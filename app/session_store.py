from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app.config import SESSIONS_DIR
from app.schemas import SessionHistoryResponse, SessionMessage


class SessionStore:
    def __init__(self, root: Path = SESSIONS_DIR) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str) -> SessionHistoryResponse:
        path = self._path_for(session_id)
        if not path.exists():
            return SessionHistoryResponse(session_id=session_id, messages=[])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return SessionHistoryResponse(session_id=session_id, messages=[])
        messages = [SessionMessage.model_validate(item) for item in payload.get("messages", [])]
        return SessionHistoryResponse(session_id=session_id, messages=messages)

    def append_exchange(self, session_id: str, *, user_message: str, assistant_message: str) -> SessionHistoryResponse:
        history = self.load(session_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        history.messages.extend(
            [
                SessionMessage(role="user", content=user_message, created_at=timestamp),
                SessionMessage(role="assistant", content=assistant_message, created_at=timestamp),
            ]
        )
        self._write(history)
        return history

    def clear(self, session_id: str) -> SessionHistoryResponse:
        history = SessionHistoryResponse(session_id=session_id, messages=[])
        self._write(history)
        return history

    def _write(self, history: SessionHistoryResponse) -> None:
        path = self._path_for(history.session_id)
        path.write_text(history.model_dump_json(indent=2), encoding="utf-8")

    def _path_for(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", session_id).strip("_") or "default"
        return self.root / f"{safe_id}.json"
