from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ChatHistoryDocBuilder:
    doc_mode: str = "prompt+response"
    max_chars: int = 6000

    def build(self, row: Dict[str, Any]) -> str:
        prompt = (row.get("prompt") or "").strip()
        response = (row.get("response") or "").strip()

        mode = (self.doc_mode or "prompt+response").lower().strip()
        if mode == "prompt":
            text = prompt or response
        elif mode == "response":
            text = response or prompt
        else:
            if response:
                text = f"User: {prompt}\nAssistant: {response}".strip()
            else:
                text = prompt or response

        return self._truncate(text or "")

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars].rstrip()
