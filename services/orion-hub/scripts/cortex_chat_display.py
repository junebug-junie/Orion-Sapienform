from __future__ import annotations

from orion.schemas.cortex.contracts import CortexChatResult


def hub_effective_chat_text(resp: CortexChatResult) -> str:
    """
    Prefer the longest non-empty final answer between CortexChatResult.final_text and
    CortexClientResult.final_text. Gateway usually sets both equal, but some paths can
    diverge; curl/HTTP exposes both via `text` + `raw`, while WS historically only sent
    `llm_response` — coalescing keeps Hub delivery faithful.
    """
    top = str(resp.final_text or "").strip()
    nested = ""
    if resp.cortex_result is not None:
        nested = str(getattr(resp.cortex_result, "final_text", None) or "").strip()
    if len(nested) > len(top):
        return nested
    return top or nested
