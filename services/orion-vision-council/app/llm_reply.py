"""Shared helpers for reading an orion-llm-gateway ChatResultPayload reply.

Both `_call_llm_raw` (main.py, the council's own metacog interpretation
calls) and `run_foveal_probe` (foveal_probe.py) RPC the identical
CHANNEL_LLM_REQUEST / ChatResultPayload contract and need to agree on how to
pull real text out of a reply. Originally duplicated -- foveal_probe.py used
`ChatResultPayload.text` (top-level content/text only) while main.py's
`_extract_chat_result_text` additionally fell back through
`choices[0].message.content` and `raw.choices[...]` shapes. That divergence
meant a reply shape only the choices-fallback path handles (a future gateway
error/legacy branch, say) would make one caller see a real answer and the
other see a false empty response -- code-review caught this drifting apart
during the foveal probe's gateway rewrite.
"""

from __future__ import annotations

from typing import Any

from orion.llm.openai_message_content import join_openai_message_content

# The embedded-error convention orion-llm-gateway's llm_backend.py uses when
# it wants to report a failure as chat content rather than a decode-level
# error (e.g. "[Error: attachments could not be read: ...]"). Shared here so
# both call sites of this reply contract check the identical literal instead
# of two copies that could silently drift apart.
GATEWAY_ERROR_PREFIX = "[Error:"


def extract_chat_result_text(payload: Any) -> str:
    if payload is None:
        return ""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, dict):
        return join_openai_message_content(payload)

    for key in ("content", "text"):
        text = join_openai_message_content(payload.get(key))
        if text:
            return text

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message") if isinstance(first.get("message"), dict) else {}
        text = join_openai_message_content(msg.get("content"))
        if text:
            return text
        text = join_openai_message_content(first.get("text"))
        if text:
            return text

    raw = payload.get("raw")
    if isinstance(raw, dict):
        raw_choices = raw.get("choices")
        if isinstance(raw_choices, list) and raw_choices:
            first = raw_choices[0] if isinstance(raw_choices[0], dict) else {}
            msg = first.get("message") if isinstance(first.get("message"), dict) else {}
            text = join_openai_message_content(msg.get("content"))
            if text:
                return text

    return ""
