from __future__ import annotations

import re

CAPTION_PROMPT = (
    "List visible objects and people. "
    "State only what is directly visible. "
    "No guesses about activity."
)

_LEGACY_PROMPT_ECHO = re.compile(r"describe this image", re.I)


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split()).strip(".,!?")


def is_caption_prompt_echo(text: str) -> bool:
    """True when caption text is the VLM prompt echoed back, not scene description."""
    raw = (text or "").strip()
    if not raw:
        return False
    if _LEGACY_PROMPT_ECHO.search(raw):
        return True
    norm = _normalize(raw)
    prompt_norm = _normalize(CAPTION_PROMPT)
    if norm == prompt_norm:
        return True
    # BLIP often echoes the prompt with minor suffix noise (e.g. trailing "s").
    if norm.startswith(prompt_norm) and len(norm) <= len(prompt_norm) + 4:
        return True
    prompt_tokens = prompt_norm.split()
    text_tokens = norm.split()
    if len(text_tokens) <= len(prompt_tokens) + 1 and text_tokens[: len(prompt_tokens)] == prompt_tokens:
        return True
    return False
