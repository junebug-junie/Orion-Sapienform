from __future__ import annotations

import re

CAPTION_PROMPT = (
    "List visible objects and people. "
    "State only what is directly visible. "
    "No guesses about activity."
)
_PROMPT_ECHO = re.compile(r"describe this image", re.I)
_STOPLIST = frozenset({"youtube", "google", "video", "watching", "webcam", "com"})


def sanitize_caption(raw: str) -> tuple[str | None, bool, str | None]:
    text = (raw or "").strip()
    if _PROMPT_ECHO.search(text):
        return None, False, "prompt_echo"
    if len(text) < 12:
        return None, False, "too_short"
    tokens = [t.strip(".,!?").lower() for t in text.split() if t.strip()]
    if not tokens:
        return None, False, "empty"
    stop_hits = sum(1 for t in tokens if t in _STOPLIST)
    if stop_hits / len(tokens) > 0.4:
        return None, False, "stoplist_ratio"
    return text, True, None
