from __future__ import annotations

from orion.vision.caption_echo import CAPTION_PROMPT, is_caption_prompt_echo

_STOPLIST = frozenset({"youtube", "google", "video", "watching", "webcam", "com"})


def sanitize_caption(raw: str) -> tuple[str | None, bool, str | None]:
    text = (raw or "").strip()
    if is_caption_prompt_echo(text):
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
