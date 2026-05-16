from __future__ import annotations

import hashlib
import re
from typing import Any

STOP_PHRASES = {
    "codex",
    "cursor",
    "orion",
    "juniper",
    "plan mode",
    "default mode",
    "today",
    "tomorrow",
    "yesterday",
}


def compact(value: Any, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip()


def bounded(value: float) -> float:
    return round(min(1.0, max(0.0, float(value or 0.0))), 3)


def stable_id(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def unique(seq: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in seq:
        item = compact(raw, 120)
        key = item.lower()
        if not item or key in seen or key in STOP_PHRASES:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out
