"""Collapse repeated Orion assistant lines in recall snippets (vector → same catchphrase many times)."""

from __future__ import annotations

import re
from typing import List

_WS = re.compile(r"\s+")


def normalize_compare_text(text: str) -> str:
    t = _WS.sub(" ", (text or "").strip())
    for a, b in (
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
    ):
        t = t.replace(a, b)
    return t.lower()


def extract_orion_assistant_from_snippet(snippet: str) -> str:
    """OrionResponse body for ExactUserText/OrionResponse rows; robust if inner quotes exist."""
    raw = _WS.sub(" ", (snippet or "").strip())
    if not raw:
        return ""
    low = raw.lower()
    key = "orionresponse:"
    pos = low.find(key)
    if pos < 0:
        return ""
    tail = raw[pos + len(key) :].strip()
    if not tail.startswith('"'):
        return tail[:800].strip()
    # Opening " … find closing " (last quote is safest for single-line vector snippets)
    inner = tail[1:]
    end = inner.rfind('"')
    if end >= 0:
        return inner[:end].strip()[:800]
    return inner.strip()[:800]


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]{3,}", text.lower()) if tok}


def materially_same_text(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    ta = _token_set(a)
    tb = _token_set(b)
    if len(ta) < 4 or len(tb) < 4:
        return normalize_compare_text(a) == normalize_compare_text(b)
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    if union == 0:
        return False
    return (inter / union) >= 0.88


def duplicate_orion_reply_assistant(a: str, b: str) -> bool:
    """True if two assistant replies are the same boilerplate (subset, near-duplicate, or matching opener)."""
    if not a or not b:
        return False
    na = normalize_compare_text(a)
    nb = normalize_compare_text(b)
    if na == nb:
        return True
    if materially_same_text(na, nb):
        return True
    # Shared long opener (typical canned intro repeated across turns)
    prefix_n = 96
    if len(na) >= 32 and len(nb) >= 32 and na[:prefix_n] == nb[:prefix_n]:
        return True
    min_sub = 28
    if len(na) >= min_sub and len(nb) >= min_sub:
        if na in nb or nb in na:
            return True
    return False


class OrionDigestDeduper:
    """Second line of defense: render skips duplicate assistant bodies even if fusion let them through."""

    def __init__(self) -> None:
        self._seen: List[str] = []

    def should_emit_snippet(self, snippet: str) -> bool:
        body = extract_orion_assistant_from_snippet(snippet)
        if len(body) < 20:
            return True
        for prev in self._seen:
            if duplicate_orion_reply_assistant(body, prev):
                return False
        self._seen.append(body)
        return True
