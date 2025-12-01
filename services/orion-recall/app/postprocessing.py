# app/postprocessing.py
from __future__ import annotations

from typing import List, Tuple

from .types import Fragment, RecallQuery


def _dedupe(fragments: List[Fragment]) -> List[Fragment]:
    seen = set()
    out: List[Fragment] = []
    for f in fragments:
        key = (f.kind, f.id, (f.text or "").strip()[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _mix_kinds(fragments: List[Fragment], max_items: int) -> List[Fragment]:
    """
    Very light kind mixing: prefer a blend of collapse + chat + assoc
    instead of 100% of one kind.
    """
    if not fragments:
        return []

    collapse = [f for f in fragments if f.kind == "collapse"]
    chat = [f for f in fragments if f.kind == "chat"]
    assoc = [f for f in fragments if f.kind == "association"]
    other = [f for f in fragments if f.kind not in ("collapse", "chat", "association")]

    out: List[Fragment] = []

    # Rough target ratios; don't overthink this.
    collapse_quota = max_items // 3
    chat_quota = max_items // 3

    out.extend(collapse[:collapse_quota])
    out.extend(chat[:chat_quota])

    remaining = max_items - len(out)
    if remaining > 0:
        pool = assoc + other + collapse[collapse_quota:] + chat[chat_quota:]
        for f in pool:
            if len(out) >= max_items:
                break
            if f not in out:
                out.append(f)

    return out[:max_items]

def postprocess_fragments(fragments: List[Fragment], q: RecallQuery) -> List[Fragment]:
    if not fragments:
        return []

    deduped = _dedupe(fragments)

    # Use the full length as the cap, i.e. no effective limit
    max_items = len(deduped)
    mixed = _mix_kinds(deduped, max_items=max_items)
    return mixed
