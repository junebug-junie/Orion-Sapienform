"""Ensure SuggestDraftV1 utterance nodes have schema:text sources before RDF/SHACL."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from orion.memory_graph.dto import SuggestDraftV1


def _uid_tail(uid: str) -> str:
    s = str(uid or "").strip()
    if ":" in s:
        return s.rsplit(":", 1)[-1]
    return s


def _keys_match(a: str, b: str) -> bool:
    a = str(a or "").strip()
    b = str(b or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    return _uid_tail(a) == _uid_tail(b)


def _lookup_text(texts: Mapping[str, str], uid: str) -> Optional[str]:
    key = str(uid or "").strip()
    if not key:
        return None
    direct = str(texts.get(key) or "").strip()
    if direct:
        return direct
    for map_key, val in texts.items():
        if _keys_match(map_key, key):
            found = str(val or "").strip()
            if found:
                return found
    return None


def _situation_label_fallback(draft: SuggestDraftV1, uid: str) -> Optional[str]:
    for sit in draft.situations or []:
        uids = [str(u or "").strip() for u in (sit.utterance_ids or [])]
        if not any(_keys_match(u, uid) for u in uids):
            continue
        label = str(sit.label or "").strip()
        if label:
            return label
    return None


def ensure_draft_utterance_text(
    draft: SuggestDraftV1,
    *,
    supplemental: Optional[Mapping[str, str]] = None,
) -> SuggestDraftV1:
    """Merge supplemental turn text and infer missing utterance_text_by_id entries."""
    merged: Dict[str, str] = dict(draft.utterance_text_by_id or {})
    if supplemental:
        for map_key, val in supplemental.items():
            key = str(map_key or "").strip()
            text = str(val or "").strip()
            if not key or not text:
                continue
            if not str(merged.get(key) or "").strip():
                merged[key] = text

    for uid in draft.utterance_ids:
        key = str(uid or "").strip()
        if not key:
            continue
        resolved = _lookup_text(merged, key) or _situation_label_fallback(draft, key)
        if resolved:
            merged[key] = resolved

    missing = [
        str(uid).strip()
        for uid in draft.utterance_ids
        if str(uid).strip() and not str(merged.get(str(uid).strip()) or "").strip()
    ]
    if missing:
        raise ValueError(f"utterance_text_missing:{','.join(missing)}")

    return draft.model_copy(update={"utterance_text_by_id": merged})
