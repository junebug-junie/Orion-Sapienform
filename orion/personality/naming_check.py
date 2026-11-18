"""Hub's structural check of one naming answer (spec section 4).

Orion was shown one ``DispositionNamingPacketV1`` and wrote one ``:DispositionNaming`` node.
``check_naming`` turns that node (as a plain dict, read back from the graph) into exactly one
``DispositionNamingV1``. It checks structure only: ids against the packet, lengths, and that every
cited evidence id and merge target was actually shown. It never repairs an id and never judges the
name itself (decision 2026-10-20: grounding is recorded, never a refusal; no veto). A failed check
still produces a row, with ``hub_check="refused"`` and the reasons, so the refusal rate is
countable. Nothing written -> ``verdict="no_answer"``.

Pure: no I/O, no clock unless the node carries no ``written_at``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
from uuid import NAMESPACE_URL, uuid5

from orion.schemas.personality import (
    NAMING_NAME_MAX_CHARS,
    NAMING_NAME_MAX_WORDS,
    NAMING_PACKET_MAX_EVIDENCE,
    NAMING_SENTENCE_MAX_CHARS,
    NAMING_SENTENCE_MIN_CHARS,
    DispositionNamingPacketV1,
    DispositionNamingV1,
    NamingRefusedReason,
)

NAME_MIN_CHARS = 2
ANSWER_VERDICTS = ("name", "reject", "merge")
# Field caps of DispositionNamingV1 itself (raw text is kept for audit up to these).
_RAW_NAME_CAP = 200
_RAW_SENTENCE_CAP = 2000
_RAW_EVIDENCE_CAP = NAMING_PACKET_MAX_EVIDENCE * 4


def naming_id_for(run_id: str, packet_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{run_id}:{packet_id}"))


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _evidence_refs(raw: Any) -> list[str]:
    """FalkorDB returns a list property as a list; a JSON string is still read. Order kept, deduped."""
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                raw = json.loads(stripped)
            except ValueError:
                raw = [stripped]
        else:
            raw = [stripped] if stripped else []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        text = _text(item)
        if text and text not in out:
            out.append(text)
    return out[:_RAW_EVIDENCE_CAP]


def _written_at(raw: Any, now: Optional[datetime]) -> datetime:
    """FalkorDB ``timestamp()`` is epoch milliseconds. Anything unreadable -> ``now``."""
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        ms = float(raw)
        if ms > 0:
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    fallback = now or datetime.now(timezone.utc)
    return fallback if fallback.tzinfo else fallback.replace(tzinfo=timezone.utc)


def _name_reasons(name: str, label_excerpt: str) -> list[NamingRefusedReason]:
    if len(name) < NAME_MIN_CHARS:
        return ["missing_name"]
    out: list[NamingRefusedReason] = []
    if (
        len(name) > NAMING_NAME_MAX_CHARS
        or len(name.split()) > NAMING_NAME_MAX_WORDS
        or "\n" in name
        or "\r" in name
    ):
        out.append("name_too_long")
    if name.casefold() == label_excerpt.strip().casefold():
        out.append("name_equals_label")
    return out


def _sentence_reasons(sentence: str) -> list[NamingRefusedReason]:
    if len(sentence) < NAMING_SENTENCE_MIN_CHARS:
        return ["empty_sentence"]
    if len(sentence) > NAMING_SENTENCE_MAX_CHARS:
        return ["sentence_too_long"]
    return []


def check_naming(
    raw: Optional[Mapping[str, Any]],
    packet: DispositionNamingPacketV1,
    run_id: str,
    *,
    now: Optional[datetime] = None,
) -> DispositionNamingV1:
    """One ``DispositionNamingV1`` for this run, always. Never raises on Orion's input."""
    node = dict(raw or {})
    verdict = _text(node.get("verdict")).lower()
    name = _text(node.get("name"))
    sentence = _text(node.get("sentence"))
    merge_with = _text(node.get("merge_with"))
    refs = _evidence_refs(node.get("evidence"))
    written_disposition = _text(node.get("disposition_id"))
    written_packet = _text(node.get("packet_id"))

    # The row is about the packet Hub fetched for this run. A disposition id Orion changed is
    # recorded as written (never repaired) and refused; an empty one is the shown id.
    disposition_id = written_disposition or packet.disposition_id
    base = dict(
        naming_id=naming_id_for(run_id, packet.packet_id),
        run_id=run_id,
        packet_id=packet.packet_id,
        disposition_id=disposition_id,
        name=name[:_RAW_NAME_CAP] or None,
        sentence=sentence[:_RAW_SENTENCE_CAP] or None,
        merge_with=merge_with or None,
        evidence_refs=refs,
        shown=packet,
        written_at=_written_at(node.get("written_at"), now),
    )

    if verdict not in ANSWER_VERDICTS:
        # Nothing written, or only the early placeholder: no answer, not a refusal.
        return DispositionNamingV1(verdict="no_answer", hub_check="accepted", refused_reasons=[], **base)

    reasons: list[NamingRefusedReason] = []
    if written_packet and written_packet != packet.packet_id:
        reasons.append("packet_mismatch")
    if disposition_id != packet.disposition_id:
        reasons.append("unknown_disposition")

    if verdict == "name":
        reasons += _name_reasons(name, packet.label_excerpt)
    if verdict in ("name", "reject"):
        reasons += _sentence_reasons(sentence)
    if verdict == "merge":
        if not merge_with:
            reasons.append("missing_merge_target")
        elif merge_with not in {n.disposition_id for n in packet.neighbors}:
            reasons.append("merge_target_not_shown")

    shown_evidence = {e.evidence_id for e in packet.evidence}
    if not refs:
        reasons.append("no_evidence_refs")
    elif any(ref not in shown_evidence for ref in refs):
        reasons.append("evidence_not_shown")

    deduped = list(dict.fromkeys(reasons))
    return DispositionNamingV1(
        verdict=verdict,  # type: ignore[arg-type]
        hub_check="refused" if deduped else "accepted",
        refused_reasons=deduped,
        **base,
    )
