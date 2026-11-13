"""Self-naming, pure side: which node Orion is shown next, and the packet Orion is shown.

Design: docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md, sections 1-2,
decisions 2026-10-20. PURE: no I/O, no logging. Evaluated by orion-durable-runs, which holds the nodes
(``GET /personality/naming/next``). Verdicts are applied by ``graph.apply_namings``.

What is due (``naming_due``), in this order:

1. an ``active`` node never shown, strongest first;
2. a node whose last shows got no usable answer (``no_answer``, or Hub refused it): again after
   ``RETRY_AFTER`` (3 days), at most ``MAX_UNUSABLE_TRIES`` (3) in a row; after that it waits for
   doubled evidence like 3;
3. a node with an applied verdict (named or rejected): again only when its ``evidence_count`` at
   least doubled since that verdict, or a source it did not have when shown now counts toward
   promotion; never sooner than ``RESHOW_MIN_GAP`` (14 days) after the last show. Calendar time alone
   never re-asks.

"Applied" is read from the node (``naming_id`` is set only by an applied NAME / REJECT), so a merge
the reducer refused as too far counts as an unusable try, not a verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Optional, Sequence

from orion.personality.graph import (
    DEFAULT_PARAMS,
    SEED_LANE,
    GraphParams,
    Node,
    cosine,
    node_cosine,
    source_key,
)
from orion.schemas.personality import (
    DISPOSITION_LABEL_MAX_CHARS,
    NAMING_NAME_MAX_CHARS,
    NAMING_PACKET_MAX_EVIDENCE,
    NAMING_PACKET_MAX_NEIGHBORS,
    NAMING_SENTENCE_MAX_CHARS,
    DispositionNamingPacketV1,
    NamingEvidenceV1,
    NamingNeighborV1,
    NamingPriorVerdictV1,
)

RETRY_AFTER = timedelta(days=3)
MAX_UNUSABLE_TRIES = 3
RESHOW_MIN_GAP = timedelta(days=14)
SITUATION_DIMS = ("arousal", "day_phase", "company")


def _get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def usable(row: Any) -> bool:
    """An answer the reducer can act on: Hub accepted it and it is not ``no_answer``."""
    return _get(row, "hub_check") == "accepted" and _get(row, "verdict") in ("name", "reject", "merge")


def _shown_sources(row: Any) -> set[str]:
    """Source keys the node had when this row's packet was shown (from the packet's lane counts).
    A journal lane may be metacog (one source with reverie), so it maps to both keys: a source is
    only "new" when no shown lane could have been it."""
    shown = _json(_get(row, "shown")) or {}
    out: set[str] = set()
    for lane in (shown.get("lane_counts") or {}) if isinstance(shown, Mapping) else ():
        out.add(source_key(lane, None))
        if lane == "journal":
            out.add(source_key("journal", {"source_kind": "metacog"}))
    return out


def _shown_count(row: Any) -> Optional[int]:
    shown = _json(_get(row, "shown")) or {}
    n = shown.get("evidence_count") if isinstance(shown, Mapping) else None
    return n if isinstance(n, int) and not isinstance(n, bool) else None


def _promoting_sources(node: Node, params: GraphParams) -> set[str]:
    return {k for k, r in node.lane_recent.items() if k != SEED_LANE and r >= params.promote_lane_min_recent}


def naming_due(nodes: Sequence[Node], naming_log: Iterable[Any], now: datetime, *,
               params: GraphParams = DEFAULT_PARAMS) -> list[Node]:
    """Nodes due a naming packet, most due first (see module docstring). ``naming_log`` holds
    ``orion_disposition_naming`` rows (mappings or ``DispositionNamingV1``)."""
    by_node: dict[str, list[Any]] = {}
    for row in naming_log:
        did = _get(row, "disposition_id")
        if isinstance(did, str) and isinstance(_get(row, "written_at"), datetime):
            by_node.setdefault(did, []).append(row)
    ranked: list[tuple[int, float, str, Node]] = []
    for node in nodes:
        if node.status != "active" or node.merged_into:
            continue
        rows = sorted(by_node.get(node.disposition_id, ()), key=lambda r: (_get(r, "written_at"), _get(r, "naming_id") or ""))
        if not rows:
            if node.shown_count <= 0 and node.naming_id is None:
                ranked.append((0, -node.weight, node.disposition_id, node))
            continue
        applied_idx = next((i for i, r in enumerate(rows) if _get(r, "naming_id") == node.naming_id), None) \
            if node.naming_id else None
        after = rows[applied_idx + 1:] if applied_idx is not None else rows
        last_at = _get(rows[-1], "written_at")
        # Every row after the applied verdict is a try that did not land (no answer, refused, a
        # merge too far, or an answer the reducer has not applied yet).
        tries = len(after)
        if 0 < tries < MAX_UNUSABLE_TRIES:
            if now - last_at >= RETRY_AFTER:
                ranked.append((1, -node.weight, node.disposition_id, node))
            continue
        # A verdict stands (or the retries ran out): only more evidence re-asks.
        if now - last_at < RESHOW_MIN_GAP:
            continue
        if tries == 0 and applied_idx is not None:
            ref, base = rows[applied_idx], node.verdict_evidence_count
        else:
            ref, base = rows[-1], _shown_count(rows[-1])
        if base is None:
            base = node.verdict_evidence_count if node.verdict_evidence_count is not None else node.evidence_count
        doubled = node.evidence_count >= 2 * max(1, base)
        new_source = bool(_promoting_sources(node, params) - _shown_sources(ref))
        if doubled or new_source:
            ranked.append((2, -node.weight, node.disposition_id, node))
    ranked.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in ranked]


def situation_mix(evidence_rows: Iterable[Any]) -> tuple[dict[str, dict[str, int]], float]:
    """Counts over the rows' ``meta.situation`` by arousal, day phase and company, plus coverage
    (rows with a situation / all rows). Rows before #2739 carry no situation; coverage says so."""
    mix: dict[str, dict[str, int]] = {d: {} for d in SITUATION_DIMS}
    total = with_sit = 0
    for row in evidence_rows:
        total += 1
        meta = _json(_get(row, "meta")) or {}
        sit = meta.get("situation") if isinstance(meta, Mapping) else None
        if not isinstance(sit, Mapping):
            continue
        with_sit += 1
        for dim in SITUATION_DIMS:
            val = sit.get(dim)
            if val is None:
                continue
            key = ("true" if val else "false") if isinstance(val, bool) else str(val)
            mix[dim][key] = mix[dim].get(key, 0) + 1
    return {d: v for d, v in mix.items() if v}, (with_sit / total if total else 0.0)


def _pick_evidence(evidence_rows: Sequence[Any]) -> list[Any]:
    """Up to ``NAMING_PACKET_MAX_EVIDENCE``: the most similar row of each lane first (lanes in order
    of their best row), then the rest by similarity. Ties break by evidence id (deterministic)."""
    rows = [r for r in evidence_rows if isinstance(_get(r, "evidence_id"), str) and _get(r, "evidence_id")
            and isinstance(_get(r, "lane"), str) and _get(r, "lane")
            and isinstance(_get(r, "excerpt"), str) and _get(r, "excerpt").strip()
            and isinstance(_get(r, "occurred_at"), datetime)]

    def sim(r: Any) -> float:
        s = _get(r, "similarity")
        return float(s) if isinstance(s, (int, float)) and math.isfinite(float(s)) else -2.0

    rows.sort(key=lambda r: (-sim(r), _get(r, "evidence_id")))
    picked, lanes = [], set()
    for r in rows:
        if _get(r, "lane") not in lanes:
            lanes.add(_get(r, "lane"))
            picked.append(r)
    chosen = {id(r) for r in picked[:NAMING_PACKET_MAX_EVIDENCE]}
    picked = picked[:NAMING_PACKET_MAX_EVIDENCE]
    picked += [r for r in rows if id(r) not in chosen][:NAMING_PACKET_MAX_EVIDENCE - len(picked)]
    return picked


def _display(node: Node) -> str:
    text = (node.name or "").strip() or (node.label or "").strip() or node.disposition_id
    return text[:DISPOSITION_LABEL_MAX_CHARS]


def neighbors(node: Node, others: Iterable[Node]) -> list[NamingNeighborV1]:
    """Up to ``NAMING_PACKET_MAX_NEIGHBORS`` other non-retired nodes nearest by centroid cosine."""
    scored = []
    for other in others:
        if other.disposition_id == node.disposition_id or other.status == "retired":
            continue
        cos = node_cosine(node, other)
        if cos is not None:
            scored.append((cos, other))
    scored.sort(key=lambda t: (-t[0], t[1].disposition_id))
    return [NamingNeighborV1(disposition_id=o.disposition_id, label=_display(o), cosine=max(-1.0, min(1.0, c)))
            for c, o in scored[:NAMING_PACKET_MAX_NEIGHBORS]]


def prior_verdict(naming_log: Iterable[Any], disposition_id: str) -> Optional[NamingPriorVerdictV1]:
    """The last usable (accepted) verdict on this node, if any."""
    rows = [r for r in naming_log if _get(r, "disposition_id") == disposition_id and usable(r)
            and isinstance(_get(r, "written_at"), datetime)]
    if not rows:
        return None
    last = max(rows, key=lambda r: (_get(r, "written_at"), _get(r, "naming_id") or ""))
    name = (_get(last, "name") or "").strip()[:NAMING_NAME_MAX_CHARS] or None
    sentence = (_get(last, "sentence") or "").strip()[:NAMING_SENTENCE_MAX_CHARS] or None
    return NamingPriorVerdictV1(verdict=_get(last, "verdict"), name=name, sentence=sentence,
                                at=_get(last, "written_at"))


def packet_id(disposition_id: str, evidence_ids: Sequence[str], generated_at: datetime) -> str:
    raw = "|".join([disposition_id, ",".join(evidence_ids), generated_at.isoformat()])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_packet(node: Node, evidence_rows: Sequence[Any], neighbor_nodes: Iterable[Node],
                 prior: Optional[NamingPriorVerdictV1], now: datetime) -> Optional[DispositionNamingPacketV1]:
    """The packet Orion is shown, or None when the node has no showable evidence. ``evidence_rows``
    are the node's ``personality_evidence`` rows (all of them: the situation mix counts every row).
    Weight is not shown: it is not a fact Orion can check against the excerpts."""
    picked = _pick_evidence(evidence_rows)
    if not picked:
        return None
    mix, coverage = situation_mix(evidence_rows)
    shown = [NamingEvidenceV1(evidence_id=_get(r, "evidence_id"), lane=_get(r, "lane"),
                              occurred_at=_get(r, "occurred_at"),
                              excerpt=_get(r, "excerpt").strip()[:DISPOSITION_LABEL_MAX_CHARS]) for r in picked]
    label = (node.label or "").strip()[:DISPOSITION_LABEL_MAX_CHARS] or node.disposition_id[:DISPOSITION_LABEL_MAX_CHARS]
    return DispositionNamingPacketV1(
        packet_id=packet_id(node.disposition_id, [e.evidence_id for e in shown], now),
        disposition_id=node.disposition_id, label_excerpt=label, origin=node.origin,
        evidence_count=max(0, int(node.evidence_count)),
        lane_counts={str(k): int(v) for k, v in node.lane_counts.items()},
        first_seen_at=node.first_seen_at, last_reinforced_at=node.last_reinforced_at,
        evidence=shown, situation_mix=mix, situation_coverage=max(0.0, min(1.0, coverage)),
        neighbors=neighbors(node, neighbor_nodes), prior_verdict=prior, generated_at=now,
    )


def grounding(name_vec: Sequence[float], disposition_id: str, nodes: Iterable[Node]) -> Optional[tuple[int, float]]:
    """Where the named node ranks (1 = best) among every non-retired centroid by cosine to the
    embedded ``name + " " + sentence``, and its cosine minus the best OTHER node's. Recorded only;
    never refuses a name (decision 3). None when the node or the vector is unusable."""
    scores: list[tuple[float, str]] = []
    for n in nodes:
        if n.status == "retired":
            continue
        cos = cosine(name_vec, n.centroid)
        if cos is not None:
            scores.append((cos, n.disposition_id))
    own = next((c for c, d in scores if d == disposition_id), None)
    if own is None:
        return None
    rank = 1 + sum(1 for c, d in scores if d != disposition_id and c > own)
    others = [c for c, d in scores if d != disposition_id]
    margin = own - max(others) if others else 0.0
    return rank, float(margin)
