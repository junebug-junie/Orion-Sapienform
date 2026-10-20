"""Personality graph reducer (spec rev 2, bottom-up). PURE: no I/O, no logging, no numpy.

Evidence (Orion's own words from each lane, plus curiosity picks) folds into nodes
("dispositions": things Orion keeps returning to). Each node keeps ``lane_recent``: per lane, a
hit counter that fades with a half-life. Its weight is DERIVED from it:

    weight = sum over lanes of base(lane) * min(lane_cap, ln(1 + lane_recent[lane]))

Log response with saturation (Weber-Fechner plus habituation): a lane's contribution grows with
the log of its recent hits and stops growing after about four (the counter is clamped there),
so one lane's volume buys nothing
more and the weight reads as HOW MANY lanes keep returning to a theme. It fades because the
counters fade. A lane counts toward promotion only with at least ``promote_lane_min_recent``
recent hits, so one echo in a second lane cannot promote a single-lane flood.

History, measured on live data (2026-10-16, read-only dry runs): gain / sqrt(1 + lifetime hits)
let reverie (~500 rows a day) crown a prediction-error theme at weight 37.9 from 244 reverie rows
in 24 h; an uncapped base / (1 + recent) still grew like ln(hits) for weeks (a 2,600-hit
reverie theme led the 7-day graph) and 6 of the top 8 nodes were one lane plus 1-4 echo hits.
Weight fades with a half-life. Nodes are born as ``candidate``, promoted to ``active`` when
at least ``promote_min_lanes`` distinct non-seed lanes back them, and ``retired`` when they
fade out or are never backed. Retired is terminal and never matched again.

Independence, not lane names: ``lane_recent`` is keyed by SOURCE (``source_key``), not by lane. Two
lanes that narrate the same upstream signal are one source. Reverie narrates the attention
broadcast coalition and the metacog journal is opened by the self-signal episode gate; both start
from ``substrate_node_prediction_error_history`` through the same attention eligibility rule. Live
2026-10-20: on the 3 active prediction-error nodes, 16 of 27 metacog journal hits had a reverie hit
on the same node within 10 minutes (1 of 27 at a 6 h or 3 h offset), so "two lanes" was one source
and promoted them. ``lane_counts`` stays per lane (what the panel shows). ``rekey_sources`` moves
nodes stored under the old lane keys onto source keys, once.

Rows mirror ``services/orion-sql-db/manual_migration_personality_graph_v1.sql``:
``Node`` -> personality_disposition, ``EvidenceRow`` -> personality_evidence,
``HistoryRow`` -> personality_weight_history. Single writer: orion-durable-runs.

Bad evidence (empty / wrong-dimension / non-finite / zero-norm embedding, unknown lane,
empty excerpt, naive or missing timestamp) is skipped, never raised. Skipped evidence that
still has a usable id and timestamp yields an ``EvidenceRow`` with ``disposition_id=None``,
``weight_delta=0`` and ``meta["skip_reason"]`` so the writer can mark it consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence, get_args

from orion.schemas.personality import (
    DISPOSITION_LABEL_MAX_CHARS,
    FRAME_MAX_DISPOSITIONS,
    ChoiceStyleV1,
    DispositionV1,
    PersonalityFrameV1,
    PersonalityLane,
)

CHOICE_LANE = "curiosity_choice"
SEED_LANE = "self_inquiry_seed"
ORIGIN_SEED = "seed_self_inquiry"
ORIGIN_RECURRENCE = "recurrence"
# One source: reverie plus the metacog journal (see module docstring).
SUBSTRATE_NARRATION = "substrate_narration"
VALID_LANES = frozenset(get_args(PersonalityLane))
_FRAME_MAX_WARNINGS = 20


@dataclass(frozen=True)
class GraphParams:
    match_threshold: float = 0.82  # cosine; bge-large-en-v1.5 1024-d unit vectors
    half_life_days: float = 14.0
    lane_gain: float = 1.0
    choice_gain: float = 2.0  # lane "curiosity_choice": a pick is stronger than a mention
    seed_weight: float = 0.5  # lane "self_inquiry_seed"
    lane_cap: float = 1.5  # max ln(1 + recent hits) one lane contributes (~4 recent hits)
    promote_min_lanes: int = 2  # distinct lanes, NOT counting "self_inquiry_seed"
    # A lane counts toward promotion with >= 2 hits, the older within one half-life (1 + 0.5).
    # 2.0 was unreachable across ticks: the first hit fades before the second lands (1.9997).
    promote_lane_min_recent: float = 1.5
    promote_min_evidence: int = 3
    candidate_ttl_days: float = 7.0  # candidate not reinforced for this long -> retired
    retire_below_weight: float = 0.25  # active node faded below this -> retired
    centroid_n_cap: int = 50  # running-mean memory, lets a node drift slowly
    history_min_rel_change: float = 0.01


DEFAULT_PARAMS = GraphParams()


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    lane: str
    source_ref: str
    occurred_at: datetime  # aware UTC
    excerpt: str  # <=160 chars, Orion's own words
    embedding: Sequence[float]
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Node:  # mirrors personality_disposition
    disposition_id: str
    origin: str
    status: str
    label: str
    label_similarity: float
    centroid: list[float]
    centroid_n: int
    weight: float
    evidence_count: int
    lane_counts: dict[str, int]
    chosen_count: int
    first_seen_at: datetime
    last_reinforced_at: datetime
    retired_at: datetime | None
    updated_at: datetime  # weight is "as of updated_at"; fade uses now - updated_at
    # Per-lane recent hits (habituation), faded with the weight, "as of updated_at".
    lane_recent: dict[str, float] = field(default_factory=dict)
    # When this graph last folded evidence into the node (wall clock, not the evidence's own time):
    # the candidate TTL runs from here, so an old seed or a backlog row is not retired at birth.
    last_folded_at: datetime | None = None
    # Weight at the last history row: history compares against it, so slow fade shows up.
    history_weight: float = 0.0


@dataclass(frozen=True)
class EvidenceRow:  # mirrors personality_evidence (minus created_at)
    evidence_id: str
    lane: str
    source_ref: str
    occurred_at: datetime
    excerpt: str
    disposition_id: str | None
    similarity: float | None
    weight_delta: float
    meta: dict


@dataclass(frozen=True)
class HistoryRow:  # mirrors personality_weight_history
    disposition_id: str
    at: datetime
    weight: float
    status: str
    evidence_count: int


@dataclass
class FoldResult:
    nodes: list[Node]  # every node whose row must be upserted
    evidence_rows: list[EvidenceRow]
    history: list[HistoryRow]


# --- math helpers -----------------------------------------------------------------------------


def _unit(vec: Sequence[Any]) -> Optional[list[float]]:
    """Unit-length float list, or None when empty / non-numeric / non-finite / zero-norm."""
    try:
        vals = [float(v) for v in vec]
    except (TypeError, ValueError):
        return None
    if not vals or not all(math.isfinite(v) for v in vals):
        return None
    norm = math.sqrt(sum(v * v for v in vals))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return [v / norm for v in vals]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _aware(dt: Any) -> Optional[datetime]:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        return None
    return dt


def _days(delta: timedelta) -> float:
    return delta.total_seconds() / 86400.0


def _lane_base(lane: str, params: GraphParams) -> float:
    if lane == CHOICE_LANE:
        return params.choice_gain
    if lane == SEED_LANE:
        return params.seed_weight
    return params.lane_gain


def _derived_weight(lane_recent: Mapping[str, float], params: GraphParams) -> float:
    return sum(
        _lane_base(lane, params) * min(params.lane_cap, math.log1p(max(0.0, r)))
        for lane, r in lane_recent.items()
    )


def source_key(lane: str, meta: Mapping[str, Any] | None) -> str:
    """The ``lane_recent`` key: the independent source behind this evidence (module docstring)."""
    if lane == "reverie" or (lane == "journal" and isinstance(meta, Mapping)
                             and meta.get("source_kind") == "metacog"):
        return SUBSTRATE_NARRATION
    return lane


def _node_id(evidence_id: str) -> str:
    return "disp-" + hashlib.sha1(evidence_id.encode("utf-8")).hexdigest()[:16]


def _clip_label(text: str) -> str:
    return text.strip()[:DISPOSITION_LABEL_MAX_CHARS]


def _copy(node: Node) -> Node:
    return replace(node, centroid=list(node.centroid), lane_counts=dict(node.lane_counts),
                   lane_recent=dict(node.lane_recent))


def _promoting_lanes(node: Node, params: GraphParams) -> int:
    """Distinct non-seed SOURCES (``lane_recent`` keys) at or above the recent floor."""
    return sum(1 for key, r in node.lane_recent.items()
               if key != SEED_LANE and r >= params.promote_lane_min_recent)


# --- tick state -------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Snap:
    weight: float
    status: str
    evidence_count: int
    label: str
    centroid_n: int


def _snap(node: Node) -> _Snap:
    return _Snap(node.weight, node.status, node.evidence_count, node.label, node.centroid_n)


class _Tick:
    """Working copies of the input nodes plus nodes born this tick, with before-snapshots."""

    def __init__(self, nodes: Sequence[Node], now: datetime):
        self.now = now
        self.nodes: list[Node] = [_copy(n) for n in nodes]
        self.before: dict[int, Optional[_Snap]] = {id(n): _snap(n) for n in self.nodes}
        self.dim: Optional[int] = next((len(n.centroid) for n in self.nodes if n.centroid), None)
        self.unit_centroids: dict[int, Optional[list[float]]] = {}
        self.evidence_rows: list[EvidenceRow] = []

    def centroid(self, node: Node) -> Optional[list[float]]:
        key = id(node)
        if key not in self.unit_centroids:
            self.unit_centroids[key] = _unit(node.centroid)
        return self.unit_centroids[key]

    def fade(self, now: datetime, params: GraphParams) -> None:
        for node in self.nodes:
            if node.status == "retired":
                continue
            updated = _aware(node.updated_at)
            dt = _days(now - updated) if updated is not None else 0.0
            if dt > 0 and params.half_life_days > 0:
                factor = 0.5 ** (dt / params.half_life_days)
                node.lane_recent = {k: v * factor for k, v in node.lane_recent.items()}
                node.weight = _derived_weight(node.lane_recent, params)
            node.updated_at = now

    def best_match(self, emb: list[float], *, eligible) -> tuple[Optional[Node], float]:
        best, best_sim = None, -2.0
        for node in self.nodes:
            if node.status == "retired" or not eligible(node):
                continue
            c = self.centroid(node)
            if c is None or len(c) != len(emb):
                continue
            sim = _dot(c, emb)
            if sim > best_sim:
                best, best_sim = node, sim
        return best, best_sim

    def reinforce(self, node: Node, ev: Evidence, emb: list[float], lane: str, key: str, sim: float,
                  params: GraphParams) -> float:
        c = self.centroid(node) or emb
        before = node.weight
        # Clamp at the level where the lane's contribution caps: hits on a capped lane refresh it
        # instead of piling up, so a theme that goes quiet fades out in weeks whatever its volume
        # (unclamped, 2,600 reverie hits stayed above the cap ~134 days after the last one).
        node.lane_recent[key] = min(node.lane_recent.get(key, 0.0) + 1.0, math.expm1(params.lane_cap))
        node.weight = _derived_weight(node.lane_recent, params)
        gain = node.weight - before
        node.evidence_count += 1
        node.lane_counts[lane] = node.lane_counts.get(lane, 0) + 1
        node.last_folded_at = self.now
        if lane == CHOICE_LANE:
            node.chosen_count += 1
        last = _aware(node.last_reinforced_at)
        node.last_reinforced_at = ev.occurred_at if last is None else max(last, ev.occurred_at)
        k = max(1, min(node.centroid_n, max(1, params.centroid_n_cap)))
        mixed = [(ci * k + ei) / (k + 1) for ci, ei in zip(c, emb)]
        new_c = _unit(mixed) or list(c)
        node.centroid = new_c
        self.unit_centroids[id(node)] = new_c
        node.centroid_n += 1
        if sim > node.label_similarity:
            label = _clip_label(ev.excerpt)
            if label:
                node.label, node.label_similarity = label, sim
        return gain

    def create(self, ev: Evidence, emb: list[float], lane: str, key: str, origin: str, now: datetime,
               params: GraphParams) -> Node:
        gain = _derived_weight({key: 1.0}, params)
        # A node born from self-inquiry alone is a seed whichever entry point created it.
        origin = ORIGIN_SEED if lane == SEED_LANE else origin
        node = Node(
            disposition_id=_node_id(ev.evidence_id), origin=origin, status="candidate",
            # The founding excerpt starts as close as a typical member (the match threshold), so
            # later evidence nearer the drifting centroid can take the label (at 1.0 it never could).
            label=_clip_label(ev.excerpt), label_similarity=params.match_threshold, centroid=list(emb), centroid_n=1,
            weight=gain, evidence_count=1, lane_counts={lane: 1},
            chosen_count=1 if lane == CHOICE_LANE else 0, first_seen_at=ev.occurred_at,
            last_reinforced_at=ev.occurred_at, retired_at=None, updated_at=now,
            lane_recent={key: 1.0}, last_folded_at=now,
        )
        # Id collision with an existing (e.g. retired) node: derive a distinct id deterministically.
        taken = {n.disposition_id for n in self.nodes}
        salt = 0
        while node.disposition_id in taken:
            salt += 1
            node.disposition_id = _node_id(f"{ev.evidence_id}#{salt}")
        self.nodes.append(node)
        self.before[id(node)] = None
        self.unit_centroids[id(node)] = list(emb)
        return node

    def skip(self, ev: Evidence, reason: str) -> None:
        eid, at = ev.evidence_id, _aware(ev.occurred_at)
        if not isinstance(eid, str) or not eid or at is None:
            return  # cannot even be recorded as consumed
        meta = dict(ev.meta) if isinstance(ev.meta, Mapping) else {}
        meta["skip_reason"] = reason
        self.evidence_rows.append(EvidenceRow(
            evidence_id=eid, lane=str(ev.lane or ""), source_ref=str(ev.source_ref or ""),
            occurred_at=at, excerpt=str(ev.excerpt or "")[:DISPOSITION_LABEL_MAX_CHARS],
            disposition_id=None, similarity=None, weight_delta=0.0, meta=meta,
        ))

    def statuses(self, now: datetime, params: GraphParams) -> None:
        for node in self.nodes:
            if node.status == "retired":
                continue
            if (node.status == "candidate"
                    and _promoting_lanes(node, params) >= params.promote_min_lanes
                    and node.evidence_count >= params.promote_min_evidence):
                node.status = "active"
            if node.status == "candidate":
                last = max((t for t in (_aware(node.last_reinforced_at), _aware(node.last_folded_at))
                            if t is not None), default=None)
                if last is not None and _days(now - last) > params.candidate_ttl_days:
                    node.status, node.retired_at, node.updated_at = "retired", now, now
            elif node.status == "active" and node.weight < params.retire_below_weight:
                node.status, node.retired_at, node.updated_at = "retired", now, now

    def result(self, now: datetime, params: GraphParams) -> FoldResult:
        changed: list[Node] = []
        history: list[HistoryRow] = []
        for node in self.nodes:
            before = self.before.get(id(node))
            if before is not None and _snap(node) == before:
                continue
            changed.append(node)
            w0 = node.history_weight if before is not None else 0.0
            status_changed = before is None or before.status != node.status
            if w0 == 0.0:
                weight_moved = node.weight != 0.0
            else:
                weight_moved = abs(node.weight - w0) / abs(w0) >= params.history_min_rel_change
            if weight_moved or status_changed:
                history.append(HistoryRow(node.disposition_id, now, node.weight, node.status,
                                          node.evidence_count))
                node.history_weight = node.weight
        return FoldResult(nodes=changed, evidence_rows=list(self.evidence_rows), history=history)


def _ingest(tick: _Tick, evidence: Sequence[Evidence], *, now: datetime, params: GraphParams,
            lane_override: Optional[str], origin: str, eligible) -> None:
    seen: set[str] = set()
    valid: list[tuple[Evidence, list[float], str]] = []
    for ev in evidence:
        if not isinstance(ev, Evidence):
            continue
        eid = ev.evidence_id
        if not isinstance(eid, str) or not eid or eid in seen:
            continue
        seen.add(eid)
        if _aware(ev.occurred_at) is None:
            continue
        lane = lane_override or ev.lane
        if lane not in VALID_LANES:
            tick.skip(ev, "unknown_lane")
            continue
        if not isinstance(ev.excerpt, str) or not ev.excerpt.strip():
            tick.skip(ev, "empty_excerpt")
            continue
        emb = _unit(ev.embedding) if ev.embedding is not None else None
        if emb is None:
            tick.skip(ev, "bad_embedding")
            continue
        if tick.dim is None:
            tick.dim = len(emb)
        elif len(emb) != tick.dim:
            tick.skip(ev, "embedding_dim_mismatch")
            continue
        valid.append((ev, emb, lane))
    valid.sort(key=lambda t: (t[0].occurred_at, t[0].evidence_id))
    for ev, emb, lane in valid:
        node, sim = tick.best_match(emb, eligible=eligible)
        if node is not None and sim >= params.match_threshold:
            delta = tick.reinforce(node, ev, emb, lane, source_key(lane, ev.meta), sim, params)
            disp_id, similarity = node.disposition_id, sim
        else:
            node = tick.create(ev, emb, lane, source_key(lane, ev.meta), origin, now, params)
            delta, disp_id, similarity = node.weight, node.disposition_id, 1.0
        tick.evidence_rows.append(EvidenceRow(
            evidence_id=ev.evidence_id, lane=lane, source_ref=str(ev.source_ref or ""),
            occurred_at=ev.occurred_at, excerpt=ev.excerpt[:DISPOSITION_LABEL_MAX_CHARS],
            disposition_id=disp_id, similarity=similarity, weight_delta=delta,
            meta=dict(ev.meta) if isinstance(ev.meta, Mapping) else {},
        ))


# --- public API -------------------------------------------------------------------------------


def apply_fade(nodes: list[Node], *, now: datetime, params: GraphParams = DEFAULT_PARAMS) -> FoldResult:
    """Fade every non-retired node to ``now`` and apply status rules. Inputs are not mutated."""
    tick = _Tick(nodes, now)
    tick.fade(now, params)
    tick.statuses(now, params)
    return tick.result(now, params)


def fold(nodes: list[Node], evidence: Sequence[Evidence], *, now: datetime,
         params: GraphParams = DEFAULT_PARAMS) -> FoldResult:
    """Fade to ``now``, fold evidence in occurred_at order, then apply status rules."""
    tick = _Tick(nodes, now)
    tick.fade(now, params)
    _ingest(tick, evidence, now=now, params=params, lane_override=None,
            origin=ORIGIN_RECURRENCE, eligible=lambda n: True)
    tick.statuses(now, params)
    return tick.result(now, params)


def seed_nodes(seeds: Sequence[Evidence], *, now: datetime,
               params: GraphParams = DEFAULT_PARAMS) -> FoldResult:
    """Each seed becomes a ``seed_self_inquiry`` candidate (lane ``self_inquiry_seed``), unless it
    matches a seed node created earlier in this same call. Seeds never count toward promotion."""
    tick = _Tick([], now)
    _ingest(tick, seeds, now=now, params=params, lane_override=SEED_LANE,
            origin=ORIGIN_SEED, eligible=lambda n: n.origin == ORIGIN_SEED)
    tick.statuses(now, params)
    return tick.result(now, params)


# Lanes whose stored ``lane_recent`` key may hold another source's hits (before ``source_key``).
REKEY_LANES = ("reverie", "journal")


def rekey_sources(nodes: Sequence[Node], hits: Mapping[str, Sequence[tuple[str, Any, datetime]]], *,
                  now: datetime, params: GraphParams = DEFAULT_PARAMS) -> FoldResult:
    """One-time move of ``lane_recent`` from lane keys to source keys (``source_key``).

    ``hits[disposition_id]`` lists that node's evidence in ``REKEY_LANES`` as (lane, meta,
    occurred_at). Those keys are rebuilt by replaying the hits (fade by the half-life between hits,
    +1 per hit, clamped as in ``reinforce``) up to the node's ``updated_at``; other keys are kept.
    An active node that no longer has ``promote_min_lanes`` sources goes back to candidate: it was
    promoted by one source counted twice. Returns every node whose ``lane_recent`` changed, with a
    history row at ``now`` when weight or status moved. Inputs are not mutated."""
    cap = math.expm1(params.lane_cap)
    changed: list[Node] = []
    history: list[HistoryRow] = []
    for orig in nodes:
        if orig.status == "retired":
            continue
        rebuilt: dict[str, float] = {}
        last: dict[str, datetime] = {}
        for lane, meta, at in sorted(hits.get(orig.disposition_id, ()), key=lambda h: h[2]):
            if lane not in REKEY_LANES or _aware(at) is None:
                continue
            key = source_key(lane, meta)
            r = rebuilt.get(key, 0.0)
            if key in last and params.half_life_days > 0:
                r *= 0.5 ** (max(0.0, _days(at - last[key])) / params.half_life_days)
            rebuilt[key], last[key] = min(r + 1.0, cap), at
        end = _aware(orig.updated_at)
        for key, at in last.items():
            if end is not None and params.half_life_days > 0:
                rebuilt[key] *= 0.5 ** (max(0.0, _days(end - at)) / params.half_life_days)
        node = _copy(orig)
        node.lane_recent = {k: v for k, v in orig.lane_recent.items() if k not in REKEY_LANES} | rebuilt
        if node.lane_recent == orig.lane_recent:
            continue
        node.weight = _derived_weight(node.lane_recent, params)
        if node.status == "active" and _promoting_lanes(node, params) < params.promote_min_lanes:
            node.status = "candidate"
        if node.status != orig.status or node.weight != orig.weight:
            history.append(HistoryRow(node.disposition_id, now, node.weight, node.status,
                                      node.evidence_count))
            node.history_weight = node.weight
        changed.append(node)
    return FoldResult(nodes=changed, evidence_rows=[], history=history)


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if _aware(dt) is None else dt


def project_frame(nodes: Sequence[Node], *, now: datetime, choice_style: ChoiceStyleV1 | None,
                  evidence_rows: int, warnings: Sequence[str] = ()) -> PersonalityFrameV1:
    counts = {"active": 0, "candidate": 0, "retired": 0}
    for n in nodes:
        if n.status in counts:
            counts[n.status] += 1
    active = sorted((n for n in nodes if n.status == "active"),
                    key=lambda n: (-n.weight, n.disposition_id))[:FRAME_MAX_DISPOSITIONS]
    dispositions = [
        DispositionV1(
            disposition_id=n.disposition_id,
            label=(n.label or "").strip()[:DISPOSITION_LABEL_MAX_CHARS] or n.disposition_id[:DISPOSITION_LABEL_MAX_CHARS],
            origin=n.origin, status="active", weight=max(0.0, float(n.weight)),
            evidence_count=max(0, int(n.evidence_count)),
            lane_counts={str(k): int(v) for k, v in n.lane_counts.items()},
            chosen_count=max(0, int(n.chosen_count)),
            first_seen_at=_as_utc(n.first_seen_at), last_reinforced_at=_as_utc(n.last_reinforced_at),
        )
        for n in active
    ]
    return PersonalityFrameV1(
        generated_at=_as_utc(now), dispositions=dispositions, choice_style=choice_style,
        active_count=counts["active"], candidate_count=counts["candidate"],
        retired_count=counts["retired"], evidence_rows=max(0, int(evidence_rows)),
        warnings=[str(w) for w in warnings][:_FRAME_MAX_WARNINGS],
    )


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return _aware(value)


def _parse_meta(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, Mapping) else {}


def choice_style_from_rows(rows: Sequence[Mapping[str, Any]], *, now: datetime,
                           window_days: int = 14) -> ChoiceStyleV1 | None:
    """Distinct curiosity choices in ``(now - window_days, now]``: took the top-ranked option
    (``chosen_best_rank == 1``) versus went their own way (any other rank, or off the list)."""
    window_days = max(1, int(window_days))
    start = now - timedelta(days=window_days)
    best: dict[str, Optional[int]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("lane", CHOICE_LANE) != CHOICE_LANE:
            continue
        at = _parse_dt(row.get("occurred_at"))
        if at is None or at <= start or at > now:
            continue
        meta = _parse_meta(row.get("meta"))
        choice_id = meta.get("choice_id")
        if not isinstance(choice_id, str) or not choice_id:
            continue
        rank = meta.get("chosen_best_rank")
        rank = rank if isinstance(rank, int) and not isinstance(rank, bool) else None
        prev = best.get(choice_id, None)
        if choice_id not in best or (rank is not None and (prev is None or rank < prev)):
            best[choice_id] = rank
    if not best:
        return None
    took_top = sum(1 for r in best.values() if r == 1)
    return ChoiceStyleV1(window_days=window_days, choices=len(best), took_top=took_top,
                         went_own_way=len(best) - took_top)
