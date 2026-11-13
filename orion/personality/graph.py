"""Personality graph reducer (spec rev 4: choices only). PURE: no I/O, no logging, no numpy.

Evidence is ONE thing: a ``PersonalityChoiceV1`` row from ``personality_choice`` (what Orion picked,
and the options Orion turned down). Rev 2's grouping of Orion's own prose (reverie, journal, reading
text, day letter, chat/outreach text, self-inquiry seeds) is retired: live 2026-11-13 it grouped job
boilerplate (59% reverie), not personality. Kill means kill: no prose path remains here.

**Direction.** A choice becomes one unit vector, its *direction*::

    direction = unit( embed(chosen text) - mean(embed(declined texts)) )

The shared part of the options (they are all curiosity priors, or all outreach topics) cancels;
what is left points from what Orion turned down toward what Orion took. A ``declined_all`` choice
uses the lane's decline phrase ("no move", "say nothing", "none") as the chosen side, so it reads as
a decline of every option. Nodes ("tendencies") are clusters of directions: a choice joins the node
whose centroid is nearest when the cosine is at least ``match_threshold``, else it founds a node.

**Weight (revealed preference, Samuelson 1938).** A choice reveals a preference only to the extent
it departs from the order it was offered in::

    w = (0.25 if chosen_market_rank == 1 else 1.0) / picks_in_decision * (0.5 if care_promoted)

``declined_all`` weighs 1.0. A choice folded after it happened (a backlog, the curiosity backfill)
is first faded by its own age. A node's ``lane_recent[lane]`` is the faded sum of its choices'
weights; its weight is their sum (``_derived_weight``), so weight reads as "weighted choices,
recently". It fades with a half-life.

**Opposition.** A choice whose direction points the other way (cosine <= -``oppose_threshold``) is
counted against a node (``opposed_count``): Orion picked the declined side of it that time. The
label carries both: "<chosen gist> over <declined gist>, n of m" with n = choices that went the
node's way and m = n + choices that went against it. A gist is a few words of the option's plain
text (``gist``); with three or more declined options only the chosen side is named (``pair_label``).
The label is computed, never an excerpt of prose, and no model writes it.

Status: born ``candidate``; ``active`` once weight >= ``promote_min_weight`` and it holds at least
``promote_min_evidence`` choices; ``retired`` when a candidate goes unreinforced past its TTL or an
active node fades below ``retire_below_weight``. Retired is terminal and never matched again.

Rows mirror ``services/orion-sql-db/manual_migration_personality_graph_v1.sql`` (+ the choice
migration's ``opposed_count``): ``Node`` -> personality_disposition, ``EvidenceRow`` ->
personality_evidence (one per folded choice, id ``choice:<choice_id>``), ``HistoryRow`` ->
personality_weight_history. Single writer: orion-durable-runs.

Bad evidence (unknown lane, unusable direction, naive timestamp) is skipped, never raised; when it
still has an id and time it yields an ``EvidenceRow`` with ``disposition_id=None`` and
``meta["skip_reason"]`` so the writer marks it consumed.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence, get_args

from orion.schemas.personality import (
    DISPOSITION_LABEL_MAX_CHARS,
    FRAME_MAX_DISPOSITIONS,
    NAMING_NAME_MAX_CHARS,
    NAMING_SENTENCE_MAX_CHARS,
    ChoiceStyleV1,
    DispositionV1,
    PersonalityFrameV1,
    PersonalityLane,
)

ORIGIN_CHOICE = "choice"
EVIDENCE_PREFIX = "choice:"  # personality_evidence.evidence_id = "choice:<choice_id>"
VALID_LANES = frozenset(get_args(PersonalityLane))
# The chosen side of a ``declined_all`` choice: the "none" option as each lane shows it.
DECLINE_TEXT = {"prime": "no move", "outreach": "say nothing"}
DECLINE_TEXT_DEFAULT = "none of these"
# Words kept from an option's text in a label. Six reads as a phrase; seven read as a sentence cut
# off mid-thought, two of them side by side (live 2026-11-13: "turn start context is stale snapshot
# over camera layer mediation not closeness (+13 more), 21 of 21").
GIST_WORDS = 6
# A label names the declined options only when there are at most this many. Past that, naming one
# loser out of many is arbitrary and "(+k more)" tells the reader nothing.
LABEL_MAX_NAMED_DECLINED = 2
# Node.orion_verdict values (self-naming).
VERDICT_NAMED = "named"
VERDICT_REJECTED = "rejected"
VERDICT_MERGED = "merged"
# Juniper's answer on a tendency (Hub Personality tab, ``personality_juniper_verdict``; rev 4 move 4). The
# reducer is handed the latest answer for each retired node Juniper retired: "no" keeps it retired and
# absorbing (``_Tick.absorb``), "yes" after a "no" revives it.
JUNIPER_NO = "no"
JUNIPER_YES = "yes"

_FRAME_MAX_WARNINGS = 20
_COUNT_SUFFIX = re.compile(r", \d+ of \d+$")
# The pre-2026-11-13 label tail: "<chosen gist> over <declined gist> (+k more)".
_MORE_SUFFIX = re.compile(r" \(\+\d+ more\)$")
# A gist must not end on a word that needs the next one ("...is", "...in", "...not"). Trimmed off the
# END of the cut only; the option's words are never rewritten or reordered.
_DANGLING = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "as", "is", "are", "was", "were", "be", "not", "no", "only", "than", "that", "this", "its", "my",
    "your", "their", "vs", "over", "into", "via", "has", "have", "had", "does", "do",
    "within", "without", "about", "between", "after", "before", "under", "per", "when", "while",
})
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class GraphParams:
    # Cosine between two choice DIRECTIONS (differences of bge-large unit vectors). Unrelated
    # directions sit near 0, unlike raw bge vectors (~0.6 baseline), so rev 2's 0.82 does not apply.
    # Calibrated on the live curiosity backlog (PR report, "Metric quality gate").
    match_threshold: float = 0.5
    oppose_threshold: float = 0.5
    half_life_days: float = 14.0
    follow_weight: float = 0.25  # picked the market's rank-1 option
    depart_weight: float = 1.0  # picked another rank, an unranked option, or declined all
    care_promoted_factor: float = 0.5  # runaway guard (rev 4 move 2)
    promote_min_weight: float = 3.0
    promote_min_evidence: int = 3
    # A lane counts as a "source" of a node (self-naming re-ask rule) at this many recent weighted choices.
    promote_lane_min_recent: float = 1.0
    candidate_ttl_days: float = 14.0  # choices are sparse (a few a day): longer than rev 2's 7
    retire_below_weight: float = 0.25
    centroid_n_cap: int = 50  # running-mean memory, lets a node drift slowly
    history_min_rel_change: float = 0.01
    # A fade-only weight change is written once it moved this much from the stored weight (``_must_write``).
    fade_write_min_rel_change: float = 0.01
    # MERGE (B into A) by Orion's own verdict only when the two centroids are at least this close.
    merge_min_cosine: float = 0.85


DEFAULT_PARAMS = GraphParams()


@dataclass(frozen=True)
class Evidence:
    """One folded choice. ``embedding`` is its direction (``choice_direction``); ``excerpt`` is its
    pair label (``pair_label``); ``weight`` is its evidence weight (``evidence_weight``)."""

    evidence_id: str
    lane: str
    source_ref: str
    occurred_at: datetime  # aware UTC
    excerpt: str
    embedding: Sequence[float]
    weight: float = 1.0
    meta: Mapping[str, Any] = field(default_factory=dict)
    # The option texts (chosen + declined). Read only by the own-name echo gate; never stored.
    text: str = ""


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
    # Per-lane faded sum of choice weights, "as of updated_at".
    lane_recent: dict[str, float] = field(default_factory=dict)
    # When this graph last folded evidence into the node (wall clock): the candidate TTL runs from here.
    last_folded_at: datetime | None = None
    # Weight at the last history row: history compares against it, so slow fade shows up.
    history_weight: float = 0.0
    # Choices whose direction went against this node (Orion picked its declined side).
    opposed_count: int = 0
    # Self-naming (design 2026-10-17-personality-orion-names-dispositions). Set only by
    # ``apply_namings``; a name never changes weight, fade or status.
    name: str | None = None
    name_sentence: str | None = None
    named_at: datetime | None = None
    naming_id: str | None = None  # the last applied naming row for this node
    orion_verdict: str | None = None  # "named" | "rejected" | "merged" (this node was merged away)
    merged_into: str | None = None
    name_grounding_rank: int | None = None
    name_grounding_margin: float | None = None
    last_shown_at: datetime | None = None
    shown_count: int = 0
    verdict_evidence_count: int | None = None  # evidence_count when the last real verdict landed


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
    if not math.isfinite(norm) or norm <= 1e-12:
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


def _derived_weight(lane_recent: Mapping[str, float], params: GraphParams = None) -> float:  # noqa: ARG001
    """A node's weight: the faded sum of its choices' weights, over lanes."""
    return sum(max(0.0, float(v)) for v in lane_recent.values())


def _node_id(evidence_id: str) -> str:
    return "disp-" + hashlib.sha1(evidence_id.encode("utf-8")).hexdigest()[:16]


def _copy(node: Node) -> Node:
    return replace(node, centroid=list(node.centroid), lane_counts=dict(node.lane_counts),
                   lane_recent=dict(node.lane_recent))


# --- choice -> evidence (pure helpers the driver calls) ---------------------------------------


def evidence_weight(chosen_market_rank: Optional[int], *, declined_all: bool = False, care_promoted: bool = False,
                    picks_in_decision: int = 1, params: GraphParams = DEFAULT_PARAMS) -> float:
    """Revealed-preference weight of one choice (module docstring)."""
    base = params.depart_weight if declined_all or chosen_market_rank != 1 else params.follow_weight
    w = base / max(1, int(picks_in_decision or 1))
    return w * params.care_promoted_factor if care_promoted else w


def choice_direction(chosen: Sequence[float], declined: Sequence[Sequence[float]]) -> Optional[list[float]]:
    """unit(unit(chosen) - mean(unit(declined))), or None when any vector is unusable, the
    dimensions differ, or the two sides are the same point."""
    c = _unit(chosen)
    ds = [_unit(d) for d in declined]
    if c is None or not ds or any(d is None or len(d) != len(c) for d in ds):
        return None
    n = float(len(ds))
    mean = [sum(col) / n for col in zip(*ds)]
    return _unit([ci - mi for ci, mi in zip(c, mean)])


def decline_text(lane: str) -> str:
    return DECLINE_TEXT.get(lane, DECLINE_TEXT_DEFAULT)


def gist(text: Any, words: int = GIST_WORDS) -> str:
    """A few words of an option's plain text (what Orion saw): the first ``words`` words,
    whitespace collapsed, with dangling function words trimmed off the end of a cut. No ellipsis."""
    parts = [p for p in _WS.sub(" ", text if isinstance(text, str) else "").strip().split(" ") if p]
    out = parts[:words]
    if len(parts) > len(out):
        while len(out) > 1 and out[-1].lower().strip(".,;:") in _DANGLING:
            out.pop()
    return " ".join(out).rstrip(".,;:-")


def pair_label(chosen_text: str, declined_texts: Sequence[str]) -> str:
    """"<chosen> over <declined>[ and <declined>]" when Orion turned down one or two options, else
    just "<chosen>". Each side is a ``gist`` of the option's text; ``declined_texts`` in shown order
    (best market rank first)."""
    rest = [t for t in declined_texts if isinstance(t, str) and t.strip()]
    chosen = gist(chosen_text)
    if not rest or len(rest) > LABEL_MAX_NAMED_DECLINED:
        return chosen[:DISPOSITION_LABEL_MAX_CHARS]
    return f"{chosen} over {' and '.join(gist(t) for t in rest)}"[:DISPOSITION_LABEL_MAX_CHARS]


def tidy_label(base: str) -> str:
    """A stored label base in the pre-2026-11-13 shape, made readable: "<a 7-word gist…> over <a
    7-word gist…> (+k more)" -> "<5-word gist>". Bases in the current shape pass through unchanged,
    so this runs on every fold and live nodes read well without a backfill."""
    s = (base or "").strip()
    many = bool(_MORE_SUFFIX.search(s))
    s = _MORE_SUFFIX.sub("", s)
    if not many and "…" not in s:
        return s
    sides = s.split(" over ")
    if many and len(sides) > 1:
        # "(+k more)" meant 2+ declined: keep the chosen side. A gist was at most 7 words (+ "…"),
        # so the declined side is the last " over " part; anything before it is the chosen side.
        return gist(" over ".join(sides[:-1]).replace("…", ""))
    return " over ".join(gist(side.replace("…", "")) for side in sides)


def label_base(label: str) -> str:
    return tidy_label(_COUNT_SUFFIX.sub("", label or ""))


def choice_label(base: str, n: int, m: int) -> str:
    """"<base>, n of m", the counts always kept whole (the base is cut to fit)."""
    suffix = f", {max(0, int(n))} of {max(0, int(m))}"
    return label_base(base)[: DISPOSITION_LABEL_MAX_CHARS - len(suffix)].rstrip() + suffix


# --- tick state -------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Snap:
    weight: float
    status: str
    evidence_count: int
    label: str
    centroid_n: int
    opposed_count: int


def _snap(node: Node) -> _Snap:
    return _Snap(node.weight, node.status, node.evidence_count, node.label, node.centroid_n, node.opposed_count)


def _must_write(before: _Snap, after: _Snap, params: GraphParams) -> bool:
    """Does a loaded node need its row rewritten this tick?

    Fade moves every live node's weight a little on every tick (exact, deterministic: 0.5 ** (dt /
    half_life) from the row's own ``updated_at``), so comparing snapshots exactly rewrote every live
    node every tick (live 2026-11-13: ``nodes_written: 118`` with ``choices_read: 0``). The stored row
    (weight, lane_recent, updated_at) stays self-consistent when a fade-only change is NOT written: the
    next tick fades from the stored ``updated_at`` and reaches the same value. So a fade-only change is
    written only once the weight moved ``fade_write_min_rel_change`` (1%) from the stored weight -- about
    every 5 h at a 14-day half-life -- which bounds how stale a reader of the raw ``weight`` column (the
    market's tendency read) can be. Any other change (evidence, status, label, oppose) is always written."""
    if replace(after, weight=before.weight) != before:
        return True
    if before.weight == 0.0:
        return after.weight != 0.0
    return abs(after.weight - before.weight) / abs(before.weight) >= params.fade_write_min_rel_change


class _Tick:
    """Working copies of the input nodes plus nodes born this tick, with before-snapshots."""

    def __init__(self, nodes: Sequence[Node], now: datetime, juniper: Optional[Mapping[str, str]] = None):
        self.now = now
        self.nodes: list[Node] = [_copy(n) for n in nodes]
        self.before: dict[int, Optional[_Snap]] = {id(n): _snap(n) for n in self.nodes}
        # Juniper's answers (``JUNIPER_NO`` / ``JUNIPER_YES``, or ``(answer, answered_at)``), only for retired
        # nodes she retired.
        # "no": the node stays retired and still matches, so a choice in its direction counts as evidence on
        # it instead of founding a new tendency she would be asked about again. It never comes back by
        # itself (fade and status rules skip retired). "yes" after her "no": revived as a candidate, and the
        # usual rules take it from there (it may promote at once, or fade out like any other).
        self.vetoed: set[int] = set()
        for node in self.nodes:
            raw = (juniper or {}).get(node.disposition_id)
            answer, answered_at = (raw if isinstance(raw, tuple) else (raw, None))
            if node.status != "retired":
                continue
            retired_at = _aware(node.retired_at)
            # A "yes" revives only a node retired BEFORE it: a node her yes revived and Orion's own rules
            # retired again (candidate TTL, fade) stays retired.
            if answer == JUNIPER_YES and answered_at is not None and retired_at is not None \
                    and _aware(answered_at) <= retired_at:
                continue
            if answer == JUNIPER_NO:
                self.vetoed.add(id(node))
            elif answer == JUNIPER_YES:
                node.status, node.retired_at, node.last_folded_at = "candidate", None, now
        live = [n for n in self.nodes if id(n) not in self.vetoed]
        self.dim = next((len(n.centroid) for n in [*live, *self.nodes] if n.centroid), None)
        self.dim: Optional[int] = None
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

    def sims(self, emb: list[float]) -> list[tuple[Node, float]]:
        out = []
        for node in self.nodes:
            if node.status == "retired" and id(node) not in self.vetoed:
                continue
            c = self.centroid(node)
            if c is None or len(c) != len(emb):
                continue
            out.append((node, _dot(c, emb)))
        return out

    def reinforce(self, node: Node, ev: Evidence, emb: list[float], lane: str, sim: float,
                  w: float, params: GraphParams, *, hit: bool = True) -> float:
        """``hit=False`` (own-name echo): the choice is counted and moves the centroid, but adds no
        weight and never becomes the label."""
        c = self.centroid(node) or emb
        before = node.weight
        if hit:
            node.lane_recent[lane] = node.lane_recent.get(lane, 0.0) + w
            node.weight = _derived_weight(node.lane_recent)
        gain = node.weight - before
        node.evidence_count += 1
        node.chosen_count += 1
        node.lane_counts[lane] = node.lane_counts.get(lane, 0) + 1
        node.last_folded_at = self.now
        last = _aware(node.last_reinforced_at)
        node.last_reinforced_at = ev.occurred_at if last is None else max(last, ev.occurred_at)
        k = max(1, min(node.centroid_n, max(1, params.centroid_n_cap)))
        new_c = _unit([(ci * k + ei) / (k + 1) for ci, ei in zip(c, emb)]) or list(c)
        node.centroid = new_c
        self.unit_centroids[id(node)] = new_c
        node.centroid_n += 1
        base = label_base(node.label)
        if hit and sim > node.label_similarity and ev.excerpt.strip():
            base, node.label_similarity = ev.excerpt, sim
        node.label = choice_label(base, node.evidence_count, node.evidence_count + node.opposed_count)
        return gain

    def absorb(self, node: Node, ev: Evidence, lane: str) -> None:
        """A choice in the direction of a tendency Juniper said No to: counted on it (the evidence row
        names it), with no weight, no centroid move and no status change. Her No was about that
        direction, so the node neither drifts nor revives."""
        node.evidence_count += 1
        node.chosen_count += 1
        node.lane_counts[lane] = node.lane_counts.get(lane, 0) + 1
        last = _aware(node.last_reinforced_at)
        node.last_reinforced_at = ev.occurred_at if last is None else max(last, ev.occurred_at)
        node.label = choice_label(label_base(node.label), node.evidence_count,
                                  node.evidence_count + node.opposed_count)

    def oppose(self, node: Node) -> None:
        node.opposed_count += 1
        node.label = choice_label(node.label, node.evidence_count, node.evidence_count + node.opposed_count)

    def create(self, ev: Evidence, emb: list[float], lane: str, w: float, now: datetime,
               params: GraphParams) -> Node:
        node = Node(
            disposition_id=_node_id(ev.evidence_id), origin=ORIGIN_CHOICE, status="candidate",
            # The founding choice starts as close as a typical member (the match threshold), so a later
            # choice nearer the drifting centroid can take the label.
            label=choice_label(ev.excerpt, 1, 1), label_similarity=params.match_threshold,
            centroid=list(emb), centroid_n=1, weight=w, evidence_count=1, lane_counts={lane: 1},
            chosen_count=1, first_seen_at=ev.occurred_at, last_reinforced_at=ev.occurred_at, retired_at=None,
            updated_at=now, lane_recent={lane: w}, last_folded_at=now,
        )
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
            if (node.status == "candidate" and node.weight >= params.promote_min_weight
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
            if before is not None and not _must_write(before, _snap(node), params):
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


def cites_own_name(node: Node, ev: Evidence) -> bool:
    """Own-name echo gate (self-naming decision 2, 2026-10-20): the choice's option texts contain
    the matched node's CURRENT name (case-insensitive exact substring)."""
    name = (node.name or "").strip().casefold()
    if not name:
        return False
    return any(name in (t or "").casefold() for t in (ev.excerpt, ev.text) if isinstance(t, str))


def _ingest(tick: _Tick, evidence: Sequence[Evidence], *, now: datetime, params: GraphParams) -> None:
    seen: set[str] = set()
    valid: list[tuple[Evidence, list[float], str, float]] = []
    for ev in evidence:
        if not isinstance(ev, Evidence):
            continue
        eid = ev.evidence_id
        if not isinstance(eid, str) or not eid or eid in seen:
            continue
        seen.add(eid)
        if _aware(ev.occurred_at) is None:
            continue
        if ev.lane not in VALID_LANES:
            tick.skip(ev, "unknown_lane")
            continue
        if not isinstance(ev.excerpt, str) or not ev.excerpt.strip():
            tick.skip(ev, "empty_label")
            continue
        try:
            w = float(ev.weight)
        except (TypeError, ValueError):
            w = float("nan")
        if not math.isfinite(w) or w <= 0.0:
            tick.skip(ev, "bad_weight")
            continue
        # A choice older than ``now`` (a backlog, the curiosity backfill) is faded by its own age, so
        # it weighs what it would weigh had it been folded when it happened.
        age = _days(now - ev.occurred_at)
        if age > 0 and params.half_life_days > 0:
            w *= 0.5 ** (age / params.half_life_days)
        emb = _unit(ev.embedding) if ev.embedding is not None else None
        if emb is None:
            tick.skip(ev, "bad_direction")
            continue
        if tick.dim is None:
            tick.dim = len(emb)
        elif len(emb) != tick.dim:
            tick.skip(ev, "direction_dim_mismatch")
            continue
        valid.append((ev, emb, ev.lane, w))
    valid.sort(key=lambda t: (t[0].occurred_at, t[0].evidence_id))
    for ev, emb, lane, w in valid:
        sims = tick.sims(emb)
        # A live tendency that matches takes the choice; only a choice no live node matches goes to the
        # nearest tendency Juniper said No to (``absorb``) instead of founding a new one.
        live_sims = [(n, x) for n, x in sims if id(n) not in tick.vetoed]
        best, best_sim = max(live_sims, key=lambda t: t[1], default=(None, -2.0))
        if best is None or best_sim < params.match_threshold:
            vetoed_best = max(((n, x) for n, x in sims if id(n) in tick.vetoed), key=lambda t: t[1],
                              default=(None, -2.0))
            if vetoed_best[0] is not None and vetoed_best[1] >= params.match_threshold:
                best, best_sim = vetoed_best
        meta = dict(ev.meta) if isinstance(ev.meta, Mapping) else {}
        if best is not None and best_sim >= params.match_threshold and id(best) in tick.vetoed:
            tick.absorb(best, ev, lane)
            meta["juniper_no"] = True
            delta, disp_id, similarity = 0.0, best.disposition_id, best_sim
        elif best is not None and best_sim >= params.match_threshold:
            echo = cites_own_name(best, ev)
            if echo:
                meta["cites_own_name"] = True
            delta = tick.reinforce(best, ev, emb, lane, best_sim, w, params, hit=not echo)
            disp_id, similarity = best.disposition_id, best_sim
        else:
            best = tick.create(ev, emb, lane, w, now, params)
            delta, disp_id, similarity = best.weight, best.disposition_id, 1.0
        live = [(n, s) for n, s in sims if id(n) not in tick.vetoed]
        opposed = sorted(n.disposition_id for n, s in live if s <= -params.oppose_threshold and n is not best)
        for node, s in live:
            if node is not best and s <= -params.oppose_threshold:
                tick.oppose(node)
        if opposed:
            meta["opposed"] = opposed[:20]
        tick.evidence_rows.append(EvidenceRow(
            evidence_id=ev.evidence_id, lane=lane, source_ref=str(ev.source_ref or ""),
            occurred_at=ev.occurred_at, excerpt=ev.excerpt[:DISPOSITION_LABEL_MAX_CHARS],
            disposition_id=disp_id, similarity=similarity, weight_delta=delta, meta=meta,
        ))


# --- public API -------------------------------------------------------------------------------


def apply_fade(nodes: list[Node], *, now: datetime, params: GraphParams = DEFAULT_PARAMS,
               juniper: Optional[Mapping[str, str]] = None) -> FoldResult:
    """Fade every non-retired node to ``now`` and apply status rules. Inputs are not mutated.
    ``juniper``: disposition_id -> Juniper's latest answer, for retired nodes she retired (``_Tick``)."""
    tick = _Tick(nodes, now, juniper)
    tick.fade(now, params)
    tick.statuses(now, params)
    return tick.result(now, params)


def fold(nodes: list[Node], evidence: Sequence[Evidence], *, now: datetime,
         params: GraphParams = DEFAULT_PARAMS, juniper: Optional[Mapping[str, str]] = None) -> FoldResult:
    """Fade to ``now``, fold choices in occurred_at order, then apply status rules. Only
    ``origin='choice'`` nodes take part; any other node passed in is left untouched. ``juniper``:
    disposition_id -> Juniper's latest answer, for retired nodes she retired: a "no" node absorbs
    choices in its direction, a "yes" after a "no" revives it (``_Tick``)."""
    tick = _Tick([n for n in nodes if n.origin == ORIGIN_CHOICE], now, juniper)
    tick.fade(now, params)
    _ingest(tick, evidence, now=now, params=params)
    tick.statuses(now, params)
    return tick.result(now, params)


def _clip(text: Optional[str], n: int) -> Optional[str]:
    t = (text or "").strip()[:n]
    return t or None


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if _aware(dt) is None else dt


def project_frame(nodes: Sequence[Node], *, now: datetime, choice_style: ChoiceStyleV1 | None,
                  evidence_rows: int, warnings: Sequence[str] = ()) -> PersonalityFrameV1:
    counts = {"active": 0, "candidate": 0, "retired": 0}
    for n in nodes:
        if n.status in counts:
            counts[n.status] += 1
    # A node Orion said is "not me" keeps matching and fading but leaves every lane's cue.
    active = sorted((n for n in nodes if n.status == "active" and n.orion_verdict != VERDICT_REJECTED),
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
            name=_clip(n.name, NAMING_NAME_MAX_CHARS), name_sentence=_clip(n.name_sentence, NAMING_SENTENCE_MAX_CHARS),
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


def choice_style_from_rows(rows: Sequence[Mapping[str, Any]], *, now: datetime,
                           window_days: int = 14) -> ChoiceStyleV1 | None:
    """Distinct decisions in ``(now - window_days, now]`` from ``personality_choice`` rows
    (``act_id``, ``choice_id``, ``chosen_market_rank``, ``declined_all``, ``occurred_at``): took the
    market's top option (rank 1) versus went their own way (another rank, unranked, or declined
    all). A multi-pick decision (one ``act_id``) counts once, by its best rank."""
    window_days = max(1, int(window_days))
    start = now - timedelta(days=window_days)
    best: dict[str, Optional[int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        at = _parse_dt(row.get("occurred_at"))
        if at is None or at <= start or at > now:
            continue
        key = row.get("act_id") or row.get("choice_id")
        if not isinstance(key, str) or not key:
            continue
        rank = None if row.get("declined_all") else row.get("chosen_market_rank")
        rank = rank if isinstance(rank, int) and not isinstance(rank, bool) else None
        prev = best.get(key)
        if key not in best or (rank is not None and (prev is None or rank < prev)):
            best[key] = rank
    if not best:
        return None
    took_top = sum(1 for r in best.values() if r == 1)
    return ChoiceStyleV1(window_days=window_days, choices=len(best), took_top=took_top,
                         went_own_way=len(best) - took_top)


# --- self-naming: Orion's verdicts applied to nodes -------------------------------------------
# Design: docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md section 5,
# decisions 2026-10-20 (grounding is recorded, never a refusal; no veto).


@dataclass(frozen=True)
class NamingRow:  # one orion_disposition_naming row (the fields the reducer reads)
    naming_id: str
    disposition_id: str
    verdict: str  # name | reject | merge | no_answer
    hub_check: str  # accepted | refused
    written_at: datetime
    name: str | None = None
    sentence: str | None = None
    merge_with: str | None = None


@dataclass(frozen=True)
class NamingOutcome:
    naming_id: str
    disposition_id: str
    verdict: str  # the row's verdict, or "refused" when Hub refused it
    apply_status: str  # applied | refused | no_answer | unknown_disposition | merge_too_far | merge_target_missing
    grounding_rank: int | None = None
    grounding_margin: float | None = None


@dataclass
class NamingApplyResult:
    nodes: list[Node]  # every node whose row must be written (full upsert)
    history: list[HistoryRow]
    outcomes: list[NamingOutcome]


def _fade_to(node: Node, now: datetime, params: GraphParams) -> None:
    updated = _aware(node.updated_at)
    dt = _days(now - updated) if updated is not None else 0.0
    if dt > 0 and params.half_life_days > 0:
        factor = 0.5 ** (dt / params.half_life_days)
        node.lane_recent = {k: v * factor for k, v in node.lane_recent.items()}
        node.weight = _derived_weight(node.lane_recent, params)
        node.updated_at = now


def cosine(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Cosine of two vectors, or None when either is unusable or the dimensions differ."""
    ua, ub = _unit(a), _unit(b)
    if ua is None or ub is None or len(ua) != len(ub):
        return None
    return _dot(ua, ub)


def node_cosine(a: Node, b: Node) -> Optional[float]:
    return cosine(a.centroid, b.centroid)


def _merge_into(a: Node, b: Node, now: datetime, params: GraphParams) -> None:
    """Pool B's existing choices into A: per-lane ``lane_recent`` and the counts are summed, so A
    weighs what the same choices would give one node."""
    _fade_to(a, now, params)
    _fade_to(b, now, params)
    for k, v in b.lane_recent.items():
        a.lane_recent[k] = a.lane_recent.get(k, 0.0) + v
    a.weight = _derived_weight(a.lane_recent, params)
    for lane, c in b.lane_counts.items():
        a.lane_counts[lane] = a.lane_counts.get(lane, 0) + c
    a.evidence_count += b.evidence_count
    a.chosen_count += b.chosen_count
    a.opposed_count += b.opposed_count
    a.label = choice_label(a.label, a.evidence_count, a.evidence_count + a.opposed_count)
    ua, ub = _unit(a.centroid), _unit(b.centroid)
    na, nb = max(1, a.centroid_n), max(1, b.centroid_n)
    if ua is not None and ub is not None and len(ua) == len(ub):
        a.centroid = _unit([(x * na + y * nb) / (na + nb) for x, y in zip(ua, ub)]) or ua
    a.centroid_n = a.centroid_n + b.centroid_n
    a.first_seen_at = min(a.first_seen_at, b.first_seen_at)
    a.last_reinforced_at = max(a.last_reinforced_at, b.last_reinforced_at)
    folded = [t for t in (a.last_folded_at, b.last_folded_at) if t is not None]
    a.last_folded_at = max(folded) if folded else None
    a.updated_at = now
    b.status, b.retired_at, b.updated_at = "retired", now, now
    b.merged_into, b.orion_verdict = a.disposition_id, VERDICT_MERGED


def apply_namings(nodes: Sequence[Node], namings: Sequence[NamingRow],
                  grounding: Mapping[str, tuple[int, float]] | None = None, *, now: datetime,
                  params: GraphParams = DEFAULT_PARAMS) -> NamingApplyResult:
    """Apply Orion's naming verdicts, in the given order, to copies of ``nodes``.

    NAME sets the name fields (and the recorded grounding rank/margin); REJECT marks the node
    ``orion_verdict='rejected'`` (it keeps matching and fading); MERGE pools B into A only when
    cosine(A, B) >= ``merge_min_cosine`` and retires B with ``merged_into=A``. A refused or empty
    answer only updates the shown bookkeeping. No verdict adds a hit, changes the fade, or moves
    a status the evidence would not give; only MERGE moves A's weight, by pooling. Promotion is not
    checked here: a merged node is promoted (if at all) by the next tick's ``fold``."""
    grounding = grounding or {}
    work = {n.disposition_id: _copy(n) for n in nodes}
    before = {k: (n.weight, n.status) for k, n in work.items()}
    touched: dict[str, Node] = {}
    outcomes: list[NamingOutcome] = []
    for row in namings:
        node = work.get(row.disposition_id)
        verdict = row.verdict if row.hub_check == "accepted" else "refused"
        if node is None or node.status == "retired":
            outcomes.append(NamingOutcome(row.naming_id, row.disposition_id, verdict, "unknown_disposition"))
            continue
        node.shown_count += 1
        at = _aware(row.written_at) or now
        node.last_shown_at = at if node.last_shown_at is None else max(node.last_shown_at, at)
        touched[node.disposition_id] = node
        if verdict == "refused" or verdict == "no_answer":
            outcomes.append(NamingOutcome(row.naming_id, row.disposition_id, verdict, verdict))
            continue
        if verdict == "name":
            rank, margin = grounding.get(row.naming_id, (None, None))
            node.name = (row.name or "").strip()[:NAMING_NAME_MAX_CHARS] or None
            node.name_sentence = (row.sentence or "").strip()[:NAMING_SENTENCE_MAX_CHARS] or None
            node.named_at, node.naming_id, node.orion_verdict = now, row.naming_id, VERDICT_NAMED
            node.name_grounding_rank, node.name_grounding_margin = rank, margin
            node.verdict_evidence_count = node.evidence_count
            outcomes.append(NamingOutcome(row.naming_id, row.disposition_id, verdict, "applied", rank, margin))
        elif verdict == "reject":
            node.orion_verdict, node.naming_id = VERDICT_REJECTED, row.naming_id
            node.verdict_evidence_count = node.evidence_count
            outcomes.append(NamingOutcome(row.naming_id, row.disposition_id, verdict, "applied"))
        elif verdict == "merge":
            target = work.get(row.merge_with or "")
            if target is None or target.status == "retired" or target is node:
                outcomes.append(NamingOutcome(row.naming_id, row.disposition_id, verdict, "merge_target_missing"))
                continue
            cos = node_cosine(target, node)
            if cos is None or cos < params.merge_min_cosine:
                outcomes.append(NamingOutcome(row.naming_id, row.disposition_id, verdict, "merge_too_far"))
                continue
            _merge_into(target, node, now, params)
            node.naming_id, node.verdict_evidence_count = row.naming_id, node.evidence_count
            touched[target.disposition_id] = target
            outcomes.append(NamingOutcome(row.naming_id, row.disposition_id, verdict, "applied"))
        else:
            outcomes.append(NamingOutcome(row.naming_id, row.disposition_id, verdict, "unknown_verdict"))
    history: list[HistoryRow] = []
    for did, n in touched.items():
        w0, st0 = before[did]
        if n.weight != w0 or n.status != st0:
            history.append(HistoryRow(did, now, n.weight, n.status, n.evidence_count))
            n.history_weight = n.weight
    return NamingApplyResult(nodes=list(touched.values()), history=history, outcomes=outcomes)
