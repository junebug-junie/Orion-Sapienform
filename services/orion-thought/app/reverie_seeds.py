"""Reveries that ask, Patch 1: the quiet gate, the three world seeds, and the seed picker.

Spec: docs/superpowers/specs/2026-11-18-reverie-questions-design.md ("The quiet gate", "Seeds").

Everything here is pure (no I/O): the worker (``app/reverie.py``) reads rows and hands them in, so each
rule is a unit test. The SQL that produces the rows is next to the builder that reads them, and is the
tested form of Patch 0's measurement (``scripts/analysis/reverie_patch0_measure.py``).

* **Quiet gate.** A reverie may start only when arousal is ``idle`` AND the attention broadcast has had
  no winner for ``min_calm_sec``. A stretch is the time since the last broadcast tick with a winner; its
  allowance is spent only when a ``reverie_question`` row is written for it (a question or a
  none_reason). A lane refusal writes no row, so the next poll tries again; after ``max_lane_refusals``
  refusals the stretch is given up (``lane_refused_stretch``).
* **Seeds.** absence (named people only; clean vs unrecognised), lost bet (calibration gaps, never one
  bad roll), far memory (centered cosine over the lived-memory index, a middle band of ranks, live
  records only). A ref a reverie used in the last ``ORION_REVERIE_THREAD_LAPSE_DAYS`` is spent.
* **Picker.** Least recently used kind first; the first kind with material wins; none -> no reverie.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from orion.schemas.reverie import (
    MAX_REVERIE_SEED_MATERIAL_CHARS,
    REVERIE_SEED_KINDS,
    ReverieSeedV1,
)

# --- the quiet gate --------------------------------------------------------------------------------

# The broadcast ticks about every 30 s. Older than this, "no winner" means "no broadcast", not calm.
MAX_TICK_AGE_SEC = 300.0
# A stretch with no winner in this window starts at its edge (bounds the "rows since" read).
STRETCH_LOOKBACK = timedelta(hours=24)
MAX_LANE_REFUSALS = 3

GATE_OPEN = "open"


@dataclass(frozen=True)
class QuietGateInputs:
    now: datetime
    arousal_level: str | None  # None when unknown (unreadable, stale, flag off)
    last_winner_at: datetime | None  # newest broadcast tick with a winner inside STRETCH_LOOKBACK
    latest_tick_at: datetime | None  # newest broadcast tick at all
    rows_since_stretch: int  # reverie_question rows created at or after the stretch start
    rows_last_24h: int
    lane_refusals_this_stretch: int = 0
    # Oldest broadcast row still in the log. When nothing won attention inside STRETCH_LOOKBACK, the stretch
    # starts here: a STABLE key (review of #3814: "now - 24 h" moved every poll, so the 3-refusal limit and
    # the no-material memo never matched). Moves only when the log's retention prunes its tail.
    oldest_tick_at: datetime | None = None


@dataclass(frozen=True)
class QuietGateDecision:
    open: bool
    reason: str
    stretch_start: datetime | None
    calm_sec: float


def stretch_start_for(
    now: datetime, last_winner_at: datetime | None, oldest_tick_at: datetime | None = None
) -> datetime:
    """The quiet stretch's start: the last tick with a winner; with no winner inside STRETCH_LOOKBACK, the
    oldest broadcast row (stable across polls); with no broadcast rows at all, now - STRETCH_LOOKBACK (the
    gate is shut then anyway: broadcast_stale)."""
    if last_winner_at is not None:
        return last_winner_at
    if oldest_tick_at is not None:
        return oldest_tick_at
    return now - STRETCH_LOOKBACK


def quiet_gate(
    i: QuietGateInputs,
    *,
    min_calm_sec: float,
    max_per_day: int,
    max_tick_age_sec: float = MAX_TICK_AGE_SEC,
    max_lane_refusals: int = MAX_LANE_REFUSALS,
) -> QuietGateDecision:
    """Whether a reverie may start now, and why not. Pure; the order of checks is the order of cost."""
    start = stretch_start_for(i.now, i.last_winner_at, i.oldest_tick_at)
    calm = max(0.0, (i.now - start).total_seconds())

    def shut(reason: str) -> QuietGateDecision:
        return QuietGateDecision(False, reason, start, calm)

    if max_per_day <= 0:
        return shut("disabled")
    if i.arousal_level is None or i.arousal_level == "unknown":
        return shut("arousal_unknown")
    if i.arousal_level != "idle":
        return shut("arousal_not_idle")
    if i.latest_tick_at is None or (i.now - i.latest_tick_at).total_seconds() > max_tick_age_sec:
        return shut("broadcast_stale")
    if calm < min_calm_sec:
        return shut("calm_too_short")
    if i.rows_last_24h >= max_per_day:
        return shut("daily_cap")
    if i.rows_since_stretch > 0:
        return shut("stretch_spent")
    if i.lane_refusals_this_stretch >= max_lane_refusals:
        return shut("lane_refused_stretch")
    return QuietGateDecision(True, GATE_OPEN, start, calm)


# Rows the gate reads (substrate_attention_broadcast_log: the same winner rule as has_attention_winner).
LAST_WINNER_SQL = """
SELECT max(generated_at) AS last_winner_at FROM substrate_attention_broadcast_log
WHERE generated_at > now() - interval '24 hours'
  AND (jsonb_array_length(coalesce(projection_json->'attended_node_ids', '[]'::jsonb)) > 0
       OR coalesce(projection_json->>'selected_open_loop_id', '') <> '')
"""
LATEST_TICK_SQL = "SELECT max(generated_at) AS latest_tick_at FROM substrate_attention_broadcast_log"
# Index-backed (idx_attention_broadcast_log_generated); read only when no winner in 24 h.
OLDEST_TICK_SQL = "SELECT min(generated_at) AS oldest_tick_at FROM substrate_attention_broadcast_log"


# --- seed picker -----------------------------------------------------------------------------------


def kinds_by_least_recent(last_used: Mapping[str, datetime | None]) -> list[str]:
    """Seed kinds, least recently used first. Never used comes first; ties keep REVERIE_SEED_KINDS order."""
    epoch = datetime.min.replace(tzinfo=timezone.utc)

    def key(kind: str) -> tuple[datetime, int]:
        at = last_used.get(kind)
        return (at if at is not None else epoch, REVERIE_SEED_KINDS.index(kind))

    return sorted(REVERIE_SEED_KINDS, key=key)


def pick_seed(
    candidates: Mapping[str, Sequence[ReverieSeedV1]], last_used: Mapping[str, datetime | None]
) -> ReverieSeedV1 | None:
    """The first seed of the least recently used kind that has material; None when no kind has any."""
    for kind in kinds_by_least_recent(last_used):
        seeds = candidates.get(kind) or ()
        if seeds:
            return seeds[0]
    return None


def _clip(text: str, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _seed(kind: str, material: str, refs: list[str], thread_key: str, **extra: Any) -> ReverieSeedV1:
    return ReverieSeedV1(
        kind=kind,  # type: ignore[arg-type]
        material=material[:MAX_REVERIE_SEED_MATERIAL_CHARS],
        refs=refs,
        thread_key=thread_key[:300],
        **extra,
    )


# --- absence ---------------------------------------------------------------------------------------

# Named PERSON individuals only (vehicles carry hash labels). A `missed` means no sighting was
# ATTRIBUTED to the individual in the window; re-identification decides that, not the raw detector. So
# each miss carries the raw census: persons_seen = inventory rows in the window that counted a person.
# 0 -> clean (nobody passed); > 0 -> unrecognised (somebody passed, not matched to them).
ABSENCE_SQL = """
SELECT e.expectation_id, e.stream_id, e.subject_key, i.label, e.day_kind, e.window_start, e.window_end,
  e.support_days, e.scored_at,
  (SELECT count(*) FROM vision_scene_inventory v WHERE v.stream_id = e.stream_id
     AND v.observed_at BETWEEN e.window_start AND e.window_end
     AND coalesce((v.counts->>'person')::int, 0) > 0) AS persons_seen
FROM vision_percept_expectation e
JOIN vision_individual i ON e.subject_key = 'individual:' || i.individual_id
WHERE e.status = 'missed' AND i.kind = 'person' AND i.label IS NOT NULL AND btrim(i.label) <> ''
  AND e.window_end > now() - interval '36 hours'
ORDER BY e.window_end DESC
LIMIT 20
"""

_DAY_KIND_WORDS = {"weekday": "on weekdays", "weekend": "on weekends", "any": "most days"}


def _local(ts: datetime, tz: ZoneInfo) -> datetime:
    return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(tz)


def absence_seeds(rows: Iterable[Mapping[str, Any]], *, tz_name: str, spent_refs: set[str]) -> list[ReverieSeedV1]:
    """One seed per named-person miss whose expectation ref is not spent, newest first.

    An unrecognised miss (a person WAS detected in the window) is never phrased as "did not come":
    the material says whether they came is unknown, and how many detections nobody was matched to."""
    tz = ZoneInfo(tz_name)
    out: list[ReverieSeedV1] = []
    for r in rows:
        ref = f"vision_percept_expectation:{r['expectation_id']}"
        label = " ".join(str(r.get("label") or "").split())
        if ref in spent_refs or not label or r.get("window_start") is None or r.get("window_end") is None:
            continue
        start, end = _local(r["window_start"], tz), _local(r["window_end"], tz)
        seen = int(r.get("persons_seen") or 0)
        evidence = "clean" if seen == 0 else "unrecognised"
        habit = (
            f"{label} usually passes the {r.get('stream_id') or 'walkway'} camera between "
            f"{start:%H:%M} and {end:%H:%M} {_DAY_KIND_WORDS.get(str(r.get('day_kind')), 'on some days')} "
            f"({int(r.get('support_days') or 0)} days of support)."
        )
        if evidence == "clean":
            what = (
                f" On {start:%A %Y-%m-%d} the camera watched that window and saw no person at all: "
                f"nobody passed."
            )
        else:
            what = (
                f" On {start:%A %Y-%m-%d} the camera detected a person in {seen} frame(s) of that window, "
                f"but none of them was recognised as {label}. Whether {label} came is unknown: it may "
                f"have been {label}, unrecognised. A look at those sightings can tell."
            )
        out.append(_seed("absence", habit + what, [ref], f"absence:{r['subject_key']}", evidence=evidence))
    return out


# --- lost bet --------------------------------------------------------------------------------------

LOST_BET_MIN_N = 20
LOST_BET_MIN_GAP = 0.15
LOST_BET_WINDOW_DAYS = 7

# One row per faculty (the bet template) x stated-p bin of tenths, over the settled bets of the window.
LOST_BET_SQL = f"""
SELECT faculty, least(9, floor(stated_p * 10))::int AS bin, count(*) AS n, avg(stated_p) AS mean_p,
  avg((status = 'happened')::int) AS rate
FROM prediction_turn_self_bet
WHERE status IN ('happened', 'not_happened') AND stated_p IS NOT NULL
  AND settled_at > now() - interval '{LOST_BET_WINDOW_DAYS} days'
GROUP BY 1, 2
"""


@dataclass(frozen=True)
class CalibrationBin:
    faculty: str
    bin: int
    n: int
    mean_p: float
    rate: float

    @property
    def gap(self) -> float:
        return self.rate - self.mean_p

    @property
    def se(self) -> float:
        p = self.mean_p
        return math.sqrt(max(p * (1.0 - p), 1e-6) / max(self.n, 1))

    @property
    def is_gap(self) -> bool:
        """n >= 20, |gap| >= 0.15 AND beyond 2 binomial standard errors (not one bad roll, not noise)."""
        return self.n >= LOST_BET_MIN_N and abs(self.gap) >= LOST_BET_MIN_GAP and abs(self.gap) >= 2 * self.se

    @property
    def span(self) -> str:
        return f"{self.bin / 10:.1f}-{(self.bin + 1) / 10:.1f}"

    def line(self) -> str:
        return (f"stated {self.span} (on average {self.mean_p:.2f}): it happened {self.rate:.0%} of the time, "
                f"n={self.n}")


def calibration_bins(rows: Iterable[Mapping[str, Any]]) -> list[CalibrationBin]:
    out = []
    for r in rows:
        try:
            out.append(CalibrationBin(str(r["faculty"]), int(r["bin"]), int(r["n"]), float(r["mean_p"]),
                                      float(r["rate"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def lost_bet_seeds(rows: Iterable[Mapping[str, Any]], *, spent_refs: set[str]) -> list[ReverieSeedV1]:
    """One seed per calibration gap whose ref is not spent, largest gap in standard errors first. The
    material shows the gap bin and the faculty's other well-filled bins, so the whole curve is visible."""
    bins = calibration_bins(rows)
    by_faculty: dict[str, list[CalibrationBin]] = {}
    for b in bins:
        by_faculty.setdefault(b.faculty, []).append(b)
    gaps = sorted((b for b in bins if b.is_gap), key=lambda b: (-abs(b.gap) / b.se, b.faculty, b.bin))
    out: list[ReverieSeedV1] = []
    for g in gaps:
        ref = f"self_bet_bin:{g.faculty}:{g.span}"
        if ref in spent_refs:
            continue
        others = [b for b in sorted(by_faculty[g.faculty], key=lambda b: b.bin)
                  if b.bin != g.bin and b.n >= LOST_BET_MIN_N]
        material = (
            f"My own self-bets that a turn will show '{g.faculty}', settled over the last "
            f"{LOST_BET_WINDOW_DAYS} days. When I {g.line()}. "
            f"That is off by {abs(g.gap):.2f}, more than two standard errors ({g.se:.3f}), so it is not one "
            f"bad roll."
        )
        if others:
            material += " The same bet at other probabilities: " + "; ".join(b.line() for b in others) + "."
        out.append(_seed("lost_bet", material, [ref], f"lost_bet:{g.faculty}:{g.span}"))
    return out


# --- far memory ------------------------------------------------------------------------------------

# Every servable lived-memory record is read through orion.memory.lived_memory.fetch_live_records (the
# same serving gate orion-recall applies: superseded, rejected, expired or removed records are never
# served). The Chroma vectors are only an index: a vector whose doc is not in that live set is dropped
# BEFORE ranking, so a superseded memory can never be ranked, let alone served.


def _unit(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def centered_cosine_ranks(anchor_id: str, vectors: Mapping[str, Sequence[float]]) -> list[tuple[str, float]]:
    """Every other doc by centered cosine to the anchor, nearest first (rank 1 = index 0).

    Raw cosine in this index is squeezed together (Patch 0: nearest 0.78, a typical doc 0.54); subtracting
    the collection mean spreads it (nearest 0.52, typical -0.01), so a band of ranks means something."""
    ids = sorted(vectors)
    if anchor_id not in vectors or len(ids) < 2:
        return []
    units = {i: _unit(vectors[i]) for i in ids}
    dim = len(units[ids[0]])
    mean = [sum(units[i][d] for i in ids) / len(ids) for d in range(dim)]
    centered = {i: _unit([units[i][d] - mean[d] for d in range(dim)]) for i in ids}
    a = centered[anchor_id]
    scored = [(i, sum(x * y for x, y in zip(a, centered[i]))) for i in ids if i != anchor_id]
    scored.sort(key=lambda s: (-s[1], s[0]))
    return scored


@dataclass(frozen=True)
class LivedDoc:
    doc_id: str
    text: str
    occurred_at: datetime | None
    created_at: datetime | None


def pick_anchor(docs: Mapping[str, LivedDoc], with_vectors: set[str]) -> LivedDoc | None:
    """The newest live memory that has a vector: what Orion most recently kept is "now" for this seed.
    Orion's own first-person memories (``episode_memory``) come first; a stance or person fact (Juniper's
    words) anchors only when no episode memory has a vector. (Smoke 2026-11-24: the newest record overall
    was a one-line person fact, a poor "now".)

    Spec deviation (documented): the spec's anchor is "the most salient item in the now". The now's items
    are temporal-self events, not lived memories, and have no vector in this index; embedding one would
    put a vector-host call on every reverie. The newest kept memory is the simplest honest stand-in."""
    pool = [d for d in docs.values() if d.doc_id in with_vectors]
    own = [d for d in pool if d.doc_id.startswith("episode_memory:")]
    pool = own or pool
    if not pool:
        return None
    floor = datetime.min.replace(tzinfo=timezone.utc)
    return max(pool, key=lambda d: (d.created_at or d.occurred_at or floor, d.doc_id))


def far_memory_seed(
    docs: Mapping[str, LivedDoc],
    vectors: Mapping[str, Sequence[float]],
    *,
    band: tuple[int, int],
    spent_refs: set[str],
) -> ReverieSeedV1 | None:
    """Anchor = newest live memory; partner = from centered-cosine ranks ``band`` (inclusive, rank 1 =
    nearest), among LIVE docs only, not spent; the oldest such memory wins (far in time, near in theme)."""
    live_vectors = {k: v for k, v in vectors.items() if k in docs}
    anchor = pick_anchor(docs, set(live_vectors))
    if anchor is None:
        return None
    ranked = centered_cosine_ranks(anchor.doc_id, live_vectors)
    lo, hi = band
    floor = datetime.max.replace(tzinfo=timezone.utc)
    window = [(rank, doc_id, sim) for rank, (doc_id, sim) in enumerate(ranked, start=1) if lo <= rank <= hi]
    window = [w for w in window if w[1] not in spent_refs]
    if not window:
        return None
    rank, partner_id, sim = min(window, key=lambda w: (docs[w[1]].occurred_at or floor, w[0]))
    partner = docs[partner_id]

    def when(d: LivedDoc) -> str:
        return d.occurred_at.date().isoformat() if d.occurred_at else "undated"

    material = (
        f"A memory I kept most recently ({when(anchor)}, {anchor.doc_id}): {_clip(anchor.text, 900)}\n"
        f"An older memory ({when(partner)}, {partner_id}), related to it but not the same "
        f"(rank {rank} by meaning, centered similarity {sim:.2f}): {_clip(partner.text, 900)}"
    )
    return _seed("far_memory", material, [anchor.doc_id, partner_id], f"far_memory:{anchor.doc_id}:{partner_id}")


# --- spent refs and threads ------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")
# The reverie organ's own deterministic "says nothing new" matcher (chain.py is_repeat_thought, retired
# with the chain; calibrated live 2026-10-16 on 2,963 consecutive reverie pairs: word-set Jaccard p50
# 0.36, p90 0.53). 0.6 only catches near-restatements. No word lists.
NEAR_DUPLICATE_JACCARD = 0.6


def is_near_duplicate(text: str, other: str, *, threshold: float = NEAR_DUPLICATE_JACCARD) -> bool:
    a = set(_WORD.findall((text or "").lower()))
    b = set(_WORD.findall((other or "").lower()))
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= threshold


def thread_id_for(thread_key: str, recent: Mapping[str, str]) -> str | None:
    """The thread a seed continues: ``recent`` maps thread_key -> thread_id of reverie rows inside the
    lapse window. None = a new thread (the caller mints the id)."""
    return recent.get(thread_key)
