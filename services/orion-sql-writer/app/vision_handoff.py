"""Street continuity: one passage across two cameras -- pure rules.

Design: docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md,
sections 2-4 and 6. I/O lives in ``app/vision_handoff_loop.py``; everything
here takes explicit inputs and returns values, so every rule has a test.

Portals come from ``config/vision_topology.yaml`` (``orion/vision/topology.py``):

- ``gap`` (door_to_walkway, walkway_to_door): a story that leaves the source
  camera toward the target opens an expectation with a predicted arrival
  window on the target camera.
- ``approach`` (approach_door): a walkway story moving toward the house
  expects a Ring motion / ring event (or a doorbell person story) soon after.
- ``overlap`` (central_walk): same-label stories on both cameras at the same
  moment are a same-person candidate. No expectation.

**Departure from ``inside``.** The detector looks every ~5 s, so a walker's
last box is usually 10-40% of the frame from the edge (35 of 36 published
walkway stories end ``inside``). A gap departure is therefore PROJECTED: a
story moving toward the portal edge (``dir_lateral``) with an exit speed
leaves through that edge ``exit_dist / exit_speed`` seconds after its last
box. ``inside`` is never read as "went into a door".

**The model** learns one number per portal and subject kind, the gap
``g = transit_sec x exit_speed - exit_dist`` in source frame widths (transit
= target story start - source story end). Travel time is then
``(exit_dist + g) / exit_speed``: slow walkers get later, wider windows.
Approach portals have no speed: ``g`` is the transit in seconds.

- Fewer than ``min_arrivals`` arrivals (met, early or late) for the portal and
  kind: the prior band,
  x0.4 to x3 of the topology's gap (approach: ``prior_sec``).
- Otherwise: the split-conformal 10%-90% band of past ``g`` (met and late
  rows; ambiguous rows never train). Counting only ``met`` toward the switch
  was a trap (2026-10-29): with a wrong prior most arrivals land early or late,
  so the learned band never took over. A kind without its own ``min_arrivals`` uses
  the person band widened by ``person_widen``.

**p-value, and the calm state it must have.** ``p`` is the probability of an
outcome at least this unlikely under the model:

- an arrival: two-sided rank of the observed ``g`` among the calibration
  ``g`` values (conformal: uniform on 0..1 when the next walker is
  exchangeable with the past ones), or, on the prior band, the two-sided tail
  of a log-normal whose 10%/90% quantiles are the band ends;
- a miss: the counted chance of no arrival for that portal and kind,
  ``(missed + 1) / (missed + arrived + 2)``.

Calm state: on ordinary days p-values of met handoffs are spread evenly over
0..1, so about 5% fall below 0.05 by chance. The rest state is reachable: an
on-time arrival in the middle of the band has ``p`` near 1.

Smallest p (granularity, not a rest-state floor): the conformal p cannot go
below ``2 / (n + 1)`` with ``n`` calibration values, so under n = 39 no
arrival can pass p < 0.05. The prior band gets the same floor as a
``min_arrivals``-value calibration set (``2 / (min_arrivals + 1)``): a hand-set x0.4..x3
guess must never call anything surprising, however short the transit (short
transits are normal on the overlapping central walk). The attention gate is
``p < attention_p`` AND (a known person OR unusual for the hour); the replay
eval checks uniformity and that a quiet hour yields zero gate passes.

**Grade** (same three-way rule as ``vision_rhythm``): ``met`` matched inside
the window, ``early`` / ``late`` inside the early margin (one window width,
never before the departure) / late margin (twice the window), ``missed`` only
when the target
camera was looking for at least ``min_coverage`` of the window, else
``unscorable``. The doorbell only sees during Ring bursts or self-looks, so
its coverage is the census frames it actually delivered.

A handoff never writes a label or merges individuals: a strong pair only
opens a capped "same person?" ask (Juniper's answer goes through the existing
answer path).
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.vision_motion import MotionConfig, MotionCrop, Passage, edge_of, exit_foot, exit_speed, usable_crops
from app.vision_person_model import unit

MODEL_VERSION = "handoff-conformal-v1"
HANDOFF_NS = uuid.UUID("3b8e2f0a-6c1d-4f7e-9a52-1d0c8e7b4a16")
HANDOFF_LOCK_KEY = 0x0A1C_0005
LOOK_KEY = "orion:vision:look:doorbell"
STREET_CHANNEL = "orion:vision:street:events"
SOURCE_KIND_SAME = "vision_same_individual"
HANDOFF_RULE_REF = "rule:handoff"

# Stories that cannot depart or be a same-moment source: props, a companion of
# a walker already in its own story (one passage, two stories), someone standing
# or a parked thing, anything out of scope. Arrival candidates exclude props and
# companions only: an arrival at a frame edge is often a short "glimpse".
NOT_A_DEPARTURE = ("prop", "companion", "standing", "parked", "out_of_scope")
NOT_AN_ARRIVAL = ("prop", "companion")

GRADED = ("met", "early", "late", "missed", "ambiguous", "unscorable")
# Arrivals: they train the gap. "early" = before the band, inside the early
# margin (a 10-90% band leaves ~10% of ordinary walkers early by design).
ARRIVED = ("met", "early", "late")
HYPOTHESES = ("detector_miss", "late", "turned_back", "entered_place", "unexplained")
RING_HYPOTHESES = ("detector_miss", "out_of_view")
# 10%-90% band (spec section 3).
BAND_LO_Q, BAND_HI_Q = 0.1, 0.9
# z of the 90% quantile: the prior band's ends are its 10% / 90% quantiles.
Z90 = 1.2815515655446004
PRIOR_LO_FACTOR, PRIOR_HI_FACTOR = 0.4, 3.0


@dataclass(frozen=True)
class HandoffConfig:
    kinds: Tuple[str, ...] = ("person", "dog", "bicycle", "stroller")
    look_kinds: Tuple[str, ...] = ("person", "dog")
    min_arrivals: int = 10
    person_widen: float = 1.5
    min_coverage: float = 0.8
    # A census span counts as watched this long past its end (census windows
    # are ~5 s every ~7 s; the rhythm loop's 30 s would credit a 5 s doorbell
    # burst with 35 s of looking).
    census_gap_sec: float = 8.0
    min_exit_speed: float = 0.005          # frame widths / s; below this nobody is leaving
    max_project_frac: float = 0.6          # farther than this from the edge: no projection
    max_window_sec: float = 600.0
    max_late_sec: float = 600.0
    hypothesis_sec: float = 600.0          # late / turned_back / entered_place deadline
    grade_lag_sec: float = 60.0
    stuck_after_sec: float = 3600.0        # target stories never caught up -> unscorable
    appearance_bars: Tuple[Tuple[str, float], ...] = (("face", 0.50), ("body", 0.92), ("osnet", 0.82))
    ambiguous_margin: float = 0.03
    attention_p: float = 0.05
    unusual_hour_max_per_day: float = 0.2
    same_ask_daily_cap: int = 2
    ask_expiry_days: float = 7.0
    ask_cooldown_days: float = 30.0
    thumb_retention_days: float = 0.0
    local_tz: str = "America/Denver"

    def bar(self, channel: Optional[str]) -> Optional[float]:
        return dict(self.appearance_bars).get(channel or "")


# ---------------------------------------------------------------------------
# Story facts (stored columns, or derived from the evidence boxes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoryFacts:
    story_id: str
    stream_id: str
    kind: str
    started_at: datetime
    ended_at: datetime
    dir_lateral: int
    dir_depth: int
    exit_edge: Optional[str]
    entry_edge: Optional[str]
    exit_foot_x: Optional[float]
    entry_foot_x: Optional[float]
    exit_speed: Optional[float]
    zone_path: Tuple[str, ...] = ()
    individual_id: Optional[str] = None
    crop_ids: Tuple[str, ...] = ()
    speed_class: Optional[str] = None
    drop_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    edges_source: str = "stored"           # stored | derived


def _parse_t(v) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def evidence_crops(evidence: Sequence[Mapping], kind: str) -> List[MotionCrop]:
    out = []
    for e in evidence or []:
        box = e.get("box")
        if not box or len(box) != 4 or not e.get("t"):
            continue
        out.append(MotionCrop(crop_id=str(e.get("crop_id") or ""), t=_parse_t(e["t"]), label=str(e.get("label") or kind),
                              box=tuple(float(b) for b in box), individual_id=e.get("individual_id")))
    return sorted(out, key=lambda c: c.t)


def majority_individual(evidence: Sequence[Mapping]) -> Optional[str]:
    ids = [e.get("individual_id") for e in evidence or [] if isinstance(e.get("individual_id"), str)]
    if not ids:
        return None
    return max(sorted(set(ids)), key=ids.count)


def facts_from_row(row: Mapping, frame: Tuple[float, float]) -> StoryFacts:
    """One motion story -> the facts the handoff rules read.

    With the edges migration (PR #2822) the stored ``exit_edge`` /
    ``exit_foot_x`` / ``exit_speed_fw_per_sec`` are used. Without it (or on a
    story written before it) they are derived from the evidence boxes with
    the same functions the motion loop uses, in ``frame`` (the story's own
    frame size if stored, else the stream's configured one)."""
    evidence = row.get("evidence") or []
    if isinstance(evidence, str):
        evidence = json.loads(evidence)
    kind = str(row.get("kind") or "")
    fw = float(row.get("frame_w") or frame[0])
    fh = float(row.get("frame_h") or frame[1])
    mcfg = MotionConfig(frame_w=fw, frame_h=fh)
    crops = evidence_crops(evidence, kind)
    entry_x = (crops[0].foot[0] / fw) if crops else None
    if row.get("exit_edge"):
        exit_e, entry_e = row.get("exit_edge"), row.get("entry_edge")
        fx = row.get("exit_foot_x")
        xs = row.get("exit_speed_fw_per_sec")
        source = "stored"
    elif crops:
        exit_e, entry_e = edge_of(crops[-1], mcfg), edge_of(crops[0], mcfg)
        fx = exit_foot(crops[-1], mcfg)[0]
        xs = exit_speed(usable_crops(Passage(kind=kind, crops=crops), mcfg), mcfg)
        source = "derived"
    else:
        exit_e = entry_e = fx = xs = None
        source = "derived"
    return StoryFacts(
        story_id=str(row["story_id"]), stream_id=str(row["stream_id"]), kind=kind,
        started_at=row["started_at"], ended_at=row["ended_at"],
        dir_lateral=int(row.get("dir_lateral") or 0), dir_depth=int(row.get("dir_depth") or 0),
        exit_edge=exit_e, entry_edge=entry_e,
        exit_foot_x=float(fx) if fx is not None else None, entry_foot_x=entry_x,
        exit_speed=float(xs) if xs is not None else None,
        zone_path=tuple(row.get("zone_path") or ()),
        individual_id=row.get("individual_id") or majority_individual(evidence),
        crop_ids=tuple(row.get("crop_ids") or [e.get("crop_id") for e in evidence if e.get("crop_id")]),
        speed_class=row.get("speed_class"), drop_reason=row.get("drop_reason"),
        created_at=row.get("created_at"), edges_source=source)


# ---------------------------------------------------------------------------
# Open: which stories depart through a portal
# ---------------------------------------------------------------------------


def edge_distance(foot_x: Optional[float], edge: str) -> Optional[float]:
    if foot_x is None:
        return None
    return float(foot_x) if edge == "left" else 1.0 - float(foot_x)


def gap_departure(f: StoryFacts, portal, cfg: HandoffConfig) -> Optional[Tuple[float, float]]:
    """(exit_dist, exit_speed) if this story leaves through the portal's edge.

    Moving toward the edge (``dir_lateral`` sign), last box on that edge or
    ``inside`` and close enough to project, and a measured exit speed. A story
    that ended on another edge (top, bottom, the far side) left some other
    way; a story with no lateral motion is not departing."""
    if f.stream_id != portal.from_stream or f.kind not in cfg.kinds or f.drop_reason == "prop":
        return None
    if portal.lateral_sign == 0 or f.dir_lateral != portal.lateral_sign:
        return None
    if portal.from_exclude_dir_depth is not None and f.dir_depth == portal.from_exclude_dir_depth:
        return None
    if f.exit_edge not in (portal.from_edge, "inside"):
        return None
    if f.exit_speed is None or f.exit_speed < cfg.min_exit_speed:
        return None
    if f.exit_edge == portal.from_edge:
        dist = 0.0
    else:
        dist = edge_distance(f.exit_foot_x, portal.from_edge)
        if dist is None or dist > cfg.max_project_frac:
            return None
    return max(0.0, dist), float(f.exit_speed)


def approach_departure(f: StoryFacts, portal, cfg: HandoffConfig) -> bool:
    return (f.stream_id == portal.from_stream and f.kind in cfg.look_kinds and f.drop_reason != "prop"
            and f.dir_depth == portal.from_dir_depth)


# ---------------------------------------------------------------------------
# Predict: bands and windows
# ---------------------------------------------------------------------------


def conformal_band(values: Sequence[float], lo_q: float = BAND_LO_Q, hi_q: float = BAND_HI_Q
                   ) -> Optional[Tuple[float, float]]:
    """Split-conformal [lo_q, hi_q] band: order statistics floor(lo_q(n+1))
    and ceil(hi_q(n+1)). A new exchangeable value falls inside with
    probability >= hi_q - lo_q. None with too few values."""
    v = sorted(float(x) for x in values)
    n = len(v)
    k_lo, k_hi = math.floor(lo_q * (n + 1)), math.ceil(hi_q * (n + 1))
    if n == 0 or k_lo < 1 or k_hi > n:
        return None
    return v[k_lo - 1], v[k_hi - 1]


def prior_band(portal) -> Tuple[float, float]:
    if portal.kind == "approach":
        return float(portal.prior_sec[0]), float(portal.prior_sec[1])
    g0 = float(portal.prior_gap_frame_widths)
    return PRIOR_LO_FACTOR * g0, PRIOR_HI_FACTOR * g0


@dataclass(frozen=True)
class Band:
    lo: float
    hi: float
    source: str                     # prior | learned | learned_person_widened
    calib: Tuple[float, ...] = ()


def choose_band(kind: str, learned: Mapping[str, Sequence[float]], portal, cfg: HandoffConfig) -> Band:
    """Own band after ``min_arrivals`` arrivals of this kind (``learned`` holds
    every met, early and late ``g``); else the person band widened; else the
    prior."""
    own = learned.get(kind) or ()
    if len(own) >= cfg.min_arrivals:
        b = conformal_band(own)
        if b:
            return Band(b[0], b[1], "learned", tuple(own))
    people = learned.get("person") or ()
    if kind != "person" and len(people) >= cfg.min_arrivals:
        b = conformal_band(people)
        if b:
            lo, hi = b
            w = cfg.person_widen
            return Band(lo - (w - 1.0) * abs(lo), hi + (w - 1.0) * abs(hi), "learned_person_widened", tuple(people))
    lo, hi = prior_band(portal)
    return Band(lo, hi, "prior")


def window_for(departed_at: datetime, dist: float, speed: float, band: Band, cfg: HandoffConfig
               ) -> Tuple[datetime, datetime, datetime]:
    """(window_start, window_end, late_until). Gap: (dist + g) / speed after
    the story ended; approach (speed 1, dist 0): g seconds. Late margin: twice
    the window, capped."""
    v = max(float(speed), 1e-6)
    a = max(0.0, (dist + band.lo) / v)
    b = max(a + 1.0, (dist + band.hi) / v)
    b = min(b, max(a + 1.0, cfg.max_window_sec))
    ws, we = departed_at + timedelta(seconds=a), departed_at + timedelta(seconds=b)
    late = min(2.0 * (b - a), cfg.max_late_sec)
    return ws, we, we + timedelta(seconds=late)


def early_from(departed_at: datetime, window_start: datetime, window_end: datetime) -> datetime:
    """Start of the early margin: one window width before the window, never before departure."""
    return max(departed_at, window_start - (window_end - window_start))


def observed_g(transit_sec: float, dist: float, speed: float, portal_kind: str) -> float:
    if portal_kind == "approach":
        return float(transit_sec)
    return float(transit_sec) * float(speed) - float(dist)


# ---------------------------------------------------------------------------
# p-values (the calm state is defined in the module docstring)
# ---------------------------------------------------------------------------


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def prior_p_floor(min_arrivals: int) -> float:
    return 2.0 / (int(min_arrivals) + 1)


def _lognormal_of(band: Band) -> Tuple[float, float]:
    lo, hi = max(band.lo, 1e-6), max(band.hi, 2e-6)
    return (math.log(lo) + math.log(hi)) / 2.0, max((math.log(hi) - math.log(lo)) / (2.0 * Z90), 1e-6)


def arrival_log_score(g: float, band: Band, speed: float, portal_kind: str) -> float:
    """Log density, in nats per second, of the actual arrival time under the
    predicted arrival distribution: the log-normal over ``g`` whose 10% / 90%
    quantiles are the band ends (prior or learned), carried to time by the
    Jacobian ``dg/dt = speed`` (approach: 1). Higher is a better prediction.
    g <= 0 (arrived before the projection allows) is clamped to 1/1000 of
    the band's low end, which scores very low."""
    mu, sigma = _lognormal_of(band)
    gg = max(float(g), max(band.lo, 1e-6) * 1e-3)
    log_pdf_g = -math.log(gg * sigma * math.sqrt(2.0 * math.pi)) - (math.log(gg) - mu) ** 2 / (2.0 * sigma ** 2)
    v = 1.0 if portal_kind == "approach" else max(float(speed), 1e-6)
    return float(log_pdf_g + math.log(v))


def p_value_arrival(g: float, band: Band, *, prior_floor: float = 0.0) -> float:
    """Two-sided p of an arrival with gap ``g``. Learned band: conformal rank
    among the calibration values. Prior band: log-normal tails with the band
    ends as its 10% / 90% quantiles."""
    if band.source != "prior" and band.calib:
        c = band.calib
        n = len(c)
        lo = (1 + sum(1 for x in c if x <= g)) / (n + 1)
        hi = (1 + sum(1 for x in c if x >= g)) / (n + 1)
        return float(min(1.0, 2.0 * min(lo, hi)))
    mu, sigma = _lognormal_of(band)
    z = (math.log(max(g, max(band.lo, 1e-6) * 1e-3)) - mu) / sigma
    return float(max(prior_floor, min(1.0, 2.0 * min(_phi(z), 1.0 - _phi(z)))))


def p_value_missed(n_missed: int, n_arrived: int) -> float:
    """Counted chance that nobody arrives, Laplace-smoothed (0.5 with no history)."""
    return (int(n_missed) + 1) / (int(n_missed) + int(n_arrived) + 2)


def unusual_for_hour(count_at_hour: int, days: float, max_per_day: float) -> bool:
    return days > 0 and (count_at_hour / days) < max_per_day


def attention_gate(p: Optional[float], *, known_person: bool, unusual_hour: bool, cfg: HandoffConfig) -> bool:
    return p is not None and p < cfg.attention_p and (known_person or unusual_hour)


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


def appearance(src: Sequence[Mapping[str, np.ndarray]], tgt: Sequence[Mapping[str, np.ndarray]]
               ) -> Tuple[Optional[float], Optional[str]]:
    """Cosine of the two sides' centroid vectors on the first channel both
    sides have: face, else body, else OSNet (the person model's channels).
    Centroids, not best pairs: a max over many crop pairs climbs with the
    number of pairs even for two strangers."""
    for ch in ("face", "body", "osnet"):
        a = [v[ch] for v in src if ch in v]
        b = [v[ch] for v in tgt if ch in v]
        if a and b and len(a[0]) == len(b[0]):
            return float(np.dot(unit(np.mean(a, axis=0)), unit(np.mean(b, axis=0)))), ch
    return None, None


def arrival_consistent(c: StoryFacts, portal, cfg: HandoffConfig) -> bool:
    """A target story that came in through the portal's ``to`` edge: moving
    the arrival way, and its first box on that edge or near enough to it."""
    if c.dir_lateral != portal.arrival_sign and c.entry_edge != portal.to_edge:
        return False
    if c.entry_edge == portal.to_edge:
        return True
    if c.entry_edge != "inside":
        return False
    d = edge_distance(c.entry_foot_x, portal.to_edge)
    return d is not None and d <= cfg.max_project_frac


@dataclass(frozen=True)
class Candidate:
    story: StoryFacts
    score: Optional[float] = None
    channel: Optional[str] = None


@dataclass(frozen=True)
class Decision:
    status: str                       # met | early | late | ambiguous | none (overlap: paired | unpaired)
    chosen: Optional[Candidate] = None
    n_candidates: int = 0


def decide_match(cands: Sequence[Candidate], *, window_start: datetime, window_end: datetime, late_until: datetime,
                 cfg: HandoffConfig, early: Optional[datetime] = None) -> Decision:
    """Best candidate in [early, late_until] (``early`` defaults to the window
    start). A unique one that clears the appearance bar wins; with none
    clearing it, a lone candidate wins on timing alone; anything else is
    ambiguous (and never trains the gap)."""
    lo = early or window_start
    live = [c for c in cands if lo <= c.story.started_at <= late_until]
    if not live:
        return Decision("none")
    strong = sorted((c for c in live if c.score is not None and cfg.bar(c.channel) is not None
                     and c.score >= cfg.bar(c.channel)), key=lambda c: -(c.score or 0.0))
    chosen: Optional[Candidate] = None
    if len(strong) == 1:
        chosen = strong[0]
    elif len(strong) > 1:
        if (strong[0].score or 0) - (strong[1].score or 0) >= cfg.ambiguous_margin:
            chosen = strong[0]
    elif len(live) == 1:
        chosen = live[0]
    if chosen is None:
        mid = window_start + (window_end - window_start) / 2
        best = min(live, key=lambda c: abs((c.story.started_at - mid).total_seconds()))
        return Decision("ambiguous", best, len(live))
    t = chosen.story.started_at
    return Decision("early" if t < window_start else "met" if t <= window_end else "late", chosen, len(live))


def grade_unmatched(coverage: float, cfg: HandoffConfig) -> str:
    """No match: missed only if the target camera was looking."""
    return "missed" if coverage >= cfg.min_coverage else "unscorable"


def overlapping(a: StoryFacts, b: StoryFacts, skew_sec: float) -> bool:
    s = timedelta(seconds=skew_sec)
    return a.started_at - s <= b.ended_at and b.started_at - s <= a.ended_at


def overlap_seconds(a: StoryFacts, b: StoryFacts) -> float:
    return (min(a.ended_at, b.ended_at) - max(a.started_at, b.started_at)).total_seconds()


def decide_overlap(cands: Sequence[Candidate], cfg: HandoffConfig) -> Decision:
    """One doorbell story vs the walkway stories that overlap it: one
    candidate -> paired; several -> the unique strong-appearance one, else
    ambiguous (the longest overlap is kept for the record, no ask)."""
    if not cands:
        return Decision("unpaired")
    if len(cands) == 1:
        return Decision("paired", cands[0], 1)
    strong = sorted((c for c in cands if c.score is not None and cfg.bar(c.channel) is not None
                     and c.score >= cfg.bar(c.channel)), key=lambda c: -(c.score or 0.0))
    if strong and (len(strong) == 1 or (strong[0].score or 0) - (strong[1].score or 0) >= cfg.ambiguous_margin):
        return Decision("paired", strong[0], len(cands))
    return Decision("ambiguous", cands[0], len(cands))


def strong_appearance(score: Optional[float], channel: Optional[str], cfg: HandoffConfig) -> bool:
    bar = cfg.bar(channel)
    return score is not None and bar is not None and score >= bar


# ---------------------------------------------------------------------------
# Misses: hypotheses and counted odds (spec section 4)
# ---------------------------------------------------------------------------


def hypothesis_priors(true_counts: Mapping[str, int], kinds: Sequence[str] = HYPOTHESES) -> Dict[str, float]:
    """Counted outcomes per portal as odds: every kind starts at 1."""
    w = {k: int(true_counts.get(k, 0)) + 1 for k in kinds}
    total = float(sum(w.values()))
    return {k: w[k] / total for k in kinds}


def hypothesis_deadlines(*, graded_at: datetime, departed_at: datetime, late_until: datetime,
                         cfg: HandoffConfig) -> Dict[str, datetime]:
    later = late_until + timedelta(seconds=cfg.hypothesis_sec)
    return {"detector_miss": graded_at,
            "late": later,
            "turned_back": max(departed_at + timedelta(seconds=cfg.hypothesis_sec), late_until),
            "entered_place": later,
            "unexplained": later}


def turned_back(src: StoryFacts, later: StoryFacts, portal) -> bool:
    """A source-camera story of the same kind, after this one, going the
    other way (gap: opposite dir_lateral; approach: moving away)."""
    if later.stream_id != src.stream_id or later.kind != src.kind or later.started_at <= src.ended_at:
        return False
    if portal.kind == "approach":
        return later.dir_depth == -int(portal.from_dir_depth or 0)
    return later.dir_lateral == -portal.lateral_sign


def entered_place(last_zone: Optional[str], place_zones: Sequence[str]) -> bool:
    return bool(last_zone) and last_zone in set(place_zones)


def settle_final(status: Mapping[str, str], *, last_zone: Optional[str], place_zones: Sequence[str]
                 ) -> Dict[str, str]:
    """At the deadline: open late / turned_back -> false; entered_place true
    only if its zone leads somewhere and nothing else came true; unexplained
    true iff nothing else is true."""
    out = dict(status)
    for k in ("late", "turned_back"):
        if out.get(k, "open") == "open":
            out[k] = "false"
    others_true = any(out.get(k) == "true" for k in ("detector_miss", "late", "turned_back"))
    if out.get("entered_place", "open") == "open":
        out["entered_place"] = "true" if (not others_true and entered_place(last_zone, place_zones)) else "false"
    explained = others_true or out["entered_place"] == "true"
    out["unexplained"] = "false" if explained else "true"
    return out


# ---------------------------------------------------------------------------
# Salience (spec section 6): who x what x surprise -- all values in one place
# ---------------------------------------------------------------------------

SALIENCE_WHAT = {
    "doorbell_ring": 1.0,
    "approach_door": 0.6,       # "arrival at door"
    "handoff_met": 0.4,
    "handoff_early": 0.4,
    "handoff_late": 0.4,
    "handoff_missed": 0.4,
    "overlap_pair": 0.2,
}
SALIENCE_WHO = {"known": 1.0, "person": 0.6, "dog": 0.4, "vehicle": 0.2, "unknown": 0.4}
# Ungraded events have no p: an approach is an arrival in itself (surprise 1);
# an overlap pair is a quiet fact (low).
SALIENCE_SURPRISE_UNGRADED = {"doorbell_ring": 1.0, "approach_door": 1.0, "overlap_pair": 0.5}
_SUBJECT_KIND = {"person": "person", "bicycle": "person", "stroller": "person", "dog": "dog", "vehicle": "vehicle"}


def subject_kind(kind: Optional[str]) -> str:
    """Motion-story kind -> StreetEventV1.subject_kind (a bicycle or stroller has a person on it)."""
    return _SUBJECT_KIND.get(kind or "", "unknown")


def salience(what: str, kind: Optional[str], *, known_person: bool, p: Optional[float]) -> float:
    who = SALIENCE_WHO["known"] if known_person else SALIENCE_WHO.get(subject_kind(kind), SALIENCE_WHO["unknown"])
    if what in SALIENCE_SURPRISE_UNGRADED:
        surprise = SALIENCE_SURPRISE_UNGRADED[what]
    else:
        surprise = 1.0 - float(p) if p is not None else 0.0
    return round(min(1.0, max(0.0, SALIENCE_WHAT[what] * who * surprise)), 4)


# ---------------------------------------------------------------------------
# Ids, words
# ---------------------------------------------------------------------------


def handoff_id(portal_id: str, *parts: str) -> str:
    return str(uuid.uuid5(HANDOFF_NS, "|".join((portal_id,) + tuple(parts))))


def correlation_for(hid: str) -> str:
    return str(uuid.uuid5(HANDOFF_NS, "corr|" + hid))


CAMERA_WORDS = {"doorbell": "door camera", "walkway": "walkway camera"}


def camera_words(stream_id: Optional[str]) -> str:
    return CAMERA_WORDS.get(stream_id or "", f"{stream_id} camera")


def narrative(status: str, *, kind: str, source_stream: str, target_stream: str,
              transit_sec: Optional[float] = None) -> str:
    who = "someone" if subject_kind(kind) == "person" else f"a {kind}"
    src, tgt = camera_words(source_stream), camera_words(target_stream)
    if status in ARRIVED and transit_sec is not None:
        s = f"{who.capitalize()} left the {src} and reached the {tgt} {round(transit_sec)} s later"
        return s + {"late": ", later than expected.", "early": ", sooner than expected."}.get(status, ", as expected.")
    if status == "missed":
        return f"{who.capitalize()} left the {src} toward the {tgt}, which was watching, and never arrived."
    if status == "paired":
        return f"The {src} and the {tgt} saw {who} at the same moment."
    return f"{who.capitalize()} left the {src} toward the {tgt} ({status})."


def same_person_question(source_stream: str, target_stream: str, gap_sec: Optional[float]) -> str:
    src, tgt = camera_words(source_stream), camera_words(target_stream)
    if gap_sec is None or abs(gap_sec) < 5:
        return f"Same person? The {src} and the {tgt} saw them at the same moment."
    return f"Same person? Seen on the {src}, then on the {tgt} {round(abs(gap_sec))} s later."


# ---------------------------------------------------------------------------
# StreetEventV1
# ---------------------------------------------------------------------------


@dataclass
class StreetFacts:
    kind: str
    hid: str
    portal_id: str
    stream_ids: List[str]
    occurred_at: datetime
    subject: Optional[str]
    individual_id: Optional[str] = None
    person_guess: Optional[str] = None
    p_value: Optional[float] = None
    gate: Optional[bool] = None
    arrival_log_score: Optional[float] = None
    refs: List[str] = field(default_factory=list)


def street_event(f: StreetFacts) -> dict:
    """A validated StreetEventV1 payload (JSON-ready). Salience from the table above."""
    from orion.schemas.vision import StreetEventV1

    sal = salience(f.kind, f.subject, known_person=bool(f.person_guess), p=f.p_value)
    refs = [f"handoff:{f.hid}", f"portal:{f.portal_id}"] + list(f.refs)
    if f.gate is not None:
        refs.append(f"attention_gate:{'pass' if f.gate else 'fail'}")
    ev = StreetEventV1(
        event_id=f"street-{f.kind}-{f.hid}", kind=f.kind, stream_ids=[s for s in f.stream_ids if s] or ["walkway"],
        occurred_at=f.occurred_at, subject_kind=subject_kind(f.subject), individual_id=f.individual_id,
        person_guess=f.person_guess, salience=sal,
        p_value=None if f.p_value is None else min(1.0, max(0.0, float(f.p_value))),
        arrival_log_score=f.arrival_log_score if f.kind in ("handoff_met", "handoff_late") else None,
        evidence_refs=refs, correlation_id=correlation_for(f.hid))
    out = ev.model_dump(mode="json")
    if out.get("arrival_log_score") is None:
        # Keep payloads without the score byte-identical to the pre-field shape.
        out.pop("arrival_log_score", None)
    return out


# ---------------------------------------------------------------------------
# Redis steering values (the key semantics are vision_rhythm's)
# ---------------------------------------------------------------------------


def merged_expect_value(existing: Optional[bytes | str], hid: str, until: datetime) -> str:
    """The expect key's value with this handoff added. Rhythm's
    ``expectation_ids`` are kept; ``handoff_until`` is the latest handoff end,
    which the rhythm loop honours before it shortens or deletes the key."""
    try:
        val = json.loads(existing) if existing else {}
        if not isinstance(val, dict):
            val = {}
    except (ValueError, TypeError):
        val = {}
    ids = [i for i in val.get("handoff_ids") or [] if isinstance(i, str)]
    if hid not in ids:
        ids.append(hid)
    val["handoff_ids"] = ids[-10:]
    prev = val.get("handoff_until")
    try:
        prev_t = datetime.fromisoformat(prev) if prev else None
    except ValueError:
        prev_t = None
    val["handoff_until"] = max(until, prev_t).isoformat() if prev_t else until.isoformat()
    return json.dumps(val)


def ttl_seconds(until: datetime, now: datetime) -> Optional[int]:
    s = (until - now).total_seconds()
    return int(math.ceil(s)) if s > 0 else None
