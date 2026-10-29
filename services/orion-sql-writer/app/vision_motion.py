"""Walkway motion stories: one plain sentence per passing thing, from box maths.

Design: docs/superpowers/specs/2026-10-16-walkway-motion-stories-proposal.md
(slice 1, approved by Juniper 2026-10-16: Hub-only, no fast-mover cadence work,
no "child" word).

Input is ``vision_crop_observation`` rows for one stream in ``observed_at``
order. Everything here is deterministic arithmetic on the detector's boxes: no
model, no embeddings, no new identity. Pure functions only (no clock, no DB);
``vision_motion_loop.py`` does the I/O.

1. **Link** boxes into passages by geometry, not by ``individual_id``. The
   first link joins a box of the same label within ``link_gap_sec`` and
   within ``first_gate_bh`` box heights, shrunk in proportion to the gap once
   the gap is longer than ``first_gate_ref_sec`` (a 10 s gap is a guess, not
   a track). Later links use a constant-velocity prediction from the last two
   boxes with a ``pred_gate_bh`` gate. Box height may change at most
   ``max_height_ratio`` per link. One box joins one passage per frame; greedy
   by distance. Individual ids then supply the NAME only.
2. **Clip.** A box that touches the left, right or bottom frame edge has no
   real foot point: it gives no zone and no travel.
3. **Kinematics** on the foot point (bottom centre), in box heights, so a far
   walker and a near walker read the same speed.
4. **Speed class** per kind; the fast class needs >= 3 unclipped detections
   (two boxes 2.5 s apart can jitter by a box height on their own).
5. **Props.** A passage is a prop (no story) when any of its crops belongs to
   a ``stationary`` individual or a configured prop individual, or when most
   of its boxes overlap a known still box. The patio skeleton (fefb24d7, seen
   2026-10-15) was never flagged ``stationary`` because its box drifted; its
   passages also travel < 0.5 box heights, so they classify as "stood" and
   standing is never published in this slice.
6. **Name** only when one named individual covers >= 2/3 of the passage's
   crops, no other named individual appears, and no crop touches the frame
   edge. Otherwise "a person".
7. **Sentence.**
8. **Frame and edges** (2026-10-20, for the street-continuity handoff:
   docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md,
   section 1). Every box is measured against ITS OWN frame size: the size the
   detector saw (``vision_crop_observation.frame_width/height``), else the
   stream's ``VISION_MOTION_FRAME_SIZE_BY_STREAM`` entry, else the global
   ``VISION_MOTION_FRAME_WIDTH/HEIGHT``. A story records where it came in and
   went out (``entry_edge`` / ``exit_edge``), the last foot point, and the
   exit speed in frame widths per second (see ``edge_of`` / ``exit_speed``).
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from orion.vision.stationary import box_iou
from orion.vision.story_naming import NOUNS as _SHARED_NOUNS
from orion.vision.story_naming import name_from_crops
from orion.vision.zones import Zone, zone_for_box

# Bump when a rule below changes meaning: Juniper's spot-check tally counts one
# version at a time, so old verdicts never vouch for new rules. Not bumped for
# the per-camera frame size and edges (2026-10-20): on the walkway (the only
# stream with verdicts) every frame is the old global 2560x1920, so no story
# reads differently; the new columns are additive facts the tally never reads.
RULES_VERSION = 1

EDGES = ("left", "right", "top", "bottom", "inside")
# Where a story's frame size came from (vision_motion_story.frame_size_source).
FRAME_SOURCES = ("detection", "stream_setting", "default")

KINDS = ("person", "dog", "bicycle", "stroller", "vehicle")
OUTSIDE = "outside the zones"

# Per kind: (still class, [(upper speed bound BH/s, class), ...], fast class).
# Calibrated in the design doc (section "Measured: speed in body-heights per
# second"): live walkers with 3+ detections top out at 0.82 BH/s; adult/child
# walking is 0.7-0.9 BH/s laterally, jogging 1.5-2.
SPEED_CLASSES: Dict[str, Tuple[str, List[Tuple[float, str]], Optional[str]]] = {
    "person": ("standing", [(1.2, "walking")], "running"),
    "dog": ("standing", [(1.5, "walking")], "running"),
    "bicycle": ("parked", [(1.5, "walked")], "riding"),
    "stroller": ("parked", [(99.0, "pushed")], None),
    "vehicle": ("parked", [(0.6, "creeping")], "driving"),
}
STILL_CLASSES = {"standing", "parked"}
# Slice 1 publishes walking-pace stories of these kinds only.
PUBLISH_KINDS = {"person", "dog", "stroller", "bicycle"}
WALKING_CLASSES = {"walking", "walked", "pushed"}
# Below this net speed (BH/s) a passage did not really travel, even when its
# net travel passes still_bh over a long dwell.
STILL_SPEED_BH = 0.08

VERBS = {
    "standing": "stood", "parked": "stayed parked", "walking": "walked", "running": "ran",
    "walked": "was walked", "riding": "was ridden", "pushed": "was pushed",
    "creeping": "crept", "driving": "drove",
}
# Shared with the Hub, which re-derives a story's name when it shows it.
NOUNS = _SHARED_NOUNS


@dataclass(frozen=True)
class MotionConfig:
    frame_w: float = 2560.0
    frame_h: float = 1920.0
    link_gap_sec: float = 11.0
    first_gate_bh: float = 4.0
    first_gate_ref_sec: float = 5.0
    pred_gate_bh: float = 2.0
    max_height_ratio: float = 2.0
    min_detections: int = 3
    fast_min_detections: int = 3
    still_bh: float = 0.5
    edge_margin_px: float = 8.0
    name_share: float = 2.0 / 3.0
    prop_iou: float = 0.3
    prop_share: float = 2.0 / 3.0
    lateral_frac: float = 0.08
    depth_frac: float = 0.10
    depth_logh: float = 0.35
    together_max_bh: float = 2.0
    together_min_shared: int = 2
    together_cos: float = 0.7
    together_min_travel_bh: float = 1.0
    # Where frame_w/frame_h came from: "default" (the global setting) or
    # "stream_setting" (VISION_MOTION_FRAME_SIZE_BY_STREAM). A box's own
    # detector frame size always wins over both (see frame_of).
    frame_source: str = "default"
    # A first/last box within this share of the frame from an edge entered or
    # left through that edge; farther in, it is "inside".
    edge_zone_frac: float = 0.05
    # Exit speed uses at most this many of the last unclipped boxes.
    exit_speed_boxes: int = 4


@dataclass(frozen=True)
class MotionCrop:
    crop_id: str
    t: datetime
    label: str
    box: Tuple[float, float, float, float]
    score: float = 0.0
    individual_id: Optional[str] = None
    thumb_ref: Optional[str] = None
    # The frame the detector saw this box in (None: not recorded).
    frame_w: Optional[float] = None
    frame_h: Optional[float] = None

    @property
    def h(self) -> float:
        return max(1.0, float(self.box[3]) - float(self.box[1]))

    @property
    def foot(self) -> Tuple[float, float]:
        return ((float(self.box[0]) + float(self.box[2])) / 2.0, float(self.box[3]))


@dataclass
class Passage:
    kind: str
    crops: List[MotionCrop] = field(default_factory=list)

    @property
    def first(self) -> MotionCrop:
        return self.crops[0]

    @property
    def last(self) -> MotionCrop:
        return self.crops[-1]


@dataclass(frozen=True)
class Kinematics:
    used: int
    dur: float
    hmed: float
    net_bh: float
    path_bh: float
    speed: float
    dx: float
    dy: float
    dlogh: float


@dataclass
class Story:
    story_id: str
    stream_id: str
    kind: str
    started_at: datetime
    ended_at: datetime
    detection_count: int
    used_detections: int
    edge_clipped_count: int
    speed_class: Optional[str]
    speed_bh_per_sec: Optional[float]
    net_travel_bh: Optional[float]
    path_bh: Optional[float]
    dir_lateral: int
    dir_depth: int
    zone_path: List[str]
    individual_id: Optional[str]
    name_label: Optional[str]
    name_withheld: Optional[str]
    companion_kinds: List[str]
    together_story_ids: List[str]
    published: bool
    drop_reason: Optional[str]
    clean: bool
    sentence: str
    crop_ids: List[str]
    evidence: List[dict]
    rules_version: int = RULES_VERSION
    # Frame and edges (manual_migration_vision_motion_story_edges_v1.sql).
    entry_edge: Optional[str] = None
    exit_edge: Optional[str] = None
    exit_foot_x: Optional[float] = None
    exit_foot_y: Optional[float] = None
    exit_speed_fw_per_sec: Optional[float] = None
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None
    frame_size_source: Optional[str] = None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


Frame = Tuple[float, float]


def frame_of(c: MotionCrop, cfg: MotionConfig) -> Frame:
    """The frame size this box was seen in: its own, else the configured one."""
    if c.frame_w and c.frame_h and c.frame_w > 0 and c.frame_h > 0:
        return float(c.frame_w), float(c.frame_h)
    return cfg.frame_w, cfg.frame_h


def is_edge_clipped(box: Sequence[float], cfg: MotionConfig, frame: Optional[Frame] = None) -> bool:
    """Touches the left, right or bottom edge: the foot point is the frame
    edge, not a foot (sample S10: y2 = 1914-1916 of 1920). The top edge does
    not move the foot point. ``frame`` is the box's own frame size (default:
    the configured one)."""
    x1, _y1, x2, y2 = (float(v) for v in box)
    fw, fh = frame or (cfg.frame_w, cfg.frame_h)
    m = cfg.edge_margin_px
    return x1 <= m or x2 >= fw - m or y2 >= fh - m


def crop_clipped(c: MotionCrop, cfg: MotionConfig) -> bool:
    return is_edge_clipped(c.box, cfg, frame_of(c, cfg))


def passage_frame(p: "Passage", cfg: MotionConfig) -> Tuple[float, float, str]:
    """(w, h, source) for a story: the commonest detector frame size among its
    boxes, else the configured size and where that came from."""
    real = [frame_of(c, cfg) for c in p.crops if c.frame_w and c.frame_h and c.frame_w > 0 and c.frame_h > 0]
    if real:
        w, h = max(sorted(set(real)), key=real.count)
        return w, h, "detection"
    return cfg.frame_w, cfg.frame_h, cfg.frame_source


def edge_of(c: MotionCrop, cfg: MotionConfig) -> str:
    """Which frame edge this box sits on, else "inside".

    left / right / bottom: the box's side is within ``edge_zone_frac`` of that
    edge (a clipped box is at distance ~0); the nearest one wins, ties in the
    order left, right, bottom. top: only when none of those, and the FOOT
    point (not the head) is within the margin of the top: a near person's head
    is often cut by the top edge while they walk across the frame, and that is
    not leaving through the top."""
    fw, fh = frame_of(c, cfg)
    x1, _y1, x2, y2 = (float(v) for v in c.box)
    f = cfg.edge_zone_frac
    cands = [(x1 / fw, 0, "left"), ((fw - x2) / fw, 1, "right"), ((fh - y2) / fh, 2, "bottom")]
    near = [cd for cd in cands if cd[0] <= f]
    if near:
        return min(near)[2]
    if y2 / fh <= f:
        return "top"
    return "inside"


def exit_foot(c: MotionCrop, cfg: MotionConfig) -> Tuple[float, float]:
    """The last box's foot point (bottom centre) as a share of its frame,
    clamped to 0..1. On a clipped box this is on the frame edge."""
    fw, fh = frame_of(c, cfg)
    x, y = c.foot
    return min(1.0, max(0.0, x / fw)), min(1.0, max(0.0, y / fh))


def exit_speed(usable: Sequence[MotionCrop], cfg: MotionConfig) -> Optional[float]:
    """Lateral foot speed over the last ``exit_speed_boxes`` unclipped boxes, in
    frame widths per second, UNSIGNED (the way out is ``exit_edge``; the
    lateral sense is ``dir_lateral``). Frame widths, not box heights: a
    cross-camera clock needs a speed that does not depend on distance. None
    with fewer than two unclipped boxes or no elapsed time. Clipped boxes are
    left out: their visible centre lags the real one."""
    tail = list(usable)[-max(2, cfg.exit_speed_boxes):]
    if len(tail) < 2:
        return None
    dt = (tail[-1].t - tail[0].t).total_seconds()
    if dt <= 0:
        return None
    (w0, _), (w1, _) = frame_of(tail[0], cfg), frame_of(tail[-1], cfg)
    return abs(tail[-1].foot[0] / w1 - tail[0].foot[0] / w0) / dt


def story_id_for(stream_id: str, first_crop_id: str) -> str:
    return hashlib.sha1(f"{stream_id}|{first_crop_id}".encode()).hexdigest()[:32]


def first_gate(cfg: MotionConfig, dt: float) -> float:
    """The first-link gate in box heights. Full width up to the reference gap
    (one baseline detector interval); beyond it, shrunk in proportion, so a
    10 s gap gets half the gate (sample S1 was a 575 px jump over 10 s)."""
    if dt <= cfg.first_gate_ref_sec:
        return cfg.first_gate_bh
    return cfg.first_gate_bh * cfg.first_gate_ref_sec / dt


# ---------------------------------------------------------------------------
# 1. Linking
# ---------------------------------------------------------------------------


def link_passages(crops: Iterable[MotionCrop], cfg: MotionConfig) -> List[Passage]:
    """Greedy constant-velocity linking over detected frames. Returns every
    passage (closed or not) in order of first detection."""
    frames: Dict[datetime, List[MotionCrop]] = {}
    for c in sorted(crops, key=lambda c: (c.t, c.crop_id)):
        if c.label in SPEED_CLASSES:
            frames.setdefault(c.t, []).append(c)
    live: List[Passage] = []
    done: List[Passage] = []
    for t in sorted(frames):
        boxes = frames[t]
        still_live = []
        for p in live:
            (done if (t - p.last.t).total_seconds() > cfg.link_gap_sec else still_live).append(p)
        live = still_live
        cands: List[Tuple[float, int, int]] = []
        for i, c in enumerate(boxes):
            for j, p in enumerate(live):
                if p.kind != c.label:
                    continue
                last = p.last
                dt = (t - last.t).total_seconds()
                if dt <= 0:
                    continue
                ratio = c.h / last.h
                if not (1.0 / cfg.max_height_ratio <= ratio <= cfg.max_height_ratio):
                    continue
                if len(p.crops) >= 2:
                    prev = p.crops[-2]
                    pdt = (last.t - prev.t).total_seconds() or 1.0
                    vx = (last.foot[0] - prev.foot[0]) / pdt
                    vy = (last.foot[1] - prev.foot[1]) / pdt
                    pred = (last.foot[0] + vx * dt, last.foot[1] + vy * dt)
                    gate = cfg.pred_gate_bh
                else:
                    pred = last.foot
                    gate = first_gate(cfg, dt)
                d = math.dist(pred, c.foot) / ((last.h + c.h) / 2.0)
                if d <= gate:
                    cands.append((d, i, j))
        cands.sort()
        used_i: Set[int] = set()
        used_j: Set[int] = set()
        for _d, i, j in cands:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            live[j].crops.append(boxes[i])
        for i, c in enumerate(boxes):
            if i not in used_i:
                live.append(Passage(c.label, [c]))
    out = done + live
    out.sort(key=lambda p: (p.first.t, p.first.crop_id))
    return out


# ---------------------------------------------------------------------------
# 3-4. Kinematics, speed, direction
# ---------------------------------------------------------------------------


def usable_crops(p: Passage, cfg: MotionConfig) -> List[MotionCrop]:
    return [c for c in p.crops if not crop_clipped(c, cfg)]


def kinematics(crops: Sequence[MotionCrop], cfg: MotionConfig) -> Optional[Kinematics]:
    """None when fewer than two unclipped boxes: nothing can be said about travel."""
    if len(crops) < 2:
        return None
    dur = (crops[-1].t - crops[0].t).total_seconds()
    hmed = statistics.median(c.h for c in crops)
    (x0, y0), (x1, y1) = crops[0].foot, crops[-1].foot
    net_bh = math.dist((x0, y0), (x1, y1)) / hmed
    path_bh = 0.0
    for a, b in zip(crops, crops[1:]):
        path_bh += math.dist(a.foot, b.foot) / ((a.h + b.h) / 2.0)
    speed = net_bh / dur if dur > 0 else 0.0
    # Each end in its own frame (a share of the frame), so a 1600 px doorbell
    # frame is not read as 0.6 of a 2560 px walkway frame.
    (w0, h0), (w1, h1) = frame_of(crops[0], cfg), frame_of(crops[-1], cfg)
    dx = (x1 - x0) / w0 if w0 == w1 else x1 / w1 - x0 / w0
    dy = (y1 - y0) / h0 if h0 == h1 else y1 / h1 - y0 / h0
    return Kinematics(used=len(crops), dur=dur, hmed=hmed, net_bh=net_bh, path_bh=path_bh, speed=speed,
                      dx=dx, dy=dy, dlogh=math.log(crops[-1].h / crops[0].h))


def speed_class(kind: str, k: Kinematics, cfg: MotionConfig) -> str:
    still, bands, fast = SPEED_CLASSES.get(kind, SPEED_CLASSES["person"])
    if k.dur <= 0 or k.net_bh < cfg.still_bh or k.speed < STILL_SPEED_BH:
        return still
    for upper, name in bands:
        if k.speed < upper:
            return name
    if fast is None:
        return bands[-1][1]
    # The fast class needs enough looks to rule out jitter.
    return fast if k.used >= cfg.fast_min_detections else bands[-1][1]


def direction(k: Kinematics, cfg: MotionConfig) -> Tuple[int, int]:
    """(lateral, depth): lateral +1 = left to right; depth +1 = toward the
    house (the camera looks out from under the porch, so down the frame and a
    growing box mean coming closer)."""
    if k.net_bh < cfg.still_bh:
        return 0, 0
    lat = 0 if abs(k.dx) < cfg.lateral_frac else (1 if k.dx > 0 else -1)
    if abs(k.dy) >= cfg.depth_frac:
        dep = 1 if k.dy > 0 else -1
    elif abs(k.dlogh) >= cfg.depth_logh:
        dep = 1 if k.dlogh > 0 else -1
    else:
        dep = 0
    return lat, dep


def direction_phrase(lat: int, dep: int) -> str:
    parts = []
    if lat:
        parts.append("left to right" if lat > 0 else "right to left")
    if dep:
        parts.append("toward the house" if dep > 0 else "away from the house")
    return " and ".join(parts)


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------


def zone_path(crops: Sequence[MotionCrop], zones: Sequence[Zone], cfg: MotionConfig, *,
              still: bool = False) -> List[str]:
    """Zone of each unclipped box's foot point, smoothed with a 3-wide
    majority filter (a box jittering on a zone edge is not a dozen
    crossings), collapsed to the zones entered. A still passage gets only its
    majority zone."""
    raw = []
    for c in crops:
        fw, fh = frame_of(c, cfg)
        z = zone_for_box(zones, c.box, fw, fh)
        raw.append(z.name if z else OUTSIDE)
    if not raw:
        return []
    if still:
        return [max(sorted(set(raw)), key=raw.count)]
    sm = []
    for i, z in enumerate(raw):
        win = raw[max(0, i - 1): i + 2]
        if len(win) == 3:
            sm.append(max(sorted(set(win)), key=lambda v: (win.count(v), v == z)))
        else:
            sm.append(z)
    seq: List[str] = []
    for z in sm:
        if not seq or seq[-1] != z:
            seq.append(z)
    return seq


def zone_phrase(zp: Sequence[str]) -> str:
    if not zp:
        return ""
    zs = [z if z == OUTSIDE else "the " + z for z in zp]
    if len(zs) == 1:
        return zs[0] if zs[0] == OUTSIDE else "on " + zs[0]
    distinct = list(dict.fromkeys(zs))
    if len(zs) >= 4 and len(distinct) == 2:
        return f"back and forth between {distinct[0]} and {distinct[1]}"
    if len(zs) == 2:
        return f"from {zs[0]} to {zs[1]}"
    return f"from {zs[0]}, " + ", ".join(f"through {z}" for z in zs[1:-1]) + f", to {zs[-1]}"


# ---------------------------------------------------------------------------
# 5. Props
# ---------------------------------------------------------------------------


PropBox = Tuple[str, Sequence[float]]   # (kind, box): a parked car's box never swallows a person


def is_prop(p: Passage, *, stationary_ids: Set[str], prop_ids: Set[str],
            prop_boxes: Sequence[PropBox], cfg: MotionConfig) -> bool:
    ids = {c.individual_id for c in p.crops if c.individual_id}
    if ids & (stationary_ids | prop_ids):
        return True
    boxes = [b for kind, b in prop_boxes if kind == p.kind]
    if not boxes:
        return False
    hits = sum(1 for c in p.crops if any(box_iou(c.box, b) >= cfg.prop_iou for b in boxes))
    return hits / len(p.crops) >= cfg.prop_share


# ---------------------------------------------------------------------------
# 6. Naming
# ---------------------------------------------------------------------------


def name_for(p: Passage, labels: Dict[str, str], cfg: MotionConfig
             ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(individual_id, label, withheld_reason). The strict rule:

    - every crop that carries an individual id carries a NAMED one, and all
      of them carry the same name (an unnamed id X next to a named one means
      the linker may have joined two people: ``other_id``; two names:
      ``two_named``; several ids of one name are fine -- one person can hold
      several ids);
    - that name covers >= name_share of the passage's crops (``low_share``);
    - no crop touches the frame edge (``edge_clipped``).
    Crops with no id at all do not block a name. The returned id is the one
    on most crops. The rule itself is orion.vision.story_naming.name_from_crops,
    shared with the Hub's read-time re-derivation."""
    return name_from_crops([(c.individual_id, crop_clipped(c, cfg)) for c in p.crops], labels,
                           cfg.name_share)


# ---------------------------------------------------------------------------
# Together
# ---------------------------------------------------------------------------


def together_pairs(passages: Sequence[Passage], cfg: MotionConfig) -> List[Tuple[int, int]]:
    """Index pairs moving together: >= together_min_shared shared instants
    (within 1 s), feet within together_max_bh at every one, both travelled,
    heading the same way (cosine >= together_cos). Vehicles never pair."""
    kins = [kinematics(usable_crops(p, cfg), cfg) for p in passages]
    out = []
    for i, a in enumerate(passages):
        for j in range(i + 1, len(passages)):
            b = passages[j]
            if "vehicle" in (a.kind, b.kind):
                continue
            if a.last.t < b.first.t or b.last.t < a.first.t:
                continue
            ka, kb = kins[i], kins[j]
            if ka is None or kb is None:
                continue
            if ka.net_bh < cfg.together_min_travel_bh or kb.net_bh < cfg.together_min_travel_bh:
                continue
            na, nb = math.hypot(ka.dx, ka.dy), math.hypot(kb.dx, kb.dy)
            if na == 0 or nb == 0 or (ka.dx * kb.dx + ka.dy * kb.dy) / (na * nb) < cfg.together_cos:
                continue
            shared = []
            for ca in a.crops:
                cb = min(b.crops, key=lambda x: abs((x.t - ca.t).total_seconds()))
                if abs((cb.t - ca.t).total_seconds()) <= 1.0:
                    shared.append(math.dist(ca.foot, cb.foot) / ((ca.h + cb.h) / 2.0))
            if len(shared) >= cfg.together_min_shared and max(shared) <= cfg.together_max_bh:
                out.append((i, j))
    return out


# ---------------------------------------------------------------------------
# 7. Sentence and story
# ---------------------------------------------------------------------------


def sentence_for(*, kind: str, sc: Optional[str], lat: int, dep: int, zp: Sequence[str], dur: float,
                 n: int, name: Optional[str], companions: Sequence[str], people_together: int) -> str:
    subj = name or NOUNS.get(kind, f"a {kind}")
    if kind == "person" and len(companions) == 1:
        subj += " with " + NOUNS.get(companions[0], f"a {companions[0]}")
    subj = subj[0].upper() + subj[1:]
    if sc is None:
        s = f"{subj} was seen"
    else:
        s = f"{subj} {VERBS.get(sc, 'moved')}"
        d = direction_phrase(lat, dep)
        if d and sc not in STILL_CLASSES:
            s += f" {d}"
    zph = zone_phrase(zp)
    if zph:
        s += " " + zph
    s += f", over {round(dur)} s ({n} detection{'s' if n != 1 else ''})"
    if people_together:
        s += f", together with {people_together} other {'person' if people_together == 1 else 'people'}"
    others = [c for c in companions] if not (kind == "person" and len(companions) == 1) else []
    if others:
        s += ", together with " + ", ".join(NOUNS.get(c, f"a {c}") for c in others)
    return s + "."


def _thumb_hex(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    h = ref.split(":", 1)[1] if ref.startswith("thumb:") else ref
    return h if len(h) == 64 and all(ch in "0123456789abcdef" for ch in h) else None


def build_stories(
    passages: Sequence[Passage], *, stream_id: str, zones: Sequence[Zone], cfg: MotionConfig,
    labels: Dict[str, str], stationary_ids: Set[str], prop_ids: Set[str],
    prop_boxes: Sequence[PropBox], clean_since: Optional[datetime],
    emit: Optional[Set[int]] = None, too_long: Optional[Set[int]] = None,
) -> List[Story]:
    """One Story per passage index in ``emit`` (default: all). ``passages`` is
    the whole linked window, so a "together" partner can be a passage that is
    not emitted this tick."""
    emit = set(range(len(passages))) if emit is None else emit
    too_long = too_long or set()
    sids = [story_id_for(stream_id, p.first.crop_id) for p in passages]
    usable = [usable_crops(p, cfg) for p in passages]
    kins = [kinematics(u, cfg) for u in usable]
    classes = [speed_class(p.kind, k, cfg) if k else None for p, k in zip(passages, kins)]
    props = [is_prop(p, stationary_ids=stationary_ids, prop_ids=prop_ids, prop_boxes=prop_boxes, cfg=cfg)
             for p in passages]
    moving = [i for i, sc in enumerate(classes) if sc and sc not in STILL_CLASSES and not props[i]]
    partners: Dict[int, List[int]] = {}
    for a, b in together_pairs([passages[i] for i in moving], cfg):
        ia, ib = moving[a], moving[b]
        partners.setdefault(ia, []).append(ib)
        partners.setdefault(ib, []).append(ia)

    out: List[Story] = []
    for i in sorted(emit):
        p, k, sc = passages[i], kins[i], classes[i]
        n = len(p.crops)
        clipped = n - len(usable[i])
        iid, label, withheld = name_for(p, labels, cfg)
        still = sc in STILL_CLASSES if sc else False
        zp = zone_path(usable[i], zones, cfg, still=still) if usable[i] else []
        lat, dep = direction(k, cfg) if (k and not still) else (0, 0)
        pidx = sorted(partners.get(i, []), key=lambda j: sids[j])
        comp = [passages[j].kind for j in pidx]
        people = sum(1 for c in comp if c == "person") if p.kind == "person" else 0
        companions = [c for c in comp if not (p.kind == "person" and c == "person")]

        if props[i]:
            reason: Optional[str] = "prop"
        elif i in too_long:
            reason = "too_long"
        elif n < cfg.min_detections:
            reason = "glimpse"
        elif k is None or len(usable[i]) < cfg.min_detections:
            # Travel rests on unclipped boxes only; too few of them is a glimpse of motion.
            reason = "edge_clipped"
        elif sc == "parked":
            reason = "parked"
        elif sc == "standing":
            reason = "standing"
        elif p.kind not in PUBLISH_KINDS or sc not in WALKING_CLASSES:
            # Slice 1 is walker stories only (Juniper, 2026-10-16): at a 5 s
            # detector cadence a vehicle "story" is mostly a parked car's
            # jitter, and fast classes are out of reach anyway.
            reason = "out_of_scope"
        elif p.kind != "person" and "person" in comp:
            # "A person with a dog walked ..." already tells it; one walk-by is
            # one story, so Juniper never checks the same event twice.
            reason = "companion"
        else:
            reason = None
        sentence = sentence_for(kind=p.kind, sc=sc, lat=lat, dep=dep, zp=zp,
                                dur=(p.last.t - p.first.t).total_seconds(), n=n, name=label,
                                companions=companions, people_together=people)
        evidence = []
        for c in p.crops:
            clip = crop_clipped(c, cfg)
            z = None if clip else zone_for_box(zones, c.box, *frame_of(c, cfg))
            evidence.append({
                "crop_id": c.crop_id, "t": c.t.isoformat(), "label": c.label, "score": round(float(c.score), 3),
                "box": [round(float(v), 1) for v in c.box], "zone": z.name if z else None,
                "clipped": clip, "individual_id": c.individual_id,
                "thumb": _thumb_hex(c.thumb_ref),
            })
        fw, fh, fsrc = passage_frame(p, cfg)
        fx, fy = exit_foot(p.last, cfg)
        xs = exit_speed(usable[i], cfg)
        out.append(Story(
            story_id=sids[i], stream_id=stream_id, kind=p.kind, started_at=p.first.t, ended_at=p.last.t,
            detection_count=n, used_detections=len(usable[i]), edge_clipped_count=clipped,
            speed_class=sc, speed_bh_per_sec=round(k.speed, 4) if k else None,
            net_travel_bh=round(k.net_bh, 3) if k else None, path_bh=round(k.path_bh, 3) if k else None,
            dir_lateral=lat, dir_depth=dep, zone_path=list(zp), individual_id=iid, name_label=label,
            name_withheld=withheld, companion_kinds=comp, together_story_ids=[sids[j] for j in pidx],
            published=reason is None, drop_reason=reason,
            clean=bool(clean_since is None or p.first.t >= clean_since),
            sentence=sentence, crop_ids=[c.crop_id for c in p.crops], evidence=evidence,
            entry_edge=edge_of(p.first, cfg), exit_edge=edge_of(p.last, cfg),
            exit_foot_x=round(fx, 4), exit_foot_y=round(fy, 4),
            exit_speed_fw_per_sec=round(xs, 5) if xs is not None else None,
            frame_w=int(round(fw)), frame_h=int(round(fh)), frame_size_source=fsrc,
        ))
    return out
