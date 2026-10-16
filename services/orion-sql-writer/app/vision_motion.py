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
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from orion.vision.stationary import box_iou
from orion.vision.zones import Zone, zone_for_box

# Bump when a rule below changes meaning: Juniper's spot-check tally counts one
# version at a time, so old verdicts never vouch for new rules.
RULES_VERSION = 1

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
# Below this net speed (BH/s) a passage did not really travel, even when its
# net travel passes still_bh over a long dwell.
STILL_SPEED_BH = 0.08

VERBS = {
    "standing": "stood", "parked": "stayed parked", "walking": "walked", "running": "ran",
    "walked": "was walked", "riding": "was ridden", "pushed": "was pushed",
    "creeping": "crept", "driving": "drove",
}
NOUNS = {"person": "a person", "dog": "a dog", "bicycle": "a bicycle", "stroller": "a stroller",
         "vehicle": "a vehicle"}


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


@dataclass(frozen=True)
class MotionCrop:
    crop_id: str
    t: datetime
    label: str
    box: Tuple[float, float, float, float]
    score: float = 0.0
    individual_id: Optional[str] = None
    thumb_ref: Optional[str] = None

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


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def is_edge_clipped(box: Sequence[float], cfg: MotionConfig) -> bool:
    """Touches the left, right or bottom edge: the foot point is the frame
    edge, not a foot (sample S10: y2 = 1914-1916 of 1920). The top edge does
    not move the foot point."""
    x1, _y1, x2, y2 = (float(v) for v in box)
    m = cfg.edge_margin_px
    return x1 <= m or x2 >= cfg.frame_w - m or y2 >= cfg.frame_h - m


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
    return [c for c in p.crops if not is_edge_clipped(c.box, cfg)]


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
    return Kinematics(used=len(crops), dur=dur, hmed=hmed, net_bh=net_bh, path_bh=path_bh, speed=speed,
                      dx=(x1 - x0) / cfg.frame_w, dy=(y1 - y0) / cfg.frame_h,
                      dlogh=math.log(crops[-1].h / crops[0].h))


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
        z = zone_for_box(zones, c.box, cfg.frame_w, cfg.frame_h)
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
    """(individual_id, label, withheld_reason). The strict rule: one named
    individual on >= name_share of the crops, no other named individual, and
    no crop touching the frame edge. Unnamed crops do not block a name."""
    named = [c.individual_id for c in p.crops if c.individual_id and labels.get(c.individual_id)]
    if not named:
        return None, None, None
    distinct = sorted(set(named))
    if len(distinct) > 1:
        return None, None, "two_named"
    if any(is_edge_clipped(c.box, cfg) for c in p.crops):
        return None, None, "edge_clipped"
    iid = distinct[0]
    if len(named) / len(p.crops) < cfg.name_share:
        return None, None, "low_share"
    return iid, labels[iid], None


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
        else:
            reason = None
        sentence = sentence_for(kind=p.kind, sc=sc, lat=lat, dep=dep, zp=zp,
                                dur=(p.last.t - p.first.t).total_seconds(), n=n, name=label,
                                companions=companions, people_together=people)
        evidence = []
        for c in p.crops:
            z = None if is_edge_clipped(c.box, cfg) else zone_for_box(zones, c.box, cfg.frame_w, cfg.frame_h)
            evidence.append({
                "crop_id": c.crop_id, "t": c.t.isoformat(), "label": c.label, "score": round(float(c.score), 3),
                "box": [round(float(v), 1) for v in c.box], "zone": z.name if z else None,
                "clipped": is_edge_clipped(c.box, cfg), "individual_id": c.individual_id,
                "thumb": _thumb_hex(c.thumb_ref),
            })
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
        ))
    return out
