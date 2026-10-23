"""Caption motion cross-check (2026-10-23, spec PR #2763, item 3 patch C).

A camera caption is a small VLM's description of ONE frame
(`services/orion-vision-window/app/caption_events.py`). It often claims motion
("A car is driving down a residential street") that one frame cannot show.
The motion-story reducer (`app/vision_motion.py`) measures motion from the
detector's boxes over time: each story has a ``speed_class`` such as
``parked`` or ``driving``. The two come from different models (VLM vs detector
track), so one can check the other.

This module is the pure half. For one ``camera_caption`` row it:

1. finds the motion claims in the caption text (`motion_claims`). The claim
   words are DERIVED from the motion vocabulary the stories already use
   (`vision_motion.SPEED_CLASSES` moving classes: walking, running, riding,
   creeping, driving, pushed). No second word list. Words outside it
   ("passing", "mowing") are not claims here and stay unchecked;
2. matches the caption's own boxes (its ``box:`` and ``crop:`` evidence refs,
   the boxes the VLM read) to the detector crops of the same frame by IoU --
   never every crop in the frame;
3. reads the speed_class of the story that holds each matched crop, and
   decides per claim: a moving story of the claimed kind -> supported; only
   still stories -> contradicted; no story -> undecided.

The clocked half (`caption_motion_step` in `app/vision_motion_loop.py`) runs
after stories are written, because a story closes after the caption is saved.

Tags written (one shared definition in `orion.situational.street_material`):
``motion_contradicted`` or ``motion_supported`` plus one
``motion_check:<kind>:<claim word>:<speed_class>`` per decided claim. A row
that cannot be decided gets nothing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.vision_motion import KINDS, SPEED_CLASSES, STILL_CLASSES
from orion.situational.street_material import (
    CAPTION_MOTION_CHECK_TAG_PREFIX,
    CAPTION_MOTION_CONTRADICTED_TAG,
    CAPTION_MOTION_SUPPORTED_TAG,
)
from orion.vision.stationary import box_iou

logger = logging.getLogger("sql-writer.vision_caption_motion")

Box = Tuple[float, float, float, float]

DEFAULT_MIN_IOU = 0.5


# ---------------------------------------------------------------------------
# Motion claims in caption text, derived from SPEED_CLASSES
# ---------------------------------------------------------------------------


def _stem(word: str) -> str:
    """'driving' -> 'driv', 'running' -> 'run', 'pushed' -> 'push'."""
    if word.endswith("ing"):
        s = word[:-3]
        if len(s) >= 3 and s[-1] == s[-2]:
            s = s[:-1]  # runn -> run
        return s
    if word.endswith("ed"):
        return word[:-2]
    return word


def _claim_vocabulary() -> Dict[str, Tuple[str, Tuple[str, ...]]]:
    """stem -> (motion class word, kinds it is a claim about).

    Every moving class of SPEED_CLASSES (the mid classes and the fast class;
    never the still class). An "-ing" class owns its stem; an "-ed" class
    (bicycle "walked", stroller "pushed") only adds a stem no "-ing" class
    owns, so "walking" is a claim about people and dogs, not about a parked
    bicycle in the same frame."""
    ing: Dict[str, Tuple[str, List[str]]] = {}
    ed: Dict[str, Tuple[str, List[str]]] = {}
    for kind, (_still, mids, fast) in SPEED_CLASSES.items():
        for word in [w for _, w in mids] + ([fast] if fast else []):
            if word in STILL_CLASSES:
                continue
            bucket = ing if word.endswith("ing") else ed
            entry = bucket.setdefault(_stem(word), (word, []))
            if kind not in entry[1]:
                entry[1].append(kind)
    out = {s: (w, tuple(k)) for s, (w, k) in ing.items()}
    for s, (w, k) in ed.items():
        out.setdefault(s, (w, tuple(k)))
    return out


CLAIM_VOCABULARY = _claim_vocabulary()
# "driv" matches drive/drives/driving/drove-less forms, never "driveway" or "driver".
_CLAIM_RES = {
    stem: re.compile(rf"\b(?:{re.escape(word)}|{re.escape(stem)}(?:e|es|s|ed|ing)?)\b", re.IGNORECASE)
    for stem, (word, _kinds) in CLAIM_VOCABULARY.items()
}


@dataclass(frozen=True)
class MotionClaim:
    word: str  # as written in the caption, lower case ("driving")
    kinds: Tuple[str, ...]  # story kinds the claim is about ("vehicle",)


def motion_claims(text: str) -> List[MotionClaim]:
    """Motion claims in a caption, in text order, one per stem."""
    found: List[Tuple[int, MotionClaim]] = []
    for stem, rx in _CLAIM_RES.items():
        m = rx.search(text or "")
        if m:
            found.append((m.start(), MotionClaim(word=m.group(0).lower(), kinds=CLAIM_VOCABULARY[stem][1])))
    return [c for _, c in sorted(found, key=lambda t: t[0])]


# ---------------------------------------------------------------------------
# Evidence refs (caption_events.build_caption_event)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptionBox:
    label: Optional[str]  # box: refs carry the gate label; crop: refs carry none
    box: Box


def _parse_box(raw: str) -> Optional[Box]:
    try:
        vals = tuple(float(v) for v in raw.split(","))
    except ValueError:
        return None
    return vals if len(vals) == 4 else None  # type: ignore[return-value]


def parse_caption_refs(refs: Iterable[str]) -> Tuple[Optional[datetime], List[CaptionBox]]:
    """(frame time, the caption's own boxes) from its evidence refs."""
    frame_ts: Optional[datetime] = None
    boxes: List[CaptionBox] = []
    for ref in refs or ():
        ref = str(ref)
        if ref.startswith("frame_ts:") and frame_ts is None:
            try:
                frame_ts = datetime.fromisoformat(ref[len("frame_ts:"):].replace("Z", "+00:00"))
            except ValueError:
                pass
        elif ref.startswith("box:"):
            label, _, coords = ref[len("box:"):].rpartition(":")
            b = _parse_box(coords)
            if b:
                boxes.append(CaptionBox(label=label.strip() or None, box=b))
        elif ref.startswith("crop:"):
            b = _parse_box(ref[len("crop:"):])
            if b:
                boxes.append(CaptionBox(label=None, box=b))
    return frame_ts, boxes


# ---------------------------------------------------------------------------
# Match and decide
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameCrop:
    crop_id: str
    label: str
    box: Box


def match_caption_crops(boxes: Sequence[CaptionBox], crops: Sequence[FrameCrop], *,
                        min_iou: float = DEFAULT_MIN_IOU) -> List[FrameCrop]:
    """The frame's crops that ARE the caption's boxes: per caption box, the
    best-IoU crop at or above ``min_iou``. A box whose label is a story kind
    only matches a crop of that kind; other labels ("mail truck", from the
    gate) and unlabelled crop refs match by IoU alone."""
    out: Dict[str, FrameCrop] = {}
    for cb in boxes:
        best: Optional[Tuple[float, FrameCrop]] = None
        for c in crops:
            if cb.label in KINDS and c.label != cb.label:
                continue
            iou = box_iou(cb.box, c.box)
            if iou >= min_iou and (best is None or iou > best[0]):
                best = (iou, c)
        if best is not None:
            out.setdefault(best[1].crop_id, best[1])
    return list(out.values())


@dataclass(frozen=True)
class ClaimCheck:
    claim: MotionClaim
    verdict: str  # "supported" | "contradicted" | "undecided" | "pending"
    kind: Optional[str] = None
    speed_class: Optional[str] = None


def check_claim(claim: MotionClaim, matched: Sequence[FrameCrop], story_class: Mapping[str, Optional[str]],
                *, settled: bool) -> ClaimCheck:
    """``story_class``: crop_id -> usable speed_class of the story that holds
    it (None = a story with no usable speed: a glimpse, a too-long or
    edge-clipped passage; see `usable_speed`). A crop absent from it has no
    story. ``settled``: no story can still close for this frame, so a
    storyless crop will stay storyless.

    Supported: any matched crop of the claimed kind is in a moving story.
    Contradicted: matched crops are in still stories and NONE of them is in a
    story with no usable speed -- a glimpse is how a passing car usually
    looks, so it may be the very car the caption saw driving. Storyless crops
    carry no evidence either way."""
    relevant = [c for c in matched if c.label in claim.kinds]
    if not relevant:
        return ClaimCheck(claim, "undecided")
    if not settled and any(c.crop_id not in story_class for c in relevant):
        return ClaimCheck(claim, "pending")
    storied = [(c.label, story_class[c.crop_id]) for c in relevant if c.crop_id in story_class]
    moving = [(k, s) for k, s in storied if s and s not in STILL_CLASSES]
    if moving:
        return ClaimCheck(claim, "supported", kind=moving[0][0], speed_class=moving[0][1])
    if any(s is None for _, s in storied):
        return ClaimCheck(claim, "undecided")
    if storied:
        return ClaimCheck(claim, "contradicted", kind=storied[0][0], speed_class=storied[0][1])
    return ClaimCheck(claim, "undecided")


@dataclass(frozen=True)
class CaptionVerdict:
    status: str  # "contradicted" | "supported" | "undecided" | "pending" | "no_claim"
    tags: Tuple[str, ...] = ()
    checks: Tuple[ClaimCheck, ...] = ()


def decide_caption(text: str, matched: Sequence[FrameCrop], story_class: Mapping[str, Optional[str]], *,
                   settled: bool) -> CaptionVerdict:
    """One caption's verdict and the tags to add. Any contradicted claim makes
    the row contradicted (one false motion claim is enough to distrust it).
    A pending claim holds the whole row back until it settles."""
    claims = motion_claims(text)
    if not claims:
        return CaptionVerdict("no_claim")
    checks = tuple(check_claim(c, matched, story_class, settled=settled) for c in claims)
    if any(ch.verdict == "pending" for ch in checks):
        return CaptionVerdict("pending", checks=checks)
    # Contradicted claims first: readers render the first detail tag.
    decided = sorted((ch for ch in checks if ch.verdict in ("supported", "contradicted")),
                     key=lambda ch: ch.verdict != "contradicted")
    if not decided:
        return CaptionVerdict("undecided", checks=checks)
    status = "contradicted" if any(ch.verdict == "contradicted" for ch in decided) else "supported"
    head = CAPTION_MOTION_CONTRADICTED_TAG if status == "contradicted" else CAPTION_MOTION_SUPPORTED_TAG
    detail = tuple(f"{CAPTION_MOTION_CHECK_TAG_PREFIX}{ch.kind}:{ch.claim.word}:{ch.speed_class}" for ch in decided)
    return CaptionVerdict(status, tags=(head, *detail), checks=checks)


# ---------------------------------------------------------------------------
# Reads and the clocked step (run by vision_motion_loop after stories close)
# ---------------------------------------------------------------------------

# A caption frame's crops are written with observed_at = the frame time; the
# window allows rounding in the ISO ref.
FRAME_MATCH_SEC = 1.0
CAPTION_ROWS_PER_TICK = 500
VERDICT_TAGS = (CAPTION_MOTION_CONTRADICTED_TAG, CAPTION_MOTION_SUPPORTED_TAG)

CAPTION_CANDIDATES_SQL = (
    "SELECT event_id, narrative, evidence_refs FROM vision_events "
    "WHERE stream_id = :s AND event_type = 'camera_caption' AND created_at >= :since AND created_at <= :until "
    "AND jsonb_typeof(evidence_refs) = 'array' "
    "AND NOT (COALESCE(tags, '[]'::jsonb) ?| CAST(:verdicts AS text[])) "
    "AND event_id <> ALL(CAST(:skip AS text[])) "
    "ORDER BY created_at DESC LIMIT :n")
FRAME_CROPS_SQL = (
    "SELECT crop_id, label, box_xyxy FROM vision_crop_observation "
    "WHERE stream_id = :s AND observed_at BETWEEN :a AND :b")
FRAME_STORIES_SQL = (
    "SELECT crop_ids, speed_class, drop_reason FROM vision_motion_story "
    "WHERE stream_id = :s AND ended_at >= :a AND started_at <= :b ORDER BY ended_at, story_id")
# Stories whose speed class says little about the moment of the frame: a
# glimpse has too few boxes, a too-long passage is cut at the window start,
# an edge-clipped one has its boxes cut by the frame edge.
UNUSABLE_DROP_REASONS = frozenset({"glimpse", "too_long", "edge_clipped"})


def usable_speed(speed_class: Optional[str], drop_reason: Optional[str]) -> Optional[str]:
    """The story's speed class when it can vouch for the frame, else None."""
    if not speed_class or (drop_reason or "") in UNUSABLE_DROP_REASONS:
        return None
    return speed_class
# Append only, and only to a row that has no verdict yet (a re-run is a no-op).
ADD_TAGS_SQL = (
    "UPDATE vision_events SET tags = COALESCE(tags, '[]'::jsonb) || CAST(:add AS jsonb) "
    "WHERE event_id = :id AND NOT (COALESCE(tags, '[]'::jsonb) ?| CAST(:verdicts AS text[]))")


def _refs(raw: Any) -> List[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return [str(r) for r in raw] if isinstance(raw, list) else []


def read_frame(conn, stream_id: str, frame_ts: datetime) -> Tuple[List[FrameCrop], Dict[str, Optional[str]]]:
    """(crops of the caption's frame, crop_id -> usable speed class of its story)."""
    from sqlalchemy import text

    a, b = frame_ts - timedelta(seconds=FRAME_MATCH_SEC), frame_ts + timedelta(seconds=FRAME_MATCH_SEC)
    crops = [FrameCrop(crop_id=r[0], label=r[1], box=tuple(float(v) for v in r[2]))  # type: ignore[arg-type]
             for r in conn.execute(text(FRAME_CROPS_SQL), {"s": stream_id, "a": a, "b": b}).fetchall()
             if r[2] and len(r[2]) == 4]
    story_class: Dict[str, Optional[str]] = {}
    if crops:
        ids = {c.crop_id for c in crops}
        for crop_ids, speed, drop in conn.execute(text(FRAME_STORIES_SQL), {"s": stream_id, "a": a, "b": b}).fetchall():
            for cid in crop_ids or ():
                if cid in ids:
                    story_class[cid] = usable_speed(speed, drop)
    return crops, story_class


def check_caption(conn, stream_id: str, narrative: str, refs: Sequence[str], *, cursor: Optional[datetime],
                  settle_sec: float, min_iou: float) -> CaptionVerdict:
    """One caption row's verdict from live reads (read only). ``cursor`` is
    the stream's motion-story cursor: a story holding a crop at the frame can
    only exist once the cursor has passed the frame, and once it is
    ``settle_sec`` past it no more stories can close for that frame."""
    if not motion_claims(narrative):
        return CaptionVerdict("no_claim")
    frame_ts, boxes = parse_caption_refs(refs)
    if frame_ts is None or not boxes:
        return CaptionVerdict("undecided")
    if cursor is None or cursor < frame_ts:
        return CaptionVerdict("pending")
    crops, story_class = read_frame(conn, stream_id, frame_ts)
    matched = match_caption_crops(boxes, crops, min_iou=min_iou)
    settled = (cursor - frame_ts).total_seconds() >= settle_sec
    return decide_caption(narrative, matched, story_class, settled=settled)


# Per process: rows already final with no tag (no claim, or undecidable).
# Losing it on restart only costs a re-read inside the lookback.
_final_rows: Dict[str, datetime] = {}


def caption_motion_step(engine, stream_id: str, ts: datetime, *, lookback_sec: float, settle_sec: float,
                        min_iou: float) -> Dict[str, int]:
    """Tag this stream's camera captions from the last ``lookback_sec`` whose
    motion claim can now be decided. Older rows are never touched."""
    from sqlalchemy import text

    counts = {"checked": 0, "tagged": 0, "contradicted": 0, "supported": 0, "undecided": 0, "pending": 0,
              "no_claim": 0}
    since = ts - timedelta(seconds=lookback_sec)
    for eid in [k for k, v in _final_rows.items() if v < since]:
        _final_rows.pop(eid, None)
    verdicts: List[Tuple[str, Tuple[str, ...]]] = []
    with engine.connect() as conn:
        cursor = conn.execute(text(
            "SELECT processed_until FROM vision_motion_story_cursor WHERE stream_id = :s"),
            {"s": stream_id}).scalar()
        if cursor is None:
            return counts  # no story was ever built for this stream: nothing can be decided
        # Final rows are skipped in SQL, before the LIMIT, and the newest rows
        # come first, so old undecidable rows can never crowd out new ones.
        rows = conn.execute(text(CAPTION_CANDIDATES_SQL), {
            "s": stream_id, "since": since, "until": ts, "verdicts": list(VERDICT_TAGS),
            "skip": sorted(_final_rows), "n": CAPTION_ROWS_PER_TICK}).fetchall()
        if len(rows) >= CAPTION_ROWS_PER_TICK:
            logger.warning("[CAPTION_MOTION] stream=%s read the per-tick cap of %d rows; older rows wait",
                           stream_id, CAPTION_ROWS_PER_TICK)
        for event_id, narrative, raw_refs in rows:
            v = check_caption(conn, stream_id, str(narrative or ""), _refs(raw_refs), cursor=cursor,
                              settle_sec=settle_sec, min_iou=min_iou)
            counts["checked"] += 1
            counts[v.status] = counts.get(v.status, 0) + 1
            if v.status in ("no_claim", "undecided"):
                _final_rows[event_id] = ts
            elif v.tags:
                verdicts.append((event_id, v.tags))
    if verdicts:
        with engine.begin() as conn:
            for event_id, tags in verdicts:
                counts["tagged"] += conn.execute(text(ADD_TAGS_SQL), {
                    "id": event_id, "add": json.dumps(list(tags)), "verdicts": list(VERDICT_TAGS)}).rowcount or 0
    return counts
