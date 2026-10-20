"""Individuals: "the same one again", from crop embeddings.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md ideas
1, 3 (ask opener + answer consumer), 6 (attention score), 9 (patio presence).

Each tick, per stream, this reads the new ``vision_crop_observation`` rows
since ``vision_individuals_cursor`` and:

1. **Clusters.** A crop with a face (walkway person crops, 2026-10-15) first
   tries the nearest same-kind individual by FACE centroid; at or above
   ``face_match_threshold`` that wins. Otherwise (no face on either side, or
   no confident face match) the crop joins the nearest same-kind
   ``vision_individual`` appearance centroid on its stream if cosine
   similarity is at or above the threshold, else opens a new individual.
   Centroids are running means of unit vectors, renormalized. Two crops from
   the same frame never join the same individual (two people side by side
   are two people).
2. **Groups sightings.** Consecutive observations of one individual no more
   than ``merge_gap_sec`` apart are one ``vision_individual_sighting``. Zone
   = the zone with the most observations; dwell = ended - started.
3. **Scores attention** on every touched sighting (app/vision_attention_score.py)
   and writes at most one ``vision_events`` row ``attention_worthy`` per sighting.
4. **Zone head counts (the patio).** Every zone with ``presence: true`` (or
   ``embed: false``) in ``config/vision_zones.yaml`` gets one
   ``substrate_embodied_presence`` row, ``<stream>:<zone name>`` (count +
   state), written only while the camera is producing census rows -- a dead
   camera leaves the row to go stale rather than writing a false "absent".
   Boxes in a no-embed zone (``zone_no_embed``; none configured since
   2026-10-15) still never become individuals.
5. **Stationary fixtures.** Each individual carries a run of "its box has not
   moved" (``orion.vision.stationary``): IoU with the run's first box at or
   above ``stationary_iou``, gaps no longer than ``stationary_max_gap_sec``.
   Held for ``stationary_min_sec`` the individual (and its current sighting)
   is ``stationary`` -- a prop such as the patio skeleton seen live
   2026-10-15. Stationary sightings are still recorded but get no attention
   score or event, do not count toward asks or rhythm subjects, and are left
   out of zone head counts. The first box that moves ends it.

The cursor is LANDING time (``vision_crop_observation.created_at``), not
``observed_at``: a crop that arrives late (bus backlog, a slow host) still
lands after the cursor and is processed; within a batch crops are still
clustered in ``observed_at`` order.

Then, across streams: apply Juniper's answers (label individuals), expire
old asks, open new asks under a DB-counted daily cap, prune retention.

Pure functions first (explicit ``now``, no clock, no DB), then the blocking
SQLAlchemy cycle, which callers MUST run in a worker thread.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import uuid
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from orion.vision.stationary import StillRun, advance, box_iou, is_stationary
from orion.vision.zones import Zone

logger = logging.getLogger("sql-writer.vision_individuals")



def no_embed_presence_id(stream_id: str, zone_name: str) -> str:
    """``<stream>:<zone>`` -- e.g. ``walkway:patio``. Derived from the zone's
    own name in config/vision_zones.yaml, so a second no-embed zone gets its
    own row rather than being folded into (or missed by) a hardcoded one."""
    return f"{stream_id}:{zone_name}"


class MigrationMissing(RuntimeError):
    """The walkway tables do not exist yet. The loop backs off instead of crash-looping."""


def is_missing_relation(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "undefinedtable" in type(getattr(exc, "orig", exc)).__name__.lower() or (
        "does not exist" in msg and ("relation" in msg or "column" in msg)
    )


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------


def normalize(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n <= 0:
        return [0.0 for _ in v]
    return [float(x) / n for x in v]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return -1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def update_centroid(centroid: Sequence[float], n: int, embedding: Sequence[float]) -> List[float]:
    """Running mean of unit vectors, renormalized to unit length."""
    e = normalize(embedding)
    c = normalize(centroid)
    return normalize([ci * n + ei for ci, ei in zip(c, e)])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropRow:
    crop_id: str
    observation_id: str
    stream_id: str
    observed_at: datetime
    label: str
    box_xyxy: Tuple[float, ...]
    zone: Optional[str]
    embedding: Optional[Tuple[float, ...]]
    embedding_ref: Optional[str]
    artifact_id: Optional[str]
    # Set by the writer (app/vision_crop_persist.py) from the zones config.
    # The DB CHECK forbids any embedding or thumbnail on such a row.
    zone_no_embed: bool = False
    thumb_ref: Optional[str] = None
    # Walkway person crops with a confident face (orion-vision-host want_crop_faces).
    face_embedding: Optional[Tuple[float, ...]] = None
    face_thumb_ref: Optional[str] = None


@dataclass
class Individual:
    individual_id: str
    stream_id: str
    kind: str
    centroid: List[float]
    centroid_n: int
    first_seen_at: datetime
    last_seen_at: datetime
    sighting_count: int = 0
    distinct_days: int = 0
    label: Optional[str] = None
    # Running mean of this individual's face vectors (None until a face is seen).
    face_centroid: Optional[List[float]] = None
    face_n: int = 0
    # The "box has not moved" run (orion.vision.stationary.StillRun).
    still_box: Optional[List[float]] = None
    still_since: Optional[datetime] = None
    still_last: Optional[datetime] = None
    stationary: bool = False


@dataclass
class Sighting:
    sighting_id: str
    individual_id: str
    stream_id: str
    started_at: datetime
    ended_at: datetime
    observation_count: int = 1
    zone_counts: Dict[str, int] = field(default_factory=dict)
    last_box_xyxy: Optional[List[float]] = None
    embedding_ref: Optional[str] = None
    evidence_ref: Optional[str] = None
    # Thumbnail of the most recent crop in this sighting that has one
    # (orion-vision-host writes one per EMBEDDED crop only). The ask card shows it.
    thumb_ref: Optional[str] = None
    # Most recent face thumbnail; the ask card prefers it over thumb_ref.
    face_thumb_ref: Optional[str] = None
    # The individual was a stationary fixture as of this sighting's newest crop.
    stationary: bool = False

    @property
    def zone(self) -> Optional[str]:
        return dominant_zone(self.zone_counts)

    @property
    def dwell_sec(self) -> float:
        return max(0.0, (self.ended_at - self.started_at).total_seconds())


_NO_ZONE = ""


def dominant_zone(zone_counts: Dict[str, int]) -> Optional[str]:
    if not zone_counts:
        return None
    best = max(zone_counts.items(), key=lambda kv: (kv[1], kv[0] != _NO_ZONE))[0]
    return best or None


def local_date(ts: datetime, tz: ZoneInfo) -> date:
    return ts.astimezone(tz).date()


class _CentroidIndex:
    """Per-(stream, kind) matrix of unit centroids, so a match is one matmul,
    not a Python loop over every individual. Rows are updated in place as
    centroids move; new individuals append a row. ``vec_of`` picks which
    centroid (appearance or face); individuals without one are not indexed."""

    def __init__(
        self,
        individuals: Iterable[Individual],
        vec_of: Callable[[Individual], Optional[Sequence[float]]] = lambda ind: ind.centroid,
    ) -> None:
        self._vec_of = vec_of
        self._ids: Dict[Tuple[str, str], List[str]] = {}
        self._rows: Dict[Tuple[str, str], List[np.ndarray]] = {}
        self._mat: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
        self._pos: Dict[Tuple[str, str], Dict[str, int]] = {}
        for ind in individuals:
            self.add(ind)

    @staticmethod
    def _unit(v: Sequence[float]) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(a))
        return a / n if n > 0 else a

    def add(self, ind: Individual) -> None:
        vec = self._vec_of(ind)
        if not vec:
            return
        key = (ind.stream_id, ind.kind)
        ids = self._ids.setdefault(key, [])
        self._pos.setdefault(key, {})[ind.individual_id] = len(ids)
        ids.append(ind.individual_id)
        self._rows.setdefault(key, []).append(self._unit(vec))
        self._mat[key] = None

    def update(self, ind: Individual) -> None:
        key = (ind.stream_id, ind.kind)
        ids = self._ids.get(key) or []
        if ind.individual_id not in ids:
            self.add(ind)
            return
        i = ids.index(ind.individual_id)
        self._rows[key][i] = self._unit(self._vec_of(ind) or ())
        m = self._mat.get(key)
        if m is not None and m.shape[1] == self._rows[key][i].shape[0]:
            m[i] = self._rows[key][i]
        else:
            self._mat[key] = None

    def nearest(self, crop: "CropRow", exclude: set, vec: Optional[Sequence[float]] = None) -> Tuple[Optional[str], float]:
        key = (crop.stream_id, crop.label)
        ids = self._ids.get(key)
        if not ids:
            return None, -2.0
        dims = {r.shape[0] for r in self._rows[key]}
        e = self._unit(vec if vec is not None else (crop.embedding or ()))
        if dims != {e.shape[0]}:
            # Mixed embedding sizes (a model change): compare only same-size rows.
            best = (None, -2.0)
            for iid, row in zip(ids, self._rows[key]):
                if iid in exclude or row.shape[0] != e.shape[0]:
                    continue
                sim = float(row @ e)
                if sim > best[1]:
                    best = (iid, sim)
            return best
        m = self._mat.get(key)
        if m is None:
            m = np.vstack(self._rows[key])
            self._mat[key] = m
        sims = m @ e
        order = np.argsort(-sims)
        for j in order:
            if ids[int(j)] not in exclude:
                return ids[int(j)], float(sims[int(j)])
        return None, -2.0

    def nearest_among(self, crop: "CropRow", allowed: Iterable[str], exclude: set) -> Tuple[Optional[str], float]:
        """``nearest`` restricted to ``allowed`` ids (the continuation lane's
        recent, close-by individuals). Same float32 unit rows, same scores."""
        key = (crop.stream_id, crop.label)
        ids = self._ids.get(key)
        if not ids:
            return None, -2.0
        e = self._unit(crop.embedding or ())
        pos = self._pos.get(key, {})
        best: Tuple[Optional[str], float] = (None, -2.0)
        for iid in sorted(set(allowed)):
            i = pos.get(iid)
            if i is None or iid in exclude:
                continue
            row = self._rows[key][i]
            if row.shape[0] != e.shape[0]:
                continue
            sim = float(row @ e)
            if sim > best[1]:
                best = (iid, sim)
        return best


def _nearest_sighting(cands: Sequence[Sighting], t: datetime, gap_sec: float) -> Optional[Sighting]:
    """The candidate sighting ``t`` falls inside (+- gap), nearest first."""
    best: Optional[Tuple[float, Sighting]] = None
    for sg in cands:
        before = (sg.started_at - t).total_seconds()
        after = (t - sg.ended_at).total_seconds()
        if before > gap_sec or after > gap_sec:
            continue
        dist = max(0.0, before, after)
        if best is None or dist < best[0]:
            best = (dist, sg)
    return best[1] if best else None


@dataclass
class BatchResult:
    individuals: Dict[str, Individual]
    sightings: Dict[str, Sighting]          # sighting_id -> sighting (touched only)
    new_individual_ids: List[str]
    patio_crops: List[CropRow]
    skipped_no_embedding: int
    assigned: int
    assigned_by_face: int = 0
    # crop_id -> was its individual a stationary fixture after this crop.
    crop_stationary: Dict[str, bool] = field(default_factory=dict)
    # individual_id -> still_since, for individuals that BECAME stationary in
    # this batch: their earlier sightings of the same run (split by gaps
    # longer than the merge gap, saved non-stationary) must be re-flagged.
    became_stationary: Dict[str, datetime] = field(default_factory=dict)
    # crop_id -> the individual it was assigned to (written to
    # vision_crop_observation.individual_id: stored membership, so a split
    # does not need a replay).
    crop_individual: Dict[str, str] = field(default_factory=dict)
    # Crops a cannot-link sent somewhere other than their best face match
    # (refused by face AND the body rule did not pick the same individual).
    face_blocked_by_cannot_link: int = 0
    # Crops that joined through the continuation lane (a recent, close-by
    # individual at >= continuation_threshold) and through re-identification
    # (any individual at >= match_threshold). face + continuation + reid +
    # new individuals = assigned.
    assigned_by_continuation: int = 0
    assigned_by_reid: int = 0


def box_centre_distance_in_heights(a: Sequence[float], b: Sequence[float]) -> float:
    """Distance between two boxes' centres, in units of their mean height.
    inf for a malformed box (no continuation can be claimed from it)."""
    if len(a) < 4 or len(b) < 4:
        return math.inf
    h = ((float(a[3]) - float(a[1])) + (float(b[3]) - float(b[1]))) / 2.0
    if h <= 0:
        return math.inf
    dx = (float(a[0]) + float(a[2])) / 2.0 - (float(b[0]) + float(b[2])) / 2.0
    dy = (float(a[1]) + float(a[3])) / 2.0 - (float(b[1]) + float(b[3])) / 2.0
    return math.hypot(dx, dy) / h


def _ts(d: datetime) -> float:
    return d.timestamp()


def _dt(t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=timezone.utc)


def advance_stationary(
    ind: Individual, box: Sequence[float], at: datetime, *, iou: float, min_sec: float, max_gap_sec: float
) -> None:
    """Fold one observation into the individual's "box has not moved" run.

    Older-than-the-run observations are ignored. While the individual is
    already stationary a gap does not break the run (the prop is still in the
    same place after a night the detector missed it); a moved box always does.
    ``min_sec <= 0`` disables the rule (nothing is ever stationary).
    """
    run = None
    if ind.still_box and ind.still_since is not None and ind.still_last is not None:
        run = StillRun(tuple(ind.still_box), _ts(ind.still_since), _ts(ind.still_last))
    nxt = advance(run, box, _ts(at), iou_threshold=iou,
                  max_gap_sec=None if ind.stationary else max_gap_sec)
    ind.still_box = [float(v) for v in nxt.box]
    ind.still_since = _dt(nxt.since)
    ind.still_last = _dt(nxt.last)
    ind.stationary = is_stationary(nxt, min_sec)

def cannot_link_ambiguous(
    face: Sequence[float], best_id: str, best_sim: float, *,
    individuals: Dict[str, "Individual"], cannot_link: Dict[str, Iterable[str]],
    margin: float, taken: Iterable[str] = (),
) -> Optional[str]:
    """The cannot-link rule (split proposal, "Preventing re-merge"): Juniper
    said ``best_id`` and some Y are different people. If the crop's face is
    within ``margin`` of Y's face centroid too, the face cannot tell them apart,
    so the face path must not assign it to ``best_id``. Returns that Y, else
    None. A Y already used in this frame cannot be the crop, so it never blocks."""
    used = set(taken)
    for y in sorted(cannot_link.get(best_id, ()) or ()):
        ind = individuals.get(y)
        if ind is None or y in used or not ind.face_centroid or len(ind.face_centroid) != len(face):
            continue
        if cosine(face, ind.face_centroid) >= best_sim - margin:
            return y
    return None


def apply_batch(
    crops: Sequence[CropRow],
    *,
    individuals: Dict[str, Individual],
    latest_sightings: Dict[str, Sighting],
    no_embed_zones: Iterable[str],
    match_threshold: float,
    merge_gap_sec: float,
    tz: ZoneInfo,
    new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    nearby_sightings: Optional[Dict[str, List[Sighting]]] = None,
    face_match_threshold: Optional[float] = 0.60,
    stationary_min_sec: float = 1800.0,
    stationary_iou: float = 0.5,
    stationary_max_gap_sec: float = 300.0,
    cannot_link: Optional[Dict[str, Iterable[str]]] = None,
    cannot_link_margin: float = 0.10,
    continuation_threshold: Optional[float] = None,
    continuation_gap_sec: float = 10.0,
    continuation_max_disp: float = 1.5,
    prior_observations: Iterable[Tuple[str, datetime, Sequence[float]]] = (),
) -> BatchResult:
    """Cluster crops into individuals and group them into sightings.

    ``individuals`` and ``latest_sightings`` (individual_id -> most recent
    sighting) are the prior state; copies are returned, inputs untouched.
    Crops are processed in ``observed_at`` order. A crop is a no-embed crop if
    its row says so (``zone_no_embed``) OR its zone name is in
    ``no_embed_zones`` -- either is enough.

    A crop joins whichever of the individual's candidate sightings it falls
    inside (+- ``merge_gap_sec``), nearest first. Candidates are the latest
    sighting, any older ones in ``nearby_sightings`` (individual_id -> the
    sightings overlapping this batch's time span, loaded from the DB), and
    every sighting this batch has already opened -- so a burst of late crops
    from one passing becomes one sighting, not one per crop.

    Faces: a crop with ``face_embedding`` joins the nearest same-kind face
    centroid at or above ``face_match_threshold`` (None disables), before
    appearance is consulted. See ``advance_stationary`` for the fixture rule.

    ``cannot_link`` (individual_id -> ids Juniper said are someone else): a
    face match is refused when ``cannot_link_ambiguous`` says the crop's face
    is as close to one of those (within ``cannot_link_margin``); the crop then
    goes through the appearance rule like a crop without a face match.

    Body matching has two lanes (docs/superpowers/specs/2026-10-16-vision-
    body-match-proposal.md). **Continuation** (``continuation_threshold`` not
    None): the candidates are non-stationary individuals with an observation
    within ``continuation_gap_sec`` of the crop (before or after it: crops
    land late) whose box is within ``continuation_max_disp`` box heights
    (centre distance, mean of the two heights) of the crop's box; the best
    of them joins at >= ``continuation_threshold``. Observations come from
    ``prior_observations`` ((individual_id, observed_at, box) of stored
    crops near this batch), each loaded sighting's ``last_box_xyxy`` at
    ``ended_at``, and every crop this batch assigns. The gap is capped at
    ``merge_gap_sec``, the window callers load sightings for. Stationary
    fixtures are left out: their box is always "recent and close", which
    would hand them passers-by at the lower line. **Re-identification**:
    otherwise the best of all individuals joins at >= ``match_threshold``.
    With the continuation lane off this is exactly the single body rule it
    replaces.
    """
    forbidden = set(no_embed_zones)
    inds = {k: replace(v, centroid=list(v.centroid),
                       face_centroid=list(v.face_centroid) if v.face_centroid else None,
                       still_box=list(v.still_box) if v.still_box else None)
            for k, v in individuals.items()}
    cands: Dict[str, List[Sighting]] = {}
    seen_ids: set = set()
    for src in (latest_sightings, *(() if nearby_sightings is None else (nearby_sightings,))):
        for iid, val in src.items():
            for sg in (val if isinstance(val, list) else [val]):
                if sg.sighting_id in seen_ids:
                    continue
                seen_ids.add(sg.sighting_id)
                cands.setdefault(iid, []).append(replace(sg, zone_counts=dict(sg.zone_counts)))
    touched: Dict[str, Sighting] = {}
    new_ids: List[str] = []
    patio: List[CropRow] = []
    skipped = 0
    assigned = 0
    used_in_observation: Dict[str, set] = {}
    index = _CentroidIndex(inds.values())
    face_index = _CentroidIndex(inds.values(), vec_of=lambda ind: ind.face_centroid)
    by_face = 0
    blocked = 0
    by_continuation = 0
    by_reid = 0
    crop_stationary: Dict[str, bool] = {}
    crop_individual: Dict[str, str] = {}
    # Time-sorted (t, individual_id, box) observations: the continuation
    # lane's "who was here, close by, within the gap".
    obs_t: List[float] = []
    obs: List[Tuple[float, str, Tuple[float, ...]]] = []
    cont_gap = min(float(continuation_gap_sec), float(merge_gap_sec))

    def _observe(iid: str, at: datetime, box: Sequence[float]) -> None:
        item = (_ts(at), iid, tuple(float(v) for v in box))
        k = bisect.bisect_right(obs_t, item[0])
        obs_t.insert(k, item[0])
        obs.insert(k, item)

    if continuation_threshold is not None:
        for iid, at, box in prior_observations:
            if iid in inds and box:
                _observe(iid, at, box)
        for iid, sgs in cands.items():
            for sg in sgs:
                if sg.last_box_xyxy and iid in inds:
                    _observe(iid, sg.ended_at, sg.last_box_xyxy)

    for crop in sorted(crops, key=lambda c: (c.observed_at, c.crop_id)):
        if crop.zone_no_embed or crop.zone in forbidden:
            patio.append(crop)       # never an individual, never an embedding
            continue
        if not crop.embedding:
            skipped += 1
            continue
        taken = used_in_observation.setdefault(crop.observation_id, set())
        best_id: Optional[str] = None
        face_refused: Optional[str] = None
        if crop.face_embedding and face_match_threshold is not None:
            fid, fsim = face_index.nearest(crop, taken, vec=crop.face_embedding)
            if fid is not None and fsim >= face_match_threshold:
                if cannot_link and cannot_link_ambiguous(
                        crop.face_embedding, fid, fsim, individuals=inds, cannot_link=cannot_link,
                        margin=cannot_link_margin, taken=taken):
                    face_refused = fid
                else:
                    best_id = fid
                    by_face += 1
        if best_id is None and continuation_threshold is not None:
            t = _ts(crop.observed_at)
            lo, hi = bisect.bisect_left(obs_t, t - cont_gap), bisect.bisect_right(obs_t, t + cont_gap)
            # Each individual's observation nearest in time to the crop.
            closest: Dict[str, Tuple[float, Tuple[float, ...]]] = {}
            for ot, iid, box in obs[lo:hi]:
                dt = abs(ot - t)
                if iid not in closest or dt < closest[iid][0]:
                    closest[iid] = (dt, box)
            near = {iid for iid, (_, box) in closest.items()
                    if iid not in taken and not inds[iid].stationary
                    and box_centre_distance_in_heights(box, crop.box_xyxy) <= continuation_max_disp}
            if near:
                cid, csim = index.nearest_among(crop, near, taken)
                if cid is not None and csim >= continuation_threshold:
                    best_id = cid
                    by_continuation += 1
        if best_id is None:
            aid, asim = index.nearest(crop, taken)
            if aid is not None and asim >= match_threshold:
                best_id = aid
                by_reid += 1
        if face_refused is not None and best_id != face_refused:
            blocked += 1   # the cannot-link changed where this crop went

        if best_id is not None:
            ind = inds[best_id]
            if local_date(crop.observed_at, tz) != local_date(ind.last_seen_at, tz) and crop.observed_at > ind.last_seen_at:
                ind.distinct_days += 1
            ind.centroid = update_centroid(ind.centroid, ind.centroid_n, crop.embedding)
            ind.centroid_n += 1
            index.update(ind)
            if crop.face_embedding:
                if ind.face_centroid:
                    ind.face_centroid = update_centroid(ind.face_centroid, ind.face_n, crop.face_embedding)
                else:
                    ind.face_centroid = normalize(crop.face_embedding)
                ind.face_n += 1
                face_index.update(ind)
            ind.last_seen_at = max(ind.last_seen_at, crop.observed_at)
        else:
            ind = Individual(
                individual_id=new_id(), stream_id=crop.stream_id, kind=crop.label,
                centroid=normalize(crop.embedding), centroid_n=1,
                first_seen_at=crop.observed_at, last_seen_at=crop.observed_at,
                sighting_count=0, distinct_days=1,
                face_centroid=normalize(crop.face_embedding) if crop.face_embedding else None,
                face_n=1 if crop.face_embedding else 0,
            )
            inds[ind.individual_id] = ind
            new_ids.append(ind.individual_id)
            index.add(ind)
            face_index.add(ind)
        taken.add(ind.individual_id)
        crop_individual[crop.crop_id] = ind.individual_id
        if continuation_threshold is not None:
            _observe(ind.individual_id, crop.observed_at, crop.box_xyxy)
        assigned += 1
        advance_stationary(ind, crop.box_xyxy, crop.observed_at, iou=stationary_iou,
                           min_sec=stationary_min_sec, max_gap_sec=stationary_max_gap_sec)
        crop_stationary[crop.crop_id] = ind.stationary

        zone_key = crop.zone or _NO_ZONE
        prev = _nearest_sighting(cands.get(ind.individual_id, []), crop.observed_at, merge_gap_sec)
        if prev is not None:
            newest = crop.observed_at >= prev.ended_at
            prev.started_at = min(prev.started_at, crop.observed_at)
            prev.ended_at = max(prev.ended_at, crop.observed_at)
            prev.observation_count += 1
            prev.zone_counts[zone_key] = prev.zone_counts.get(zone_key, 0) + 1
            if newest:
                prev.last_box_xyxy = list(crop.box_xyxy)
                prev.embedding_ref = crop.embedding_ref
                prev.evidence_ref = f"crop:{crop.crop_id}"
                prev.stationary = ind.stationary
            if crop.thumb_ref and (newest or not prev.thumb_ref):
                prev.thumb_ref = crop.thumb_ref
            if crop.face_thumb_ref and (newest or not prev.face_thumb_ref):
                prev.face_thumb_ref = crop.face_thumb_ref
            s = prev
        else:
            s = Sighting(
                sighting_id=new_id(), individual_id=ind.individual_id, stream_id=crop.stream_id,
                started_at=crop.observed_at, ended_at=crop.observed_at, observation_count=1,
                zone_counts={zone_key: 1}, last_box_xyxy=list(crop.box_xyxy),
                embedding_ref=crop.embedding_ref,
                evidence_ref=f"crop:{crop.crop_id}",
                thumb_ref=crop.thumb_ref,
                face_thumb_ref=crop.face_thumb_ref,
                stationary=ind.stationary,
            )
            cands.setdefault(ind.individual_id, []).append(s)
            ind.sighting_count += 1
        touched[s.sighting_id] = s

    return BatchResult(
        individuals=inds, sightings=touched, new_individual_ids=new_ids,
        patio_crops=patio, skipped_no_embedding=skipped, assigned=assigned,
        assigned_by_face=by_face, crop_stationary=crop_stationary,
        became_stationary=_mark_run_sightings(inds, individuals, cands, touched),
        crop_individual=crop_individual, face_blocked_by_cannot_link=blocked,
        assigned_by_continuation=by_continuation, assigned_by_reid=by_reid,
    )


def _mark_run_sightings(
    inds: Dict[str, Individual],
    before: Dict[str, Individual],
    cands: Dict[str, List[Sighting]],
    touched: Dict[str, Sighting],
) -> Dict[str, datetime]:
    """Individuals that turned stationary in this batch: flag every sighting of
    the same still run (ended at/after ``still_since``) this batch knows about,
    and return them so the DB pass can flag older ones. Without this, the
    first 30 minutes of a prop -- split into many sightings by detector gaps
    longer than the merge gap -- stay ordinary sightings that count toward asks."""
    flipped: Dict[str, datetime] = {}
    for iid, ind in inds.items():
        prior = before.get(iid)
        if not ind.stationary or ind.still_since is None or (prior is not None and prior.stationary):
            continue
        flipped[iid] = ind.still_since
        for sg in cands.get(iid, []):
            if sg.ended_at >= ind.still_since and not sg.stationary:
                sg.stationary = True
                touched[sg.sighting_id] = sg
    return flipped


def head_count_crops(
    crops: Sequence[CropRow],
    zone_name: str,
    *,
    crop_stationary: Dict[str, bool],
    stationary_boxes: Sequence[Tuple[str, Sequence[float]]],
    iou: float,
) -> List[CropRow]:
    """The crops in ``zone_name`` that count as someone being there.

    A crop whose individual is a stationary fixture is left out. A crop with
    no individual (no vector, no-embed zone) is left out when its box sits on
    a stationary individual's box of the same kind -- the prop seen by a
    frame whose crop was not embedded is still the prop."""
    out: List[CropRow] = []
    for c in crops:
        if c.zone != zone_name:
            continue
        if c.crop_id in crop_stationary:
            if crop_stationary[c.crop_id]:
                continue
        elif any(kind == c.label and box_iou(c.box_xyxy, box) >= iou for kind, box in stationary_boxes):
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Patio presence (idea 9): count + state, no pictures, no vectors
# ---------------------------------------------------------------------------


def patio_count(patio_crops: Sequence[CropRow]) -> Tuple[Optional[datetime], int]:
    """(last observed_at, number of patio boxes in that last observation)."""
    if not patio_crops:
        return None, 0
    last = max(c.observed_at for c in patio_crops)
    per_obs: Dict[str, int] = {}
    for c in patio_crops:
        if c.observed_at == last:
            per_obs[c.observation_id] = per_obs.get(c.observation_id, 0) + 1
    return last, max(per_obs.values())


def patio_snapshot(
    *,
    prev: Optional[Dict[str, Any]],
    batch_last_seen_at: Optional[datetime],
    batch_count: int,
    now: datetime,
    present_sec: float,
    grace_sec: float,
    zone: str = "patio",
) -> Dict[str, Any]:
    """Same keys as orion-vision-window's presence snapshot, plus the zone count.

    ``subject`` is ``{"count": n}`` -- how many, never who (names live on
    sightings, not on this row).
    """
    prev = prev or {}
    last_seen_at = batch_last_seen_at
    count = batch_count
    prev_seen = datetime.fromisoformat(prev["last_seen_at"]) if prev.get("last_seen_at") else None
    # Late crops can land in a batch whose newest patio box is OLDER than what
    # the row already says: last-seen never moves backwards, and the count
    # stays the one that goes with the newer sighting.
    if prev_seen is not None and (last_seen_at is None or last_seen_at < prev_seen):
        last_seen_at = prev_seen
        count = int((prev.get("subject") or {}).get("count") or 0)
    last_seen_sec = (now - last_seen_at).total_seconds() if last_seen_at is not None else None
    if last_seen_sec is not None and last_seen_sec <= present_sec:
        state = "present"
    elif last_seen_sec is not None and last_seen_sec <= grace_sec:
        state = "recent"
    else:
        state = "absent"
        count = 0
    state_since = now
    if prev.get("state") == state and prev.get("state_since"):
        state_since = datetime.fromisoformat(prev["state_since"])
    return {
        "state": state,
        "since_sec": round((now - state_since).total_seconds(), 1),
        "last_seen_sec": round(last_seen_sec, 1) if last_seen_sec is not None else None,
        "subject": {"count": count},
        "identity_uncertain": False,
        "identity_confirmed": False,
        "zone": zone,
        # Readers find these rows by a flag, not by the zone's name.
        # head_count_zone is the honest name. no_embed_zone is the name
        # orion/situational/perception_reader.py still filters on: kept True
        # on every head-count row until that reader reads head_count_zone,
        # even though no configured zone is no-embed since 2026-10-15.
        "head_count_zone": True,
        "no_embed_zone": True,
        "state_since": state_since.isoformat(),
        "last_seen_at": last_seen_at.isoformat() if last_seen_at is not None else None,
    }


# ---------------------------------------------------------------------------
# Asks (idea 3): opener rule, cap, wording
# ---------------------------------------------------------------------------


def should_ask(
    *, label: Optional[str], sighting_count: int, distinct_days: int, min_sightings: int, min_days: int
) -> bool:
    return not label and sighting_count >= min_sightings and distinct_days >= min_days


def remaining_ask_budget(*, daily_cap: int, asked_today: int) -> int:
    return max(0, int(daily_cap) - int(asked_today))


def local_day_start(now: datetime, tz: ZoneInfo) -> datetime:
    local = now.astimezone(tz)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def circular_mean_minute(minutes: Sequence[float]) -> Tuple[Optional[int], float]:
    """(mean minute-of-day, resultant length R in [0,1]). R near 0 = no usual time."""
    if not minutes:
        return None, 0.0
    ang = [2 * math.pi * (m % 1440) / 1440.0 for m in minutes]
    s = sum(math.sin(a) for a in ang) / len(ang)
    c = sum(math.cos(a) for a in ang) / len(ang)
    r = math.hypot(s, c)
    mean = (math.atan2(s, c) % (2 * math.pi)) * 1440.0 / (2 * math.pi)
    return int(round(mean)) % 1440, r


# orion-vision-host's crop thumbnail ref: "thumb:<sha256 of the JPEG bytes>".
# The Hub serves it at /api/vision/crop-thumbs/<sha256>.
THUMB_REF_PREFIX = "thumb:"

_PEOPLE_KINDS = {"person", "man", "woman", "child", "kid"}


def ask_question(*, kind: str, sighting_count: int, distinct_days: int, mean_minute: Optional[int], r: float) -> str:
    who = f"this same {kind}"
    if mean_minute is not None and r >= 0.5:
        # Round to 5 minutes: "07:40", not a false-precision "07:43".
        m = int(5 * round(mean_minute / 5.0)) % 1440
        when = f"usually around {m // 60:02d}:{m % 60:02d}"
    else:
        when = "at different times of day"
    tail = "Do you know who this is?" if kind in _PEOPLE_KINDS else f"Do you know whose {kind} this is?"
    return f"I have seen {who} {sighting_count} times over {distinct_days} days, {when}. {tail}"


def ask_image_refs(
    sightings: Sequence[Tuple],
    *,
    now: datetime,
    thumb_retention_days: float,
    ask_expiry_days: float,
    limit: int = 3,
    faces_only: bool = False,
) -> List[str]:
    """The pictures for an ask, best first: FACE thumbnails newest first, then
    crop thumbnails newest first, no repeats, at most ``limit``.

    ``sightings`` are ``(thumb_ref, ended_at, zone_no_embed[, face_thumb_ref])``,
    any order. A thumbnail is a crop of ONE embedded box (or the face inside
    it), never the whole frame. ``thumb_retention_days`` <= 0 means the host
    keeps thumbnails forever (the default since 2026-10-20), so every one is
    offered, however old. Otherwise it must outlive the ask: the host prunes
    thumbnails ``thumb_retention_days`` after they were last written, and the
    ask stays open ``ask_expiry_days``, so only a thumb from a sighting that
    ended within the difference is offered. Empty -> the card shows no picture.
    """
    horizon: Optional[datetime] = None
    if thumb_retention_days > 0:
        if thumb_retention_days <= ask_expiry_days:
            # Any thumbnail could be pruned while the ask is still open.
            logger.warning("ask_image_ref_disabled thumb_retention_days=%s <= ask_expiry_days=%s",
                           thumb_retention_days, ask_expiry_days)
            return []
        horizon = now - timedelta(days=thumb_retention_days - ask_expiry_days)
    faces: List[Tuple[datetime, str]] = []
    crops: List[Tuple[datetime, str]] = []
    for row in sightings:
        thumb_ref, ended_at, no_embed = row[0], row[1], row[2]
        face_ref = row[3] if len(row) > 3 else None
        if no_embed or ended_at is None or (horizon is not None and ended_at < horizon):
            continue
        for bucket, ref in ((faces, face_ref), (crops, None if faces_only else thumb_ref)):
            if ref and str(ref).startswith(THUMB_REF_PREFIX):
                bucket.append((ended_at, str(ref)))
    out: List[str] = []
    for bucket in (faces, crops):
        for _, ref in sorted(bucket, key=lambda tr: tr[0], reverse=True):
            if ref not in out:
                out.append(ref)
    return out[:max(0, int(limit))]


def ask_image_ref(
    sightings: Sequence[Tuple],
    *,
    now: datetime,
    thumb_retention_days: float,
    ask_expiry_days: float,
) -> Optional[str]:
    """The single best picture (``ask_image_refs``' first), for ``image_ref``."""
    refs = ask_image_refs(sightings, now=now, thumb_retention_days=thumb_retention_days,
                          ask_expiry_days=ask_expiry_days, limit=1)
    return refs[0] if refs else None


# ---------------------------------------------------------------------------
# Visual confirmation asks (2026-10-15): "who is this?" from faces, and
# "are these the same person?". Juniper asked for Orion to confirm what it
# sees so it can learn; every answer to a pair ask is a labeled calibration
# point for the face match threshold (vision_individual_pair_verdict).
# ---------------------------------------------------------------------------

SOURCE_KIND_INDIVIDUAL = "vision_individual"
SOURCE_KIND_SAME = "vision_same_individual"
# Juniper's "someone else" / "not sure" about one person on a Hub walkway story
# card (2026-10-18). Carries only picture verdicts (answer_detail.pictures, each
# with its crop_id): never a name for the individual, so it is not a
# "who is this?" answer and never stops the opener from asking one. A NAME typed
# on a story card is a plain SOURCE_KIND_INDIVIDUAL answer with
# answer_detail.story_id. Both are written by the Hub (scripts/motion_story_routes.py).
SOURCE_KIND_STORY_PERSON = "vision_story_person"
# evidence_refs marker on a "who is this?" ask opened by the face rule, so the
# daily caps can tell the two openers apart without another column.
FACE_RULE_REF = "rule:face"
PAIR_SEP = "|"


def should_ask_by_face(*, label: Optional[str], kind: str, face_n: int, stationary: bool, min_faces: int) -> bool:
    """A person whose face Orion has seen clearly ``min_faces`` times is worth
    a "who is this?" the same day: a face is strong evidence that the
    individual is one real person, where the 10-sightings / 5-days rule waits
    for appearance clusters to prove they hold. ``min_faces <= 0`` disables."""
    return (min_faces > 0 and not label and not stationary and kind in _PEOPLE_KINDS
            and int(face_n) >= int(min_faces))


def pair_ref(a: str, b: str) -> str:
    lo, hi = sorted((a, b))
    return f"{lo}{PAIR_SEP}{hi}"


def parse_pair_ref(ref: str) -> Optional[Tuple[str, str]]:
    parts = str(ref or "").split(PAIR_SEP)
    if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] >= parts[1]:
        return None
    return parts[0], parts[1]


@dataclass(frozen=True)
class FaceCandidate:
    individual_id: str
    kind: str
    face_centroid: Tuple[float, ...]
    face_n: int
    label: Optional[str] = None


def same_groups(same_pairs: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    """individual_id -> one representative id, over Juniper's 'same' verdicts
    (transitively: a=b and b=c puts a, b, c in one group)."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in same_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {x: find(x) for x in list(parent)}


def same_person_pairs(
    cands: Sequence[FaceCandidate], *, min_cosine: float, skip_refs: Iterable[str] = (),
    groups: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str, float]]:
    """Pairs worth "are these the same person?", most alike first.

    Every pair of people whose face centroids are at least ``min_cosine``
    alike, whatever the match threshold: a pair above the threshold that the
    reducer still kept apart (it matches crop to centroid, one crop at a time)
    is as worth asking about as one just under it. Skipped: pairs in
    ``skip_refs`` (already asked or answered), pairs where both are already
    named (Juniper said who each one is), different face sizes (a model change).

    ``groups`` (``same_groups``): people Juniper already said are one person
    count as one. A pair inside a group is never asked, and between two
    groups only the most alike pair is -- "is X the same as A?" and "is X the
    same as B?" are one question when A and B are the same person."""
    skip = set(skip_refs)
    grp = groups or {}
    g = lambda iid: grp.get(iid, iid)  # noqa: E731
    seen_groups = set()
    for ref in skip:
        pr = parse_pair_ref(ref)
        if pr:
            seen_groups.add(tuple(sorted((g(pr[0]), g(pr[1])))))
    people = [c for c in cands if c.kind in _PEOPLE_KINDS and c.face_centroid]
    out: List[Tuple[str, str, float]] = []
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            a, b = people[i], people[j]
            if a.individual_id == b.individual_id or (a.label and b.label):
                continue
            if len(a.face_centroid) != len(b.face_centroid):
                continue
            if pair_ref(a.individual_id, b.individual_id) in skip:
                continue
            sim = cosine(a.face_centroid, b.face_centroid)
            if sim >= min_cosine:
                lo, hi = sorted((a.individual_id, b.individual_id))
                out.append((lo, hi, sim))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    kept: List[Tuple[str, str, float]] = []
    for a, b, sim in out:
        key = tuple(sorted((g(a), g(b))))
        if key[0] == key[1] or key in seen_groups:
            continue
        seen_groups.add(key)
        kept.append((a, b, sim))
    return kept


_YES = {"yes", "y", "same", "same person", "yes same person", "yes, same person"}
_NO = {"no", "n", "different", "different people", "not the same", "no different people", "no, different people"}


def parse_yes_no(answer: Optional[str]) -> Optional[bool]:
    """Juniper's answer to a pair ask: True = same person, False = different,
    None = not a yes/no (applied as nothing; the pair is not asked again)."""
    a = " ".join(str(answer or "").lower().replace(".", " ").replace("!", " ").split())
    if a in _YES:
        return True
    if a in _NO:
        return False
    return None


def face_ask_question(*, face_n: int, sighting_count: int) -> str:
    times = "once" if sighting_count == 1 else f"{sighting_count} times"
    return (f"I saw this person's face clearly ({face_n} good looks) on the walkway, {times} so far. "
            "Do you know who this is?")


def same_person_question(*, face_n_a: int, face_n_b: int) -> str:
    return ("Are these the same person? Their faces look alike to me, but I am not sure "
            f"(left: {face_n_a} good looks, right: {face_n_b}).")


def picture_face_split(refs: Sequence[str], face_by_ref: Dict[str, Sequence[float]],
                       band: float) -> Dict[str, Any]:
    """Does one "who is this?" card show more than one face?

    ``face_by_ref``: picture ref -> face embedding (for the pictures that have
    one). The card is ``impure`` when two of its faces are less alike than
    ``band`` -- the low edge of the measured same-person band -- and ``None``
    (unknown) when fewer than two pictures have a face."""
    faces = [(r, face_by_ref[r]) for r in refs if face_by_ref.get(r)]
    sims = [cosine(faces[i][1], faces[j][1]) for i in range(len(faces)) for j in range(i + 1, len(faces))
            if len(faces[i][1]) == len(faces[j][1])]
    lo = min(sims) if sims else None
    return {"pictures": len(refs), "with_face": len(faces),
            "min_face_cosine": None if lo is None else round(float(lo), 3),
            "impure": None if lo is None else bool(lo < band)}


def individual_display_label(ind_label: Optional[str], kind: str, individual_id: str) -> str:
    return ind_label or f"{kind} #{individual_id[:6]}"


# ---------------------------------------------------------------------------
# Postgres. Blocking; run in a worker thread.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndividualsConfig:
    # Re-identification line: body cosine to ANY same-kind individual's
    # centroid (after the face lane and the continuation lane). 0.90 sits
    # above sqrt(p) of a centroid of several people (p = 0.71-0.75 between
    # different people on the walkway), so a mixed centroid stops attracting
    # new people (docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md).
    match_threshold: float = 0.90
    # Continuation lane: a crop within continuation_max_disp box heights of
    # an individual's last box, seen within continuation_gap_sec, joins the
    # best such individual at >= continuation_threshold. Off + match_threshold
    # 0.80 is the rule before 2026-10-16 exactly.
    continuation_enabled: bool = True
    continuation_threshold: float = 0.88
    continuation_gap_sec: float = 10.0
    continuation_max_disp: float = 1.5
    merge_gap_sec: float = 60.0
    batch_rows: int = 5000
    lookback_ceiling_sec: float = 3600.0
    crop_retention_days: float = 7.0
    sighting_retention_days: float = 90.0
    patio_present_sec: float = 120.0
    patio_grace_sec: float = 600.0
    attention_threshold: float = 0.70
    ask_min_sightings: int = 10
    ask_min_days: int = 5
    ask_expiry_days: float = 7.0
    ask_daily_cap: int = 2
    local_tz: str = "America/Denver"
    camera_alive_sec: float = 300.0
    # Crops newer than this are left for the next tick: bus/threaded writes
    # commit out of observed_at order, and the cursor must not pass a crop
    # that has not landed yet.
    settle_sec: float = 30.0
    # Only individuals seen this recently (plus every labeled one) are match
    # candidates. Bounds per-tick cost as passers-by accumulate.
    candidate_days: float = 30.0
    # After an ask expires unanswered, do not re-ask about that individual for this long.
    ask_cooldown_days: float = 30.0
    # orion-vision-host keeps crop thumbnails this long (its own prune, by
    # file age); 0 = forever, so any thumbnail may be shown. A positive value:
    # an ask only shows a thumbnail young enough to outlive the ask itself,
    # taken within (thumb_retention_days - ask_expiry_days).
    thumb_retention_days: float = 0.0
    # Faces (walkway person crops): cosine similarity to the nearest same-kind
    # face centroid that wins assignment over appearance. <= 0 disables.
    face_match_threshold: float = 0.60
    # Stationary fixtures (orion.vision.stationary): a box that keeps IoU >=
    # stationary_iou with its run's first box, gaps <= stationary_max_gap_sec,
    # for stationary_min_sec is a prop, not a visitor. min_sec <= 0 disables.
    stationary_min_sec: float = 1800.0
    stationary_iou: float = 0.5
    stationary_max_gap_sec: float = 300.0
    # Visual confirmation asks (2026-10-15). A person with at least this many
    # face crops gets "who is this?" the same day, and only people with this
    # many faces are compared for "are these the same person?". <= 0 disables both.
    ask_face_min_faces: int = 5
    # Face-centroid cosine at or above which two people are asked about.
    # Measured live 2026-10-15: two faces in ONE frame (known different
    # people) median 0.27, 75th pct 0.41; the same face over <= 3 s median
    # 0.65, 10th pct 0.39. 0.40 is where the two overlap.
    same_ask_min_face_cosine: float = 0.40
    # Daily cap for these face asks together (both kinds). The 10-sightings
    # rule keeps its own cap (ask_daily_cap); memory confirmation has its own.
    confirm_ask_daily_cap: int = 6
    # Splits on "someone else" picture answers (app/vision_split.py) and the
    # matcher's cannot-link rule. False: verdicts are recorded only.
    split_on_verdict_enabled: bool = True
    split_follow_min_face_cosine: float = 0.60
    split_follow_margin: float = 0.10
    split_max_move_fraction: float = 0.5
    cannot_link_margin: float = 0.10


# Columns manual_migration_walkway_privacy_v2.sql adds; the cycle refuses to
# run (MigrationMissing -> back off) without them.
V2_COLUMNS = (
    ("vision_crop_observation", "face_embedding"),
    ("vision_crop_observation", "face_thumb_ref"),
    ("vision_individual", "face_centroid"),
    ("vision_individual", "stationary"),
    ("vision_individual_sighting", "face_thumb_ref"),
    ("vision_individual_sighting", "stationary"),
)


# Two sql-writer instances must never run the same cycle at once: cursors,
# counts and the ask cap all assume a single writer.
INDIVIDUALS_LOCK_KEY = 0x0A1C_0001


# What manual_migration_vision_confirm_asks_v1.sql adds.
V3_COLUMNS = (("orion_ask", "image_refs"),)
V3_TABLES = ("vision_individual_pair_verdict",)
V4_COLUMNS = (("orion_ask", "answer_detail"), ("vision_individual", "label_copied"))
V4_TABLES = ("vision_ask_picture_verdict",)
# What manual_migration_vision_split_v1.sql adds.
V5_COLUMNS = (("vision_crop_observation", "individual_id"),)
V5_TABLES = ("vision_individual_split",)


def _check_tables(conn) -> None:
    from sqlalchemy import text

    missing = conn.execute(text(
        "SELECT t FROM unnest(ARRAY['vision_crop_observation','vision_individual',"
        "'vision_individual_sighting','vision_individuals_cursor','orion_ask']) AS t "
        "WHERE to_regclass(t) IS NULL"
    )).fetchall()
    if missing:
        raise MigrationMissing(
            "missing tables " + ",".join(r[0] for r in missing)
            + " -- apply services/orion-sql-db/manual_migration_walkway_camera_v1.sql"
        )
    have = {(r[0], r[1]) for r in conn.execute(text(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = ANY(:t)"),
        {"t": sorted({t for t, _ in V2_COLUMNS})}).fetchall()}
    absent = [f"{t}.{c}" for t, c in V2_COLUMNS if (t, c) not in have]
    if absent:
        raise MigrationMissing(
            "missing columns " + ",".join(absent)
            + " -- apply services/orion-sql-db/manual_migration_walkway_privacy_v2.sql"
        )
    have3 = {(r[0], r[1]) for r in conn.execute(text(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = ANY(:t)"),
        {"t": sorted({t for t, _ in V3_COLUMNS})}).fetchall()}
    absent3 = [f"{t}.{c}" for t, c in V3_COLUMNS if (t, c) not in have3]
    absent3 += [r[0] for r in conn.execute(text(
        "SELECT t FROM unnest(CAST(:t AS text[])) AS t WHERE to_regclass(t) IS NULL"),
        {"t": list(V3_TABLES)}).fetchall()]
    if absent3:
        raise MigrationMissing(
            "missing " + ",".join(absent3)
            + " -- apply services/orion-sql-db/manual_migration_vision_confirm_asks_v1.sql"
        )
    have4 = {(r[0], r[1]) for r in conn.execute(text(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = ANY(:t)"),
        {"t": sorted({t for t, _ in V4_COLUMNS})}).fetchall()}
    absent4 = [f"{t}.{c}" for t, c in V4_COLUMNS if (t, c) not in have4]
    absent4 += [r[0] for r in conn.execute(text(
        "SELECT t FROM unnest(CAST(:t AS text[])) AS t WHERE to_regclass(t) IS NULL"),
        {"t": list(V4_TABLES)}).fetchall()]
    if absent4:
        raise MigrationMissing(
            "missing " + ",".join(absent4)
            + " -- apply services/orion-sql-db/manual_migration_vision_ask_names_v1.sql"
        )
    have5 = {(r[0], r[1]) for r in conn.execute(text(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = ANY(:t)"),
        {"t": sorted({t for t, _ in V5_COLUMNS})}).fetchall()}
    absent5 = [f"{t}.{c}" for t, c in V5_COLUMNS if (t, c) not in have5]
    absent5 += [r[0] for r in conn.execute(text(
        "SELECT t FROM unnest(CAST(:t AS text[])) AS t WHERE to_regclass(t) IS NULL"),
        {"t": list(V5_TABLES)}).fetchall()]
    if absent5:
        raise MigrationMissing(
            "missing " + ",".join(absent5)
            + " -- apply services/orion-sql-db/manual_migration_vision_split_v1.sql"
        )


def run_one_individuals_cycle(
    *, postgres_uri: str, cfg: IndividualsConfig, zones_by_stream: Dict[str, List[Zone]],
    now: Optional[datetime] = None,
) -> Tuple[dict, List[dict]]:
    """One full tick. Returns (summary, opened_asks as OrionAskV1 dicts to publish)."""
    from sqlalchemy import create_engine, text

    ts = now or datetime.now(timezone.utc)
    tz = ZoneInfo(cfg.local_tz)
    summary: Dict[str, Any] = {
        "streams": 0, "crops_read": 0, "assigned": 0, "new_individuals": 0,
        "sightings_touched": 0, "attention_events": 0, "patio_writes": 0,
        "answers_applied": 0, "asks_expired": 0, "asks_opened": 0,
        "assigned_by_face": 0, "stationary_crops": 0, "face_blocked_by_cannot_link": 0,
        "splits": 0, "splits_refused": 0, "splits_skipped": 0,
        "pruned_crops": 0, "pruned_sightings": 0, "pruned_individuals": 0,
    }
    opened: List[dict] = []
    engine = create_engine(postgres_uri, pool_pre_ping=True)
    lock_conn = None
    try:
        lock_conn = engine.connect()
        if not lock_conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": INDIVIDUALS_LOCK_KEY}).scalar():
            summary["skipped"] = "another instance holds the individuals lock"
            return summary, []
        lock_conn.commit()  # the session-level lock outlives this; do not sit idle in a transaction
        with engine.begin() as conn:
            _check_tables(conn)
            streams = {r[0] for r in conn.execute(text(
                "SELECT DISTINCT stream_id FROM vision_crop_observation "
                "WHERE created_at > :since"), {"since": ts - timedelta(seconds=cfg.lookback_ceiling_sec)}).fetchall()}
            streams |= {r[0] for r in conn.execute(text("SELECT stream_id FROM vision_individuals_cursor")).fetchall()}
            # Zone head counts must be able to decay to "absent" on a quiet street.
            streams |= {s for s, zs in zones_by_stream.items() if any(z.counts_presence for z in zs)}

        for stream_id in sorted(streams):
            try:
                _one_stream(engine, stream_id, ts, tz, cfg, zones_by_stream.get(stream_id, []), summary)
                summary["streams"] += 1
            except MigrationMissing:
                raise
            except Exception as exc:
                if is_missing_relation(exc):
                    raise MigrationMissing(str(exc)) from exc
                logger.warning("individuals_stream_failed stream=%s error=%s", stream_id, exc)

        with engine.begin() as conn:
            summary["answers_applied"] = _apply_answers(conn, ts)
        if cfg.split_on_verdict_enabled:
            from app.vision_split import apply_splits

            with engine.begin() as conn:
                for k, v in apply_splits(conn, ts, cfg).items():
                    summary[k] = summary.get(k, 0) + v
        with engine.begin() as conn:
            summary["asks_expired"] = conn.execute(text(
                "UPDATE orion_ask SET status='expired' WHERE status='open' "
                "AND expires_at IS NOT NULL AND expires_at < :now"), {"now": ts}).rowcount or 0
        with engine.begin() as conn:
            opened = _open_asks(conn, ts, tz, cfg)
            summary["asks_opened"] = len(opened)
        with engine.begin() as conn:
            _prune(conn, ts, cfg, summary)
    except Exception as exc:
        if not isinstance(exc, MigrationMissing) and is_missing_relation(exc):
            raise MigrationMissing(str(exc)) from exc
        raise
    finally:
        if lock_conn is not None:
            try:
                lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": INDIVIDUALS_LOCK_KEY})
                lock_conn.close()
            except Exception:
                pass
        engine.dispose()
    return summary, opened


# The body-match proposal expects about 500 new person individuals a day on
# the walkway (most of them small). Three times that means the matcher is
# shattering passings: say so in the log.
NEW_PERSON_INDIVIDUALS_WARN_PER_DAY = 1500
FRAGMENTING_WARN_EVERY_SEC = 3600.0
_last_fragmenting_check: Dict[str, datetime] = {}


def _warn_if_fragmenting(conn, stream_id: str, now: datetime) -> Optional[int]:
    """At most once an hour per stream (one count query, one warning)."""
    from sqlalchemy import text

    prev = _last_fragmenting_check.get(stream_id)
    if prev is not None and (now - prev).total_seconds() < FRAGMENTING_WARN_EVERY_SEC:
        return None
    _last_fragmenting_check[stream_id] = now
    n = int(conn.execute(text(
        "SELECT count(*) FROM vision_individual WHERE stream_id=:s AND kind='person' AND first_seen_at > :since"),
        {"s": stream_id, "since": now - timedelta(hours=24)}).scalar() or 0)
    if n > NEW_PERSON_INDIVIDUALS_WARN_PER_DAY:
        logger.warning("[VISION_MATCH] stream=%s fragmenting: %d new person individuals in 24 h (> %d)",
                       stream_id, n, NEW_PERSON_INDIVIDUALS_WARN_PER_DAY)
    return n


def _row_to_crop(r) -> CropRow:
    emb = r.embedding
    return CropRow(
        crop_id=r.crop_id, observation_id=r.observation_id, stream_id=r.stream_id,
        observed_at=r.observed_at, label=r.label, box_xyxy=tuple(r.box_xyxy or ()),
        zone=r.zone, embedding=tuple(emb) if emb else None,
        embedding_ref=r.embedding_ref, artifact_id=r.artifact_id,
        zone_no_embed=bool(r.zone_no_embed), thumb_ref=r.thumb_ref,
        face_embedding=tuple(r.face_embedding) if r.face_embedding else None,
        face_thumb_ref=r.face_thumb_ref,
    )


def _one_stream(engine, stream_id: str, now: datetime, tz: ZoneInfo, cfg: IndividualsConfig,
                zones: List[Zone], summary: dict) -> None:
    from sqlalchemy import text

    forbidden = {z.name for z in zones if not z.embed}
    head_count_zones = {z.name for z in zones if z.counts_presence}
    with engine.begin() as conn:
        cur = conn.execute(text(
            "SELECT last_created_at FROM vision_individuals_cursor WHERE stream_id=:s"), {"s": stream_id}).fetchone()
        since = cur[0] if cur else now - timedelta(seconds=cfg.lookback_ceiling_sec)
        # The cursor is LANDING time (created_at, the DB clock), not the edge
        # frame time: a crop observed long ago that lands now is still read.
        # settle_sec covers insert transactions that began before `upto` but
        # had not committed yet.
        upto = now - timedelta(seconds=cfg.settle_sec)
        rows = conn.execute(text(
            "SELECT crop_id, observation_id, stream_id, observed_at, label, box_xyxy, zone, "
            "embedding, embedding_ref, artifact_id, zone_no_embed, thumb_ref, face_embedding, face_thumb_ref, "
            "created_at "
            "FROM vision_crop_observation "
            "WHERE stream_id=:s AND created_at > :since AND created_at <= :upto "
            "ORDER BY created_at, crop_id LIMIT :lim"),
            {"s": stream_id, "since": since, "upto": upto, "lim": cfg.batch_rows}).fetchall()
        landed = [(r.created_at, _row_to_crop(r)) for r in rows]
        full = len(landed) >= cfg.batch_rows
        # A full batch may have cut one landing instant in half; leave the
        # rows at the last created_at for the next tick (unless that is all
        # there is).
        if full:
            last_ts = landed[-1][0]
            trimmed = [lc for lc in landed if lc[0] < last_ts]
            if trimmed:
                landed = trimmed
        crops = [c for _, c in landed]
        summary["crops_read"] += len(crops)

        batch = None
        if crops:
            oldest_seen = min(c.observed_at for c in crops)
            kinds = sorted({c.label for c in crops})
            ind_rows = conn.execute(text(
                "SELECT individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, last_seen_at, "
                "sighting_count, distinct_days, label, face_centroid, face_n, still_box, still_since, "
                "still_last, stationary FROM vision_individual "
                "WHERE stream_id=:s AND kind = ANY(:k) AND (label IS NOT NULL OR last_seen_at > :recent)"),
                {"s": stream_id, "k": kinds,
                 "recent": oldest_seen - timedelta(days=cfg.candidate_days)}).fetchall()
            individuals = {r.individual_id: Individual(
                individual_id=r.individual_id, stream_id=r.stream_id, kind=r.kind,
                centroid=list(r.centroid), centroid_n=r.centroid_n, first_seen_at=r.first_seen_at,
                last_seen_at=r.last_seen_at, sighting_count=r.sighting_count,
                distinct_days=r.distinct_days, label=r.label,
                face_centroid=list(r.face_centroid) if r.face_centroid else None, face_n=int(r.face_n or 0),
                still_box=list(r.still_box) if r.still_box else None, still_since=r.still_since,
                still_last=r.still_last, stationary=bool(r.stationary)) for r in ind_rows}
            # Every sighting overlapping this batch's span (+- gap), not just
            # each individual's latest: a late crop must be able to rejoin the
            # older sighting it belongs to.
            newest_seen = max(c.observed_at for c in crops)
            gap = timedelta(seconds=cfg.merge_gap_sec)
            s_rows = conn.execute(text(
                "SELECT sighting_id, individual_id, stream_id, started_at, ended_at, "
                "observation_count, zone_counts, last_box_xyxy, embedding_ref, evidence_ref, thumb_ref, "
                "face_thumb_ref, stationary "
                "FROM vision_individual_sighting WHERE stream_id=:s AND ended_at >= :after "
                "AND started_at <= :before"),
                {"s": stream_id, "after": oldest_seen - gap, "before": newest_seen + gap}).fetchall()
            nearby: Dict[str, List[Sighting]] = {}
            for r in s_rows:
                nearby.setdefault(r.individual_id, []).append(Sighting(
                    sighting_id=r.sighting_id, individual_id=r.individual_id, stream_id=r.stream_id,
                    started_at=r.started_at, ended_at=r.ended_at, observation_count=r.observation_count,
                    zone_counts=dict(r.zone_counts or {}), last_box_xyxy=list(r.last_box_xyxy or []) or None,
                    embedding_ref=r.embedding_ref, evidence_ref=r.evidence_ref,
                    thumb_ref=r.thumb_ref, face_thumb_ref=r.face_thumb_ref, stationary=bool(r.stationary)))

            cannot = load_cannot_link(conn) if cfg.split_on_verdict_enabled else {}
            prior_obs: List[Tuple[str, datetime, Sequence[float]]] = []
            if cfg.continuation_enabled:
                # Where each individual was, crop by crop, around this batch
                # (stored membership): a late crop is compared with where its
                # person was at that moment, not where they were last.
                cg = timedelta(seconds=min(cfg.continuation_gap_sec, cfg.merge_gap_sec))
                prior_obs = [(r.individual_id, r.observed_at, list(r.box_xyxy or ())) for r in conn.execute(text(
                    "SELECT individual_id, observed_at, box_xyxy FROM vision_crop_observation "
                    "WHERE stream_id=:s AND individual_id IS NOT NULL AND observed_at >= :lo AND observed_at <= :hi"),
                    {"s": stream_id, "lo": oldest_seen - cg, "hi": newest_seen + cg}).fetchall()]
            batch = apply_batch(
                crops, individuals=individuals, latest_sightings={}, nearby_sightings=nearby,
                cannot_link=cannot, cannot_link_margin=cfg.cannot_link_margin,
                no_embed_zones=forbidden, match_threshold=cfg.match_threshold,
                merge_gap_sec=cfg.merge_gap_sec, tz=tz,
                face_match_threshold=cfg.face_match_threshold if cfg.face_match_threshold > 0 else None,
                stationary_min_sec=cfg.stationary_min_sec, stationary_iou=cfg.stationary_iou,
                stationary_max_gap_sec=cfg.stationary_max_gap_sec,
                continuation_threshold=cfg.continuation_threshold if cfg.continuation_enabled else None,
                continuation_gap_sec=min(cfg.continuation_gap_sec, cfg.merge_gap_sec),
                continuation_max_disp=cfg.continuation_max_disp,
                prior_observations=prior_obs,
            )
            logger.info("[VISION_MATCH] stream=%s face=%d continuation=%d reid=%d new=%d",
                        stream_id, batch.assigned_by_face, batch.assigned_by_continuation,
                        batch.assigned_by_reid, len(batch.new_individual_ids))
            summary["assigned_by_continuation"] = (summary.get("assigned_by_continuation", 0)
                                                   + batch.assigned_by_continuation)
            summary["assigned_by_reid"] = summary.get("assigned_by_reid", 0) + batch.assigned_by_reid
            summary["assigned"] += batch.assigned
            summary["assigned_by_face"] = summary.get("assigned_by_face", 0) + batch.assigned_by_face
            summary["stationary_crops"] = summary.get("stationary_crops", 0) + sum(
                1 for v in batch.crop_stationary.values() if v)
            summary["face_blocked_by_cannot_link"] = (summary.get("face_blocked_by_cannot_link", 0)
                                                      + batch.face_blocked_by_cannot_link)
            summary["new_individuals"] += len(batch.new_individual_ids)
            summary["sightings_touched"] += len(batch.sightings)
            changed_ids = {s.individual_id for s in batch.sightings.values()}
            for iid in changed_ids:
                _upsert_individual(conn, batch.individuals[iid])
            for s in batch.sightings.values():
                _upsert_sighting(conn, s, forbidden)
            _write_membership(conn, batch.crop_individual)
            if batch.new_individual_ids:
                _warn_if_fragmenting(conn, stream_id, now)
            for iid, since in batch.became_stationary.items():
                conn.execute(text(
                    "UPDATE vision_individual_sighting SET stationary=true, attention_score=NULL, "
                    "attention_components=CAST(:c AS jsonb) "
                    "WHERE individual_id=:i AND ended_at >= :since AND NOT stationary"),
                    {"c": json.dumps({"_basis": "stationary"}), "i": iid, "since": since})
            zone_rare = {z.name: z.dwell_rare_sec for z in zones}
            for s in batch.sightings.values():
                if _score_and_maybe_event(conn, s, batch.individuals[s.individual_id], zone_rare, tz, cfg):
                    summary["attention_events"] += 1

        # Everything that landed up to `upto` has been read (unless the batch
        # was full), so the cursor can advance on a quiet street too -- the
        # rhythm loop reads it as "individuals are processed through here".
        new_cursor = landed[-1][0] if (landed and full) else max(upto, since)
        conn.execute(text(
            "INSERT INTO vision_individuals_cursor (stream_id, last_created_at, updated_at) "
            "VALUES (:s, :t, now()) ON CONFLICT (stream_id) DO UPDATE SET "
            "last_created_at = GREATEST(vision_individuals_cursor.last_created_at, EXCLUDED.last_created_at), "
            "updated_at = now()"),
            {"s": stream_id, "t": new_cursor})

        no_embed_crops = batch.patio_crops if batch else []
        # One head-count row per presence zone: every zone the config names
        # (so a quiet one can decay to absent) plus any a no-embed row names.
        zone_names = set(head_count_zones) | {c.zone for c in no_embed_crops if c.zone}
        stationary_boxes = [(ind.kind, ind.still_box) for ind in (batch.individuals.values() if batch else [])
                            if ind.stationary and ind.still_box]
        for zone_name in sorted(zone_names):
            # Its own savepoint: a missing presence table (a different
            # migration) must not abort clustering, asks, or pruning.
            try:
                with conn.begin_nested():
                    in_zone = head_count_crops(
                        crops, zone_name, crop_stationary=batch.crop_stationary if batch else {},
                        stationary_boxes=stationary_boxes, iou=cfg.stationary_iou)
                    if _write_patio_presence(conn, stream_id, zone_name, in_zone, now, cfg):
                        summary["patio_writes"] += 1
            except Exception as exc:
                logger.warning("no_embed_presence_write_failed stream=%s zone=%s error=%s "
                               "(substrate_embodied_presence: manual_migration_embodied_presence_v1.sql)",
                               stream_id, zone_name, exc)


def _upsert_individual(conn, ind: Individual) -> None:
    from sqlalchemy import text

    conn.execute(text("""
        INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at,
            last_seen_at, sighting_count, distinct_days, face_centroid, face_n, still_box, still_since,
            still_last, stationary, updated_at)
        VALUES (:id, :s, :k, :c, :n, :f, :l, :sc, :dd, :fc, :fn, :sb, :ss, :sl, :st, now())
        ON CONFLICT (individual_id) DO UPDATE SET centroid=EXCLUDED.centroid, centroid_n=EXCLUDED.centroid_n,
            last_seen_at=EXCLUDED.last_seen_at, sighting_count=EXCLUDED.sighting_count,
            distinct_days=EXCLUDED.distinct_days, face_centroid=EXCLUDED.face_centroid,
            face_n=EXCLUDED.face_n, still_box=EXCLUDED.still_box, still_since=EXCLUDED.still_since,
            still_last=EXCLUDED.still_last, stationary=EXCLUDED.stationary, updated_at=now()
    """), {"id": ind.individual_id, "s": ind.stream_id, "k": ind.kind, "c": ind.centroid,
           "n": ind.centroid_n, "f": ind.first_seen_at, "l": ind.last_seen_at,
           "sc": ind.sighting_count, "dd": ind.distinct_days, "fc": ind.face_centroid,
           "fn": ind.face_n, "sb": ind.still_box, "ss": ind.still_since, "sl": ind.still_last,
           "st": ind.stationary})


def _write_membership(conn, crop_individual: Dict[str, str]) -> int:
    """vision_crop_observation.individual_id for every crop this batch assigned,
    in the batch's own transaction (manual_migration_vision_split_v1.sql)."""
    from sqlalchemy import text

    if not crop_individual:
        return 0
    cids = sorted(crop_individual)
    return conn.execute(text(
        "UPDATE vision_crop_observation c SET individual_id = m.iid "
        "FROM unnest(CAST(:c AS text[]), CAST(:i AS text[])) AS m(cid, iid) WHERE c.crop_id = m.cid"),
        {"c": cids, "i": [crop_individual[c] for c in cids]}).rowcount or 0


def load_cannot_link(conn) -> Dict[str, set]:
    """individual_id -> ids Juniper said are a different person (every
    'different' pair verdict, including the ones a split writes)."""
    from sqlalchemy import text

    out: Dict[str, set] = {}
    for a, b in conn.execute(text(
            "SELECT individual_a, individual_b FROM vision_individual_pair_verdict WHERE verdict='different'")).fetchall():
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def sighting_row(s: Sighting, no_embed_zones: Iterable[str]) -> Dict[str, Any]:
    """Pure: the upsert parameters. A sighting whose dominant zone is a
    no-embed zone (cannot happen from apply_batch, which never clusters those
    crops -- this is the belt to the DB CHECK's braces) carries no refs."""
    no_embed = s.zone in set(no_embed_zones)
    return {"id": s.sighting_id, "iid": s.individual_id, "s": s.stream_id, "z": s.zone,
            "st": s.started_at, "en": s.ended_at, "dw": s.dwell_sec, "oc": s.observation_count,
            "box": s.last_box_xyxy, "er": None if no_embed else s.embedding_ref, "ev": s.evidence_ref,
            "th": None if no_embed else s.thumb_ref, "ne": no_embed,
            "fth": None if no_embed else s.face_thumb_ref, "sta": bool(s.stationary),
            "zc": json.dumps(s.zone_counts)}


def _upsert_sighting(conn, s: Sighting, no_embed_zones: Iterable[str] = ()) -> None:
    from sqlalchemy import text

    conn.execute(text("""
        INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, zone, started_at, ended_at,
            dwell_sec, observation_count, last_box_xyxy, embedding_ref, evidence_ref, thumb_ref, zone_no_embed,
            face_thumb_ref, stationary, zone_counts, updated_at)
        VALUES (:id, :iid, :s, :z, :st, :en, :dw, :oc, :box, :er, :ev, :th, :ne, :fth, :sta,
            CAST(:zc AS jsonb), now())
        ON CONFLICT (sighting_id) DO UPDATE SET zone=EXCLUDED.zone, started_at=EXCLUDED.started_at,
            ended_at=EXCLUDED.ended_at,
            dwell_sec=EXCLUDED.dwell_sec, observation_count=EXCLUDED.observation_count,
            last_box_xyxy=EXCLUDED.last_box_xyxy, embedding_ref=EXCLUDED.embedding_ref,
            evidence_ref=EXCLUDED.evidence_ref, thumb_ref=EXCLUDED.thumb_ref,
            zone_no_embed=EXCLUDED.zone_no_embed, face_thumb_ref=EXCLUDED.face_thumb_ref,
            stationary=EXCLUDED.stationary, zone_counts=EXCLUDED.zone_counts, updated_at=now()
    """), sighting_row(s, no_embed_zones))


def _history(conn, sql: str, params: dict, tz: ZoneInfo, basis: str):
    from sqlalchemy import text

    from app.vision_attention_score import hour_history

    rows = conn.execute(text(sql), params).fetchall()
    return hour_history(((r[0].astimezone(tz).hour, r[0].astimezone(tz).date().isoformat()) for r in rows), basis)


def _score_and_maybe_event(conn, s: Sighting, ind: Individual, zone_rare: Dict[str, Optional[float]],
                           tz: ZoneInfo, cfg: IndividualsConfig) -> bool:
    from sqlalchemy import text

    from app.vision_attention_score import attention_narrative, score_sighting

    if s.stationary:
        # A fixture, not a visit: recorded, never scored, never an event.
        conn.execute(text(
            "UPDATE vision_individual_sighting SET attention_score=NULL, "
            "attention_components=CAST(:c AS jsonb) WHERE sighting_id=:id"),
            {"c": json.dumps({"_basis": "stationary"}), "id": s.sighting_id})
        return False
    since = s.started_at - timedelta(days=cfg.sighting_retention_days)
    ind_hist = _history(conn, (
        "SELECT started_at FROM vision_individual_sighting WHERE individual_id=:i "
        "AND sighting_id <> :sid AND started_at >= :since"),
        {"i": ind.individual_id, "sid": s.sighting_id, "since": since}, tz, "individual")
    kind_hist = _history(conn, (
        "SELECT s.started_at FROM vision_individual_sighting s JOIN vision_individual i "
        "ON i.individual_id = s.individual_id WHERE s.stream_id=:st AND i.kind=:k "
        "AND s.sighting_id <> :sid AND s.started_at >= :since"),
        {"st": s.stream_id, "k": ind.kind, "sid": s.sighting_id, "since": since}, tz, "kind")
    local_start = s.started_at.astimezone(tz)
    result = score_sighting(
        labeled=bool(ind.label), local_hour=local_start.hour, dwell_sec=s.dwell_sec,
        dwell_rare_sec=zone_rare.get(s.zone or ""), prior_sightings=max(0, ind.sighting_count - 1),
        individual_history=ind_hist, kind_history=kind_hist,
    )
    components = dict(result.components)
    components["_basis"] = result.basis
    conn.execute(text(
        "UPDATE vision_individual_sighting SET attention_score=:sc, attention_components=CAST(:c AS jsonb) "
        "WHERE sighting_id=:id"), {"sc": result.score, "c": json.dumps(components), "id": s.sighting_id})
    if result.score < cfg.attention_threshold:
        return False
    narrative = attention_narrative(kind=ind.kind, zone=s.zone, stream_id=s.stream_id,
                                    local_time=local_start, dwell_sec=s.dwell_sec, result=result)
    tags = [s.stream_id, "walkway_camera", ind.kind] + ([s.zone] if s.zone else [])
    res = conn.execute(text(
        "INSERT INTO vision_events (event_id, event_type, narrative, entities, tags, confidence, salience, "
        "evidence_refs, stream_id, created_at) VALUES (:id, 'attention_worthy', :n, CAST(:e AS jsonb), "
        "CAST(:t AS jsonb), :sc, :sc, CAST(:ev AS jsonb), :sid, now()) ON CONFLICT (event_id) DO NOTHING"),
        {"id": f"attention-{s.sighting_id}", "sid": s.stream_id, "n": narrative, "e": json.dumps([ind.kind]),
         "t": json.dumps(tags), "sc": result.score,
         "ev": json.dumps([f"sighting:{s.sighting_id}", f"individual:{ind.individual_id}"])})
    return bool(res.rowcount)


def _write_patio_presence(conn, stream_id: str, zone_name: str, patio_crops: Sequence[CropRow],
                          now: datetime, cfg: IndividualsConfig) -> bool:
    from sqlalchemy import text

    alive = conn.execute(text(
        "SELECT 1 FROM vision_scene_inventory WHERE stream_id=:s AND observed_at > :t LIMIT 1"),
        {"s": stream_id, "t": now - timedelta(seconds=cfg.camera_alive_sec)}).fetchone()
    if not alive and not patio_crops:
        return False  # camera silent: let the row go stale, do not assert "absent"
    pid = no_embed_presence_id(stream_id, zone_name)
    prev_row = conn.execute(text(
        "SELECT presence_json FROM substrate_embodied_presence WHERE presence_id=:p"), {"p": pid}).fetchone()
    prev = prev_row[0] if prev_row else None
    if isinstance(prev, str):
        prev = json.loads(prev)
    last, count = patio_count(patio_crops)
    snap = patio_snapshot(prev=prev, batch_last_seen_at=last, batch_count=count, now=now,
                          present_sec=cfg.patio_present_sec, grace_sec=cfg.patio_grace_sec, zone=zone_name)
    conn.execute(text("""
        INSERT INTO substrate_embodied_presence (presence_id, generated_at, presence_json, updated_at)
        VALUES (:p, now(), CAST(:j AS jsonb), now())
        ON CONFLICT (presence_id) DO UPDATE SET generated_at=EXCLUDED.generated_at,
            presence_json=EXCLUDED.presence_json, updated_at=EXCLUDED.updated_at
    """), {"p": pid, "j": json.dumps(snap)})
    return True


def _clean_name(value: Any) -> str:
    # orion.substrate.adapters.vision_individual.MAX_LABEL_LENGTH; the Hub refuses longer.
    return " ".join(str(value or "").split())[:120]


def _detail(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _set_typed_label(conn, individual_id: str, label: str, ask_id: str, at: datetime) -> bool:
    """A name Juniper typed for this individual (a "who is this?" answer, or a
    name on a "same person?" card). The last typed name wins: it replaces a
    copied name and an earlier typed one; replacing a typed one is logged."""
    from sqlalchemy import text

    prev = conn.execute(text(
        "SELECT label, label_copied, label_ask_id FROM vision_individual WHERE individual_id=:i FOR UPDATE"),
        {"i": individual_id}).fetchone()
    if prev is None:
        logger.warning("ask_answer_for_missing_individual ask_id=%s individual_id=%s", ask_id, individual_id)
        return False
    if prev[0] and prev[0] != label and not prev[1]:
        logger.info("[VISION_LABEL_REPLACED] individual_id=%s old=%r old_ask_id=%s new=%r ask_id=%s",
                    individual_id, prev[0], prev[2], label, ask_id)
    conn.execute(text(
        "UPDATE vision_individual SET label=:l, labeled_at=:at, label_ask_id=:a, label_copied=false, "
        "updated_at=now() WHERE individual_id=:i"), {"l": label, "at": at, "a": ask_id, "i": individual_id})
    return True


def _propagate_label(conn, individual_id: str, label: str, ask_id: str, at: datetime, *,
                     replace_copied: bool = False) -> List[str]:
    """Give ``label`` to everyone Juniper said is the same person as
    ``individual_id`` -- the whole group, transitively (a=b, b=c: naming a
    names c too), matching ``same_groups`` used by the opener. Never overwrites
    a name she typed for that individual; ``replace_copied`` (the label is one
    she just typed) also replaces names that were only copied. Returns the ids
    it labelled."""
    from sqlalchemy import text

    pairs = [(r[0], r[1]) for r in conn.execute(text(
        "WITH RECURSIVE g(id) AS (SELECT CAST(:i AS text) UNION "
        "  SELECT CASE WHEN v.individual_a = g.id THEN v.individual_b ELSE v.individual_a END "
        "  FROM vision_individual_pair_verdict v JOIN g ON g.id IN (v.individual_a, v.individual_b) "
        "  WHERE v.verdict = 'same') "
        "SELECT CAST(:i AS text), id FROM g"), {"i": individual_id}).fetchall()]
    group = sorted({b for _, b in pairs} - {individual_id})
    if not group:
        return []
    if replace_copied:
        # Two typed names in one confirmed group: copies follow the newest, but say so.
        others = sorted({r[0] for r in conn.execute(text(
            "SELECT label FROM vision_individual WHERE individual_id = ANY(:ids) AND label IS NOT NULL "
            "AND NOT label_copied AND lower(label) <> lower(:l)"), {"ids": group, "l": label}).fetchall()})
        if others:
            logger.warning("[VISION_GROUP_NAME_CONFLICT] individual_id=%s new=%r other_typed=%r ask_id=%s",
                           individual_id, label, others, ask_id)
    return [r[0] for r in conn.execute(text(
        "UPDATE vision_individual SET label=:l, labeled_at=:at, label_ask_id=:a, label_copied=true, updated_at=now() "
        "WHERE individual_id = ANY(:ids) AND (label IS NULL OR (:rc AND label_copied)) "
        "RETURNING individual_id"),
        {"l": label, "at": at, "a": ask_id, "ids": group, "rc": bool(replace_copied)}).fetchall()]


def _close_who_is_this(conn, individual_ids: Sequence[str], label: str, ask_id: str, now: datetime) -> int:
    """Close any still-open "who is this?" card for people who just got ``label``
    from a name Juniper typed elsewhere, so she is not asked again. Marked
    applied: the label is already set (a copied one must not become typed)."""
    from sqlalchemy import text

    if not individual_ids:
        return 0
    n = conn.execute(text(
        "UPDATE orion_ask SET status='answered', answer=:l, answered_at=:now, applied_at=:now, "
        "answer_detail=jsonb_build_object('resolved_by', CAST(:a AS text)) "
        "WHERE source_kind=:k AND status='open' AND source_ref = ANY(:ids)"),
        {"l": label, "now": now, "a": ask_id, "k": SOURCE_KIND_INDIVIDUAL, "ids": list(individual_ids)}).rowcount or 0
    if n:
        logger.info("[VISION_ASK_CLOSED_BY_NAME] n=%d by_ask_id=%s ids=%s", n, ask_id, ",".join(individual_ids))
    return n


def _apply_typed_name(conn, individual_id: str, label: str, ask_id: str, at: datetime, now: datetime) -> bool:
    """One typed name, applied the same way wherever it was typed (a "who is
    this?" card, a walkway story card, a name box on a "same person?" card): a
    'same' link to an existing individual she already typed that name for
    (``_link_same_name``), then the individual, then everyone confirmed the same
    person, then their open cards. A typed name that replaces a different
    typed name first drops the same-name links the old name made."""
    _unlink_old_name(conn, individual_id, label, ask_id)
    _link_same_name(conn, individual_id, label, ask_id, at)
    if not _set_typed_label(conn, individual_id, label, ask_id, at):
        return False
    copied = _propagate_label(conn, individual_id, label, ask_id, at, replace_copied=True)
    _close_who_is_this(conn, [individual_id, *copied], label, ask_id, now)
    return True


PICTURE_VERDICTS = ("same", "different", "unsure")


def picture_rows(ask_id: str, individual_id: str, detail: Dict[str, Any], answered_at: datetime) -> List[dict]:
    """``answer_detail.pictures`` -> vision_ask_picture_verdict rows (pure; no crop ids yet)."""
    out: List[dict] = []
    seen = set()
    for p in detail.get("pictures") or []:
        if not isinstance(p, dict):
            continue
        ref, verdict = str(p.get("image_ref") or ""), str(p.get("verdict") or "")
        if not ref or ref in seen or verdict not in PICTURE_VERDICTS:
            continue
        seen.add(ref)
        name = _clean_name(p.get("name")) or None
        # A story card names the crop itself (its pictures are crops, not thumbnails).
        crop_id = str(p.get("crop_id") or "") or None
        out.append({"ask": ask_id, "ref": ref, "idx": int(p.get("index") or 0), "ind": individual_id,
                    "v": verdict, "name": name if verdict != "unsure" else None, "at": answered_at,
                    "crop_id": crop_id})
    return out


def _record_pictures(conn, rows: List[dict]) -> int:
    """Keep each picture's verdict. Nothing here moves a crop: app/vision_split.py
    acts on the 'different' rows later in the same cycle (when enabled)."""
    from sqlalchemy import text

    if not rows:
        return 0
    refs = [r["ref"] for r in rows]
    crop = {}
    for ref, crop_id in conn.execute(text(
            "SELECT DISTINCT ON (ref) ref, crop_id FROM ("
            "  SELECT face_thumb_ref AS ref, crop_id, observed_at FROM vision_crop_observation "
            "  WHERE face_thumb_ref = ANY(:r) UNION ALL "
            "  SELECT thumb_ref, crop_id, observed_at FROM vision_crop_observation WHERE thumb_ref = ANY(:r)) c "
            "ORDER BY ref, observed_at DESC"), {"r": refs}).fetchall():
        crop[ref] = crop_id
    for r in rows:
        conn.execute(text(
            "INSERT INTO vision_ask_picture_verdict (ask_id, image_ref, picture_index, individual_id, verdict, name, "
            "crop_id, answered_at) VALUES (:ask, :ref, :idx, :ind, :v, :name, :crop, :at) "
            "ON CONFLICT (ask_id, image_ref) DO NOTHING"),
            {k: v for k, v in {**r, "crop": r.get("crop_id") or crop.get(r["ref"])}.items() if k != "crop_id"})
    others = [r for r in rows if r["v"] != "same"]
    if others:
        logger.info("[VISION_PICTURE_VERDICT] ask_id=%s individual_id=%s pictures=%d not_same=%s",
                    rows[0]["ask"], rows[0]["ind"], len(rows),
                    ",".join(f"{r['idx']}:{r['v']}" for r in others))
    return len(rows)


def same_name_target(conn, individual_id: str, name: str, *, exclude: Sequence[str] = ()) -> Optional[str]:
    """An existing individual (same stream and kind as ``individual_id``) Juniper
    already TYPED ``name`` for, that is not already confirmed the same person
    (a 'same' path between them), has no pair verdict with it, and whose
    confirmed group has no 'different' (cannot-link) with this one's group --
    a link must never join two people she said are different, even through
    others. First by labeled_at. None when there is none."""
    from sqlalchemy import text

    me = conn.execute(text("SELECT stream_id, kind FROM vision_individual WHERE individual_id=:i"),
                      {"i": individual_id}).fetchone()
    if me is None:
        return None
    skip = [individual_id, *exclude]
    # A candidate with any pair verdict with this individual is skipped (a 'different' she gave stays).
    cands = [r[0] for r in conn.execute(text(
        "SELECT v.individual_id FROM vision_individual v WHERE v.stream_id=:s AND v.kind=:k "
        "AND v.label IS NOT NULL AND NOT v.label_copied AND lower(v.label) = lower(:n) "
        "AND NOT (v.individual_id = ANY(:skip)) "
        "AND NOT EXISTS (SELECT 1 FROM vision_individual_pair_verdict p WHERE "
        "  (p.individual_a = v.individual_id AND p.individual_b = :me) OR "
        "  (p.individual_b = v.individual_id AND p.individual_a = :me)) "
        "ORDER BY v.labeled_at NULLS LAST, v.individual_id"),
        {"s": me[0], "k": me[1], "n": name, "skip": skip, "me": individual_id}).fetchall()]
    if not cands:
        return None
    verdicts = conn.execute(text(
        "SELECT individual_a, individual_b, verdict FROM vision_individual_pair_verdict "
        "WHERE verdict IN ('same', 'different')")).fetchall()
    groups = same_groups((a, b) for a, b, v in verdicts if v == "same")
    g = lambda i: groups.get(i, i)  # noqa: E731
    apart = {frozenset((g(a), g(b))) for a, b, v in verdicts if v == "different"}
    mine = g(individual_id)
    return next((c for c in cands if g(c) != mine and frozenset((g(c), mine)) not in apart), None)


def _link_same_name(conn, individual_id: str, name: str, ask_id: str, at: datetime) -> Optional[str]:
    """A typed name (any card): a 'same' link to an existing
    individual Juniper already typed that name for (the split's same-name link,
    app/vision_split.py). Never over an existing pair verdict (a 'different'
    she gave stays). Returns the linked id, or None."""
    from sqlalchemy import text

    other = same_name_target(conn, individual_id, name)
    if other is None:
        return None
    a, b = sorted((individual_id, other))
    linked = conn.execute(text(
        "INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, face_cosine, ask_id, "
        "decided_at) VALUES (:a, :b, 'same', NULL, :ask, :at) ON CONFLICT (individual_a, individual_b) DO NOTHING "
        "RETURNING individual_a"), {"a": a, "b": b, "ask": ask_id, "at": at}).fetchone()
    if linked is None:
        return None
    logger.info("[VISION_SAME_NAME_LINK] individual_id=%s other=%s name=%r ask_id=%s",
                individual_id, other, name, ask_id)
    return other


def _unlink_old_name(conn, individual_id: str, label: str, ask_id: str) -> int:
    """She corrected a typed name ("Maya" -> "Mya"): the 'same' links that only
    came from the old name go. Those are rows involving this individual written
    by a typed-name ask (source_kind vision_individual: a who-is-this or story
    answer, or a split it caused), never her own "same person?" answers. No-op
    when the old name was copied or differs only in case."""
    from sqlalchemy import text

    prev = conn.execute(text(
        "SELECT label, label_copied FROM vision_individual WHERE individual_id=:i"), {"i": individual_id}).fetchone()
    if prev is None or not prev[0] or prev[1] or prev[0].lower() == label.lower():
        return 0
    gone = conn.execute(text(
        "DELETE FROM vision_individual_pair_verdict p WHERE p.verdict='same' "
        "AND :i IN (p.individual_a, p.individual_b) AND EXISTS (SELECT 1 FROM orion_ask q "
        "  WHERE q.ask_id = p.ask_id AND q.source_kind = :k) "
        "RETURNING CASE WHEN p.individual_a = :i THEN p.individual_b ELSE p.individual_a END"),
        {"i": individual_id, "k": SOURCE_KIND_INDIVIDUAL}).fetchall()
    if gone:
        logger.info("[VISION_SAME_NAME_UNLINK] individual_id=%s old=%r new=%r ask_id=%s others=%s",
                    individual_id, prev[0], label, ask_id, ",".join(sorted(r[0] for r in gone)))
    return len(gone)


def _apply_answers(conn, now: datetime) -> int:
    """Every unapplied "who is this?" and "same person?" answer, oldest first, so
    when two answers name one individual the later one wins. Every typed name
    also links the individual to an existing one Juniper gave the same name
    (``_apply_typed_name``); a name typed on a walkway story card
    (answer_detail.story_id) is a plain "who is this?" answer; a story card's
    "someone else" / "not sure" records its picture verdicts only."""
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT ask_id, source_kind, source_ref, answer, answered_at, answer_detail FROM orion_ask "
        "WHERE source_kind = ANY(:k) AND status='answered' AND applied_at IS NULL "
        "ORDER BY answered_at NULLS FIRST, ask_id"),
        {"k": [SOURCE_KIND_INDIVIDUAL, SOURCE_KIND_SAME, SOURCE_KIND_STORY_PERSON]}).fetchall()
    for ask_id, kind, ref, answer, answered_at, detail in rows:
        at = answered_at or now
        if kind == SOURCE_KIND_INDIVIDUAL:
            label = _clean_name(answer)
            if label:
                _apply_typed_name(conn, ref, label, ask_id, at, now)
                _record_pictures(conn, picture_rows(ask_id, ref, _detail(detail), at))
        elif kind == SOURCE_KIND_STORY_PERSON:
            _record_pictures(conn, picture_rows(ask_id, ref, _detail(detail), at))
        else:
            _apply_pair_answer(conn, ask_id, ref, answer, at, _detail(detail), now)
        conn.execute(text("UPDATE orion_ask SET applied_at=:now WHERE ask_id=:a"), {"now": now, "a": ask_id})
    return len(rows) + _heal_pair_names(conn, now)


def _heal_pair_names(conn, now: datetime) -> int:
    """Names on "same person?" answers that were marked applied without being
    applied: a sql-writer built before answer_detail applies the verdict and
    sets applied_at, but ignores the names (Hub deployed first). A name is
    unapplied when its individual's label is older than the answer (or absent);
    one this writer applied has labeled_at == answered_at, a later one is newer."""
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT a.ask_id, a.source_ref, a.answer, a.answered_at, a.answer_detail FROM orion_ask a "
        "WHERE a.source_kind = :k AND a.status = 'answered' AND a.applied_at IS NOT NULL "
        "AND a.answered_at IS NOT NULL AND a.answer_detail -> 'names' IS NOT NULL "
        "AND EXISTS (SELECT 1 FROM jsonb_each_text(a.answer_detail -> 'names') n "
        "  JOIN vision_individual v ON v.individual_id = n.key "
        "  WHERE v.labeled_at IS NULL OR v.labeled_at < a.answered_at) "
        "ORDER BY a.answered_at, a.ask_id"), {"k": SOURCE_KIND_SAME}).fetchall()
    healed = 0
    for ask_id, ref, answer, answered_at, detail in rows:
        pair = parse_pair_ref(ref)
        if pair is None:
            continue
        names = pair_names(pair, parse_yes_no(answer), _detail(detail))
        stale = {r[0] for r in conn.execute(text(
            "SELECT individual_id FROM vision_individual WHERE individual_id = ANY(:ids) "
            "AND (labeled_at IS NULL OR labeled_at < :at)"), {"ids": list(names), "at": answered_at}).fetchall()}
        for iid in pair:
            if iid in names and iid in stale:
                _apply_typed_name(conn, iid, names[iid], ask_id, answered_at, now)
                healed += 1
        logger.warning("[VISION_PAIR_NAMES_HEALED] ask_id=%s names=%d", ask_id, len(stale & set(names)))
    return healed


def pair_names(pair: Tuple[str, str], same: Optional[bool], detail: Dict[str, Any]) -> Dict[str, str]:
    """The typed names on a "same person?" answer that can be applied: only for
    the two faces on the card, never on a "not a yes/no" answer, and one name
    for both on "same" (if two different names got stored, none is applied)."""
    if same is None:
        return {}
    raw = detail.get("names") if isinstance(detail.get("names"), dict) else {}
    names = {iid: _clean_name(raw.get(iid)) for iid in pair if _clean_name(raw.get(iid))}
    if same and names:
        if len(set(names.values())) != 1:
            return {}
        name = next(iter(names.values()))
        names = {iid: name for iid in pair}
    return names


def _apply_pair_answer(conn, ask_id: str, ref: str, answer: Optional[str], at: datetime,
                       detail: Dict[str, Any], now: datetime) -> None:
    """One "are these the same person?" answer -> one vision_individual_pair_verdict
    row, with the face similarity at that moment (the calibration point).
    'same' links the two (no merge, no delete); 'different' keeps the pair from
    being asked again. A name typed on the card is applied as a typed name for
    that face (both faces on 'same'). With no name, 'same' carries a name
    across if exactly one of them has one."""
    from sqlalchemy import text

    pair = parse_pair_ref(ref)
    same = parse_yes_no(answer)
    if pair is None or same is None:
        logger.warning("pair_answer_not_applied ask_id=%s source_ref=%s answer=%r", ask_id, ref, answer)
        return
    a, b = pair
    inds = {r[0]: r for r in conn.execute(text(
        "SELECT individual_id, face_centroid, label FROM vision_individual WHERE individual_id = ANY(:ids)"),
        {"ids": [a, b]}).fetchall()}
    fa = inds.get(a)[1] if a in inds else None
    fb = inds.get(b)[1] if b in inds else None
    sim = cosine(list(fa), list(fb)) if fa and fb and len(fa) == len(fb) else None
    conn.execute(text(
        "INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, face_cosine, "
        "ask_id, decided_at) VALUES (:a, :b, :v, :c, :ask, :at) ON CONFLICT (individual_a, individual_b) "
        "DO UPDATE SET verdict=EXCLUDED.verdict, face_cosine=EXCLUDED.face_cosine, ask_id=EXCLUDED.ask_id, "
        "decided_at=EXCLUDED.decided_at"),
        {"a": a, "b": b, "v": "same" if same else "different",
         "c": None if sim is None else round(float(sim), 4), "ask": ask_id, "at": at})
    names = pair_names(pair, same, detail)
    logger.info("[VISION_PAIR_VERDICT] ask_id=%s a=%s b=%s verdict=%s face_cosine=%s names=%d",
                ask_id, a, b, "same" if same else "different", sim, len(names))
    if names:
        for iid in (a, b):
            if iid in names:
                _apply_typed_name(conn, iid, names[iid], ask_id, at, now)
    elif same:
        labels = {iid: inds[iid][2] for iid in inds if inds[iid][2]}
        if len(labels) == 1:
            (named, label), = labels.items()
            _propagate_label(conn, named, label, ask_id, at)


# The slow "who is this?" rule (10 sightings over 5 days) counts only its own
# asks against ask_daily_cap. orion_ask is shared: since 2026-10-06 the memory
# confirmation loop also opens cards there under its own cap (5 open), and
# face asks (FACE_RULE_REF, vision_same_individual) have confirm_ask_daily_cap.
ASKED_TODAY_SQL = (
    "SELECT count(*) FROM orion_ask WHERE source_kind = 'vision_individual' AND created_at >= :d "
    "AND NOT (evidence_refs @> CAST(:face AS jsonb)) "
    # A name Juniper typed on a walkway story card is not a question Orion asked (2026-10-18).
    "AND NOT coalesce(answer_detail ? 'story_id', false)"
)
CONFIRM_ASKED_TODAY_SQL = (
    "SELECT count(*) FROM orion_ask WHERE created_at >= :d AND (source_kind = :same "
    "OR (source_kind = 'vision_individual' AND evidence_refs @> CAST(:face AS jsonb)))"
)

# An earlier ask about the same subject blocks a new one: open, answered or
# dismissed for good; expired only for the cooldown.
_NOT_ASKED_SQL = (
    "NOT EXISTS (SELECT 1 FROM orion_ask a WHERE a.source_kind={kind} AND a.source_ref={ref} "
    "AND (a.status IN ('open','answered','dismissed') OR (a.status = 'expired' AND a.created_at > :cool)))"
)


def _insert_ask(conn, ask) -> bool:
    from sqlalchemy import text

    inserted = conn.execute(text("""
        INSERT INTO orion_ask (ask_id, asked_of, question, evidence_refs, image_ref, image_refs, status,
            created_at, expires_at, source_kind, source_ref)
        VALUES (:id, :of, :q, CAST(:ev AS jsonb), :img, CAST(:imgs AS jsonb), 'open', :c, :x, :sk, :sr)
        ON CONFLICT (source_kind, source_ref) WHERE status = 'open' DO NOTHING
        RETURNING ask_id
    """), {"id": ask.ask_id, "of": ask.asked_of, "q": ask.question, "ev": json.dumps(ask.evidence_refs),
           "img": ask.image_ref, "imgs": json.dumps(ask.image_refs), "c": ask.created_at, "x": ask.expires_at,
           "sk": ask.source_kind, "sr": ask.source_ref}).fetchone()
    return bool(inserted)


def _sighting_rows(conn, individual_id: str):
    from sqlalchemy import text

    return conn.execute(text(
        "SELECT sighting_id, started_at, thumb_ref, ended_at, zone_no_embed, face_thumb_ref "
        "FROM vision_individual_sighting "
        "WHERE individual_id=:i AND NOT stationary ORDER BY started_at DESC"), {"i": individual_id}).fetchall()


def _images(srows, now: datetime, cfg: IndividualsConfig, *, limit: int = 3, faces_only: bool = False) -> List[str]:
    return ask_image_refs([(row[2], row[3], bool(row[4]), row[5]) for row in srows], now=now,
                          thumb_retention_days=cfg.thumb_retention_days, ask_expiry_days=cfg.ask_expiry_days,
                          limit=limit, faces_only=faces_only)


def _open_asks(conn, now: datetime, tz: ZoneInfo, cfg: IndividualsConfig) -> List[dict]:
    """Face asks first (their own cap), then the slow 10-sightings rule (its cap)."""
    out = _open_face_asks(conn, now, tz, cfg)
    out.extend(_open_sighting_asks(conn, now, tz, cfg))
    return out


def _open_face_asks(conn, now: datetime, tz: ZoneInfo, cfg: IndividualsConfig) -> List[dict]:
    from sqlalchemy import text

    from orion.schemas.ask import OrionAskV1

    if cfg.ask_face_min_faces <= 0:
        return []
    day = local_day_start(now, tz)
    face_marker = json.dumps([FACE_RULE_REF])
    asked = conn.execute(text(CONFIRM_ASKED_TODAY_SQL),
                         {"d": day, "same": SOURCE_KIND_SAME, "face": face_marker}).scalar() or 0
    budget = remaining_ask_budget(daily_cap=cfg.confirm_ask_daily_cap, asked_today=int(asked))
    if budget <= 0:
        return []
    cool = now - timedelta(days=cfg.ask_cooldown_days)
    people = sorted(_PEOPLE_KINDS)
    out: List[dict] = []
    expires = now + timedelta(days=cfg.ask_expiry_days)

    # 1. "Are these the same person?" -- first, so the same person is not
    # asked "who is this?" twice under two ids.
    frows = conn.execute(text(
        "SELECT individual_id, kind, face_centroid, face_n, label FROM vision_individual "
        "WHERE face_centroid IS NOT NULL AND face_n >= :f AND NOT stationary AND kind = ANY(:k) "
        "AND (label IS NOT NULL OR last_seen_at > :recent)"),
        {"f": cfg.ask_face_min_faces, "k": people,
         "recent": now - timedelta(days=cfg.candidate_days)}).fetchall()
    cands = [FaceCandidate(r[0], r[1], tuple(r[2]), int(r[3] or 0), r[4]) for r in frows if r[2]]
    by_id = {c.individual_id: c for c in cands}
    skip = {r[0] for r in conn.execute(text(
        "SELECT source_ref FROM orion_ask WHERE source_kind=:k AND (status IN ('open','answered') "
        "OR (status IN ('expired','dismissed') AND created_at > :cool))"),
        {"k": SOURCE_KIND_SAME, "cool": cool}).fetchall()}
    verdicts = conn.execute(text(
        "SELECT individual_a, individual_b, verdict FROM vision_individual_pair_verdict")).fetchall()
    skip |= {pair_ref(r[0], r[1]) for r in verdicts}
    groups = same_groups((r[0], r[1]) for r in verdicts if r[2] == "same")
    for a, b, sim in same_person_pairs(cands, min_cosine=cfg.same_ask_min_face_cosine, skip_refs=skip,
                                       groups=groups):
        if len(out) >= budget:
            break
        faces = []
        for iid in (a, b):
            refs = _images(_sighting_rows(conn, iid), now, cfg, limit=1, faces_only=True)
            if refs:
                faces.append(refs[0])
        if len(faces) != 2 or faces[0] == faces[1]:
            continue  # the card must show both faces
        ask = OrionAskV1(
            ask_id=f"ask-{uuid.uuid4().hex}",
            question=same_person_question(face_n_a=by_id[a].face_n, face_n_b=by_id[b].face_n),
            evidence_refs=[f"individual:{a}", f"individual:{b}", f"face_cosine:{sim:.3f}"],
            image_ref=faces[0], image_refs=faces, created_at=now, expires_at=expires,
            source_kind=SOURCE_KIND_SAME, source_ref=pair_ref(a, b),
        )
        if _insert_ask(conn, ask):
            out.append(ask.model_dump(mode="json"))

    # 2. "Who is this?" for a face seen often enough, the same day. Not for
    # someone in an open "same person?" ask: the answer may make it one ask.
    if len(out) >= budget:
        return out
    rows = conn.execute(text(f"""
        SELECT i.individual_id, i.kind, i.face_n, i.sighting_count, i.label, i.stationary
        FROM vision_individual i
        WHERE i.label IS NULL AND NOT i.stationary AND i.face_n >= :f AND i.kind = ANY(:k)
          AND i.last_seen_at > :recent
          AND {_NOT_ASKED_SQL.format(kind="'vision_individual'", ref="i.individual_id")}
          AND NOT EXISTS (SELECT 1 FROM orion_ask p WHERE p.source_kind=:same AND p.status='open'
                          AND (p.source_ref LIKE i.individual_id || '|%' OR p.source_ref LIKE '%|' || i.individual_id))
        ORDER BY i.face_n DESC, i.individual_id LIMIT :b
    """), {"f": cfg.ask_face_min_faces, "k": people, "recent": now - timedelta(days=cfg.candidate_days),
           "cool": cool, "same": SOURCE_KIND_SAME, "b": budget - len(out)}).fetchall()
    # One "who is this?" per person: individuals Juniper said are the same
    # share an answer (_propagate_label), so a group already asked is skipped.
    asked_groups = {groups.get(r[0], r[0]) for r in conn.execute(text(
        "SELECT source_ref FROM orion_ask WHERE source_kind='vision_individual' "
        "AND status IN ('open','answered')")).fetchall()}
    for iid, kind, face_n, sc, label, stationary in rows:
        if len(out) >= budget:
            break
        if groups.get(iid, iid) in asked_groups:
            continue
        if not should_ask_by_face(label=label, kind=kind, face_n=face_n, stationary=bool(stationary),
                                  min_faces=cfg.ask_face_min_faces):
            continue
        srows = _sighting_rows(conn, iid)
        images = _images(srows, now, cfg)
        if not images or not images[0].startswith(THUMB_REF_PREFIX) or not any(
                row[5] == images[0] for row in srows):
            continue  # a face ask without the face is not worth Juniper's time
        ask = OrionAskV1(
            ask_id=f"ask-{uuid.uuid4().hex}",
            question=face_ask_question(face_n=int(face_n), sighting_count=len(srows) or int(sc)),
            evidence_refs=[FACE_RULE_REF] + [f"sighting:{row[0]}" for row in srows[:5]],
            image_ref=images[0], image_refs=images, created_at=now, expires_at=expires,
            source_kind=SOURCE_KIND_INDIVIDUAL, source_ref=iid,
        )
        if _insert_ask(conn, ask):
            out.append(ask.model_dump(mode="json"))
            asked_groups.add(groups.get(iid, iid))
    return out


def _open_sighting_asks(conn, now: datetime, tz: ZoneInfo, cfg: IndividualsConfig) -> List[dict]:
    from sqlalchemy import text

    from orion.schemas.ask import OrionAskV1

    asked_today = conn.execute(text(ASKED_TODAY_SQL), {"d": local_day_start(now, tz),
                                                       "face": json.dumps([FACE_RULE_REF])}).scalar() or 0
    budget = remaining_ask_budget(daily_cap=cfg.ask_daily_cap, asked_today=int(asked_today))
    if budget <= 0:
        return []
    # Only real visits count: stationary sightings (a prop) are excluded from
    # the counts, and an individual that is a fixture right now is not asked about.
    cands = conn.execute(text(f"""
        SELECT i.individual_id, i.kind, v.n, v.d, i.label FROM vision_individual i
        JOIN LATERAL (
            SELECT count(*) AS n, count(DISTINCT (s.started_at AT TIME ZONE :tz)::date) AS d
            FROM vision_individual_sighting s
            WHERE s.individual_id = i.individual_id AND NOT s.stationary
        ) v ON true
        WHERE i.label IS NULL AND NOT i.stationary AND i.sighting_count >= :n AND i.distinct_days >= :d
          AND v.n >= :n AND v.d >= :d
          AND {_NOT_ASKED_SQL.format(kind="'vision_individual'", ref="i.individual_id")}
        ORDER BY v.n DESC, i.individual_id LIMIT :b
    """), {"n": cfg.ask_min_sightings, "d": cfg.ask_min_days, "b": budget, "tz": str(tz.key),
           "cool": now - timedelta(days=cfg.ask_cooldown_days)}).fetchall()
    out: List[dict] = []
    for iid, kind, sc, dd, label in cands:
        if not should_ask(label=label, sighting_count=sc, distinct_days=dd,
                          min_sightings=cfg.ask_min_sightings, min_days=cfg.ask_min_days):
            continue
        srows = _sighting_rows(conn, iid)
        minutes = [r[1].astimezone(tz).hour * 60 + r[1].astimezone(tz).minute for r in srows]
        mean, r = circular_mean_minute(minutes)
        images = _images(srows, now, cfg)
        ask = OrionAskV1(
            ask_id=f"ask-{uuid.uuid4().hex}",
            question=ask_question(kind=kind, sighting_count=sc, distinct_days=dd, mean_minute=mean, r=r),
            evidence_refs=[f"sighting:{row[0]}" for row in srows[:5]],
            image_ref=images[0] if images else None, image_refs=images, created_at=now,
            expires_at=now + timedelta(days=cfg.ask_expiry_days),
            source_kind=SOURCE_KIND_INDIVIDUAL, source_ref=iid,
        )
        if _insert_ask(conn, ask):
            out.append(ask.model_dump(mode="json"))
    return out


def _prune(conn, now: datetime, cfg: IndividualsConfig, summary: dict) -> None:
    from sqlalchemy import text

    crop_cut = now - timedelta(days=cfg.crop_retention_days)
    sight_cut = now - timedelta(days=cfg.sighting_retention_days)
    summary["pruned_crops"] = conn.execute(text(
        "DELETE FROM vision_crop_observation WHERE crop_id IN (SELECT crop_id FROM vision_crop_observation "
        "WHERE observed_at < :c LIMIT 20000)"), {"c": crop_cut}).rowcount or 0
    summary["pruned_sightings"] = conn.execute(text(
        "DELETE FROM vision_individual_sighting WHERE sighting_id IN (SELECT sighting_id FROM "
        "vision_individual_sighting WHERE started_at < :c LIMIT 20000)"), {"c": sight_cut}).rowcount or 0
    # Unlabeled individuals nobody has seen for the whole sighting retention
    # go too. Labeled ones stay: Juniper named them.
    summary["pruned_individuals"] = conn.execute(text(
        "DELETE FROM vision_individual i WHERE i.label IS NULL AND i.last_seen_at < :c "
        "AND NOT EXISTS (SELECT 1 FROM orion_ask a WHERE a.status='open' AND ("
        "(a.source_kind='vision_individual' AND a.source_ref=i.individual_id) OR "
        "(a.source_kind='vision_same_individual' AND (a.source_ref LIKE i.individual_id || '|%' "
        "OR a.source_ref LIKE '%|' || i.individual_id))))"), {"c": sight_cut}).rowcount or 0
