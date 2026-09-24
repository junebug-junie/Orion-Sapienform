"""Individuals: "the same one again", from crop embeddings.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md ideas
1, 3 (ask opener + answer consumer), 6 (attention score), 9 (patio presence).

Each tick, per stream, this reads the new ``vision_crop_observation`` rows
since ``vision_individuals_cursor`` and:

1. **Clusters.** A crop with an embedding joins the nearest same-kind
   ``vision_individual`` centroid on its stream if cosine similarity is at or
   above the threshold, else opens a new individual. The centroid is a
   running mean of unit vectors, renormalized. Two crops from the same frame
   never join the same individual (two people side by side are two people).
2. **Groups sightings.** Consecutive observations of one individual no more
   than ``merge_gap_sec`` apart are one ``vision_individual_sighting``. Zone
   = the zone with the most observations; dwell = ended - started.
3. **Scores attention** on every touched sighting (app/vision_attention_score.py)
   and writes at most one ``vision_events`` row ``attention_worthy`` per sighting.
4. **No-embed zones (the patio).** Boxes in a no-embed zone
   (``vision_crop_observation.zone_no_embed``, set by the writer from
   ``config/vision_zones.yaml``) never become individuals. They only drive
   one ``substrate_embodied_presence`` row per no-embed zone,
   ``<stream>:<zone name>`` (count + state, ``no_embed_zone: true``), and
   only while the camera is producing census rows -- a dead camera leaves
   the row to go stale rather than writing a false "absent".

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

import json
import logging
import math
import uuid
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

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
    centroids move; new individuals append a row."""

    def __init__(self, individuals: Iterable[Individual]) -> None:
        self._ids: Dict[Tuple[str, str], List[str]] = {}
        self._rows: Dict[Tuple[str, str], List[np.ndarray]] = {}
        self._mat: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
        for ind in individuals:
            self.add(ind)

    @staticmethod
    def _unit(v: Sequence[float]) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(a))
        return a / n if n > 0 else a

    def add(self, ind: Individual) -> None:
        key = (ind.stream_id, ind.kind)
        self._ids.setdefault(key, []).append(ind.individual_id)
        self._rows.setdefault(key, []).append(self._unit(ind.centroid))
        self._mat[key] = None

    def update(self, ind: Individual) -> None:
        key = (ind.stream_id, ind.kind)
        i = self._ids[key].index(ind.individual_id)
        self._rows[key][i] = self._unit(ind.centroid)
        m = self._mat.get(key)
        if m is not None:
            m[i] = self._rows[key][i]

    def nearest(self, crop: "CropRow", exclude: set) -> Tuple[Optional[str], float]:
        key = (crop.stream_id, crop.label)
        ids = self._ids.get(key)
        if not ids:
            return None, -2.0
        dims = {r.shape[0] for r in self._rows[key]}
        e = self._unit(crop.embedding or ())
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


@dataclass
class BatchResult:
    individuals: Dict[str, Individual]
    sightings: Dict[str, Sighting]          # sighting_id -> sighting (touched only)
    new_individual_ids: List[str]
    patio_crops: List[CropRow]
    skipped_no_embedding: int
    assigned: int


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
) -> BatchResult:
    """Cluster crops into individuals and group them into sightings.

    ``individuals`` and ``latest_sightings`` (individual_id -> most recent
    sighting) are the prior state; copies are returned, inputs untouched.
    Crops are processed in ``observed_at`` order. A crop is a no-embed crop if
    its row says so (``zone_no_embed``) OR its zone name is in
    ``no_embed_zones`` -- either is enough.

    A late crop (observed before the individual's latest sighting ended, e.g.
    it landed after the cursor passed its time) joins that sighting only if it
    falls inside it (+- ``merge_gap_sec``); otherwise it is its own, older
    sighting and does not displace the latest one.
    """
    forbidden = set(no_embed_zones)
    inds = {k: replace(v, centroid=list(v.centroid)) for k, v in individuals.items()}
    latest = {k: replace(v, zone_counts=dict(v.zone_counts)) for k, v in latest_sightings.items()}
    touched: Dict[str, Sighting] = {}
    new_ids: List[str] = []
    patio: List[CropRow] = []
    skipped = 0
    assigned = 0
    used_in_observation: Dict[str, set] = {}
    index = _CentroidIndex(inds.values())

    for crop in sorted(crops, key=lambda c: (c.observed_at, c.crop_id)):
        if crop.zone_no_embed or crop.zone in forbidden:
            patio.append(crop)       # never an individual, never an embedding
            continue
        if not crop.embedding:
            skipped += 1
            continue
        taken = used_in_observation.setdefault(crop.observation_id, set())
        best_id, best_sim = index.nearest(crop, taken)

        if best_id is not None and best_sim >= match_threshold:
            ind = inds[best_id]
            if local_date(crop.observed_at, tz) != local_date(ind.last_seen_at, tz) and crop.observed_at > ind.last_seen_at:
                ind.distinct_days += 1
            ind.centroid = update_centroid(ind.centroid, ind.centroid_n, crop.embedding)
            ind.centroid_n += 1
            index.update(ind)
            ind.last_seen_at = max(ind.last_seen_at, crop.observed_at)
        else:
            ind = Individual(
                individual_id=new_id(), stream_id=crop.stream_id, kind=crop.label,
                centroid=normalize(crop.embedding), centroid_n=1,
                first_seen_at=crop.observed_at, last_seen_at=crop.observed_at,
                sighting_count=0, distinct_days=1,
            )
            inds[ind.individual_id] = ind
            new_ids.append(ind.individual_id)
            index.add(ind)
        taken.add(ind.individual_id)
        assigned += 1

        zone_key = crop.zone or _NO_ZONE
        prev = latest.get(ind.individual_id)
        within = prev is not None and (
            (prev.started_at - crop.observed_at).total_seconds() <= merge_gap_sec
            and (crop.observed_at - prev.ended_at).total_seconds() <= merge_gap_sec
        )
        if within:
            newest = crop.observed_at >= prev.ended_at
            prev.started_at = min(prev.started_at, crop.observed_at)
            prev.ended_at = max(prev.ended_at, crop.observed_at)
            prev.observation_count += 1
            prev.zone_counts[zone_key] = prev.zone_counts.get(zone_key, 0) + 1
            if newest:
                prev.last_box_xyxy = list(crop.box_xyxy)
                prev.embedding_ref = crop.embedding_ref
                prev.evidence_ref = f"crop:{crop.crop_id}"
            if crop.thumb_ref and (newest or not prev.thumb_ref):
                prev.thumb_ref = crop.thumb_ref
            s = prev
        else:
            s = Sighting(
                sighting_id=new_id(), individual_id=ind.individual_id, stream_id=crop.stream_id,
                started_at=crop.observed_at, ended_at=crop.observed_at, observation_count=1,
                zone_counts={zone_key: 1}, last_box_xyxy=list(crop.box_xyxy),
                embedding_ref=crop.embedding_ref,
                evidence_ref=f"crop:{crop.crop_id}",
                thumb_ref=crop.thumb_ref,
            )
            # A late crop older than the latest sighting is its own, earlier
            # sighting; it must not become "latest" and swallow newer crops.
            if prev is None or s.ended_at >= prev.ended_at:
                latest[ind.individual_id] = s
            ind.sighting_count += 1
        touched[s.sighting_id] = s

    return BatchResult(
        individuals=inds, sightings=touched, new_individual_ids=new_ids,
        patio_crops=patio, skipped_no_embedding=skipped, assigned=assigned,
    )


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
    """Same keys as orion-vision-window's presence snapshot, plus the patio count.

    ``subject`` is ``{"count": n}`` -- how many, never who.
    """
    prev = prev or {}
    last_seen_at = batch_last_seen_at
    count = batch_count
    if last_seen_at is None and prev.get("last_seen_at"):
        last_seen_at = datetime.fromisoformat(prev["last_seen_at"])
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
        # Readers find these rows by this flag, not by the zone's name.
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


def ask_image_ref(
    sightings: Sequence[Tuple[Optional[str], datetime, bool]],
    *,
    now: datetime,
    thumb_retention_days: float,
    ask_expiry_days: float,
) -> Optional[str]:
    """The thumbnail for an ask: the individual's most recent crop thumbnail.

    ``sightings`` are ``(thumb_ref, ended_at, zone_no_embed)``, any order. A
    thumbnail is a crop of ONE embedded box (orion-vision-host never writes
    one for a no-embed zone), never the whole frame, so an ask cannot show
    Juniper the patio. It must also outlive the ask: the host prunes
    thumbnails ``thumb_retention_days`` after they were last written, and the
    ask stays open ``ask_expiry_days``, so only a thumb from a sighting that
    ended within the difference is offered. None -> the card shows no picture.
    """
    horizon = now - timedelta(days=max(0.0, thumb_retention_days - ask_expiry_days))
    best: Optional[Tuple[datetime, str]] = None
    for thumb_ref, ended_at, no_embed in sightings:
        if no_embed or not thumb_ref or not str(thumb_ref).startswith(THUMB_REF_PREFIX):
            continue
        if ended_at is None or ended_at < horizon:
            continue
        if best is None or ended_at > best[0]:
            best = (ended_at, str(thumb_ref))
    return best[1] if best else None


def individual_display_label(ind_label: Optional[str], kind: str, individual_id: str) -> str:
    return ind_label or f"{kind} #{individual_id[:6]}"


# ---------------------------------------------------------------------------
# Postgres. Blocking; run in a worker thread.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndividualsConfig:
    match_threshold: float = 0.80
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
    # file age). An ask only shows a thumbnail young enough to outlive the
    # ask itself: taken within (thumb_retention_days - ask_expiry_days).
    thumb_retention_days: float = 14.0


# Two sql-writer instances must never run the same cycle at once: cursors,
# counts and the ask cap all assume a single writer.
INDIVIDUALS_LOCK_KEY = 0x0A1C_0001


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
            # Patio presence must be able to decay to "absent" on a quiet street.
            streams |= {s for s, zs in zones_by_stream.items() if any(not z.embed for z in zs)}

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


def _row_to_crop(r) -> CropRow:
    emb = r.embedding
    return CropRow(
        crop_id=r.crop_id, observation_id=r.observation_id, stream_id=r.stream_id,
        observed_at=r.observed_at, label=r.label, box_xyxy=tuple(r.box_xyxy or ()),
        zone=r.zone, embedding=tuple(emb) if emb else None,
        embedding_ref=r.embedding_ref, artifact_id=r.artifact_id,
        zone_no_embed=bool(r.zone_no_embed), thumb_ref=r.thumb_ref,
    )


def _one_stream(engine, stream_id: str, now: datetime, tz: ZoneInfo, cfg: IndividualsConfig,
                zones: List[Zone], summary: dict) -> None:
    from sqlalchemy import text

    forbidden = {z.name for z in zones if not z.embed}
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
            "embedding, embedding_ref, artifact_id, zone_no_embed, thumb_ref, created_at "
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
                "sighting_count, distinct_days, label FROM vision_individual "
                "WHERE stream_id=:s AND kind = ANY(:k) AND (label IS NOT NULL OR last_seen_at > :recent)"),
                {"s": stream_id, "k": kinds,
                 "recent": oldest_seen - timedelta(days=cfg.candidate_days)}).fetchall()
            individuals = {r.individual_id: Individual(
                individual_id=r.individual_id, stream_id=r.stream_id, kind=r.kind,
                centroid=list(r.centroid), centroid_n=r.centroid_n, first_seen_at=r.first_seen_at,
                last_seen_at=r.last_seen_at, sighting_count=r.sighting_count,
                distinct_days=r.distinct_days, label=r.label) for r in ind_rows}
            s_rows = conn.execute(text(
                "SELECT DISTINCT ON (individual_id) sighting_id, individual_id, stream_id, started_at, ended_at, "
                "observation_count, zone_counts, last_box_xyxy, embedding_ref, evidence_ref, thumb_ref "
                "FROM vision_individual_sighting WHERE stream_id=:s AND ended_at >= :after "
                "ORDER BY individual_id, ended_at DESC"),
                {"s": stream_id, "after": oldest_seen - timedelta(seconds=cfg.merge_gap_sec)}).fetchall()
            latest = {r.individual_id: Sighting(
                sighting_id=r.sighting_id, individual_id=r.individual_id, stream_id=r.stream_id,
                started_at=r.started_at, ended_at=r.ended_at, observation_count=r.observation_count,
                zone_counts=dict(r.zone_counts or {}), last_box_xyxy=list(r.last_box_xyxy or []) or None,
                embedding_ref=r.embedding_ref, evidence_ref=r.evidence_ref,
                thumb_ref=r.thumb_ref) for r in s_rows}

            batch = apply_batch(
                crops, individuals=individuals, latest_sightings=latest, no_embed_zones=forbidden,
                match_threshold=cfg.match_threshold, merge_gap_sec=cfg.merge_gap_sec, tz=tz,
            )
            summary["assigned"] += batch.assigned
            summary["new_individuals"] += len(batch.new_individual_ids)
            summary["sightings_touched"] += len(batch.sightings)
            changed_ids = {s.individual_id for s in batch.sightings.values()}
            for iid in changed_ids:
                _upsert_individual(conn, batch.individuals[iid])
            for s in batch.sightings.values():
                _upsert_sighting(conn, s, forbidden)
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
        # One presence row per no-embed zone: every zone the config names (so
        # a quiet one can decay to absent) plus any a flagged row names.
        zone_names = set(forbidden) | {c.zone for c in no_embed_crops if c.zone}
        for zone_name in sorted(zone_names):
            # Its own savepoint: a missing presence table (a different
            # migration) must not abort clustering, asks, or pruning.
            try:
                with conn.begin_nested():
                    in_zone = [c for c in no_embed_crops if c.zone == zone_name]
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
            last_seen_at, sighting_count, distinct_days, updated_at)
        VALUES (:id, :s, :k, :c, :n, :f, :l, :sc, :dd, now())
        ON CONFLICT (individual_id) DO UPDATE SET centroid=EXCLUDED.centroid, centroid_n=EXCLUDED.centroid_n,
            last_seen_at=EXCLUDED.last_seen_at, sighting_count=EXCLUDED.sighting_count,
            distinct_days=EXCLUDED.distinct_days, updated_at=now()
    """), {"id": ind.individual_id, "s": ind.stream_id, "k": ind.kind, "c": ind.centroid,
           "n": ind.centroid_n, "f": ind.first_seen_at, "l": ind.last_seen_at,
           "sc": ind.sighting_count, "dd": ind.distinct_days})


def sighting_row(s: Sighting, no_embed_zones: Iterable[str]) -> Dict[str, Any]:
    """Pure: the upsert parameters. A sighting whose dominant zone is a
    no-embed zone (cannot happen from apply_batch, which never clusters those
    crops -- this is the belt to the DB CHECK's braces) carries no refs."""
    no_embed = s.zone in set(no_embed_zones)
    return {"id": s.sighting_id, "iid": s.individual_id, "s": s.stream_id, "z": s.zone,
            "st": s.started_at, "en": s.ended_at, "dw": s.dwell_sec, "oc": s.observation_count,
            "box": s.last_box_xyxy, "er": None if no_embed else s.embedding_ref, "ev": s.evidence_ref,
            "th": None if no_embed else s.thumb_ref, "ne": no_embed,
            "zc": json.dumps(s.zone_counts)}


def _upsert_sighting(conn, s: Sighting, no_embed_zones: Iterable[str] = ()) -> None:
    from sqlalchemy import text

    conn.execute(text("""
        INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, zone, started_at, ended_at,
            dwell_sec, observation_count, last_box_xyxy, embedding_ref, evidence_ref, thumb_ref, zone_no_embed,
            zone_counts, updated_at)
        VALUES (:id, :iid, :s, :z, :st, :en, :dw, :oc, :box, :er, :ev, :th, :ne, CAST(:zc AS jsonb), now())
        ON CONFLICT (sighting_id) DO UPDATE SET zone=EXCLUDED.zone, started_at=EXCLUDED.started_at,
            ended_at=EXCLUDED.ended_at,
            dwell_sec=EXCLUDED.dwell_sec, observation_count=EXCLUDED.observation_count,
            last_box_xyxy=EXCLUDED.last_box_xyxy, embedding_ref=EXCLUDED.embedding_ref,
            evidence_ref=EXCLUDED.evidence_ref, thumb_ref=EXCLUDED.thumb_ref,
            zone_no_embed=EXCLUDED.zone_no_embed, zone_counts=EXCLUDED.zone_counts, updated_at=now()
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


def _apply_answers(conn, now: datetime) -> int:
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT ask_id, source_ref, answer, answered_at FROM orion_ask WHERE source_kind='vision_individual' "
        "AND status='answered' AND applied_at IS NULL")).fetchall()
    for ask_id, individual_id, answer, answered_at in rows:
        label = (answer or "").strip()
        if label:
            n = conn.execute(text(
                "UPDATE vision_individual SET label=:l, labeled_at=:at, label_ask_id=:a, updated_at=now() "
                "WHERE individual_id=:i"), {"l": label, "at": answered_at or now, "a": ask_id, "i": individual_id}).rowcount
            if not n:
                logger.warning("ask_answer_for_missing_individual ask_id=%s individual_id=%s", ask_id, individual_id)
        conn.execute(text("UPDATE orion_ask SET applied_at=:now WHERE ask_id=:a"), {"now": now, "a": ask_id})
    return len(rows)


def _open_asks(conn, now: datetime, tz: ZoneInfo, cfg: IndividualsConfig) -> List[dict]:
    from sqlalchemy import text

    from orion.schemas.ask import OrionAskV1

    asked_today = conn.execute(text("SELECT count(*) FROM orion_ask WHERE created_at >= :d"),
                               {"d": local_day_start(now, tz)}).scalar() or 0
    budget = remaining_ask_budget(daily_cap=cfg.ask_daily_cap, asked_today=int(asked_today))
    if budget <= 0:
        return []
    cands = conn.execute(text("""
        SELECT i.individual_id, i.kind, i.sighting_count, i.distinct_days, i.label FROM vision_individual i
        WHERE i.label IS NULL AND i.sighting_count >= :n AND i.distinct_days >= :d
          AND NOT EXISTS (SELECT 1 FROM orion_ask a WHERE a.source_kind='vision_individual'
                          AND a.source_ref=i.individual_id
                          AND (a.status IN ('open','answered','dismissed')
                               OR (a.status = 'expired' AND a.created_at > :cool)))
        ORDER BY i.sighting_count DESC, i.individual_id LIMIT :b
    """), {"n": cfg.ask_min_sightings, "d": cfg.ask_min_days, "b": budget,
           "cool": now - timedelta(days=cfg.ask_cooldown_days)}).fetchall()
    out: List[dict] = []
    for iid, kind, sc, dd, label in cands:
        if not should_ask(label=label, sighting_count=sc, distinct_days=dd,
                          min_sightings=cfg.ask_min_sightings, min_days=cfg.ask_min_days):
            continue
        srows = conn.execute(text(
            "SELECT sighting_id, started_at, thumb_ref, ended_at, zone_no_embed FROM vision_individual_sighting "
            "WHERE individual_id=:i ORDER BY started_at DESC"), {"i": iid}).fetchall()
        minutes = [r[1].astimezone(tz).hour * 60 + r[1].astimezone(tz).minute for r in srows]
        mean, r = circular_mean_minute(minutes)
        image_ref = ask_image_ref(
            [(row[2], row[3], bool(row[4])) for row in srows], now=now,
            thumb_retention_days=cfg.thumb_retention_days, ask_expiry_days=cfg.ask_expiry_days)
        ask = OrionAskV1(
            ask_id=f"ask-{uuid.uuid4().hex}",
            question=ask_question(kind=kind, sighting_count=sc, distinct_days=dd, mean_minute=mean, r=r),
            evidence_refs=[f"sighting:{row[0]}" for row in srows[:5]],
            image_ref=image_ref, created_at=now, expires_at=now + timedelta(days=cfg.ask_expiry_days),
            source_kind="vision_individual", source_ref=iid,
        )
        inserted = conn.execute(text("""
            INSERT INTO orion_ask (ask_id, asked_of, question, evidence_refs, image_ref, status, created_at,
                expires_at, source_kind, source_ref)
            VALUES (:id, :of, :q, CAST(:ev AS jsonb), :img, 'open', :c, :x, :sk, :sr)
            ON CONFLICT (source_kind, source_ref) WHERE status = 'open' DO NOTHING
            RETURNING ask_id
        """), {"id": ask.ask_id, "of": ask.asked_of, "q": ask.question, "ev": json.dumps(ask.evidence_refs),
               "img": ask.image_ref, "c": ask.created_at, "x": ask.expires_at, "sk": ask.source_kind,
               "sr": ask.source_ref}).fetchone()
        if inserted:
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
        "AND NOT EXISTS (SELECT 1 FROM orion_ask a WHERE a.source_kind='vision_individual' "
        "AND a.source_ref=i.individual_id AND a.status='open')"), {"c": sight_cut}).rowcount or 0
