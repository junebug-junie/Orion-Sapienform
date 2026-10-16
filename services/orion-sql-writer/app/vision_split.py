"""Split a walkway individual on Juniper's "someone else" picture answers.

docs/superpowers/specs/2026-10-16-vision-individual-split-proposal.md, the
recommended next patch. A "who is this?" card shows several pictures; each
picture verdict is a ``vision_ask_picture_verdict`` row (written by
``vision_individuals._record_pictures``). For every 'different' picture whose
crop is still in the asked individual, this:

1. **Groups** pictures by typed name (case-insensitive); unnamed 'different'
   pictures of one ask form one group. One split per (individual, ask, group).
2. **Moves** the named crops plus member FACE crops that pass both follow
   conditions (``plan_split``). Body-only crops never follow: at this camera
   distance the body embedding cannot tell these people apart.
3. **Keeps identity on the confirmed side.** The asked individual keeps its
   id, label and substrate entity node. The moved crops get a new id.
4. **Rewrites** both individuals' centroids/counts and the source's affected
   sightings (count, zones, and pictures that pointed at a moved crop), and
   gives the new individual its own sightings.
5. **Links.** A 'different' pair verdict (cannot-link) from the new
   individual to the source, always; with a typed name, the new individual
   gets it as a typed label and a 'same' link to an existing individual with
   that name (no crop merge).
6. **Records** the exact before/after in ``vision_individual_split`` so that
   ``undo_split`` (scripts/undo_vision_split.py) can reverse it.

Stored membership (``vision_crop_observation.individual_id``) starts with
manual_migration_vision_split_v1.sql. A crop older than that has NULL: it is
taken as a member when a verdict names it (Juniper looked at it on this
individual's card), but it can never be a face FOLLOWER -- nothing says which
individual it belongs to. Without a backfill, today's splits move only the
pictures she named.

Pure functions first; the DB pass runs in the individuals cycle's own
transaction (single writer: the cycle's advisory lock).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from app.vision_individuals import (
    IndividualsConfig,
    _set_typed_label,
    cosine,
    local_date,
    normalize,
    parse_pair_ref,
    same_groups,
)

logger = logging.getLogger("sql-writer.vision_split")

MOVED_BY_VERDICT = "verdict"
MOVED_BY_FACE = "face_follow"


# ---------------------------------------------------------------------------
# Pure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitCrop:
    crop_id: str
    observed_at: datetime
    embedding: Optional[Tuple[float, ...]] = None
    face_embedding: Optional[Tuple[float, ...]] = None


@dataclass
class SplitPlan:
    moved: List[Tuple[str, str]] = field(default_factory=list)   # (crop_id, moved_by)
    refused: Optional[str] = None
    followers_dropped: bool = False

    @property
    def by_verdict(self) -> int:
        return sum(1 for _, m in self.moved if m == MOVED_BY_VERDICT)

    @property
    def by_face(self) -> int:
        return sum(1 for _, m in self.moved if m == MOVED_BY_FACE)


def mean_unit(vecs: Iterable[Sequence[float]]) -> Optional[List[float]]:
    """Sum of unit vectors, renormalized. None for no vectors."""
    vs = [normalize(v) for v in vecs if v]
    if not vs:
        return None
    dim = len(vs[0])
    vs = [v for v in vs if len(v) == dim]
    return normalize([sum(v[i] for v in vs) for i in range(dim)])


def remove_from_centroid(centroid: Sequence[float], n: int, removed: Sequence[Sequence[float]]) -> Optional[List[float]]:
    """Take ``removed`` out of a running-mean centroid of ``n`` unit vectors
    (the inverse of ``vision_individuals.update_centroid``'s step). Used only
    when the individual's members are not all stored; None when nothing is left."""
    left = int(n) - len(removed)
    if left <= 0 or not centroid:
        return None
    c = normalize(centroid)
    acc = [ci * n for ci in c]
    for e in removed:
        u = normalize(e)
        if len(u) != len(acc):
            continue
        acc = [a - ui for a, ui in zip(acc, u)]
    return normalize(acc)


def add_to_centroid(centroid: Optional[Sequence[float]], n: int, added: Sequence[Sequence[float]]) -> Optional[List[float]]:
    """Put ``added`` back into a running-mean centroid of ``n`` unit vectors."""
    if not added:
        return list(centroid) if centroid else None
    if not centroid or n <= 0:
        return mean_unit(added)
    acc = [ci * n for ci in normalize(centroid)]
    for e in added:
        u = normalize(e)
        if len(u) == len(acc):
            acc = [a + ui for a, ui in zip(acc, u)]
    return normalize(acc)


def plan_split(
    *,
    source_n: int,
    move_seed: Sequence[SplitCrop],
    keep_seed: Sequence[SplitCrop],
    members: Sequence[SplitCrop],
    source_face_centroid: Optional[Sequence[float]] = None,
    follow_min_face_cosine: float = 0.60,
    follow_margin: float = 0.10,
    max_move_fraction: float = 0.5,
    exclude: Iterable[str] = (),
) -> SplitPlan:
    """Which crops leave the source.

    (a) Every ``move_seed`` crop (the pictures Juniper called someone else).
    (b) A ``members`` crop WITH a face, when the move seed has a face, its face
        cosine to the move seed's face centroid is >= ``follow_min_face_cosine``
        and beats the keep seed's face centroid (the confirmed pictures; the
        source's own face centroid if none of them has a face) by >=
        ``follow_margin``. Body-only crops never follow.

    Refused (``refused='majority'``) when more than ``max_move_fraction`` of
    the source's ``source_n`` crops would move: identity follows Juniper's
    answer, never a body-embedding majority."""
    plan = SplitPlan(moved=[(c.crop_id, MOVED_BY_VERDICT) for c in move_seed])
    move_face = mean_unit(c.face_embedding for c in move_seed if c.face_embedding)
    if move_face is not None and follow_min_face_cosine > 0:
        keep_face = mean_unit(c.face_embedding for c in keep_seed if c.face_embedding)
        if keep_face is None and source_face_centroid:
            keep_face = list(source_face_centroid)
        skip = {c.crop_id for c in move_seed} | {c.crop_id for c in keep_seed} | set(exclude)
        for c in sorted(members, key=lambda m: (m.observed_at, m.crop_id)):
            if c.crop_id in skip or not c.face_embedding or len(c.face_embedding) != len(move_face):
                continue
            s_move = cosine(c.face_embedding, move_face)
            s_keep = (cosine(c.face_embedding, keep_face)
                      if keep_face and len(keep_face) == len(c.face_embedding) else -1.0)
            if s_move >= follow_min_face_cosine and s_move - s_keep >= follow_margin:
                plan.moved.append((c.crop_id, MOVED_BY_FACE))
                skip.add(c.crop_id)
    limit = max_move_fraction * source_n
    if source_n > 0 and len(plan.moved) > limit:
        if plan.by_face and plan.by_verdict <= limit:
            # Only Orion's own inference pushed it over: move just what Juniper named.
            plan.moved = [m for m in plan.moved if m[1] == MOVED_BY_VERDICT]
            plan.followers_dropped = True
        else:
            plan.refused = "majority"
    return plan


@dataclass(frozen=True)
class PictureVerdict:
    picture_index: int
    verdict: str
    name: Optional[str]
    crop_id: Optional[str]
    group_key: str = ""


def verdict_groups(rows: Sequence[PictureVerdict]) -> Tuple[List[str], Dict[str, Tuple[Optional[str], List[str]]]]:
    """(keep crop ids, {group_key: (display name, move crop ids)}) for one ask.
    Keep = the 'same' pictures. One group per typed name (case-insensitive key,
    the first spelling is kept); every unnamed 'different' picture shares the
    '' group. 'unsure' pictures are neither."""
    keep: List[str] = []
    groups: Dict[str, Tuple[Optional[str], List[str]]] = {}
    for r in sorted(rows, key=lambda x: x.picture_index):
        if r.verdict == "same" and r.crop_id:
            keep.append(r.crop_id)
        elif r.verdict == "different":
            name, ids = groups.get(r.group_key, (r.name or None, []))
            if r.crop_id:
                ids.append(r.crop_id)
            groups[r.group_key] = (name, ids)
    return keep, groups


def group_sightings(crops: Sequence[Dict[str, Any]], *, individual_id: str, merge_gap_sec: float,
                    new_id=lambda: uuid.uuid4().hex) -> List[Dict[str, Any]]:
    """The new individual's sightings: its crops in observed_at order, a gap
    longer than ``merge_gap_sec`` starts a new one (apply_batch's rule).
    Pictures are the newest member's that has one."""
    out: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for c in sorted(crops, key=lambda x: (x["observed_at"], x["crop_id"])):
        if cur is None or (c["observed_at"] - cur["ended_at"]).total_seconds() > merge_gap_sec:
            cur = {"sighting_id": new_id(), "individual_id": individual_id, "stream_id": c["stream_id"],
                   "started_at": c["observed_at"], "ended_at": c["observed_at"], "observation_count": 0,
                   "zone_counts": {}, "last_box_xyxy": None, "embedding_ref": None, "evidence_ref": None,
                   "thumb_ref": None, "face_thumb_ref": None}
            out.append(cur)
        cur["ended_at"] = c["observed_at"]
        cur["observation_count"] += 1
        z = c.get("zone") or ""
        cur["zone_counts"][z] = cur["zone_counts"].get(z, 0) + 1
        cur["last_box_xyxy"] = list(c.get("box_xyxy") or []) or None
        cur["embedding_ref"] = c.get("embedding_ref")
        cur["evidence_ref"] = f"crop:{c['crop_id']}"
        if c.get("thumb_ref"):
            cur["thumb_ref"] = c["thumb_ref"]
        if c.get("face_thumb_ref"):
            cur["face_thumb_ref"] = c["face_thumb_ref"]
    for s in out:
        zc = s["zone_counts"]
        best = max(zc.items(), key=lambda kv: (kv[1], kv[0] != "")) if zc else ("", 0)
        s["zone"] = best[0] or None
        s["dwell_sec"] = max(0.0, (s["ended_at"] - s["started_at"]).total_seconds())
    return out


def face_threshold_calibration(same: Sequence[float], different: Sequence[float], *, min_pairs: int = 20,
                               max_false_accept: float = 0.05, min_recall: float = 0.50,
                               pad: float = 0.02) -> Dict[str, Any]:
    """The split proposal's rule for moving VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD.

    'propose' only with >= ``min_pairs`` labelled 'same' AND 'different' face
    pairs, and a threshold t that lets through at most ``max_false_accept`` of
    the 'different' pairs (cosine >= t) while keeping >= ``min_recall`` of the
    'same' pairs; the proposal is the smallest such t plus ``pad``. Otherwise
    'hold'. Nothing writes the threshold: a proposal goes to Juniper."""
    s, d = sorted(float(x) for x in same), sorted(float(x) for x in different)
    out: Dict[str, Any] = {"same_pairs": len(s), "different_pairs": len(d),
                           "min_same": s[0] if s else None, "max_different": d[-1] if d else None,
                           "overlap": bool(s and d and d[-1] >= s[0])}
    if len(s) < min_pairs or len(d) < min_pairs:
        return {**out, "verdict": "hold",
                "reason": f"need >= {min_pairs} labelled pairs of each kind (same={len(s)}, different={len(d)})"}
    for t in sorted(set(s) | set(d)):
        fa = sum(1 for x in d if x >= t) / len(d)
        recall = sum(1 for x in s if x >= t) / len(s)
        if fa <= max_false_accept and recall >= min_recall:
            return {**out, "verdict": "propose", "threshold": round(t + pad, 3),
                    "false_accept": round(fa, 3), "recall": round(recall, 3)}
    return {**out, "verdict": "hold", "reason": "no threshold separates the labelled pairs"}


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

_CROP_COLS = ("crop_id, stream_id, observed_at, label, individual_id, embedding, face_embedding, zone, "
              "box_xyxy, thumb_ref, face_thumb_ref, embedding_ref")


def reanchored_body(conn, individual_id: str, rows: Sequence[Any]) -> Optional[List[Any]]:
    """After scripts/reanchor_vision_individuals.py rewrote this individual's
    body centroid, only some member crops are part of it: the run's
    ``crop_ids`` and crops that landed after the run. Returns the subset of
    ``rows`` that is, or None when no re-anchor is in force (none recorded,
    the latest one was undone, or the audit table does not exist)."""
    from sqlalchemy import text

    if not conn.execute(text("SELECT to_regclass('vision_individual_reanchor') IS NOT NULL")).scalar():
        return None
    last = conn.execute(text(
        "SELECT action, created_at, crop_ids FROM vision_individual_reanchor WHERE individual_id=:i "
        "ORDER BY created_at DESC LIMIT 1"), {"i": individual_id}).fetchone()
    if last is None or last.action != "reanchor":
        return None
    basis = set(last.crop_ids or [])
    ids = [r.crop_id for r in rows]
    landed = {r[0]: r[1] for r in conn.execute(text(
        "SELECT crop_id, created_at FROM vision_crop_observation WHERE crop_id = ANY(:c)"), {"c": ids}).fetchall()}
    return [r for r in rows if r.crop_id in basis
            or (landed.get(r.crop_id) is not None and landed[r.crop_id] > last.created_at)]


def _jsonb_row(conn, table: str, key: str, value: str) -> Optional[Dict[str, Any]]:
    from sqlalchemy import text

    r = conn.execute(text(f"SELECT to_jsonb(t) FROM {table} t WHERE {key} = :v"), {"v": value}).fetchone()
    return None if r is None else r[0]


def _as_split_crop(r) -> SplitCrop:
    return SplitCrop(crop_id=r.crop_id, observed_at=r.observed_at,
                     embedding=tuple(r.embedding) if r.embedding else None,
                     face_embedding=tuple(r.face_embedding) if r.face_embedding else None)


def _record(conn, *, split_id: str, source: str, ask_id: str, group_key: str, name: Optional[str],
            status: str, reason: Optional[str] = None, new_id: Optional[str] = None,
            moved: Sequence[dict] = (), before: Any = None, after: Any = None,
            linked: Sequence[dict] = (), out_of_retention: int = 0) -> bool:
    from sqlalchemy import text

    return bool(conn.execute(text(
        "INSERT INTO vision_individual_split (split_id, source_individual_id, new_individual_id, ask_id, "
        "group_key, name, status, reason, moved, before, after, linked_pair, sightings_out_of_retention) "
        "VALUES (:sid, :src, :new, :ask, :gk, :name, :st, :reason, CAST(:moved AS jsonb), CAST(:before AS jsonb), "
        "CAST(:after AS jsonb), CAST(:linked AS jsonb), :oor) "
        "ON CONFLICT (source_individual_id, ask_id, group_key) DO NOTHING RETURNING split_id"),
        {"sid": split_id, "src": source, "new": new_id, "ask": ask_id, "gk": group_key, "name": name,
         "st": status, "reason": reason, "moved": json.dumps(list(moved)),
         "before": None if before is None else json.dumps(before, default=str),
         "after": None if after is None else json.dumps(after, default=str),
         "linked": json.dumps(list(linked)), "oor": int(out_of_retention)}).fetchone())


def pending_split_groups(conn) -> List[Tuple[str, str, str, List[PictureVerdict], datetime]]:
    """(ask_id, source individual, group_key, all picture verdicts of that ask,
    answered_at) for every 'different' group with no vision_individual_split row yet."""
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT v.ask_id, v.individual_id, v.picture_index, v.verdict, v.name, v.crop_id, v.answered_at, "
        "lower(coalesce(v.name, '')) AS gk FROM vision_ask_picture_verdict v "
        "WHERE v.ask_id IN (SELECT d.ask_id FROM vision_ask_picture_verdict d WHERE d.verdict = 'different' "
        "  AND NOT EXISTS (SELECT 1 FROM vision_individual_split s WHERE s.source_individual_id = d.individual_id "
        "  AND s.ask_id = d.ask_id AND s.group_key = lower(coalesce(d.name, '')))) "
        "ORDER BY v.answered_at, v.ask_id, v.picture_index")).fetchall()
    by_ask: Dict[Tuple[str, str], List[Any]] = {}
    for r in rows:
        by_ask.setdefault((r.ask_id, r.individual_id), []).append(r)
    out = []
    for (ask_id, src), rs in by_ask.items():
        pvs = [PictureVerdict(r.picture_index, r.verdict, r.name, r.crop_id,
                              r.gk if r.verdict == "different" else "") for r in rs]
        done = {x[0] for x in conn.execute(text(
            "SELECT group_key FROM vision_individual_split WHERE source_individual_id=:s AND ask_id=:a"),
            {"s": src, "a": ask_id}).fetchall()}
        for gk in sorted({p.group_key for p in pvs if p.verdict == "different"} - done):
            out.append((ask_id, src, gk, pvs, rs[0].answered_at))
    return out


def apply_splits(conn, now: datetime, cfg: IndividualsConfig) -> Dict[str, int]:
    """Every pending 'different' picture group -> one split / refusal / skip
    row. Each in its own savepoint: one failure never blocks the others (it is
    retried next cycle, since it left no row)."""
    out = {"splits": 0, "splits_refused": 0, "splits_skipped": 0}
    if not cfg.split_on_verdict_enabled:
        return out
    tz = ZoneInfo(cfg.local_tz)
    for ask_id, src, gk, pvs, answered_at in pending_split_groups(conn):
        try:
            with conn.begin_nested():
                status = split_one(conn, ask_id=ask_id, source_id=src, group_key=gk, verdicts=pvs,
                                   answered_at=answered_at or now, now=now, cfg=cfg, tz=tz)
            out[{"split": "splits", "refused": "splits_refused"}.get(status, "splits_skipped")] += 1
        except Exception as exc:
            logger.warning("vision_split_failed ask_id=%s source=%s group=%r error=%s", ask_id, src, gk, exc)
            # A visible end state: one 'skipped' row, never a retry every tick.
            try:
                with conn.begin_nested():
                    _record(conn, split_id=f"split-{uuid.uuid4().hex}", source=src, ask_id=ask_id, group_key=gk,
                            name=None, status="skipped", reason=f"error: {type(exc).__name__}: {str(exc)[:200]}")
                out["splits_skipped"] += 1
            except Exception as exc2:
                logger.warning("vision_split_error_row_failed ask_id=%s error=%s", ask_id, exc2)
    return out


def split_one(conn, *, ask_id: str, source_id: str, group_key: str, verdicts: Sequence[PictureVerdict],
              answered_at: datetime, now: datetime, cfg: IndividualsConfig, tz: ZoneInfo,
              new_id: Optional[str] = None, split_id: Optional[str] = None) -> str:
    """One split. Returns 'split', 'refused' or 'skipped' (always writes a row)."""
    from sqlalchemy import text

    split_id = split_id or f"split-{uuid.uuid4().hex}"
    keep_ids, groups = verdict_groups(verdicts)
    name, move_ids = groups.get(group_key, (None, []))
    base = dict(split_id=split_id, source=source_id, ask_id=ask_id, group_key=group_key, name=name)

    def skip(reason: str, status: str = "skipped", **kw) -> str:
        _record(conn, status=status, reason=reason, **base, **kw)
        tag = "[VISION_SPLIT_REFUSED]" if status == "refused" else "[VISION_SPLIT_SKIPPED]"
        logger.info("%s split_id=%s ask_id=%s source=%s group=%r reason=%s", tag, split_id, ask_id,
                    source_id, group_key, reason)
        return status

    src = conn.execute(text(
        "SELECT individual_id, stream_id, kind, label, centroid, centroid_n, face_centroid, face_n, "
        "sighting_count FROM vision_individual WHERE individual_id=:i FOR UPDATE"), {"i": source_id}).fetchone()
    if src is None:
        return skip("source_missing")
    if name and src.label and name.strip().lower() == src.label.strip().lower():
        # "Someone else" but given the asked individual's own name: contradictory, never act on it.
        return skip("name_is_source_label")

    want = sorted(set(move_ids) | set(keep_ids))
    rows = {r.crop_id: r for r in conn.execute(text(
        f"SELECT {_CROP_COLS} FROM vision_crop_observation WHERE crop_id = ANY(:c)"), {"c": want}).fetchall()}
    seeds, why = [], []
    for cid in move_ids:
        r = rows.get(cid)
        if r is None:
            why.append(f"{cid}:pruned")
        elif r.individual_id not in (None, source_id):
            why.append(f"{cid}:not_a_member")
        elif not r.embedding:
            why.append(f"{cid}:no_embedding")
        else:
            seeds.append(r)
    if not move_ids:
        return skip("no_crop_recorded")
    both = sorted(set(move_ids) & set(keep_ids))
    if both:
        return skip("contradictory " + ",".join(both))
    if not seeds:
        return skip("no_member_crop " + ",".join(why))
    keep_seed = [_as_split_crop(rows[c]) for c in keep_ids if c in rows]

    members = conn.execute(text(
        f"SELECT {_CROP_COLS} FROM vision_crop_observation WHERE individual_id=:i AND embedding IS NOT NULL"),
        {"i": source_id}).fetchall()
    plan = plan_split(
        source_n=int(src.centroid_n), move_seed=[_as_split_crop(r) for r in seeds], keep_seed=keep_seed,
        members=[_as_split_crop(r) for r in members],
        source_face_centroid=list(src.face_centroid) if src.face_centroid else None,
        follow_min_face_cosine=cfg.split_follow_min_face_cosine, follow_margin=cfg.split_follow_margin,
        max_move_fraction=cfg.split_max_move_fraction)
    by_id = {r.crop_id: r for r in list(members) + seeds}
    moved_rows = [by_id[c] for c, _ in plan.moved]
    moved_doc = [{"crop_id": c, "moved_by": m, "prior_individual_id": by_id[c].individual_id}
                 for c, m in plan.moved]
    if plan.refused:
        return skip(plan.refused, status="refused", moved=moved_doc)

    new_id = new_id or uuid.uuid4().hex
    moved_set = {r.crop_id for r in moved_rows}
    before_src = _jsonb_row(conn, "vision_individual", "individual_id", source_id)

    # --- source sightings that held a moved crop -----------------------------------------------
    affected: Dict[str, Any] = {}
    in_sighting: Dict[str, List[Any]] = {}
    for r in moved_rows:
        for s in conn.execute(text(
                "SELECT sighting_id FROM vision_individual_sighting WHERE individual_id=:i "
                "AND started_at <= :t AND ended_at >= :t"), {"i": source_id, "t": r.observed_at}).fetchall():
            in_sighting.setdefault(s[0], []).append(r)
    before_sightings = []
    for sid in sorted(in_sighting):
        row = _jsonb_row(conn, "vision_individual_sighting", "sighting_id", sid)
        if row is not None:
            affected[sid] = row
            before_sightings.append(row)
    oor = conn.execute(text(
        "SELECT count(*) FROM vision_individual_sighting WHERE individual_id=:i AND ended_at < :c"),
        {"i": source_id, "c": now - timedelta(days=cfg.crop_retention_days)}).scalar() or 0

    # --- new individual ---------------------------------------------------------------------------
    m_emb = [list(r.embedding) for r in moved_rows]
    m_face = [list(r.face_embedding) for r in moved_rows if r.face_embedding]
    new_sightings = group_sightings([r._asdict() for r in moved_rows], individual_id=new_id,
                                    merge_gap_sec=cfg.merge_gap_sec)
    first = min(r.observed_at for r in moved_rows)
    last = max(r.observed_at for r in moved_rows)
    days = len({local_date(r.observed_at, tz) for r in moved_rows})
    conn.execute(text(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
        "last_seen_at, sighting_count, distinct_days, face_centroid, face_n, stationary, updated_at) "
        "VALUES (:i, :s, :k, :c, :n, :f, :l, :sc, :dd, :fc, :fn, false, now())"),
        {"i": new_id, "s": src.stream_id, "k": src.kind, "c": mean_unit(m_emb), "n": len(m_emb), "f": first,
         "l": last, "sc": len(new_sightings), "dd": days, "fc": mean_unit(m_face), "fn": len(m_face)})
    for s in new_sightings:
        conn.execute(text(
            "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, zone, started_at, "
            "ended_at, dwell_sec, observation_count, last_box_xyxy, embedding_ref, evidence_ref, thumb_ref, "
            "face_thumb_ref, zone_counts, attention_components, updated_at) VALUES (:sighting_id, :individual_id, "
            ":stream_id, :zone, :started_at, :ended_at, :dwell_sec, :observation_count, :last_box_xyxy, "
            ":embedding_ref, :evidence_ref, :thumb_ref, :face_thumb_ref, CAST(:zc AS jsonb), "
            "CAST(:ac AS jsonb), now())"),
            {**s, "zc": json.dumps(s["zone_counts"]), "ac": json.dumps({"_basis": "split", "split_id": split_id})})
    conn.execute(text("UPDATE vision_crop_observation SET individual_id=:n WHERE crop_id = ANY(:c)"),
                 {"n": new_id, "c": sorted(moved_set)})

    # --- source: centroids and counts ------------------------------------------------------------
    stored = {r.crop_id for r in members}
    total_known = stored | moved_set
    remaining = [r for r in members if r.crop_id not in moved_set]
    n_left = int(src.centroid_n) - len(moved_rows)
    f_left = max(0, int(src.face_n or 0) - len(m_face))
    in_body = reanchored_body(conn, source_id, moved_rows)
    if in_body is not None:
        # Re-anchored: the body centroid holds only the run's face crops and
        # later joins; subtract just the moved crops that are part of it.
        method = "subtract_reanchored"
        n_left = int(src.centroid_n) - len(in_body)
        c_new = (remove_from_centroid(list(src.centroid), int(src.centroid_n), [list(r.embedding) for r in in_body])
                 if in_body else list(src.centroid))
        f_new = (remove_from_centroid(list(src.face_centroid), int(src.face_n or 0), m_face)
                 if src.face_centroid and m_face else (list(src.face_centroid) if src.face_centroid else None))
    elif len(total_known) == int(src.centroid_n):
        method = "exact"
        c_new = mean_unit(list(r.embedding) for r in remaining)
        f_new = mean_unit(list(r.face_embedding) for r in remaining if r.face_embedding)
    else:
        method = "subtract"
        c_new = remove_from_centroid(list(src.centroid), int(src.centroid_n), m_emb)
        f_new = (remove_from_centroid(list(src.face_centroid), int(src.face_n or 0), m_face)
                 if src.face_centroid and m_face else (list(src.face_centroid) if src.face_centroid else None))
    if f_left == 0:
        f_new = None

    # --- source sightings: counts, zones, pictures that pointed at a moved crop ------------------
    moved_thumbs = {r.thumb_ref for r in moved_rows if r.thumb_ref}
    moved_faces = {r.face_thumb_ref for r in moved_rows if r.face_thumb_ref}
    moved_ev = {f"crop:{c}" for c in moved_set}
    deleted = 0
    for sid, row in affected.items():
        gone = in_sighting[sid]
        count = int(row["observation_count"]) - len(gone)
        if count <= 0:
            conn.execute(text("DELETE FROM vision_individual_sighting WHERE sighting_id=:s"), {"s": sid})
            deleted += 1
            continue
        zc = dict(row.get("zone_counts") or {})
        for r in gone:
            z = r.zone or ""
            if zc.get(z, 0) > 1:
                zc[z] -= 1
            else:
                zc.pop(z, None)
        stay = sorted((r for r in remaining if row_in(r, row)), key=lambda r: (r.observed_at, r.crop_id))
        upd = {"s": sid, "oc": count, "zc": json.dumps(zc), "th": row.get("thumb_ref"),
               "fth": row.get("face_thumb_ref"), "ev": row.get("evidence_ref"),
               "er": row.get("embedding_ref"), "box": row.get("last_box_xyxy")}
        if upd["th"] in moved_thumbs:
            upd["th"] = next((r.thumb_ref for r in reversed(stay) if r.thumb_ref), None)
        if upd["fth"] in moved_faces:
            upd["fth"] = next((r.face_thumb_ref for r in reversed(stay) if r.face_thumb_ref), None)
        if upd["ev"] in moved_ev:
            newest = stay[-1] if stay else None
            upd["ev"] = f"crop:{newest.crop_id}" if newest else None
            upd["er"] = newest.embedding_ref if newest else None
            upd["box"] = list(newest.box_xyxy) if newest and newest.box_xyxy else None
        upd["st"], upd["en"] = _parse_ts(row["started_at"]), _parse_ts(row["ended_at"])
        if stay and len(stay) == count:
            # Every crop left in this sighting is a known member: its bounds are exact.
            upd["st"], upd["en"] = stay[0].observed_at, stay[-1].observed_at
        upd["dw"] = max(0.0, (upd["en"] - upd["st"]).total_seconds())
        zone = max(zc.items(), key=lambda kv: (kv[1], kv[0] != ""))[0] if zc else ""
        conn.execute(text(
            "UPDATE vision_individual_sighting SET observation_count=:oc, zone_counts=CAST(:zc AS jsonb), "
            "started_at=:st, ended_at=:en, dwell_sec=:dw, "
            "zone=:z, thumb_ref=:th, face_thumb_ref=:fth, evidence_ref=:ev, embedding_ref=:er, "
            "last_box_xyxy=:box, updated_at=now() WHERE sighting_id=:s"), {**upd, "z": zone or None})
    conn.execute(text(
        "UPDATE vision_individual SET centroid=:c, centroid_n=:n, face_centroid=:fc, face_n=:fn, "
        "sighting_count=GREATEST(0, sighting_count - :d), first_seen_at=COALESCE(:f, first_seen_at), "
        "last_seen_at=COALESCE(:l, last_seen_at), distinct_days=COALESCE(:dd, distinct_days), "
        "updated_at=now() WHERE individual_id=:i"),
        {"c": c_new or list(src.centroid), "n": max(n_left, 0), "fc": f_new, "fn": f_left, "d": deleted,
         "i": source_id,
         # Exact membership: the seen fields come from the crops that stay.
         "f": min(r.observed_at for r in remaining) if method == "exact" and remaining else None,
         "l": max(r.observed_at for r in remaining) if method == "exact" and remaining else None,
         "dd": len({local_date(r.observed_at, tz) for r in remaining}) if method == "exact" and remaining else None})

    # --- links and name ----------------------------------------------------------------------------
    after_src = _jsonb_row(conn, "vision_individual", "individual_id", source_id)
    face_cos = (cosine(after_src["face_centroid"], mean_unit(m_face))
                if after_src.get("face_centroid") and m_face else None)
    body_cos = cosine(after_src["centroid"], mean_unit(m_emb))
    linked = [_write_pair(conn, source_id, new_id, "different", face_cos, ask_id, now)]
    same_with = None
    if name:
        _set_typed_label(conn, new_id, name, ask_id, answered_at)
        same_with = _same_name_target(conn, name, src, new_id)
        if same_with:
            linked.append(_write_pair(conn, same_with, new_id, "same", None, ask_id, now))
    linked = [x for x in linked if x]

    after = {"method": method, "source": after_src,
             "new": _jsonb_row(conn, "vision_individual", "individual_id", new_id),
             "sightings": {sid: _jsonb_row(conn, "vision_individual_sighting", "sighting_id", sid)
                           for sid in affected},
             "new_sightings": [s["sighting_id"] for s in new_sightings],
             "body_cosine": round(float(body_cos), 4),
             "face_cosine": None if face_cos is None else round(float(face_cos), 4)}
    before = {"source": before_src, "sightings": before_sightings}
    if not _record(conn, status="split", new_id=new_id, moved=moved_doc, before=before, after=after,
                   linked=linked, out_of_retention=int(oor), **base):
        raise RuntimeError("split already recorded")  # rolls this savepoint back
    logger.info("[VISION_SPLIT] split_id=%s ask_id=%s source=%s new=%s name=%r moved_verdict=%d "
                "moved_follow=%d stay=%d body_cos=%.3f face_cos=%s method=%s same_link=%s "
                "sightings_rewritten=%d sightings_deleted=%d",
                split_id, ask_id, source_id, new_id, name, plan.by_verdict, plan.by_face, max(n_left, 0),
                body_cos, "n/a" if face_cos is None else f"{face_cos:.3f}", method, same_with or "-",
                len(affected) - deleted, deleted)
    return "split"


def row_in(crop, sighting_row: Dict[str, Any]) -> bool:
    t = crop.observed_at
    st = _parse_ts(sighting_row["started_at"])
    en = _parse_ts(sighting_row["ended_at"])
    return st <= t <= en


def _parse_ts(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v
    d = datetime.fromisoformat(str(v))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _write_pair(conn, x: str, y: str, verdict: str, face_cos: Optional[float], ask_id: str,
                at: datetime) -> Optional[dict]:
    """One pair verdict the split wrote (None if the pair already had one)."""
    from sqlalchemy import text

    a, b = sorted((x, y))
    r = conn.execute(text(
        "INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, face_cosine, ask_id, "
        "decided_at) VALUES (:a, :b, :v, :c, :ask, :at) ON CONFLICT (individual_a, individual_b) DO NOTHING "
        "RETURNING individual_a"),
        {"a": a, "b": b, "v": verdict, "c": None if face_cos is None else round(float(face_cos), 4),
         "ask": ask_id, "at": at}).fetchone()
    return {"individual_a": a, "individual_b": b, "verdict": verdict, "ask_id": ask_id} if r else None


def _same_name_target(conn, name: str, src, new_id: str) -> Optional[str]:
    """An existing individual (same stream and kind) Juniper already TYPED this
    name for, that is not on the source's confirmed side (a 'same' path to the
    source would make the new individual the source again)."""
    from sqlalchemy import text

    cands = [r[0] for r in conn.execute(text(
        "SELECT individual_id FROM vision_individual WHERE stream_id=:s AND kind=:k AND label IS NOT NULL "
        "AND NOT label_copied AND lower(label) = lower(:n) AND individual_id NOT IN (:src, :new) "
        "ORDER BY labeled_at NULLS LAST, individual_id"),
        {"s": src.stream_id, "k": src.kind, "n": name, "src": src.individual_id, "new": new_id}).fetchall()]
    if not cands:
        return None
    verdicts = conn.execute(text(
        "SELECT individual_a, individual_b FROM vision_individual_pair_verdict WHERE verdict='same'")).fetchall()
    groups = same_groups((a, b) for a, b in verdicts)
    g_src = groups.get(src.individual_id, src.individual_id)
    return next((c for c in cands if groups.get(c, c) != g_src), None)


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


def _columns(conn, table: str) -> List[str]:
    from sqlalchemy import text

    return [r[0] for r in conn.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
        "AND table_name = :t ORDER BY ordinal_position"), {"t": table}).fetchall()]


def _restore_individual(conn, row: Dict[str, Any]) -> None:
    from sqlalchemy import text

    cols = [c for c in _columns(conn, "vision_individual") if c != "individual_id"]
    conn.execute(text(
        f"UPDATE vision_individual t SET ({', '.join(cols)}) = (SELECT {', '.join('r.' + c for c in cols)} "
        "FROM jsonb_populate_record(NULL::vision_individual, CAST(:j AS jsonb)) r) WHERE t.individual_id = :i"),
        {"j": json.dumps(row), "i": row["individual_id"]})


def _restore_sighting(conn, row: Dict[str, Any]) -> None:
    from sqlalchemy import text

    conn.execute(text("DELETE FROM vision_individual_sighting WHERE sighting_id=:s"), {"s": row["sighting_id"]})
    conn.execute(text(
        "INSERT INTO vision_individual_sighting "
        "SELECT * FROM jsonb_populate_record(NULL::vision_individual_sighting, CAST(:j AS jsonb))"),
        {"j": json.dumps(row)})


def row_ask(row) -> Optional[str]:
    return getattr(row, "ask_id", None)


def undo_split(conn, split_id: str, *, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Reverse one split inside ``conn``'s transaction (the caller commits or
    rolls back: scripts/undo_vision_split.py rolls back unless --apply).

    The source is restored to ``before`` exactly when nothing touched it since
    the split; otherwise (new crops joined it since) the moved crops are added
    back to its centroid and counts so that later crops are not lost. Refuses
    when crops joined the NEW individual since the split, naming them."""
    from sqlalchemy import text

    now = now or datetime.now(timezone.utc)
    from app.vision_individuals import INDIVIDUALS_LOCK_KEY

    # The individuals cycle holds this key while it reads and rewrites
    # individuals; an undo committed in between would be overwritten.
    conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": INDIVIDUALS_LOCK_KEY})
    row = conn.execute(text(
        "SELECT split_id, source_individual_id, new_individual_id, ask_id, status, moved, before, after, linked_pair, "
        "undone_at FROM vision_individual_split WHERE split_id=:s FOR UPDATE"), {"s": split_id}).fetchone()
    if row is None:
        return {"ok": False, "reason": "no such split"}
    if row.status != "split":
        return {"ok": False, "reason": f"status is {row.status}: nothing was moved"}
    if row.undone_at is not None:
        return {"ok": False, "reason": f"already undone at {row.undone_at.isoformat()}"}
    src, new = row.source_individual_id, row.new_individual_id
    moved = list(row.moved or [])
    moved_ids = [m["crop_id"] for m in moved]
    before, after = dict(row.before or {}), dict(row.after or {})

    joined = [r[0] for r in conn.execute(text(
        "SELECT crop_id FROM vision_crop_observation WHERE individual_id=:n AND NOT (crop_id = ANY(:m)) "
        "ORDER BY observed_at"), {"n": new, "m": moved_ids}).fetchall()]
    if joined:
        return {"ok": False, "reason": "crops joined the new individual since the split; Juniper decides",
                "joined_crops": joined}
    new_now = _jsonb_row(conn, "vision_individual", "individual_id", new)
    if new_now is not None and int(new_now.get("centroid_n") or 0) != len(moved_ids):
        return {"ok": False, "reason": "the new individual's count changed since the split",
                "centroid_n": new_now.get("centroid_n"), "moved": len(moved_ids)}

    if new_now is not None and new_now.get("label_ask_id") not in (None, row_ask(row)):
        return {"ok": False, "reason": "Juniper named the new individual after the split; Juniper decides",
                "label_ask_id": new_now.get("label_ask_id")}
    written = {(p["individual_a"], p["individual_b"]) for p in (row.linked_pair or [])}
    later = [f"{a}|{b}:{v}" for a, b, v in conn.execute(text(
        "SELECT individual_a, individual_b, verdict FROM vision_individual_pair_verdict "
        "WHERE :n IN (individual_a, individual_b)"), {"n": new}).fetchall() if (a, b) not in written]
    if later:
        return {"ok": False, "reason": "pair verdicts about the new individual were answered after the split",
                "pair_verdicts": later}
    if _jsonb_row(conn, "vision_individual", "individual_id", src) is None:
        return {"ok": False, "reason": "the source individual was pruned since the split: nothing to give back to"}

    report: Dict[str, Any] = {"ok": True, "split_id": split_id, "source": src, "new": new,
                              "crops_restored": 0, "sightings_restored": 0, "sightings_merged": 0}
    src_now = _jsonb_row(conn, "vision_individual", "individual_id", src)
    after_sightings = after.get("sightings") or {}
    # Crops back first (exact restore of the source needs nothing else to change).
    for m in moved:
        report["crops_restored"] += conn.execute(text(
            "UPDATE vision_crop_observation SET individual_id=:p WHERE crop_id=:c AND individual_id=:n"),
            {"p": m.get("prior_individual_id"), "c": m["crop_id"], "n": new}).rowcount or 0
    # Sightings: a source sighting nothing touched since the split goes back to
    # `before` exactly; one that changed since gets the moved crops' counts back.
    moved_rows = conn.execute(text(
        f"SELECT {_CROP_COLS} FROM vision_crop_observation WHERE crop_id = ANY(:c)"), {"c": moved_ids}).fetchall()
    restored_deleted = 0
    for b in before.get("sightings") or []:
        sid = b["sighting_id"]
        cur = _jsonb_row(conn, "vision_individual_sighting", "sighting_id", sid)
        if cur is None and after_sightings.get(sid) is not None:
            continue  # pruned since the split: nothing to give back
        if cur == after_sightings.get(sid):
            _restore_sighting(conn, b)
            report["sightings_restored"] += 1
            if after_sightings.get(sid) is None:
                restored_deleted += 1
        else:
            gone = [r for r in moved_rows if row_in(r, b)]
            zc = dict(cur.get("zone_counts") or {})
            for r in gone:
                zc[r.zone or ""] = zc.get(r.zone or "", 0) + 1
            conn.execute(text(
                "UPDATE vision_individual_sighting SET observation_count = observation_count + :k, "
                "zone_counts = CAST(:zc AS jsonb), updated_at = now() WHERE sighting_id=:s"),
                {"k": len(gone), "zc": json.dumps(zc), "s": sid})
            report["sightings_merged"] += 1
    if src_now is not None and src_now == after.get("source"):
        _restore_individual(conn, before["source"])
        report["source"] = {"id": src, "restore": "exact"}
    elif src_now is not None:
        in_body = reanchored_body(conn, src, moved_rows)
        # Re-anchored since: only crops that are part of the re-anchored body centroid go back into it.
        emb = [list(r.embedding) for r in (moved_rows if in_body is None else in_body) if r.embedding]
        faces = [list(r.face_embedding) for r in moved_rows if r.face_embedding]
        n, fn = int(src_now["centroid_n"]), int(src_now.get("face_n") or 0)
        conn.execute(text(
            "UPDATE vision_individual SET centroid=:c, centroid_n=:n, face_centroid=:fc, face_n=:fn, "
            "sighting_count = sighting_count + :d, first_seen_at = LEAST(first_seen_at, :f), "
            "last_seen_at = GREATEST(last_seen_at, :l), updated_at=now() WHERE individual_id=:i"),
            {"c": add_to_centroid(src_now["centroid"], n, emb), "n": n + len(emb),
             "fc": add_to_centroid(src_now.get("face_centroid"), fn, faces), "fn": fn + len(faces),
             "d": restored_deleted, "f": min(r.observed_at for r in moved_rows) if moved_rows else now,
             "l": max(r.observed_at for r in moved_rows) if moved_rows else now, "i": src})
        report["source"] = {"id": src, "restore": "delta (it changed since the split)"}
    for p in row.linked_pair or []:
        conn.execute(text(
            "DELETE FROM vision_individual_pair_verdict WHERE individual_a=:a AND individual_b=:b "
            "AND verdict=:v AND ask_id=:ask"),
            {"a": p["individual_a"], "b": p["individual_b"], "v": p["verdict"], "ask": p["ask_id"]})
    # The new individual and its sightings (cascade). Its label goes with it.
    conn.execute(text("DELETE FROM vision_individual WHERE individual_id=:n"), {"n": new})
    asks = [r[0] for r in conn.execute(text(
        "UPDATE orion_ask SET status='expired' WHERE status='open' AND (source_ref=:n OR source_ref LIKE :n || '|%' "
        "OR source_ref LIKE '%|' || :n) RETURNING ask_id"), {"n": new}).fetchall()]
    if asks:
        report["expired_open_asks_about_new_individual"] = asks
    conn.execute(text("UPDATE vision_individual_split SET undone_at=:t WHERE split_id=:s"), {"t": now, "s": split_id})
    logger.info("[VISION_SPLIT_UNDONE] split_id=%s source=%s new=%s crops=%d source_restore=%s",
                split_id, src, new, report["crops_restored"], report["source"].get("restore"))
    return report


__all__ = [
    "SplitCrop", "SplitPlan", "PictureVerdict", "plan_split", "verdict_groups", "group_sightings",
    "mean_unit", "remove_from_centroid", "add_to_centroid", "apply_splits", "split_one", "undo_split",
    "pending_split_groups", "parse_pair_ref",
]
