"""Walkway person learning, Phase 1: albums, learned bars, guesses.

Orion learns who people are from Juniper's answers. Each named person gets an
album of crops (+1 "this is them", -1 "this is not them"), built only from her
answers. Per person and channel (face | body | osnet) a confidence bar is
fitted from the album itself:

- a cluster's score for a person: the MEDIAN over its crops of each crop's
  mean top-k cosine to that person's +1 album (album crops of the same
  cluster left out), lowered when a -1 crop is closer than the nearest +1;
- ``accept_at``: conformal bound on false accepts at ``alpha``, fitted on
  CLUSTER scores of everyone who is not this person: the street (every
  unnamed recent cluster), other named people's clusters, and clusters with
  crops Juniper said are not them. The street is the population a guess is
  made on; calibrating only on the other named people let ~9% of street
  clusters through on live data (2026-10-20). Unnamed clusters that really
  are the person only raise the bar (conservative). A cluster is accepted
  when its score is above the bound;
- the street is both what the bar is fitted on and what is guessed on, so
  about alpha of street clusters clear any armed bar. alpha is therefore
  split across people (Bonferroni: alpha / number of people), and a bar is
  armed only when at least ``MIN_GENUINE_CLUSTERS`` of the person's own
  clusters, each held out together with every album crop from its days,
  exist and at least half of them score above it: the person must beat the
  street across days, not once by luck. A channel that only matches same-day
  clothes accepts nobody;
- ``reject_below``: alpha-quantile of cross-day genuine CROP scores (needs
  >= 19). Crops of one cluster are correlated, so that n is nominal; the
  bar is applied to a cluster median against the full album, which only
  makes rejects rarer (conservative).

A person with no armed bar still gets ``unsure`` guesses: those are the
input to the capped "Is this X?" asks, the way albums grow.

Three bands: accept ("probably X"), unsure (ask), reject (not X). A guess
never writes ``vision_individual.label``.

Design: docs/superpowers/specs/2026-10-20-walkway-person-learning-design.md.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger("sql-writer.vision_person_model")

MODEL_VERSION = "median-topk-street-v1"
CHANNELS = ("face", "body", "osnet")
RECONCILED_SOURCES = ("cluster_label", "pair_same", "picture_different")
PEOPLE_KINDS = ("person", "man", "woman", "child", "kid")
PERSON_TABLES = ("vision_person_exemplar", "vision_person_bar", "vision_person_attribution",
                 "vision_person_model_eval")
# Purity: a cluster needs this many body vectors before a 2-way split is meaningful.
PURITY_MIN_CROPS = 4
# Crops per unnamed cluster that are scored (newest first).
ATTRIBUTION_CROPS = 20
# An accept bar is armed only with this many held-out own clusters, at least half above it.
MIN_GENUINE_CLUSTERS = 3


@dataclass(frozen=True)
class PersonModelConfig:
    alpha: float = 0.05
    top_k: int = 3
    fallback_channel: str = "body"
    candidate_days: float = 30.0
    local_tz: str = "America/Denver"


@dataclass(frozen=True)
class Crop:
    crop_id: str
    cluster: str
    day: str                      # local date, "YYYY-MM-DD"
    vecs: Dict[str, np.ndarray]   # channel -> unit vector (channels the crop has)


@dataclass(frozen=True)
class Bar:
    person: str
    channel: str
    accept_at: Optional[float]
    reject_below: Optional[float]
    n_genuine: int
    n_impostor: int


# --- pure maths -------------------------------------------------------------------------------------


def unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


def conformal_accept_at(impostor: Sequence[float], alpha: float) -> Optional[float]:
    """The ceil((1-a)(n+1))-th smallest impostor score: a new impostor scores
    above it with probability <= a. None when there are too few scores."""
    n = len(impostor)
    k = math.ceil((1.0 - alpha) * (n + 1))
    if k > n:
        return None
    return float(sorted(impostor)[k - 1])


def conformal_reject_below(genuine: Sequence[float], alpha: float) -> Optional[float]:
    """The floor(a(n+1))-th smallest genuine score: a new genuine scores below
    it with probability <= a. None when there are too few scores."""
    n = len(genuine)
    k = math.floor(alpha * (n + 1))
    if k < 1:
        return None
    return float(sorted(genuine)[k - 1])


def band(score: float, bar: Bar) -> str:
    if bar.accept_at is not None and score > bar.accept_at:
        return "accept"
    if bar.reject_below is not None and score < bar.reject_below:
        return "reject"
    return "unsure"


def split_cosine(vectors: Sequence[np.ndarray], iters: int = 20) -> Optional[float]:
    """2-means split of unit vectors; cosine between the two halves' centroids.
    None when there are too few vectors or one half is under 2."""
    if len(vectors) < PURITY_MIN_CROPS:
        return None
    x = np.stack(vectors)
    a, b = x[0], x[int(np.argmin(x @ x[0]))]
    g = None
    for _ in range(iters):
        g = (x @ a) >= (x @ b)
        if g.sum() < 2 or (~g).sum() < 2:
            return None
        a, b = unit(x[g].mean(axis=0)), unit(x[~g].mean(axis=0))
    return float(a @ b)


def same_sighting_floor(groups: Iterable[Sequence[np.ndarray]], pct: float = 10.0) -> Optional[float]:
    """The pct-th percentile of crop-vs-crop cosines inside one sighting (the
    same person seconds apart): two halves of a cluster less alike than this
    look like two people."""
    sims: List[float] = []
    for vs in groups:
        if len(vs) < 2:
            continue
        m = np.stack(vs)
        s = m @ m.T
        sims.extend(s[np.triu_indices(len(vs), 1)].tolist())
    if len(sims) < 20:
        return None
    return float(np.percentile(sims, pct))


class _Side:
    """One person's crops on one channel, as a matrix plus cluster/day labels."""

    def __init__(self, crops: Sequence[Crop], channel: str):
        cs = [c for c in crops if channel in c.vecs]
        self.m = np.stack([c.vecs[channel] for c in cs]) if cs else None
        self.cluster = np.array([c.cluster for c in cs])
        self.day = np.array([c.day for c in cs])

    def __len__(self) -> int:
        return 0 if self.m is None else self.m.shape[0]


def _score_one(v: np.ndarray, pos: _Side, pos_mask: np.ndarray, neg: _Side, neg_mask: np.ndarray,
               top_k: int) -> Optional[float]:
    """One crop's score: the mean of its top-k cosines to the (masked) +1 album,
    lowered by how far its nearest (masked) -1 crop is closer than its nearest
    +1 crop. Rows are unit vectors; masks select rows without copying."""
    if not pos_mask.any():
        return None
    sims = (pos.m @ v)[pos_mask]
    top = -np.sort(-sims)[:min(top_k, sims.shape[0])]
    score = float(top.mean())
    if len(neg) and neg_mask.any():
        score -= max(0.0, float((neg.m @ v)[neg_mask].max()) - float(top[0]))
    return score


@dataclass(frozen=True)
class Fit:
    bar: Bar
    genuine_crops: List[float]       # cross-day, per crop (reject bar)
    genuine_clusters: List[float]    # the person's own clusters, held out with their days
    impostor_clusters: List[float]   # street + other people + ruled-out clusters (accept bar)


def _by_cluster(crops: Iterable[Crop]) -> Dict[str, List[Crop]]:
    out: Dict[str, List[Crop]] = {}
    for c in crops:
        out.setdefault(c.cluster, []).append(c)
    return out


def score_cluster(crops: Sequence[Crop], person: str, channel: str, album: Dict[str, Dict[int, List[Crop]]],
                  cfg: PersonModelConfig, sides: Optional[Dict[Tuple[str, str, int], _Side]] = None
                  ) -> Optional[float]:
    """Median over the cluster's crops of their score against the person's album
    (album crops from this same cluster left out). Same formula as _score_one."""
    def side(pol: int) -> _Side:
        key = (person, channel, pol)
        if sides is not None and key in sides:
            return sides[key]
        sd = _Side(album.get(person, {}).get(pol, []), channel)
        if sides is not None:
            sides[key] = sd
        return sd

    qs = [c.vecs[channel] for c in crops if channel in c.vecs]
    pos, neg = side(1), side(-1)
    if not qs or not len(pos):
        return None
    names = np.array(sorted({c.cluster for c in crops}))
    cols = pos.m[~np.isin(pos.cluster, names)]
    if not cols.shape[0]:
        return None
    q = np.stack(qs)
    sims = q @ cols.T
    top = -np.sort(-sims, axis=1)[:, :min(cfg.top_k, cols.shape[0])]
    score = top.mean(axis=1)
    if len(neg):
        ncols = neg.m[~np.isin(neg.cluster, names)]
        if ncols.shape[0]:
            score = score - np.maximum(0.0, (q @ ncols.T).max(axis=1) - top[:, 0])
    return float(np.median(score))


def fit_bar(person: str, channel: str, album: Dict[str, Dict[int, List[Crop]]], cfg: PersonModelConfig,
            street: Optional[Dict[str, List[Crop]]] = None) -> Fit:
    """Fit one person's bar on one channel. ``album``: person -> polarity -> crops;
    ``street``: unnamed recent clusters (cluster -> crops). The accept bound
    uses alpha / number of people in the album."""
    own = album.get(person, {})
    pos = _Side(own.get(1, []), channel)
    neg = _Side(own.get(-1, []), channel)
    genuine: List[float] = []
    for c in own.get(1, []):
        if channel not in c.vecs:
            continue
        s = _score_one(c.vecs[channel], pos, (pos.cluster != c.cluster) & (pos.day != c.day),
                       neg, neg.cluster != c.cluster, cfg.top_k)
        if s is not None:
            genuine.append(s)
    genuine_clusters: List[float] = []
    for k, cs in _by_cluster(own.get(1, [])).items():
        days = {c.day for c in cs}
        held = {**album, person: {**own, 1: [c for c in own.get(1, []) if c.day not in days]}}
        s = score_cluster(cs, person, channel, held, cfg)
        if s is not None:
            genuine_clusters.append(s)
    own_clusters = {c.cluster for c in own.get(1, [])}
    ruled_out = _by_cluster(own.get(-1, []))
    groups = list(_by_cluster(c for other, pol in album.items() if other != person
                              for c in pol.get(1, [])).values())
    groups += list(ruled_out.values())
    groups += [cs for cid, cs in (street or {}).items() if cid not in own_clusters and cid not in ruled_out]
    sides: Dict[Tuple[str, str, int], _Side] = {(person, channel, 1): pos, (person, channel, -1): neg}
    impostor = [s for s in (score_cluster(cs, person, channel, album, cfg, sides) for cs in groups)
                if s is not None]
    accept_at = conformal_accept_at(impostor, cfg.alpha / max(1, len(album)))
    above = sum(1 for g in genuine_clusters if accept_at is not None and g > accept_at)
    if len(genuine_clusters) < MIN_GENUINE_CLUSTERS or 2 * above < len(genuine_clusters):
        accept_at = None
    bar = Bar(person, channel, accept_at, conformal_reject_below(genuine, cfg.alpha),
              len(genuine), len(impostor))
    return Fit(bar, genuine, genuine_clusters, impostor)


def pick_channel(crops: Sequence[Crop], person: str, bars: Dict[Tuple[str, str], Bar],
                 album: Dict[str, Dict[int, List[Crop]]], cfg: PersonModelConfig) -> Optional[str]:
    """Face when both sides have faces and the face bar can decide (has an
    accept or reject bound), else the fallback, else face with an open bar."""
    face = (any("face" in c.vecs for c in crops) and (person, "face") in bars
            and any("face" in c.vecs for c in album.get(person, {}).get(1, [])))
    decides = lambda b: b.accept_at is not None or b.reject_below is not None  # noqa: E731
    if face and decides(bars[(person, "face")]):
        return "face"
    if (person, cfg.fallback_channel) in bars:
        return cfg.fallback_channel
    return "face" if face else None


def attribute(clusters: Dict[str, List[Crop]], album: Dict[str, Dict[int, List[Crop]]],
              bars: Dict[Tuple[str, str], Bar], blocked: Dict[str, set], cfg: PersonModelConfig
              ) -> List[dict]:
    """One guess per unnamed cluster: its best-scoring person, banded. A
    cluster accepted for two people is downgraded to unsure. ``blocked``:
    cluster -> casefolded names Juniper said it is not."""
    out: List[dict] = []
    persons = sorted(album)
    sides: Dict[Tuple[str, str, int], _Side] = {}
    for cid, crops in sorted(clusters.items()):
        scored = []
        for p in persons:
            if p.strip().casefold() in blocked.get(cid, set()):
                continue
            ch = pick_channel(crops, p, bars, album, cfg)
            if ch is None:
                continue
            s = score_cluster(crops, p, ch, album, cfg, sides)
            if s is None:
                continue
            b = band(s, bars[(p, ch)])
            if b != "reject":
                scored.append((b == "accept", s, p, ch, b))
        if not scored:
            continue
        accepted = [x for x in scored if x[0]]
        best = max(accepted or scored, key=lambda x: x[1])
        b = "unsure" if len(accepted) > 1 else best[4]
        out.append({"individual_id": cid, "person_label": best[2], "channel": best[3],
                    "score": round(best[1], 4), "band": b})
    return out


def exemplar_id(person: str, crop_id: str, source: str) -> str:
    return hashlib.md5(f"{person}|{crop_id}|{source}".encode()).hexdigest()


def canonical_names(labels: Iterable[str]) -> Dict[str, str]:
    """casefolded name -> one display form (the most common casing; ties: sorted first)."""
    counts: Dict[str, Dict[str, int]] = {}
    for lbl in labels:
        name = (lbl or "").strip()
        if name:
            counts.setdefault(name.casefold(), {}).setdefault(name, 0)
            counts[name.casefold()][name] += 1
    return {k: sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for k, v in counts.items()}


# --- database ----------------------------------------------------------------------------------------


def tables_present(conn) -> bool:
    from sqlalchemy import text

    return not conn.execute(text(
        "SELECT t FROM unnest(CAST(:t AS text[])) AS t WHERE to_regclass(t) IS NULL"),
        {"t": list(PERSON_TABLES)}).fetchall()


def _load_crops(conn, where: str, params: dict, tz: ZoneInfo) -> List[Crop]:
    from sqlalchemy import text

    out = []
    for r in conn.execute(text(
            "SELECT crop_id, individual_id, observed_at, embedding, osnet_embedding, face_embedding "
            f"FROM vision_crop_observation c WHERE {where}"), params).fetchall():
        vecs = {ch: unit(v) for ch, v in zip(("body", "osnet", "face"), (r[3], r[4], r[5])) if v}
        if vecs:
            out.append(Crop(r[0], r[1] or "", r[2].astimezone(tz).date().isoformat(), vecs))
    return out


def desired_exemplars(conn, tz: ZoneInfo) -> Tuple[List[dict], dict]:
    """The album implied by Juniper's answers right now (reconciled sources only).
    Returns (rows, stats)."""
    from sqlalchemy import text

    named = conn.execute(text(
        "SELECT individual_id, label, label_copied, label_ask_id FROM vision_individual "
        "WHERE label IS NOT NULL AND kind = ANY(:k)"), {"k": list(PEOPLE_KINDS)}).fetchall()
    names = canonical_names(r[1] for r in named)
    crops = _load_crops(conn, "individual_id = ANY(:i)", {"i": [r[0] for r in named]}, tz) if named else []
    by_cluster: Dict[str, List[Crop]] = {}
    for c in crops:
        by_cluster.setdefault(c.cluster, []).append(c)
    sighting_groups = conn.execute(text(
        "SELECT c.crop_id, s.sighting_id FROM vision_crop_observation c JOIN vision_individual_sighting s "
        "ON s.individual_id = c.individual_id AND c.observed_at BETWEEN s.started_at AND s.ended_at "
        "WHERE c.individual_id = ANY(:i)"), {"i": list(by_cluster)}).fetchall() if by_cluster else []
    body = {c.crop_id: c.vecs["body"] for c in crops if "body" in c.vecs}
    groups: Dict[str, List[np.ndarray]] = {}
    for crop_id, sid in sighting_groups:
        if crop_id in body:
            groups.setdefault(sid, []).append(body[crop_id])
    floor = same_sighting_floor(groups.values())
    rows: List[dict] = []
    stats = {"named_clusters": len(named), "impure_clusters": 0, "purity_floor": floor}
    for iid, label, copied, ask_id in named:
        if not label.strip():
            continue
        cl = by_cluster.get(iid, [])
        split = split_cosine([c.vecs["body"] for c in cl if "body" in c.vecs])
        if floor is not None and split is not None and split < floor:
            stats["impure_clusters"] += 1        # looks like two people: abstain until Juniper answers
            continue
        person = names[label.strip().casefold()]
        source = "pair_same" if copied else "cluster_label"
        rows += [{"person": person, "crop_id": c.crop_id, "polarity": 1, "source": source, "ask_id": ask_id}
                 for c in cl]
    for crop_id, ask_id, label in conn.execute(text(
            "SELECT v.crop_id, v.ask_id, i.label FROM vision_ask_picture_verdict v "
            "JOIN vision_individual i ON i.individual_id = v.individual_id "
            "WHERE v.verdict = 'different' AND v.crop_id IS NOT NULL AND i.label IS NOT NULL")).fetchall():
        if not label.strip():
            continue
        person = names.get(label.strip().casefold(), label.strip())
        rows.append({"person": person, "crop_id": crop_id, "polarity": -1, "source": "picture_different",
                     "ask_id": ask_id})
    return rows, stats


def reconcile_exemplars(conn, rows: List[dict], now: datetime) -> Tuple[int, int]:
    """Insert new / revive revoked rows; revoke reconciled rows no longer implied."""
    from sqlalchemy import text

    added = 0
    ids = []
    for r in rows:
        eid = exemplar_id(r["person"], r["crop_id"], r["source"])
        ids.append(eid)
        added += conn.execute(text(
            "INSERT INTO vision_person_exemplar (exemplar_id, person_label, crop_id, polarity, source, ask_id, added_at) "
            "VALUES (:e, :p, :c, :pol, :s, :a, :t) ON CONFLICT (exemplar_id) DO UPDATE SET revoked_at = NULL "
            "WHERE vision_person_exemplar.revoked_at IS NOT NULL"),
            {"e": eid, "p": r["person"], "c": r["crop_id"], "pol": r["polarity"], "s": r["source"],
             "a": r["ask_id"], "t": now}).rowcount or 0
    revoked = conn.execute(text(
        "UPDATE vision_person_exemplar SET revoked_at = :t WHERE revoked_at IS NULL "
        "AND source = ANY(:s) AND NOT (exemplar_id = ANY(:ids))"),
        {"t": now, "s": list(RECONCILED_SOURCES), "ids": ids}).rowcount or 0
    return added, revoked


def build_album(ex: Sequence[Tuple[str, str, int]], crops: Dict[str, Crop]) -> Dict[str, Dict[int, List[Crop]]]:
    """(person, crop_id, polarity) rows -> person -> polarity -> crops. Names
    that differ only in casing are one person (rows keep the spelling they were
    written with). A crop that is both +1 and -1 for a person counts as -1;
    vanished crops drop."""
    names = canonical_names(p for p, _, _ in ex)
    ex = [(names.get(p.strip().casefold(), p), c, pol) for p, c, pol in ex]
    negs = {(p, c) for p, c, pol in ex if pol == -1}
    album: Dict[str, Dict[int, List[Crop]]] = {}
    seen = set()
    for person, crop_id, pol in ex:
        c = crops.get(crop_id)
        if c is None or (pol == 1 and (person, crop_id) in negs) or (person, crop_id, pol) in seen:
            continue
        seen.add((person, crop_id, pol))
        album.setdefault(person, {}).setdefault(int(pol), []).append(c)
    return album


def load_crops_by_id(conn, crop_ids: Iterable[str], tz: ZoneInfo) -> Dict[str, Crop]:
    ids = sorted(set(crop_ids))
    return {c.crop_id: c for c in _load_crops(conn, "crop_id = ANY(:i)", {"i": ids}, tz)} if ids else {}


def load_album(conn, tz: ZoneInfo) -> Tuple[Dict[str, Dict[int, List[Crop]]], str]:
    """Active exemplars with their vectors, and a fingerprint of the album."""
    from sqlalchemy import text

    ex = conn.execute(text(
        "SELECT exemplar_id, person_label, crop_id, polarity FROM vision_person_exemplar "
        "WHERE revoked_at IS NULL ORDER BY exemplar_id")).fetchall()
    fingerprint = hashlib.md5(("|".join(r[0] for r in ex) + "|" + MODEL_VERSION).encode()).hexdigest()
    rows = [(r[1], r[2], int(r[3])) for r in ex]
    return build_album(rows, load_crops_by_id(conn, (r[1] for r in rows), tz)), fingerprint


def fit_all(album: Dict[str, Dict[int, List[Crop]]], cfg: PersonModelConfig,
            street: Optional[Dict[str, List[Crop]]] = None) -> Dict[Tuple[str, str], Bar]:
    bars = {}
    for person in sorted(album):
        for ch in CHANNELS:
            if any(ch in c.vecs for c in album[person].get(1, [])):
                bars[(person, ch)] = fit_bar(person, ch, album, cfg, street).bar
    return bars


def _write_bars(conn, bars: Dict[Tuple[str, str], Bar], fingerprint: str, now: datetime) -> None:
    from sqlalchemy import text

    conn.execute(text("DELETE FROM vision_person_bar"))
    for b in bars.values():
        conn.execute(text(
            "INSERT INTO vision_person_bar (person_label, channel, accept_at, reject_below, n_genuine, n_impostor, "
            "album_fingerprint, model_version, fitted_at) VALUES (:p, :c, :a, :r, :g, :i, :f, :v, :t)"),
            {"p": b.person, "c": b.channel, "a": b.accept_at, "r": b.reject_below, "g": b.n_genuine,
             "i": b.n_impostor, "f": fingerprint, "v": MODEL_VERSION, "t": now})


def unnamed_clusters(conn, now: datetime, cfg: PersonModelConfig, tz: ZoneInfo) -> Dict[str, List[Crop]]:
    from sqlalchemy import text

    ids = [r[0] for r in conn.execute(text(
        "SELECT individual_id FROM vision_individual WHERE label IS NULL AND NOT stationary "
        "AND kind = ANY(:k) AND last_seen_at > :since"),
        {"k": list(PEOPLE_KINDS), "since": now - timedelta(days=cfg.candidate_days)}).fetchall()]
    if not ids:
        return {}
    crops = _load_crops(conn, "crop_id IN (SELECT crop_id FROM (SELECT crop_id, row_number() OVER "
                        "(PARTITION BY individual_id ORDER BY observed_at DESC) AS n FROM vision_crop_observation "
                        "WHERE individual_id = ANY(:i)) x WHERE n <= :n)", {"i": ids, "n": ATTRIBUTION_CROPS}, tz)
    out: Dict[str, List[Crop]] = {}
    for c in crops:
        out.setdefault(c.cluster, []).append(c)
    return out


def _blocked(conn) -> Dict[str, set]:
    """cluster -> persons Juniper said it is not (a -1 exemplar on one of its crops)."""
    from sqlalchemy import text

    out: Dict[str, set] = {}   # cluster -> casefolded names
    for iid, person in conn.execute(text(
            "SELECT DISTINCT c.individual_id, e.person_label FROM vision_person_exemplar e "
            "JOIN vision_crop_observation c ON c.crop_id = e.crop_id "
            "WHERE e.polarity = -1 AND e.revoked_at IS NULL AND c.individual_id IS NOT NULL")).fetchall():
        out.setdefault(iid, set()).add(person.strip().casefold())
    return out


def write_attributions(conn, guesses: List[dict], now: datetime) -> Tuple[int, int]:
    from sqlalchemy import text

    keep = []
    for g in guesses:
        keep.append(f"{g['individual_id']}|{g['person_label']}")
        conn.execute(text(
            "INSERT INTO vision_person_attribution (individual_id, person_label, channel, score, band, "
            "model_version, decided_at) VALUES (:i, :p, :c, :s, :b, :v, :t) "
            "ON CONFLICT (individual_id, person_label, model_version) DO UPDATE SET channel = EXCLUDED.channel, "
            "score = EXCLUDED.score, band = EXCLUDED.band, decided_at = EXCLUDED.decided_at, revoked_at = NULL"),
            {"i": g["individual_id"], "p": g["person_label"], "c": g["channel"], "s": g["score"], "b": g["band"],
             "v": MODEL_VERSION, "t": now})
    revoked = conn.execute(text(
        "UPDATE vision_person_attribution SET revoked_at = :t WHERE revoked_at IS NULL AND "
        "(model_version <> :v OR NOT ((individual_id || '|' || person_label) = ANY(:k)))"),
        {"t": now, "v": MODEL_VERSION, "k": keep}).rowcount or 0
    return len(guesses), revoked


def run_person_model(conn, now: datetime, cfg: PersonModelConfig) -> dict:
    """One cycle: reconcile the album, refit the bars (the street they are
    calibrated on changes every cycle), re-guess."""
    tz = ZoneInfo(cfg.local_tz)
    if not tables_present(conn):
        return {"person_model": "skipped: apply services/orion-sql-db/manual_migration_vision_person_album_v1.sql"}
    rows, stats = desired_exemplars(conn, tz)
    added, revoked = reconcile_exemplars(conn, rows, now)
    album, fingerprint = load_album(conn, tz)
    street = unnamed_clusters(conn, now, cfg, tz)
    bars = fit_all(album, cfg, street)
    _write_bars(conn, bars, fingerprint, now)
    guesses = attribute(street, album, bars, _blocked(conn), cfg)
    n, gone = write_attributions(conn, guesses, now)
    return {
        "person_exemplars_added": added, "person_exemplars_revoked": revoked,
        "person_impure_clusters": stats["impure_clusters"],
        "person_people": len(album), "person_street_clusters": len(street),
        "person_accept_bars": sum(1 for b in bars.values() if b.accept_at is not None), "person_guesses": n,
        "person_accepts": sum(1 for g in guesses if g["band"] == "accept"),
        "person_guesses_revoked": gone,
    }
