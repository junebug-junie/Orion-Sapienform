#!/usr/bin/env python3
"""Eval: does the shadow OSNet body vector tell people apart better than SigLIP?

docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md, "A better body
vector". READ ONLY: one READ ONLY transaction on the live DB, nothing written,
nothing copied, no pictures opened.

Uses only person crops that carry BOTH vectors (``embedding`` = SigLIP,
``osnet_embedding`` = OSNet, from manual_migration_vision_osnet_shadow_v1.sql),
so both models are scored on exactly the same crops and the same pairs.

Pair labels come from the camera's geometry, never from either model, so
neither model can grade itself:

- **Different people, same frame.** Two boxes in one frame are two people.
  Pairs whose boxes overlap (IoU > 0.3, or one mostly inside the other) are
  left out: they can be one person boxed twice.
- **Same person, next frame (track pair).** A crop and a crop in the next
  frame of the same stream, at most ``--max-gap-sec`` later, whose box centre
  moved at most one box height and whose height changed by less than 25%,
  matched both ways (mutual nearest) with no close second candidate (the
  runner-up is at least 1.5x farther). The design doc's proxy.
- **Juniper's verdicts.** Answered "who is this?" pictures
  (``vision_ask_picture_verdict``) whose crops both carry both vectors, expanded
  into same/different pairs by name the way the split replay eval does.

For each model it prints the cosine quantiles of each pair kind, the AUC (the
chance a random same-person pair scores above a random different-people pair;
0.5 = no better than chance), and the "different people let through" rate at
the line that keeps 90% of same-person pairs. Higher AUC and lower pass rate
win.

Insufficient data is reported as such, with the counts, and exits 2. Expect
that until about a day after orion-vision-host starts producing OSNet vectors.

    python evals/run_osnet_vs_siglip_eval.py --live postgresql://postgres:...@localhost:55432/conjourney

Exit: 0 = scored; 2 = cannot score yet (columns missing or too few pairs).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

MODELS = ("siglip", "osnet")
DEFAULT_MIN_PAIRS = 200
DEFAULT_MAX_GAP_SEC = 3.0
KEEP_SAME = 0.90


@dataclass
class Crop:
    crop_id: str
    observation_id: str
    stream_id: str
    observed_at: datetime
    box: Tuple[float, float, float, float]
    vecs: Dict[str, np.ndarray]


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 and np.isfinite(n) else a


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _iou_and_containment(a, b) -> Tuple[float, float]:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    small = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0), (inter / small if small > 0 else 0.0)


def _centre(b) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _height(b) -> float:
    return max(1e-6, b[3] - b[1])


def frames(crops: Sequence[Crop]) -> Dict[str, List[List[Crop]]]:
    """stream -> frames (crops of one observation), oldest first."""
    by_obs: Dict[Tuple[str, str], List[Crop]] = defaultdict(list)
    for c in crops:
        by_obs[(c.stream_id, c.observation_id)].append(c)
    out: Dict[str, List[List[Crop]]] = defaultdict(list)
    for (stream, _), fr in by_obs.items():
        out[stream].append(fr)
    for stream in out:
        out[stream].sort(key=lambda fr: (fr[0].observed_at, fr[0].observation_id))
    return out


def same_frame_pairs(by_stream: Dict[str, List[List[Crop]]]) -> List[Tuple[Crop, Crop]]:
    pairs = []
    for frs in by_stream.values():
        for fr in frs:
            for i in range(len(fr)):
                for j in range(i + 1, len(fr)):
                    iou, contain = _iou_and_containment(fr[i].box, fr[j].box)
                    if iou > 0.3 or contain > 0.7:
                        continue  # possibly one person boxed twice
                    pairs.append((fr[i], fr[j]))
    return pairs


def _nearest(c: Crop, cands: Sequence[Crop]) -> Tuple[Optional[int], float, float]:
    """Index of the geometric nearest candidate, its distance, and the runner-up's distance."""
    cx, cy = _centre(c.box)
    ds = []
    for k, d in enumerate(cands):
        h_ratio = _height(d.box) / _height(c.box)
        if not (0.8 <= h_ratio <= 1.25):
            continue
        dx, dy = _centre(d.box)
        dist = ((dx - cx) ** 2 + (dy - cy) ** 2) ** 0.5
        if dist <= _height(c.box):
            ds.append((dist, k))
    if not ds:
        return None, float("inf"), float("inf")
    ds.sort()
    return ds[0][1], ds[0][0], (ds[1][0] if len(ds) > 1 else float("inf"))


def track_pairs(by_stream: Dict[str, List[List[Crop]]], max_gap_sec: float) -> List[Tuple[Crop, Crop]]:
    pairs = []
    for frs in by_stream.values():
        for a_fr, b_fr in zip(frs, frs[1:]):
            gap = (b_fr[0].observed_at - a_fr[0].observed_at).total_seconds()
            if gap <= 0 or gap > max_gap_sec:
                continue
            for a in a_fr:
                k, best, second = _nearest(a, b_fr)
                if k is None or second < 1.5 * max(best, 1e-6):
                    continue
                b = b_fr[k]
                back, best_b, second_b = _nearest(b, a_fr)
                if back is None or a_fr[back] is not a or second_b < 1.5 * max(best_b, 1e-6):
                    continue
                pairs.append((a, b))
    return pairs


def auc(same: Sequence[float], diff: Sequence[float]) -> Optional[float]:
    """P(same > diff) + 0.5 P(tie), by ranks (Mann-Whitney U)."""
    if not same or not diff:
        return None
    s = np.asarray(same, dtype=np.float64)
    d = np.asarray(diff, dtype=np.float64)
    allv = np.concatenate([s, d])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[: len(s)].sum() - len(s) * (len(s) + 1) / 2.0
    return float(u / (len(s) * len(d)))


def _q(v: Sequence[float]) -> Dict[str, Optional[float]]:
    if not v:
        return {"n": 0, "p10": None, "p50": None, "p90": None}
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(len(a)), "p10": round(float(np.percentile(a, 10)), 3),
            "p50": round(float(np.percentile(a, 50)), 3), "p90": round(float(np.percentile(a, 90)), 3)}


def separation(same: Sequence[float], diff: Sequence[float]) -> Dict[str, object]:
    """Quantiles, AUC, and the different-people pass rate at the line keeping KEEP_SAME of same pairs."""
    out: Dict[str, object] = {"same": _q(same), "different": _q(diff), "auc": None,
                              "line_keeping_90pct_same": None, "different_let_through_at_line": None}
    a = auc(same, diff)
    if a is not None:
        line = float(np.percentile(np.asarray(same, dtype=np.float64), (1.0 - KEEP_SAME) * 100.0))
        out["auc"] = round(a, 4)
        out["line_keeping_90pct_same"] = round(line, 3)
        out["different_let_through_at_line"] = round(float(np.mean(np.asarray(diff) >= line)), 4)
    return out


def score_pairs(same_pairs, diff_pairs) -> Dict[str, Dict[str, object]]:
    res = {}
    for m in MODELS:
        same = [_cos(a.vecs[m], b.vecs[m]) for a, b in same_pairs]
        diff = [_cos(a.vecs[m], b.vecs[m]) for a, b in diff_pairs]
        res[m] = separation(same, diff)
    return res


def verdict_pairs(pictures, by_crop: Dict[str, Crop], clean_name) -> Tuple[list, list, int]:
    """(same pairs, different pairs, pictures usable) from answered picture verdicts.

    ``pictures``: rows with ask_id, picture_index, verdict ('same'/'different'),
    name (the picture's own name), answer (the card's typed answer), crop_id.
    A 'same' picture is the person the card's answer names; a 'different' one
    is the person its own name names. An unnamed picture is only known to
    differ from pictures on its own card. Same expansion as the split replay
    eval's calibration_pairs.
    """
    people = []
    for p in pictures:
        crop = by_crop.get(p.crop_id) if p.crop_id else None
        if crop is None:
            continue
        who = clean_name(p.answer if p.verdict == "same" else p.name).lower() or None
        people.append((p.ask_id, who or f"unnamed:{p.ask_id}:{p.picture_index}", who is not None, crop))
    same, diff = [], []
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            a, b = people[i], people[j]
            if a[3].crop_id == b[3].crop_id:
                continue
            if (not a[2] or not b[2]) and a[0] != b[0]:
                continue
            (same if a[1] == b[1] else diff).append((a[3], b[3]))
    return same, diff, len(people)


def _columns_present(conn) -> bool:
    from sqlalchemy import text

    have = {r[0] for r in conn.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
        "AND table_name = 'vision_crop_observation'")).fetchall()}
    return {"osnet_embedding", "osnet_model_id", "embedding_model_id"} <= have


def load_crops(conn, since: datetime, stream: Optional[str]) -> Tuple[List[Crop], Dict[str, object]]:
    from sqlalchemy import text

    where = "label = 'person' AND observed_at >= :since AND zone_no_embed = false"
    params: Dict[str, object] = {"since": since}
    if stream:
        where += " AND stream_id = :s"
        params["s"] = stream
    counts = conn.execute(text(
        f"SELECT count(*) FILTER (WHERE embedding IS NOT NULL) AS body, "
        f"count(*) FILTER (WHERE osnet_embedding IS NOT NULL) AS osnet, "
        f"count(*) FILTER (WHERE embedding IS NOT NULL AND osnet_embedding IS NOT NULL) AS both_, "
        f"min(observed_at) FILTER (WHERE osnet_embedding IS NOT NULL) AS first_osnet, "
        f"max(observed_at) FILTER (WHERE osnet_embedding IS NOT NULL) AS last_osnet "
        f"FROM vision_crop_observation WHERE {where}"), params).one()
    models = conn.execute(text(
        f"SELECT coalesce(embedding_model_id, '(not recorded)') AS b, coalesce(osnet_model_id, '-') AS o, "
        f"count(*) AS n FROM vision_crop_observation WHERE {where} AND osnet_embedding IS NOT NULL "
        f"GROUP BY 1, 2 ORDER BY 3 DESC"), params).fetchall()
    rows = conn.execute(text(
        f"SELECT crop_id, observation_id, stream_id, observed_at, box_xyxy, embedding, osnet_embedding "
        f"FROM vision_crop_observation WHERE {where} AND embedding IS NOT NULL AND osnet_embedding IS NOT NULL "
        f"ORDER BY observed_at, crop_id"), params)
    crops = []
    for r in rows:
        if not r.box_xyxy or len(r.box_xyxy) != 4:
            continue
        crops.append(Crop(r.crop_id, r.observation_id, r.stream_id, r.observed_at,
                          tuple(float(v) for v in r.box_xyxy),
                          {"siglip": _unit(r.embedding), "osnet": _unit(r.osnet_embedding)}))
    coverage = {
        "person_crops_with_body_vector": int(counts.body),
        "person_crops_with_osnet_vector": int(counts.osnet),
        "person_crops_with_both": int(counts.both_),
        "first_osnet_at": counts.first_osnet.isoformat() if counts.first_osnet else None,
        "last_osnet_at": counts.last_osnet.isoformat() if counts.last_osnet else None,
        "models": [{"body": m.b, "osnet": m.o, "crops": int(m.n)} for m in models],
    }
    return crops, coverage


def evaluate(crops: Sequence[Crop], pictures, clean_name, *, max_gap_sec: float, min_pairs: int) -> Dict[str, object]:
    by_stream = frames(crops)
    diff_pairs = same_frame_pairs(by_stream)
    same_pairs = track_pairs(by_stream, max_gap_sec)
    by_crop = {c.crop_id: c for c in crops}
    v_same, v_diff, v_pics = verdict_pairs(pictures, by_crop, clean_name)
    report: Dict[str, object] = {
        "pairs": {"same_person_next_frame": len(same_pairs), "different_people_same_frame": len(diff_pairs),
                  "verdict_pictures_with_both_vectors": v_pics, "verdict_same": len(v_same),
                  "verdict_different": len(v_diff)},
        "min_pairs": min_pairs,
    }
    enough = len(same_pairs) >= min_pairs and len(diff_pairs) >= min_pairs
    report["sufficient"] = enough
    if enough:
        report["geometry"] = score_pairs(same_pairs, diff_pairs)
        g = report["geometry"]
        report["winner"] = ("osnet" if g["osnet"]["auc"] > g["siglip"]["auc"] else
                            "siglip" if g["siglip"]["auc"] > g["osnet"]["auc"] else "tie")
    if v_same or v_diff:
        report["verdicts"] = score_pairs(v_same, v_diff)
    return report


def _print(report: Dict[str, object]) -> None:
    cov = report.get("coverage") or {}
    print(f"window: since {report['since']} stream={report['stream'] or 'all'}")
    print(f"person crops: body vector {cov.get('person_crops_with_body_vector')}, "
          f"OSNet vector {cov.get('person_crops_with_osnet_vector')}, both {cov.get('person_crops_with_both')} "
          f"(OSNet {cov.get('first_osnet_at')} .. {cov.get('last_osnet_at')})")
    for m in cov.get("models") or []:
        print(f"  body model {m['body']} + {m['osnet']}: {m['crops']} crops")
    pairs = report.get("pairs") or {}
    print(f"pairs: same person next frame {pairs.get('same_person_next_frame')}, "
          f"different people same frame {pairs.get('different_people_same_frame')}, "
          f"verdict same/different {pairs.get('verdict_same')}/{pairs.get('verdict_different')}")
    if not report.get("sufficient"):
        print(f"INSUFFICIENT DATA: need >= {report.get('min_pairs')} pairs of each geometry kind "
              f"with both vectors. {report.get('reason') or ''}".rstrip())
    for section in ("geometry", "verdicts"):
        res = report.get(section)
        if not res:
            continue
        print(f"\n{section}:")
        for m in MODELS:
            r = res[m]
            print(f"  {m:6s} same p10/p50/p90 {r['same']['p10']}/{r['same']['p50']}/{r['same']['p90']} (n={r['same']['n']})"
                  f"  different {r['different']['p10']}/{r['different']['p50']}/{r['different']['p90']} "
                  f"(n={r['different']['n']})  AUC {r['auc']}  line {r['line_keeping_90pct_same']} lets "
                  f"{r['different_let_through_at_line']} of different people through")
    if report.get("winner"):
        print(f"\nseparation winner on geometry pairs (AUC): {report['winner'].upper()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", required=True, help="DSN (read in a READ ONLY transaction)")
    ap.add_argument("--stream", default="walkway", help="stream_id, or '' for every stream")
    ap.add_argument("--since-hours", type=float, default=24.0 * 8)
    ap.add_argument("--max-gap-sec", type=float, default=DEFAULT_MAX_GAP_SEC)
    ap.add_argument("--min-pairs", type=int, default=DEFAULT_MIN_PAIRS)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from sqlalchemy import create_engine, text

    from app.vision_individuals import _clean_name

    since = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)
    report: Dict[str, object] = {"generated_at": datetime.now(timezone.utc).isoformat(),
                                 "since": since.isoformat(), "stream": args.stream or None}
    engine = create_engine(args.live)
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        if not _columns_present(conn):
            report.update(sufficient=False, reason="manual_migration_vision_osnet_shadow_v1.sql is not applied")
            crops, pictures = [], []
        else:
            crops, report["coverage"] = load_crops(conn, since, args.stream or None)
            pictures = conn.execute(text(
                "SELECT p.ask_id, p.picture_index, p.verdict, p.name, p.crop_id, a.answer "
                "FROM vision_ask_picture_verdict p JOIN orion_ask a ON a.ask_id = p.ask_id "
                "WHERE p.verdict IN ('same', 'different') AND p.crop_id IS NOT NULL "
                "ORDER BY p.ask_id, p.picture_index")).fetchall()
            report.update(evaluate(crops, pictures, _clean_name, max_gap_sec=args.max_gap_sec,
                                   min_pairs=args.min_pairs))
    engine.dispose()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print(report)
    return 0 if report.get("sufficient") else 2


if __name__ == "__main__":
    raise SystemExit(main())
