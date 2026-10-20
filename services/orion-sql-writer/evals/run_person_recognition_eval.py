#!/usr/bin/env python3
"""Eval: is Orion learning who people are? (person learning, Phase 1)

docs/superpowers/specs/2026-10-20-walkway-person-learning-design.md,
"Acceptance checks". Leave-one-cluster-out over every cluster Juniper's
answers put in an album: the cluster's crops are removed from every album,
every bar is refitted without them, and the cluster is then guessed like an
unnamed one. Outcomes:

- correct accept: "probably <its own person>" (the cluster AND its person's
  other crops from the same days are held out, so clothes cannot carry it);
- WRONG accept: "probably <someone else>" -- the gate is zero of these;
- unsure: Orion would ask; reject: no person fits; no album: its person has
  no other cluster to learn from (cannot be recognised yet).

Also reported: street accepts, how many unnamed clusters get "probably X"
with the full album. An armed bar lets about alpha / people of the street
through by construction, so treat that number as likely wrong accepts until
answers confirm them. Per channel: cross-day separation, the AUC of genuine CLUSTER
scores (each of a person's clusters held out with every album crop from its
days) against impostor cluster scores (the street, other people, ruled-out
clusters), the populations the accept bar is fitted on. 0.5 = no better than
chance. A channel that only separates same-day clothes shows ~0.5 here.

Album rule is the live one (``desired_exemplars``: named clusters that pass
the purity split, plus "different" pictures as -1). READ ONLY on --live unless
--write, which inserts one ``vision_person_model_eval`` row. With a previous
row of a different model_version, the two are compared on the same clusters
with an exact McNemar test: a new version must be better at p < 0.05, not
just have a higher average.

    python evals/run_person_recognition_eval.py --live postgresql://postgres:...@localhost:55432/conjourney [--write]

Exit: 0 = gate passed (no wrong accepts); 1 = wrong accepts; 2 = nothing to evaluate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from app import vision_person_model as pm  # noqa: E402


def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """P(random genuine > random impostor), ties count half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar: b, c = discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def evaluate(album: Dict[str, Dict[int, List[pm.Crop]]], cfg: pm.PersonModelConfig,
             street: Dict[str, List[pm.Crop]]) -> dict:
    clusters: Dict[str, Tuple[str, List[pm.Crop]]] = {}
    for person, pol in album.items():
        for c in pol.get(1, []):
            clusters.setdefault(c.cluster, (person, []))[1].append(c)
    per_cluster = {}
    counts = {"correct_accept": 0, "wrong_accept": 0, "unsure": 0, "unsure_right_person": 0,
              "reject": 0, "no_album": 0}
    for cid, (truth, crops) in sorted(clusters.items()):
        # Hold out the cluster AND every crop of its person from the same days
        # (same clothes), as fit_bar's own gate does.
        days = {c.day for c in crops}
        held = {p: {pol: [c for c in cs if c.cluster != cid and not (p == truth and c.day in days)]
                    for pol, cs in pols.items()}
                for p, pols in album.items()}
        held = {p: pols for p, pols in held.items() if pols.get(1)}
        if truth not in held:
            counts["no_album"] += 1
            per_cluster[cid] = {"truth": truth, "outcome": "no_album"}
            continue
        bars = pm.fit_all(held, cfg, street)
        newest = sorted(crops, key=lambda c: c.day, reverse=True)[:pm.ATTRIBUTION_CROPS]
        guess = pm.attribute({cid: newest}, held, bars, {}, cfg)
        if not guess:
            outcome = "reject"
        elif guess[0]["band"] == "accept":
            outcome = "correct_accept" if guess[0]["person_label"] == truth else "wrong_accept"
        else:
            outcome = "unsure"
            counts["unsure_right_person"] += guess[0]["person_label"] == truth
        counts[outcome] += 1
        per_cluster[cid] = {"truth": truth, "outcome": outcome,
                            "guess": guess[0] if guess else None}
    separation = {}
    for ch in pm.CHANNELS:
        gen, imp, accept_bars = [], [], 0
        for person in album:
            if any(ch in c.vecs for c in album[person].get(1, [])):
                fit = pm.fit_bar(person, ch, album, cfg, street)
                gen += fit.genuine_clusters
                imp += fit.impostor_clusters
                accept_bars += fit.bar.accept_at is not None
        separation[ch] = {"cross_day_cluster_auc": auc(gen, imp), "n_genuine_clusters": len(gen),
                          "n_impostor_clusters": len(imp), "people_with_accept_bar": accept_bars}
    # Street accepts: guesses "probably X" on unnamed clusters with the full
    # album. About alpha / people of the street clears an armed bar by
    # construction, so read this as the bar's likely wrong-accept count until
    # Juniper's answers say otherwise.
    full_bars = pm.fit_all(album, cfg, street)
    street_guess = pm.attribute(street, album, full_bars, {}, cfg)
    street_accepts: Dict[str, int] = {}
    for g in street_guess:
        if g["band"] == "accept":
            street_accepts[g["person_label"]] = street_accepts.get(g["person_label"], 0) + 1
    scored = len(clusters) - counts["no_album"]
    return {
        "model_version": pm.MODEL_VERSION, "alpha": cfg.alpha, "top_k": cfg.top_k,
        "fallback_channel": cfg.fallback_channel, "people": len(album), "clusters": len(clusters),
        "street_clusters": len(street),
        **counts,
        "correct_accept_rate": round(counts["correct_accept"] / scored, 4) if scored else None,
        "unsure_rate": round(counts["unsure"] / scored, 4) if scored else None,
        "separation": separation, "per_cluster": per_cluster,
        "street_accepts": sum(street_accepts.values()), "street_accepts_by_person": street_accepts,
        "street_unsure": sum(1 for g in street_guess if g["band"] == "unsure"),
        "gate_passed": counts["wrong_accept"] == 0,
    }


def compare_previous(conn, metrics: dict) -> Optional[dict]:
    from sqlalchemy import text

    if conn.execute(text("SELECT to_regclass('vision_person_model_eval')")).scalar() is None:
        return None
    row = conn.execute(text(
        "SELECT model_version, metrics FROM vision_person_model_eval WHERE model_version <> :v "
        "ORDER BY ran_at DESC LIMIT 1"), {"v": metrics["model_version"]}).fetchone()
    if row is None:
        return None
    prev = row[1].get("per_cluster", {})
    common = sorted(set(prev) & set(metrics["per_cluster"]))
    ok = lambda d: d["outcome"] == "correct_accept"  # noqa: E731
    b = sum(1 for c in common if ok(metrics["per_cluster"][c]) and not ok(prev[c]))
    c_ = sum(1 for c in common if ok(prev[c]) and not ok(metrics["per_cluster"][c]))
    p = mcnemar_p(b, c_)
    return {"previous_version": row[0], "common_clusters": len(common), "new_only_right": b,
            "previous_only_right": c_, "mcnemar_p": round(p, 4), "better": b > c_ and p < 0.05}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from sqlalchemy import create_engine, text

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", required=True)
    ap.add_argument("--write", action="store_true", help="insert one vision_person_model_eval row")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--fallback-channel", default="body")
    ap.add_argument("--tz", default="America/Denver")
    args = ap.parse_args(argv)
    cfg = pm.PersonModelConfig(alpha=args.alpha, top_k=args.top_k, fallback_channel=args.fallback_channel,
                               local_tz=args.tz)
    tz = ZoneInfo(args.tz)
    engine = create_engine(args.live)
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rows, stats = pm.desired_exemplars(conn, tz)
        ex = [(r["person"], r["crop_id"], r["polarity"]) for r in rows]
        album = pm.build_album(ex, pm.load_crops_by_id(conn, (r[1] for r in ex), tz))
        if not album:
            print(json.dumps({"verdict": "nothing to evaluate", **stats}))
            return 2
        street = pm.unnamed_clusters(conn, datetime.now(timezone.utc), cfg, tz)
        metrics = evaluate(album, cfg, street)
        metrics["album"] = {**stats, "exemplars": len(ex)}
        metrics["vs_previous"] = compare_previous(conn, metrics)
    summary = {k: v for k, v in metrics.items() if k != "per_cluster"}
    print(json.dumps(summary, indent=2, default=str))
    if args.write:
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO vision_person_model_eval (run_id, model_version, ran_at, metrics) "
                "VALUES (:r, :v, :t, CAST(:m AS jsonb))"),
                {"r": f"eval-{uuid.uuid4().hex}", "v": pm.MODEL_VERSION, "t": datetime.now(timezone.utc),
                 "m": json.dumps(metrics, default=str)})
    return 0 if metrics["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
