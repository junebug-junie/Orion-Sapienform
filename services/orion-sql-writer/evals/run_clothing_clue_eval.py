#!/usr/bin/env python3
"""Read-only: do Juniper's clothing / hair words separate a named person from the street? (2026-11-25)

Design: docs/superpowers/specs/2026-11-24-walkway-clue-evidence-design.md (clothes, hair, build:
"no answer key yet"). The answer key here is the person album itself (``desired_exemplars``):
pictures of a person she named. No new labels.

Each phrase from ``clothing_phrases.json`` (e.g. "a person wearing a blue denim shirt") was
turned into a SigLIP text vector once (``scripts/generate_clothing_phrase_vectors.py``). Every
stored person crop already has a SigLIP image vector. Per crop:

    score = cos(crop, phrase) - cos(crop, neutral "a photo of a person")

Per passage (one cluster, stream, minute; frames of one walk past are not independent) the
mean. Per (person, phrase): AUC of the person's album passages against street passages
(unnamed clusters, same window). 0.5 = the phrase says nothing; 1.0 = every one of their
passages scores above every street passage.

A phrase is a usable clue for a person only when the AUC is high AND clearly above what the
same phrase gives other named people (otherwise it reads the camera or the light, not them).
``--days`` limits both sides to recent passages (clothes change with the season).

Nothing is written. Names are printed to the terminal only.

    PYTHONPATH=. python services/orion-sql-writer/evals/run_clothing_clue_eval.py [--days 30] [--min-passages 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

DEFAULT_PG = "postgresql://postgres:postgres@localhost:55432/conjourney"
PHRASES = HERE.with_name("clothing_phrases.json")
VECTORS = HERE.with_name("clothing_phrase_vectors.json")


def load_text() -> Tuple[str, np.ndarray, Dict[str, np.ndarray]]:
    spec, vec = json.loads(PHRASES.read_text()), json.loads(VECTORS.read_text())
    sys.path.insert(0, str(HERE.parents[3] / "scripts"))
    from generate_clothing_phrase_vectors import phrase_hash

    if vec.get("phrase_hash") != phrase_hash(spec) or vec.get("model_id") != spec["model_id"]:
        raise SystemExit(f"{VECTORS.name} is stale: regenerate with scripts/generate_clothing_phrase_vectors.py")
    return (spec["model_id"], np.asarray(vec["neutral"], dtype=np.float64),
            {k: np.asarray(v, dtype=np.float64) for k, v in vec["phrases"].items()})


def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """P(a random pos > a random neg), ties count half. None when either side is empty."""
    if not pos or not neg:
        return None
    neg_sorted = np.sort(np.asarray(neg, dtype=np.float64))
    total = 0.0
    for p in pos:
        lo = np.searchsorted(neg_sorted, p, side="left")
        hi = np.searchsorted(neg_sorted, p, side="right")
        total += lo + 0.5 * (hi - lo)
    return float(total / (len(pos) * len(neg)))


def passage_scores(crops: Iterable[Tuple[str, np.ndarray]], neutral: np.ndarray,
                   phrases: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
    """(passage key, unit crop vector) -> phrase -> one mean score per passage."""
    by: Dict[str, List[np.ndarray]] = {}
    for key, v in crops:
        by.setdefault(key, []).append(v)
    out: Dict[str, List[float]] = {k: [] for k in phrases}
    for vs in by.values():
        m = np.stack(vs)
        base = m @ neutral
        for k, t in phrases.items():
            out[k].append(float(np.mean(m @ t - base)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    from sqlalchemy import create_engine, text

    from app import vision_person_model as pm

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pg", default=os.getenv("ORION_EVAL_PG_URI", DEFAULT_PG))
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--min-passages", type=int, default=10)
    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--combine", action="append", default=[], metavar="NAME=phrase,phrase",
                    help="also score the mean of these phrases for this person (names stay out of the repo)")
    args = ap.parse_args(argv)
    model_id, neutral, phrases = load_text()
    cfg = pm.PersonModelConfig(local_tz=args.tz, candidate_days=args.days)
    tz = ZoneInfo(args.tz)
    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).astimezone(tz).date()
    with create_engine(args.pg).connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rows, _ = pm.desired_exemplars(conn, tz)
        ex = [(r["person"], r["crop_id"], r["polarity"]) for r in rows]
        album = pm.build_album(ex, pm.load_crops_by_id(conn, (r[1] for r in ex), tz))
        street, _ = pm.street_pools(conn, datetime.now(timezone.utc), cfg, tz)
        ids = [c.crop_id for p in album.values() for c in p.get(1, [])] + [c.crop_id for cs in street.values() for c in cs]
        obs = pm.load_crop_clues(conn, ids, ("stroller",))   # used only for each crop's passage key

    def usable(crops):
        for c in crops:
            v = c.vecs.get("body")
            if v is not None and c.body_model == model_id and c.crop_id in obs and date.fromisoformat(c.day) >= since:
                yield obs[c.crop_id][0], v

    street_s = passage_scores(usable(c for cs in street.values() for c in cs), neutral, phrases)
    n_street = len(next(iter(street_s.values()), []))
    people = {p: passage_scores(usable(pols.get(1, [])), neutral, phrases) for p, pols in album.items()}
    people = {p: s for p, s in people.items() if len(next(iter(s.values()), [])) >= args.min_passages}
    print(f"last {args.days:g} days; street passages {n_street}; people with >= {args.min_passages} passages: {len(people)}")
    if not n_street or not people:
        print("PROBLEM nothing to compare")
        return 1
    table = {p: {k: auc(s[k], street_s[k]) for k in phrases} for p, s in people.items()}
    for k in phrases:
        vals = sorted(((table[p][k], p) for p in table), reverse=True)
        others = [a for a, _ in vals]
        line = ", ".join(f"{p} {a:.2f} (n={len(people[p][k])})" for a, p in vals)
        print(f"{k}: {line}  | median {float(np.median(others)):.2f}")
    for spec in args.combine:
        name, _, keys = spec.partition("=")
        ks = [k for k in keys.split(",") if k in phrases]
        if name not in people or not ks:
            print(f"combine {spec!r}: unknown person or phrases")
            continue
        own = np.mean([people[name][k] for k in ks], axis=0).tolist()
        st = np.mean([street_s[k] for k in ks], axis=0).tolist()
        rank = sorted(((auc(np.mean([people[p][k] for k in ks], axis=0).tolist(), st) or 0, p) for p in people),
                      reverse=True)
        print(f"combined {'+'.join(ks)}: {name} {auc(own, st):.2f}; rank {[p for _, p in rank].index(name) + 1}"
              f" of {len(rank)}; next best {next(a for a, p in rank if p != name):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
