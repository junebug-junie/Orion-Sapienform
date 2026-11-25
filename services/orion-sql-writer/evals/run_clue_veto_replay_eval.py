#!/usr/bin/env python3
"""Replay: would the clue veto have dropped the "Is this X?" questions Juniper said no to? (2026-11-24)

Design: docs/superpowers/specs/2026-11-24-walkway-clue-evidence-design.md, patch 3.

For every answered ``vision_person_confirm`` ask (source_ref ``<cluster>|<name>``,
answer yes / no), the cluster's crops are read, the asked name's album is taken
WITHOUT that cluster's crops (her yes must not vouch for itself), and
``vision_person_model.clue_llr`` is computed exactly as the live cycle does. The
ask counts as "vetoed" when it is at or below ``-clue_veto_llr``.

Gate (exit 0 only when all hold):
- vetoed questions she answered "yes" are at most 10% of all her "yes" answers
  (the veto must not cost many right guesses);
- among vetoed questions, the share she answered "no" is higher than her overall
  "no" share (the veto must pick out wrong guesses, not random ones).

READ ONLY. Names are printed to the terminal only.

    PYTHONPATH=. python services/orion-sql-writer/evals/run_clue_veto_replay_eval.py --live postgresql://...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from app import vision_person_model as pm  # noqa: E402

DEFAULT_PG = "postgresql://postgres:postgres@localhost:55432/conjourney"
MAX_YES_COST = 0.10


def gate(table: Dict[str, Counter]) -> dict:
    """table: {"vetoed"|"kept": Counter(yes=.., no=..)} -> verdict dict."""
    yes_all = table["vetoed"]["yes"] + table["kept"]["yes"]
    no_all = table["vetoed"]["no"] + table["kept"]["no"]
    vetoed = table["vetoed"]["yes"] + table["vetoed"]["no"]
    yes_cost = table["vetoed"]["yes"] / yes_all if yes_all else 0.0
    no_share_all = no_all / (yes_all + no_all) if yes_all + no_all else 0.0
    no_share_vetoed = table["vetoed"]["no"] / vetoed if vetoed else None
    return {"answered": yes_all + no_all, "vetoed": vetoed, "vetoed_yes": table["vetoed"]["yes"],
            "vetoed_no": table["vetoed"]["no"], "yes_cost": round(yes_cost, 4),
            "no_share_overall": round(no_share_all, 4),
            "no_share_vetoed": None if no_share_vetoed is None else round(no_share_vetoed, 4),
            "passed": bool(vetoed) and yes_cost <= MAX_YES_COST and no_share_vetoed > no_share_all}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from sqlalchemy import create_engine, text

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", default=os.getenv("ORION_EVAL_PG_URI", DEFAULT_PG))
    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--veto-llr", type=float, default=pm.PersonModelConfig.clue_veto_llr,
                    help="sensitivity only; the live value is PersonModelConfig.clue_veto_llr")
    args = ap.parse_args(argv)
    cfg = pm.PersonModelConfig(local_tz=args.tz, clue_veto_llr=args.veto_llr)
    tz = ZoneInfo(args.tz)
    with create_engine(args.live).connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rows, _stats = pm.desired_exemplars(conn, tz)
        ex = [(r["person"], r["crop_id"], r["polarity"]) for r in rows]
        album = pm.build_album(ex, pm.load_crops_by_id(conn, (r[1] for r in ex), tz))
        street, _face = pm.street_pools(conn, datetime.now(timezone.utc), cfg, tz)
        asks = conn.execute(text(
            "SELECT split_part(source_ref, '|', 1), split_part(source_ref, '|', 2), lower(btrim(answer)) "
            "FROM orion_ask WHERE source_kind = 'vision_person_confirm' AND status = 'answered' "
            "AND lower(btrim(answer)) IN ('yes', 'no')")).fetchall()
        clusters = {cid: pm._load_crops(conn, "individual_id = :i", {"i": cid}, tz) for cid in {a[0] for a in asks}}
        obs = pm.load_crop_clues(conn, [c.crop_id for pols in album.values() for c in pols.get(1, [])]
                                 + [c.crop_id for cs in street.values() for c in cs]
                                 + [c.crop_id for cs in clusters.values() for c in cs], cfg.clue_kinds)
    names = {p.casefold(): p for p in album}
    _cc, _pc, street_counts = pm.clue_inputs({}, street, obs)
    table: Dict[str, Counter] = {"vetoed": Counter(), "kept": Counter()}
    per_name: Dict[str, Counter] = {}
    examples: List[dict] = []
    for cid, name, answer in asks:
        person = names.get(name.strip().casefold())
        crops = clusters.get(cid) or []
        llr = None
        if person is not None and crops:
            mine = pm.clue_counts(obs[c.crop_id] for c in crops if c.crop_id in obs)
            theirs = pm.clue_counts(obs[c.crop_id] for c in album[person].get(1, [])
                                    if c.cluster != cid and c.crop_id in obs)
            llr = pm.clue_llr(mine, theirs, street_counts, cfg.clue_min_passages)
        vetoed = llr is not None and llr <= -cfg.clue_veto_llr
        table["vetoed" if vetoed else "kept"][answer] += 1
        if vetoed:
            per_name.setdefault(name, Counter())[answer] += 1
            if answer == "yes":
                examples.append({"cluster": cid, "name": name, "llr": round(llr, 2)})
    out = gate(table)
    out.update({"veto_llr": cfg.clue_veto_llr, "min_passages": cfg.clue_min_passages,
                "clue_kinds": list(cfg.clue_kinds), "street_clue_passages": {k: {"with": v[0], "of": v[1]} for k, v in street_counts.items()},
                "vetoed_by_name": {k: dict(v) for k, v in sorted(per_name.items())},
                "vetoed_yes_examples": examples[:10]})
    print(json.dumps(out, indent=2))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
