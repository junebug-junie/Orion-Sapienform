"""Family A per-bucket gate replay on live history, read-only.

Re-runs the issue-time hygiene gate (``family_a.gate_named_inputs`` on the 24 h window) for every 5-min
origin in the last ``--hours``, with the bands / floors of the run that was serving at that time, and
compares the refusals with what the live loop stored in ``prediction_joint_surprise``. With
``--compare-module`` it also runs a second copy of ``family_a.py`` (e.g. main's, via
``git show origin/main:services/orion-prediction/app/family_a.py > /tmp/fa_old.py``) for a before/after.

    cd services/orion-prediction
    POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_family_a_gate_replay_eval.py --hours 24 \\
        [--compare-module /tmp/fa_old.py]

Frozen-sensor check (#3370): runs selected before that fix stored no ``stats.max_day_mode_share``; for
those the ceiling is recomputed with the current ``screen_series`` from the 14 days before the run's
``fitted_at`` (what its select would have stored), and the per-run ceilings are printed.

Exit 0 when the current gate refuses no bucket for a degeneracy reason (one_value / few_distinct /
role_changed) or for flat_beyond_training, 1 otherwise. On healthy sensors both are false refusals; a
real frozen sensor shows up as flat_beyond_training and fails the run on purpose (look at it).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from app import family_a as fa
from app.family_a_load import load_frame

CONFIG = Path(__file__).resolve().parents[3] / "config" / "prediction" / "family_a.v1.yaml"
DEGENERACY = ("one_value", "few_distinct", "role_changed", "flat_beyond_training")
SELECT_DAYS = 14


def _module(path: str):
    spec = importlib.util.spec_from_file_location("family_a_compare", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["family_a_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _manifest(manifest: dict) -> tuple[dict, dict, dict, dict]:
    m = manifest.get("inputs", manifest)
    d = {k: v for k, v in m.items() if isinstance(v, dict) and v.get("status") in (fa.TARGET, fa.COVARIATE)}
    return ({k: v["status"] for k, v in d.items()}, {k: v["stats"]["band"] for k, v in d.items()},
            {k: v["stats"]["min"] for k, v in d.items()},
            {k: v["stats"].get("max_day_mode_share") for k, v in d.items()})


def _ceilings(frame: pd.DataFrame, named: dict, fitted_at: datetime, hyg) -> dict:
    """What a select at ``fitted_at`` would store (manifests from before #3370 lack it)."""
    w = frame.loc[(frame.index >= fitted_at - timedelta(days=SELECT_DAYS)) & (frame.index < fitted_at)]
    return {k: fa.screen_series(w[k], hyg, k).stats.get("max_day_mode_share") for k in named if k in w.columns}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--compare-module", default=None)
    ap.add_argument("--freeze", default=None, metavar="INPUT",
                    help="live smoke for #3370: hold this input at its median (a normal-looking, non-floor "
                         "value) from the replay start on, and report when each gate first refuses it")
    args = ap.parse_args()
    uri = os.environ.get("POSTGRES_URI")
    if not uri:
        ap.error("POSTGRES_URI is required")
    import psycopg

    cfg = fa.parse_config(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    end -= timedelta(minutes=end.minute % cfg.grain_min)
    start = end - timedelta(hours=args.hours)
    with psycopg.connect(uri) as c:
        c.execute("SET TRANSACTION READ ONLY")
        runs = c.execute("SELECT fitted_at, metrics->'manifest' FROM prediction_model_run WHERE kind = 'champion_blend' "
                         "AND model = 'family_a_blend' AND fitted_at < %s ORDER BY fitted_at", (end,)).fetchall()
        stored = Counter(r[0] for r in c.execute(
            "SELECT status FROM prediction_joint_surprise WHERE family = 'family_a' AND horizon_min = %s "
            "AND origin_ts >= %s AND origin_ts < %s", (cfg.horizons_min[0], start, end)).fetchall())
    runs = [(t.astimezone(timezone.utc), m) for t, m in runs]          # psycopg returns Etc/UTC
    first_fit = min((t for t, _m in runs if t <= end), default=start)
    lo = min(start - timedelta(hours=24), first_fit - timedelta(days=SELECT_DAYS))
    lo = lo.replace(second=0, microsecond=0) - timedelta(minutes=lo.minute % cfg.grain_min)   # on the grain grid
    frame, _ = load_frame(uri, cfg.inputs, cfg.grain_min, lo, end)
    if args.freeze:
        held = float(frame.loc[frame.index < start, args.freeze].median())
        frame.loc[frame.index >= start, args.freeze] = held
        print(f"froze {args.freeze} at {held:g} from {start:%Y-%m-%d %H:%M}Z")
    first_flat: dict = {}
    gates = {"current": fa}
    if args.compare_module:
        gates["compare"] = _module(args.compare_module)
    passed, refused = Counter(), {k: Counter() for k in gates}
    grain = timedelta(minutes=cfg.grain_min)
    ceil_cache: dict = {}
    for o in pd.date_range(start, end - grain, freq=f"{cfg.grain_min}min"):
        e = o.to_pydatetime() + grain
        now = e + timedelta(seconds=60)
        serving = [m for t, m in runs if t <= now]
        if not serving:
            continue
        fitted, manifest = [(t, m) for t, m in runs if t <= now][-1]
        named, bands, floors, ceilings = _manifest(manifest)
        if any(v is None for v in ceilings.values()):
            if fitted not in ceil_cache:
                ceil_cache[fitted] = _ceilings(frame, named, fitted, cfg.hygiene)
            ceilings = {k: (v if v is not None else ceil_cache[fitted].get(k)) for k, v in ceilings.items()}
        w = frame.loc[(frame.index >= e - timedelta(hours=24)) & (frame.index < e)]
        seen = {col: (w[col].dropna().index[-1].to_pydatetime() + grain) if w[col].notna().any() else None
                for col in w.columns}
        for name, mod in gates.items():
            try:
                extra = {"flat_ceilings": ceilings} if hasattr(mod, "flat_limit") else {}
                mod.gate_named_inputs(named, w, mod.Hygiene(**{k: v for k, v in cfg.hygiene.__dict__.items()
                                                              if k in mod.Hygiene.__dataclass_fields__}),
                                      seen, now, bands=bands, floors=floors, **extra)
                passed[name] += 1
            except mod.HygieneRefused as exc:
                refused[name][json.dumps(exc.refusals, sort_keys=True)] += 1
                if args.freeze and args.freeze in exc.refusals:
                    first_flat.setdefault(name, (o.to_pydatetime() - start, exc.refusals[args.freeze]))
    print(f"window [{start:%Y-%m-%d %H:%M}Z, {end:%Y-%m-%d %H:%M}Z), stored joint rows "
          f"({cfg.horizons_min[0]} min horizon): {dict(stored)}")
    for t, c in sorted(ceil_cache.items()):
        print(f"recomputed ceilings for the run fitted {t:%Y-%m-%d %H:%M}Z: "
              f"{ {k: v for k, v in sorted(c.items(), key=lambda kv: -(kv[1] or 0))[:4]} } (top 4)")
    for name in gates:
        print(f"{name}: pass={passed[name]} refused={sum(refused[name].values())} reasons={dict(refused[name])}")
    if args.freeze:
        for name in gates:
            hit = first_flat.get(name)
            print(f"{name}: first refused the frozen input after {hit[0]} ({list(hit[1])})" if hit
                  else f"{name}: never refused the frozen input (it read as calm)")
        return 0 if first_flat.get("current") else 1
    bad = [k for k in refused["current"] if any(r in k for r in DEGENERACY)]
    print("PASS" if not bad else f"FAIL: degeneracy refusals at issue time: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
