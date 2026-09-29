#!/usr/bin/env python3
"""Replay Candidate A's node competition with and without the staleness fade.

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -AtF $'\\t' -c "
    with f as (select distinct on (date_trunc('minute',generated_at)) generated_at,
                 source_field_tick_id, frame_json
               from substrate_attention_frames
               where generated_at > now()-interval '72 hours'
               order by date_trunc('minute',generated_at), generated_at)
    select f.generated_at, x->>'target_id', x->'dominant_channels'->>'prediction_error',
           x->'reasons'->>0,
           fs.field_json->'node_vector_updated_at'->(x->>'target_id')->>'prediction_error'
    from f join substrate_field_state fs on fs.tick_id = f.source_field_tick_id
         and fs.generated_at > now()-interval '73 hours',
         jsonb_array_elements(coalesce(f.frame_json->'node_targets','[]'::jsonb)
                              || coalesce(f.frame_json->'suppressed_targets','[]'::jsonb)) x
    where x->>'target_id' like 'node:substrate.%'" > frames.tsv
    python scripts/analysis/replay_candidate_a_staleness_fade.py --frames frames.tsv

One frame per minute. For each of the five Candidate A targets the frame records
the error it used and its precision (reason text); raw salience is
``precision * |error|`` exactly as live. The reading's age comes from the field
state the frame was built from (``node_vector_updated_at[target].prediction_error``,
the time the digester applied that domain's last receipt -- the receipts
themselves are pruned after 30 min). "after" multiplies each error by the live
``prediction_error_staleness_factor(age)``; precision is unchanged by the fade
(it depends on variances, not on the current error), so the competition can be
recomputed without re-running the baselines. Top-1 is the argmax of raw salience
(min-max normalisation preserves the order).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from orion.attention.field_attention.candidate_precision_weighted import (  # noqa: E402
    PREDICTION_ERROR_STALENESS_HORIZON_SEC,
    prediction_error_staleness_factor,
)
from orion.attention.field_attention.selectors import PREDICTION_ERROR_NATIVE_TARGETS  # noqa: E402

_PRECISION_RE = re.compile(r"precision ([0-9.eE+-]+)")


def _ts(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def load(path: Path) -> dict[datetime, dict[str, tuple[float, float, datetime | None]]]:
    frames: dict[datetime, dict[str, tuple[float, float, datetime | None]]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        gen, target, err, reason, updated = parts[:5]
        if target not in PREDICTION_ERROR_NATIVE_TARGETS or not err:
            continue
        m = _PRECISION_RE.search(reason)
        if not m:
            continue
        frames[_ts(gen)][target] = (float(err), float(m.group(1)), _ts(updated))
    return frames


def replay(frames, *, stale_sec: float = PREDICTION_ERROR_STALENESS_HORIZON_SEC) -> dict:
    out = {
        "minutes": 0,
        "before": Counter(),
        "after": Counter(),
        "before_stale": Counter(),
        "after_stale": Counter(),
        "changed": 0,
        # Minutes where every competitor's raw salience is 0 (before / after the
        # fade): normalize_across_targets reads that set as 0.0, not a 1.0 tie.
        "before_all_zero": 0,
        "after_all_zero": 0,
    }
    for gen, targets in sorted(frames.items()):
        if not targets:
            continue
        out["minutes"] += 1
        before = {t: p * abs(e) for t, (e, p, _u) in targets.items()}
        after = {
            t: p * abs(e) * prediction_error_staleness_factor(u, now=gen)
            for t, (e, p, u) in targets.items()
        }
        ages = {t: ((gen - u).total_seconds() if u else None) for t, (_e, _p, u) in targets.items()}
        b = max(before, key=before.get)
        a = max(after, key=after.get)
        out["before"][b] += 1
        out["after"][a] += 1
        if ages[b] is not None and ages[b] > stale_sec:
            out["before_stale"][b] += 1
        if ages[a] is not None and ages[a] > stale_sec:
            out["after_stale"][a] += 1
        if a != b:
            out["changed"] += 1
        if max(before.values()) < 1e-12:
            out["before_all_zero"] += 1
        if max(after.values()) < 1e-12:
            out["after_all_zero"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--stale-min", type=float, default=PREDICTION_ERROR_STALENESS_HORIZON_SEC / 60.0,
                    help="age (minutes) above which a top-1 reading is counted as stale")
    args = ap.parse_args()
    frames = load(args.frames)
    r = replay(frames, stale_sec=args.stale_min * 60.0)
    span = sorted(frames)
    print(f"span {span[0].isoformat()} .. {span[-1].isoformat()}  minutes={r['minutes']}")
    print(f"top-1 changed on {r['changed']} minutes")
    m = f"{args.stale_min:g}"
    print(f"target | before top-1 (on a reading >{m} min old) | after top-1 (>{m} min old)")
    for t in PREDICTION_ERROR_NATIVE_TARGETS:
        print(
            f"{t} | {r['before'][t]} ({r['before_stale'][t]}) | {r['after'][t]} ({r['after_stale'][t]})"
        )
    print(f"minutes with every raw salience 0: before={r['before_all_zero']} after={r['after_all_zero']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
