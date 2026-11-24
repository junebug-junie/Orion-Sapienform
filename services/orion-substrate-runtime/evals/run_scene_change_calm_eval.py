"""7-day calm check for camera scene change (PE retire Patch 3, design rev 4 acceptance checks).

Read-only. Live only (needs POSTGRES_URI; from the host use the published port, e.g.
``postgresql://postgres:<pw>@localhost:55432/conjourney``).

    PYTHONDONTWRITEBYTECODE=1 POSTGRES_URI=... python services/orion-substrate-runtime/evals/run_scene_change_calm_eval.py --days 7

Exit 0 = every gate passed, 1 = a gate failed, 3 = insufficient data (not deployed, migration not
applied, or a camera with fewer than ``--min-rows`` v2 rows), 2 = no POSTGRES_URI.

Gates:

1. History, ``substrate_node_signal_history`` rows with ``scale = 'v2'`` ONLY (``v2_partial`` rows,
   written while stage 2 held less than 24 h, are excluded from calm checks and win share), per
   camera over the window: exact-zero share in [0.58, 0.72] (theory 1 - 1/e = 0.632), share above
   2.0 nats in [0.025, 0.065] (theory e^-3 = 0.050), max <= ln(289) - 1 = 4.67.
2. Curiosity, ``substrate_endogenous_curiosity_candidates`` over the same window: each camera seeds
   (a ``source:scene_change`` candidate naming it) in [2.5 %, 6.5 %] of candidate sets; camera
   seeds total <= 250 per 7 days; in sets holding 3 or more candidates, no other seed source
   (focal node + source note) with >= 10 seeds in the 7 days before ``--cutover`` loses more than
   20 % of its count.

Reported, not gated: each camera's attention win share (share of
``substrate_attention_broadcast_log`` ticks whose attended_node_ids name it), 7 days before
``--cutover`` vs the window, the window starting no earlier than the camera's first ``v2`` row.

``--cutover`` defaults to the earliest camera row with scale v2 / v2_partial (the deploy).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

CAMERAS = ("node:substrate.perception", "node:substrate.perception_walkway")
CAP_NATS = math.log(289) - 1.0
EXACT_ZERO_BAND = (0.58, 0.72)
CROSS_BAND = (0.025, 0.065)
CROSS_NATS = 2.0
SEED_SHARE_BAND = (0.025, 0.065)
MAX_CAMERA_SEEDS = 250
FULL_SET = 3
MAX_DISPLACEMENT = 0.20
MIN_BEFORE_SEEDS = 10

# Every read of a camera's signal rows names its scale (scripts/check_pe_alarm_readers.py).
CUTOVER_SQL = (
    "SELECT min(observed_at) FROM substrate_node_signal_history "
    "WHERE node_id = ANY(%s) AND scale IN ('v2','v2_partial')"
)
FIRST_V2_SQL = (
    "SELECT node_id, min(observed_at) FROM substrate_node_signal_history "
    "WHERE node_id = ANY(%s) AND scale = 'v2' GROUP BY node_id"
)
ROWS_SQL = (
    "SELECT node_id, value FROM substrate_node_signal_history "
    "WHERE node_id = ANY(%s) AND scale = 'v2' AND observed_at >= %s AND observed_at < %s"
)
SETS_SQL = (
    "SELECT generated_at, candidates_json FROM substrate_endogenous_curiosity_candidates "
    "WHERE generated_at >= %s AND generated_at < %s"
)
WINS_SQL = (
    "SELECT generated_at, projection_json->'attended_node_ids' FROM substrate_attention_broadcast_log "
    "WHERE generated_at >= %s AND generated_at < %s"
)


@dataclass(frozen=True)
class Check:
    name: str
    value: float | None
    ok: bool | None  # None = insufficient data
    detail: str = ""


def history_checks(node_id: str, values: Sequence[float], *, min_rows: int) -> list[Check]:
    n = len(values)
    if n < min_rows:
        return [Check(f"{node_id} history", float(n), None, f"insufficient: {n} v2 rows < {min_rows}")]
    zero = sum(1 for v in values if v == 0.0) / n
    cross = sum(1 for v in values if v > CROSS_NATS) / n
    top = max(values)
    return [
        Check(f"{node_id} exact-zero share", zero, EXACT_ZERO_BAND[0] <= zero <= EXACT_ZERO_BAND[1],
              f"band {EXACT_ZERO_BAND}, theory 0.632, n={n}"),
        Check(f"{node_id} share > {CROSS_NATS} nats", cross, CROSS_BAND[0] <= cross <= CROSS_BAND[1],
              f"band {CROSS_BAND}, theory 0.050, n={n}"),
        Check(f"{node_id} max", top, top <= CAP_NATS + 1e-9, f"cap {CAP_NATS:.3f}"),
    ]


def _seeds(candidates: object) -> list[tuple[str, str]]:
    """(focal node, source note) per candidate in one persisted set."""
    out = []
    for c in candidates if isinstance(candidates, list) else []:
        if not isinstance(c, dict):
            continue
        refs = c.get("focal_node_refs") or []
        source = next((n for n in (c.get("notes") or []) if str(n).startswith("source:")), "source:?")
        out.append((str(refs[0]) if refs else "", str(source)))
    return out


def curiosity_checks(after_sets: Sequence[object], before_sets: Sequence[object]) -> list[Check]:
    n_sets = len(after_sets)
    if not n_sets:
        return [Check("curiosity candidate sets", 0.0, None, "insufficient: no candidate sets")]
    checks: list[Check] = []
    camera_total = 0
    for camera in CAMERAS:
        hits = sum(1 for s in after_sets if (camera, "source:scene_change") in _seeds(s))
        camera_total += sum(_seeds(s).count((camera, "source:scene_change")) for s in after_sets)
        share = hits / n_sets
        checks.append(Check(f"{camera} seeds in share of sets", share,
                            SEED_SHARE_BAND[0] <= share <= SEED_SHARE_BAND[1],
                            f"band {SEED_SHARE_BAND}, theory 0.050, sets={n_sets}"))
    checks.append(Check("camera seeds total (7 d)", float(camera_total), camera_total <= MAX_CAMERA_SEEDS,
                        f"<= {MAX_CAMERA_SEEDS}"))

    def full_counts(sets: Iterable[object]) -> Counter:
        c: Counter = Counter()
        for s in sets:
            seeds = _seeds(s)
            if len(seeds) >= FULL_SET:
                c.update(k for k in seeds if k[0] not in CAMERAS)
        return c

    before, after = full_counts(before_sets), full_counts(after_sets)
    worst = None
    for key, n_before in before.items():
        if n_before < MIN_BEFORE_SEEDS:
            continue
        drop = 1.0 - after.get(key, 0) / n_before
        if worst is None or drop > worst[1]:
            worst = (key, drop)
    if worst is None:
        checks.append(Check("displacement in full sets", None, None,
                            f"insufficient: no seed source with >= {MIN_BEFORE_SEEDS} seeds before"))
    else:
        checks.append(Check("displacement in full sets (worst drop)", worst[1], worst[1] <= MAX_DISPLACEMENT,
                            f"{worst[0][0]} {worst[0][1]}, <= {MAX_DISPLACEMENT}"))
    return checks


def win_share(rows: Sequence[tuple[datetime, object]], node_id: str, since: datetime | None) -> tuple[float | None, int]:
    ticks = [ids for t, ids in rows if since is None or t >= since]
    if not ticks:
        return None, 0
    return sum(1 for ids in ticks if isinstance(ids, list) and node_id in ids) / len(ticks), len(ticks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--cutover", type=str, default=None, help="ISO time of the deploy (default: first new-scale row)")
    ap.add_argument("--min-rows", type=int, default=500)
    args = ap.parse_args(argv)
    uri = os.environ.get("POSTGRES_URI")
    if not uri:
        print("POSTGRES_URI not set (live-only eval)", file=sys.stderr)
        return 2
    import psycopg2

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    with psycopg2.connect(uri) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        if args.cutover:
            cutover = datetime.fromisoformat(args.cutover.replace("Z", "+00:00"))
        else:
            try:
                cur.execute(CUTOVER_SQL, (list(CAMERAS),))
            except psycopg2.errors.UndefinedColumn:
                print("substrate_node_signal_history has no scale column: "
                      "manual_migration_node_signal_history_scale_v1.sql not applied (insufficient)")
                return 3
            cutover = cur.fetchone()[0]
        if cutover is None:
            print("no camera scene-change rows yet: Patch 3 not deployed (insufficient)")
            return 3
        start = max(start, cutover)
        cur.execute(FIRST_V2_SQL, (list(CAMERAS),))
        first_v2 = dict(cur.fetchall())
        cur.execute(ROWS_SQL, (list(CAMERAS), start, now))
        rows = cur.fetchall()
        cur.execute(SETS_SQL, (start, now))
        after_sets = [c for _, c in cur.fetchall()]
        cur.execute(SETS_SQL, (cutover - timedelta(days=7), cutover))
        before_sets = [c for _, c in cur.fetchall()]
        cur.execute(WINS_SQL, (cutover - timedelta(days=7), now))
        wins = cur.fetchall()

    print(f"window {start:%Y-%m-%dT%H:%MZ} .. {now:%Y-%m-%dT%H:%MZ} (cutover {cutover:%Y-%m-%dT%H:%MZ}); "
          f"history rows scale='v2' only")
    checks: list[Check] = []
    for camera in CAMERAS:
        checks += history_checks(camera, [float(v) for n, v in rows if n == camera], min_rows=args.min_rows)
    checks += curiosity_checks(after_sets, before_sets)
    for c in checks:
        verdict = "INSUFFICIENT" if c.ok is None else "PASS" if c.ok else "FAIL"
        value = "-" if c.value is None else f"{c.value:.4f}"
        print(f"{verdict:12s} {c.name}: {value}  ({c.detail})")
    for camera in CAMERAS:
        before, nb = win_share([(t, i) for t, i in wins if t < cutover], camera, None)
        after, na = win_share([(t, i) for t, i in wins if t >= cutover], camera, first_v2.get(camera))
        print(f"INFO         {camera} attention win share: before {before} (ticks {nb}) -> after {after} "
              f"(ticks {na}, from first v2 row {first_v2.get(camera)})")
    if any(c.ok is False for c in checks):
        return 1
    if any(c.ok is None for c in checks):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
