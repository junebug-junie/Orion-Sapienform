#!/usr/bin/env python3
"""Phase A of Causal Geometry v1: read-only observed-geometry measurement.

See docs/superpowers/specs/2026-07-16-causal-geometry-v1-design.md for the full
three-phase design. This script is Phase A only -- it measures the *observed*
causal structure between organ time-series already in Postgres and compares it
against the *designed* structure in
`config/field/orion_field_topology.v1.yaml`. It does not publish to the bus and
does not write to Postgres.

As of the follow-up producer-wiring change, the core measurement logic (data
fetch, resampling, lagged correlation, surrogate significance, divergence) has
moved to `orion/substrate/causal_geometry_engine.py`, an importable module, so
`orion/substrate/causal_geometry_producer.py` (called periodically from
`services/orion-field-digester/app/worker.py`) can call the exact same code
this CLI uses instead of duplicating it. This file re-exports every engine
name it previously defined, so existing callers/tests of this module are
unaffected. What remains here is CLI-only: argument parsing and human/JSON
output formatting.

What this measures
-------------------
Four independently-computed organ signal tables, verified live against
Postgres on 2026-07-16 (the design doc's original assumption of a generic
"state-journaler organ rollup" table was WRONG -- the real per-tick journaler
table only stores avg_valence/avg_arousal/avg_coherence/avg_novelty/
pct_missing/pct_stale/avg_distress, not the drive/salience/pressure channels
this feature needs):

1. `drive_audits` (jsonb `drive_pressures`, 6 keys: coherence, continuity,
   capability, relational, predictive, autonomy) -> channels `drive:<key>`.
   Timestamp: `COALESCE(observed_at, created_at)` (matches the table's own
   `idx_drive_audits_window` index expression). As of 2026-07-16 this table
   only has ~10.5 hours of history -- do not assume a 7-day window is always
   satisfiable; --window-hours is a request, not a guarantee, and this script
   reports the actual n_samples achieved per pair rather than pretending.
2. `attention_salience_trace` (scalar `salience` column, no sub-keys) ->
   single channel `attention:salience`. Timestamp: `created_at`.
3. `orion_biometrics_summary` (jsonb `pressures`, keys observed live: cpu,
   mem, net, disk, swap, power, gpu_mem, thermal, gpu_util) -> channels
   `biometrics:<key>`. Timestamp column is `character varying`, cast with
   `timestamp::timestamptz` in SQL. This table has 3 distinct `node` values
   (athena/atlas/circe) with wildly different row counts (athena is ~95% of
   all rows as of 2026-07-16) -- averaging across nodes would blend unrelated
   physical/software signal sources, so this script restricts to a single
   node via --biometrics-node (default: athena, the dominant node). There is
   no index on the `timestamp` column, so this query is a full scan filtered
   by node + cast timestamp; acceptable for a periodic offline report, not
   something to run in a hot path.
4. `self_state_predictions` (jsonb `prediction_json.predicted_dimension_scores`,
   keys observed live on 2026-07-16: agency_readiness, coherence,
   continuity_pressure, execution_pressure, field_intensity,
   introspection_pressure, reasoning_pressure, reliability_pressure,
   resource_pressure, social_pressure, transport_integrity, uncertainty --
   this is the actual key set from a live query, not the design doc's
   original guess, and is enumerated per-row rather than hardcoded, so a
   future added dimension is picked up automatically) -> channels
   `self_state:<key>`. Timestamp: `generated_at`.

Resampling
----------
All channels are bucketed onto a common grid of BUCKET_SECONDS (see the
constant below) using the mean of all raw points falling in each bucket.
Bucket size is a fixed module-level constant, not a per-run tuning knob,
because it was chosen from the actual observed sampling density of the
sparsest channel that matters (attention_salience_trace, ~1 point per ~5
minutes over its 8-day history as of 2026-07-16) -- a smaller bucket would
just manufacture empty buckets for that channel; a larger one would throw
away resolution in the much denser channels (self_state_predictions and
drive_audits both sample roughly every few seconds to ~10 seconds).

Lagged cross-correlation
------------------------
For every unordered pair of channels, this script tests both directions
(source leads target, and target leads source) across LAG_GRID_SECONDS,
using Pearson correlation on mean-aggregated, common-grid-aligned buckets.
The zero-lag co-sampled overlap size is used as the min-samples eligibility
gate (positive lags can only shrink alignment further, never grow it, so
zero-lag overlap is the maximum possible n for a pair). Pairs below
--min-samples are marked `insufficient_data` in the divergence list and
never produce a fabricated edge.

Significance: circular time-shift surrogates
---------------------------------------------
For the single (direction, lag) combination with the largest |r| for a pair,
this script draws --n-surrogates circular shifts of the target array (numpy
.roll by a uniformly-sampled non-zero offset), recomputes Pearson r each
time, and computes p_raw = (count(|r_surrogate| >= |r_observed|) + 1) /
(--n-surrogates + 1) (add-one smoothing, standard permutation-test
convention, avoids ever reporting p=0.0 off a finite sample).

Multiple-comparisons correction: p_raw alone understates the real false
positive rate, because "winner" was already selected as the best of 2
directions x len(LAG_GRID_SECONDS) lags -- i.e. up to 10 comparisons per
pair before the surrogate test ever runs. Reporting p_raw as-is would be
exactly the anti-conservative "statistical mirage" the design spec warns
against: picking the best of 10 candidates and then testing only that one
against a null built for a single comparison inflates the apparent
significance. This script applies a Bonferroni correction --
p = min(1.0, p_raw * n_comparisons) where n_comparisons = 2 *
len(LAG_GRID_SECONDS) -- before comparing against --alpha. Bonferroni is
conservative (some real edges near the boundary will be missed) rather than
a full max-statistic reconstruction (redoing the lag/direction search per
surrogate), which would be tighter but costs ~10x more surrogate
evaluations for a report-only nightly script; conservative-but-cheap was
chosen deliberately for v1. An edge is only emitted into `edges` if the
corrected p < --alpha.

Designed-vs-observed divergence
--------------------------------
Every other YAML edge/channel_map entry with no observed channel aliasing to
it also becomes a designed_only entry. Every significant observed edge whose
endpoints don't alias to any capability channel becomes observed_only.
Because two different capability channels (e.g. reasoning_pressure and
execution_pressure) can point at the *same* YAML edge (same source_id/
target_id, different channel_map key), this script disambiguates divergence
entries for those cases by suffixing target_id with `#<capability_channel>`,
e.g. `capability:orchestration#reasoning_pressure` -- otherwise two distinct
comparisons against the same edge would collide under one (source_id,
target_id) key.

No empty-shell output
----------------------
If literally zero edges clear the significance bar (e.g. an empty/near-empty
DB in CI, or a --window-hours too short to accumulate --min-samples anywhere),
`insufficient_data=True` is set on the snapshot and `notes` explains why. This
script never renders a plausible-looking graph backed by noise.

Usage
-----
    python scripts/causal_geometry_report.py
    python scripts/causal_geometry_report.py --window-hours 24 --json
    ORION_CAUSAL_GEOMETRY_PG_URI=postgresql://... python scripts/causal_geometry_report.py

This is a report-only diagnostic, not a pass/fail gate (mirrors
scripts/check_activation_saturation.py's report-only-by-default posture and
scripts/drive_state_divergence_audit.py's "always exits 0" convention for a
successful measurement run). It exits 2 only if the Postgres connection or
query itself fails -- there is no --fail-above-style content gate here, this
script only measures and reports, it never merges, blends, mutates the
lattice, or decides anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/causal_geometry_report.py` puts scripts/ on
# sys.path[0], which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/check_single_consumer_channels.py
# / scripts/drive_state_divergence_audit.py). This repo happens to have a
# scripts/platform/ package, which shadows stdlib `platform` -- and stdlib
# `uuid` imports `platform` internally, so this fix must run *before*
# anything (including stdlib) is imported below, not just before the
# repo-local imports.
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from typing import List, Optional  # noqa: E402

from orion.schemas.causal_geometry import CausalGeometrySnapshotV1  # noqa: E402

# Re-exported from the engine module so existing callers/tests of this file
# (e.g. tests/test_causal_geometry_report.py, which references cgr.<name>)
# keep working unchanged after the extraction.
from orion.substrate.causal_geometry_engine import (  # noqa: E402,F401
    BUCKET_SECONDS,
    DEFAULT_ALPHA,
    DEFAULT_BIOMETRICS_NODE,
    DEFAULT_FIELD_TOPOLOGY_PATH,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_N_SURROGATES,
    DEFAULT_POSTGRES_URI,
    DEFAULT_SEED,
    DEFAULT_WINDOW_HOURS,
    LAG_GRID_SECONDS,
    ChannelPoints,
    ChannelSeries,
    DirectionResult,
    PairResult,
    _aligned_arrays,
    _best_direction,
    _best_strength_for_channel,
    _bucket_idx_to_datetime,
    _pearson,
    build_capability_channel_index,
    build_divergence,
    build_edges,
    build_snapshot,
    bucketize,
    circular_shift_pvalue,
    compute_pairwise_results,
    fetch_channels,
    load_field_topology,
)


# --- CLI ---------------------------------------------------------------------


def _print_human_report(snapshot: CausalGeometrySnapshotV1) -> None:
    print(f"causal_geometry_report: snapshot_id={snapshot.snapshot_id}")
    print(f"window: {snapshot.window_start.isoformat()} .. {snapshot.window_end.isoformat()}")
    print(f"designed_topology_version: {snapshot.designed_topology_version}")
    print(f"insufficient_data: {snapshot.insufficient_data}")
    for note in snapshot.notes:
        print(f"note: {note}")
    print()

    if snapshot.edges:
        header = f"{'source_id':<28} {'target_id':<28} {'lag_sec':>8} {'strength':>9} {'p_value':>9} {'n':>6}"
        print("Observed edges (significant):")
        print(header)
        print("-" * len(header))
        for edge in sorted(snapshot.edges, key=lambda e: -abs(e.strength)):
            print(
                f"{edge.source_id:<28} {edge.target_id:<28} {edge.lag_sec:>8} "
                f"{edge.strength:>9.3f} {edge.significance:>9.4f} {edge.n_samples:>6}"
            )
        print()
    else:
        print("Observed edges (significant): none.")
        print()

    if snapshot.divergence:
        header = (
            f"{'source_id':<28} {'target_id':<34} {'status':<16} "
            f"{'observed':>9} {'designed':>9} {'delta':>9}"
        )
        print("Divergence (designed vs. observed):")
        print(header)
        print("-" * len(header))

        def _fmt(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v:.3f}"

        for entry in snapshot.divergence:
            print(
                f"{entry.source_id:<28} {entry.target_id:<34} {entry.status:<16} "
                f"{_fmt(entry.observed_strength):>9} {_fmt(entry.designed_weight):>9} {_fmt(entry.delta):>9}"
            )
    else:
        print("Divergence: no entries.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--postgres-uri",
        default=os.getenv("ORION_CAUSAL_GEOMETRY_PG_URI", DEFAULT_POSTGRES_URI),
        help=(
            "Postgres DSN. Defaults to $ORION_CAUSAL_GEOMETRY_PG_URI, falling back to "
            f"{DEFAULT_POSTGRES_URI!r} (this repo's standard local Postgres convention -- "
            "see services/orion-hub/.env_example's POSTGRES_URI / RECALL_PG_DSN)."
        ),
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=DEFAULT_WINDOW_HOURS,
        help=(
            f"lookback window in hours (default {DEFAULT_WINDOW_HOURS}, i.e. 7 days). This "
            "is a request, not a guarantee -- if a table has less history than this, the "
            "actual co-sampled n_samples per pair is reported honestly, never inflated."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"minimum co-sampled points (at zero lag) required to test a pair (default {DEFAULT_MIN_SAMPLES}).",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=DEFAULT_N_SURROGATES,
        help=f"number of circular-shift surrogates for significance testing (default {DEFAULT_N_SURROGATES}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"significance threshold on the surrogate p-value (default {DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--biometrics-node",
        default=DEFAULT_BIOMETRICS_NODE,
        help=(
            f"which orion_biometrics_summary.node to read (default {DEFAULT_BIOMETRICS_NODE!r}, "
            "the dominant node by row count as of 2026-07-16 -- averaging across nodes would "
            "blend unrelated physical/software signal sources)."
        ),
    )
    parser.add_argument(
        "--field-topology-path",
        default=DEFAULT_FIELD_TOPOLOGY_PATH,
        help="path to the designed field topology YAML (default: config/field/orion_field_topology.v1.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed for the circular-shift surrogate RNG, for reproducible runs (default {DEFAULT_SEED}).",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    if not args.postgres_uri.strip():
        print(
            "causal_geometry_report: no --postgres-uri given and $ORION_CAUSAL_GEOMETRY_PG_URI is unset.",
            file=sys.stderr,
        )
        return 2

    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=args.window_hours)

    try:
        channels, table_row_counts = fetch_channels(
            args.postgres_uri, window_start, biometrics_node=args.biometrics_node
        )
    except Exception as exc:
        print(f"causal_geometry_report: fetching channels failed -- {exc}", file=sys.stderr)
        return 2

    try:
        topology = load_field_topology(args.field_topology_path)
    except Exception as exc:
        print(f"causal_geometry_report: loading field topology failed -- {exc}", file=sys.stderr)
        return 2

    snapshot, _pair_results = build_snapshot(
        channels,
        topology,
        window_start=window_start,
        window_end=window_end,
        min_samples=args.min_samples,
        n_surrogates=args.n_surrogates,
        alpha=args.alpha,
        seed=args.seed,
        table_row_counts=table_row_counts,
    )

    if args.json:
        print(json.dumps(snapshot.model_dump(mode="json"), indent=2, default=str))
    else:
        _print_human_report(snapshot)

    return 0


if __name__ == "__main__":
    sys.exit(main())
