#!/usr/bin/env python3
"""Phase A of Causal Geometry v1: read-only observed-geometry measurement.

See docs/superpowers/specs/2026-07-16-causal-geometry-v1-design.md for the full
three-phase design. This script is Phase A only -- it measures the *observed*
causal structure between organ time-series already in Postgres and compares it
against the *designed* structure in
`config/field/orion_field_topology.v1.yaml`. It does not publish to the bus,
does not write to Postgres, and runs no reducer/worker process (see the
design doc's "Files likely to touch" -- Phase B/C are separate rungs).

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
`config/field/orion_field_topology.v1.yaml` edges carry a `channel_map` from
a node/capability-local channel name (e.g. `reasoning_load`) to a shared
*capability channel* name (e.g. `reasoning_pressure`) listed under
`capability_channels`. This script builds a reverse index from capability
channel name -> list of (source_id, target_id, weight) YAML edges, then
checks every observed channel's suffix (the part after the first `:`, e.g.
`self_state:reasoning_pressure` -> `reasoning_pressure`) against that index.
A match is "plausible" purely by exact string identity -- e.g.
`self_state:reasoning_pressure`, `self_state:execution_pressure`, and
`self_state:reliability_pressure` all match capability channel names
verbatim (verified against a live sample on 2026-07-16; see
docs/PR-field-lattice-cap-cap-edges.md for how those specific capability
channels are wired into the lattice). No other observed channel prefix
(drive:*, attention:*, biometrics:*) happened to collide with a capability
channel name in the live data sampled while building this script, but the
matching is generic (name-based, not hardcoded to a fixed list), so it will
pick up new collisions automatically if the schemas drift closer together.

For a matched capability channel, if this script found a *significant*
edge touching that observed channel, the divergence entry gets
status="both" (observed_strength = that edge's strength, designed_weight =
the YAML edge's weight). If no significant edge was found, status is
"designed_only" (the YAML wiring exists but nothing observed backs it yet --
that is still meaningfully different from "no YAML edge exists at all", but
the divergence entry schema doesn't have a slot for "designed but
unconfirmed" vs "designed and no attempt was even matched", so both cases
collapse to designed_only; the human-readable table and JSON `notes` list
which capability channels had zero co-sampled observed data behind them).
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
import uuid  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from orion.schemas.causal_geometry import (  # noqa: E402
    CausalGeometryDivergenceEntryV1,
    CausalGeometryEdgeV1,
    CausalGeometrySnapshotV1,
)

# --- documented, fixed knobs (see module docstring for rationale) ----------

# 5-minute buckets: matches the sampling density of the sparsest channel that
# matters for cross-correlation (attention_salience_trace, ~1 point per ~5
# minutes over its full history as of 2026-07-16).
BUCKET_SECONDS = 300

# 0, 5, 10, 15, 30 minutes. Kept small: this is lagged correlation over
# organ-scale signals, not a long-horizon forecast; the design doc explicitly
# defers transfer entropy / Granger-style modeling to a later phase.
LAG_GRID_SECONDS: Tuple[int, ...] = (0, 300, 600, 900, 1800)

DEFAULT_WINDOW_HOURS = 168.0
DEFAULT_MIN_SAMPLES = 30
DEFAULT_N_SURROGATES = 200
DEFAULT_ALPHA = 0.05
DEFAULT_BIOMETRICS_NODE = "athena"
DEFAULT_SEED = 1337

# Local Postgres convention used across this repo for host-side access, e.g.
# services/orion-hub/.env_example's RECALL_PG_DSN / POSTGRES_URI and
# services/orion-cortex-orch/.env_example's commented RECALL_PG_DSN, both
# postgresql://postgres:postgres@127.0.0.1:55432/conjourney (POSTGRES_PORT
# default 55432 per services/orion-sql-db/.env_example and .env_example at
# repo root).
DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"

DEFAULT_FIELD_TOPOLOGY_PATH = str(
    Path(_REPO_ROOT) / "config" / "field" / "orion_field_topology.v1.yaml"
)

ChannelPoints = List[Tuple[datetime, float]]
ChannelSeries = Dict[str, ChannelPoints]


# --- data fetch (real DB) ---------------------------------------------------


def fetch_channels(
    postgres_uri: str,
    window_start: datetime,
    *,
    biometrics_node: str = DEFAULT_BIOMETRICS_NODE,
) -> Tuple[ChannelSeries, Dict[str, int]]:
    """Pull and flatten all four organ signal tables into named channels.

    Uses psycopg2 (already a dependency of several services, e.g.
    services/orion-sql-writer, and available in this repo's shared dev venv;
    no new dependency added). Each row's jsonb column is flattened into one
    channel per key (`drive:<key>`, `biometrics:<key>`, `self_state:<key>`);
    `attention_salience_trace.salience` has no sub-keys and becomes the
    single scalar channel `attention:salience`.

    Returns `(channels, table_row_counts)`. `table_row_counts` maps each
    source table name to the raw row count pulled in the window -- a table
    with 0 rows in the window contributes zero channels and would otherwise
    be silently invisible from the rest of the report (verified live on
    2026-07-16: an 8h window found 0 attention_salience_trace rows, since
    that table's most recent row was already >8h old -- without this count,
    "27 channels found" gives no hint that a whole source table went dark).
    """
    import psycopg2

    channels: ChannelSeries = {}
    table_row_counts: Dict[str, int] = {}

    def _append(channel_id: str, ts: datetime, value: Any) -> None:
        if value is None or isinstance(value, bool):
            return
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            return
        channels.setdefault(channel_id, []).append((ts, fvalue))

    conn = psycopg2.connect(postgres_uri)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(observed_at, created_at) AS ts, drive_pressures
                FROM drive_audits
                WHERE COALESCE(observed_at, created_at) >= %s
                ORDER BY ts
                """,
                (window_start,),
            )
            rows = cur.fetchall()
            table_row_counts["drive_audits"] = len(rows)
            for ts, drive_pressures in rows:
                for key, value in (drive_pressures or {}).items():
                    _append(f"drive:{key}", ts, value)

            cur.execute(
                """
                SELECT created_at AS ts, salience
                FROM attention_salience_trace
                WHERE created_at >= %s
                ORDER BY ts
                """,
                (window_start,),
            )
            rows = cur.fetchall()
            table_row_counts["attention_salience_trace"] = len(rows)
            for ts, salience in rows:
                _append("attention:salience", ts, salience)

            cur.execute(
                """
                SELECT timestamp::timestamptz AS ts, pressures
                FROM orion_biometrics_summary
                WHERE node = %s AND timestamp::timestamptz >= %s
                ORDER BY ts
                """,
                (biometrics_node, window_start),
            )
            rows = cur.fetchall()
            table_row_counts["orion_biometrics_summary"] = len(rows)
            for ts, pressures in rows:
                for key, value in (pressures or {}).items():
                    _append(f"biometrics:{key}", ts, value)

            cur.execute(
                """
                SELECT generated_at AS ts, prediction_json->'predicted_dimension_scores' AS scores
                FROM self_state_predictions
                WHERE generated_at >= %s
                ORDER BY ts
                """,
                (window_start,),
            )
            rows = cur.fetchall()
            table_row_counts["self_state_predictions"] = len(rows)
            for ts, scores in rows:
                for key, value in (scores or {}).items():
                    _append(f"self_state:{key}", ts, value)
    finally:
        conn.close()

    for points in channels.values():
        points.sort(key=lambda p: p[0])
    return channels, table_row_counts


# --- resampling + lagged correlation ----------------------------------------


def bucketize(points: ChannelPoints, bucket_seconds: int = BUCKET_SECONDS) -> Dict[int, float]:
    """Mean-aggregate raw (timestamp, value) points onto a bucket-index grid.

    Bucket index is `int(ts.timestamp() // bucket_seconds)`. Empty buckets are
    simply absent from the returned dict -- this script never interpolates or
    forward-fills, since a fabricated value in a gap could manufacture a
    spurious correlation exactly opposite of this feature's purpose.
    """
    grouped: Dict[int, List[float]] = {}
    for ts, value in points:
        idx = int(ts.timestamp() // bucket_seconds)
        grouped.setdefault(idx, []).append(value)
    return {idx: float(np.mean(vals)) for idx, vals in grouped.items()}


def _aligned_arrays(
    source_buckets: Dict[int, float], target_buckets: Dict[int, float], lag_buckets: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair up source[idx] with target[idx + lag_buckets] for every idx present
    in both, i.e. target observed `lag_buckets` buckets after source."""
    xs: List[float] = []
    ys: List[float] = []
    for idx in sorted(source_buckets.keys()):
        j = idx + lag_buckets
        if j in target_buckets:
            xs.append(source_buckets[idx])
            ys.append(target_buckets[j])
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(r):
        return 0.0
    return r


class DirectionResult:
    __slots__ = ("source_id", "target_id", "lag_sec", "r", "n", "xs", "ys")

    def __init__(self, source_id: str, target_id: str, lag_sec: int, r: float, n: int, xs: np.ndarray, ys: np.ndarray):
        self.source_id = source_id
        self.target_id = target_id
        self.lag_sec = lag_sec
        self.r = r
        self.n = n
        self.xs = xs
        self.ys = ys


def _best_direction(
    source_id: str,
    target_id: str,
    source_buckets: Dict[int, float],
    target_buckets: Dict[int, float],
    lag_grid_seconds: Tuple[int, ...],
    bucket_seconds: int,
) -> Optional[DirectionResult]:
    best: Optional[DirectionResult] = None
    for lag_sec in lag_grid_seconds:
        lag_buckets = lag_sec // bucket_seconds
        xs, ys = _aligned_arrays(source_buckets, target_buckets, lag_buckets)
        if len(xs) < 2:
            continue
        r = _pearson(xs, ys)
        if r is None:
            continue
        if best is None or abs(r) > abs(best.r):
            best = DirectionResult(source_id, target_id, lag_sec, r, len(xs), xs, ys)
    return best


def circular_shift_pvalue(
    xs: np.ndarray,
    ys: np.ndarray,
    observed_r: float,
    *,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Circular time-shift surrogate p-value for a Pearson correlation.

    Shifts `ys` by a uniformly-sampled non-zero circular offset (numpy.roll)
    `n_surrogates` times, recomputes Pearson r against the fixed `xs` each
    time, and returns the add-one-smoothed fraction of surrogate |r| >=
    observed |r|. Add-one smoothing (standard permutation-test convention)
    means this never reports p=0.0 off a finite surrogate sample.
    """
    n = len(ys)
    if n < 2:
        return 1.0
    if rng is None:
        rng = np.random.default_rng()
    observed_abs = abs(observed_r)
    count_ge = 0
    for _ in range(n_surrogates):
        shift = int(rng.integers(1, n)) if n > 1 else 0
        shifted = np.roll(ys, shift)
        r = _pearson(xs, shifted)
        if r is not None and abs(r) >= observed_abs:
            count_ge += 1
    return (count_ge + 1) / (n_surrogates + 1)


class PairResult:
    __slots__ = ("channel_a", "channel_b", "n_lag0", "winner", "significant", "p_value")

    def __init__(self, channel_a: str, channel_b: str, n_lag0: int):
        self.channel_a = channel_a
        self.channel_b = channel_b
        self.n_lag0 = n_lag0
        self.winner: Optional[DirectionResult] = None
        self.significant = False
        self.p_value: Optional[float] = None


def compute_pairwise_results(
    channel_buckets: Dict[str, Dict[int, float]],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    alpha: float = DEFAULT_ALPHA,
    lag_grid_seconds: Tuple[int, ...] = LAG_GRID_SECONDS,
    bucket_seconds: int = BUCKET_SECONDS,
    seed: int = DEFAULT_SEED,
) -> List[PairResult]:
    """Compute lagged cross-correlation + surrogate significance for every
    unordered pair of channels.

    Eligibility gate: zero-lag co-sampled overlap size (the maximum possible
    n for a pair, since positive lags only shrink alignment further). Pairs
    below `min_samples` get a `PairResult` with `winner=None` and are never
    given a fabricated edge -- callers must check `n_lag0 < min_samples`
    before treating the absence of `winner` as "not correlated" instead of
    "not enough data to know".
    """
    channel_ids = sorted(channel_buckets.keys())
    results: List[PairResult] = []
    rng = np.random.default_rng(seed)

    for i, a in enumerate(channel_ids):
        for b in channel_ids[i + 1 :]:
            bucket_a = channel_buckets[a]
            bucket_b = channel_buckets[b]
            n_lag0 = len(set(bucket_a.keys()) & set(bucket_b.keys()))
            result = PairResult(a, b, n_lag0)
            if n_lag0 < min_samples:
                results.append(result)
                continue

            fwd = _best_direction(a, b, bucket_a, bucket_b, lag_grid_seconds, bucket_seconds)
            bwd = _best_direction(b, a, bucket_b, bucket_a, lag_grid_seconds, bucket_seconds)
            candidates = [c for c in (fwd, bwd) if c is not None]
            if not candidates:
                results.append(result)
                continue
            winner = max(candidates, key=lambda c: abs(c.r))
            p_raw = circular_shift_pvalue(
                winner.xs, winner.ys, winner.r, n_surrogates=n_surrogates, rng=rng
            )
            # Bonferroni correction for the implicit multiple comparisons across
            # both directions x the full lag grid (see module docstring's
            # "Significance: circular time-shift surrogates" section).
            n_comparisons = 2 * len(lag_grid_seconds)
            p_value = min(1.0, p_raw * n_comparisons)
            result.winner = winner
            result.p_value = p_value
            result.significant = p_value < alpha
            results.append(result)

    return results


def _bucket_idx_to_datetime(idx: int, bucket_seconds: int = BUCKET_SECONDS) -> datetime:
    return datetime.fromtimestamp(idx * bucket_seconds, tz=timezone.utc)


def build_edges(
    pair_results: List[PairResult],
    *,
    window_start: datetime,
    window_end: datetime,
    bucket_seconds: int = BUCKET_SECONDS,
) -> List[CausalGeometryEdgeV1]:
    edges: List[CausalGeometryEdgeV1] = []
    for result in pair_results:
        if not result.significant or result.winner is None or result.p_value is None:
            continue
        w = result.winner
        edges.append(
            CausalGeometryEdgeV1(
                source_id=w.source_id,
                target_id=w.target_id,
                lag_sec=w.lag_sec,
                strength=max(-1.0, min(1.0, w.r)),
                significance=result.p_value,
                n_samples=w.n,
                window_start=window_start,
                window_end=window_end,
            )
        )
    return edges


# --- designed topology loading + divergence ---------------------------------


def load_field_topology(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_capability_channel_index(topology: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Reverse index: capability channel name (e.g. "reasoning_pressure") ->
    list of YAML edges (source_id, target_id, weight, edge_type) whose
    channel_map maps some local channel onto that capability channel name."""
    index: Dict[str, List[Dict[str, Any]]] = {}
    for edge in topology.get("edges", []) or []:
        channel_map = edge.get("channel_map") or {}
        for _local_channel, cap_channel in channel_map.items():
            index.setdefault(cap_channel, []).append(
                {
                    "source_id": edge.get("source_id"),
                    "target_id": edge.get("target_id"),
                    "weight": edge.get("weight"),
                    "edge_type": edge.get("edge_type"),
                }
            )
    return index


def _best_strength_for_channel(channel_id: str, edges: List[CausalGeometryEdgeV1]) -> Optional[float]:
    best: Optional[float] = None
    for edge in edges:
        if edge.source_id == channel_id or edge.target_id == channel_id:
            if best is None or abs(edge.strength) > abs(best):
                best = edge.strength
    return best


def build_divergence(
    edges: List[CausalGeometryEdgeV1],
    pair_results: List[PairResult],
    topology: Dict[str, Any],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> List[CausalGeometryDivergenceEntryV1]:
    """Compare observed edges against `config/field/orion_field_topology.v1.yaml`.

    See the module docstring's "Designed-vs-observed divergence" section for
    the full matching rule. Summary: an observed channel's suffix (after the
    first ':') is checked against every capability channel name used in the
    YAML's edge channel_maps; an exact string match is treated as a plausible
    alias. Divergence entries for YAML edges get `#<capability_channel>`
    appended to target_id to disambiguate the case where two different
    capability channels point at the same (source_id, target_id) YAML edge.
    """
    cap_index = build_capability_channel_index(topology)
    entries: List[CausalGeometryDivergenceEntryV1] = []
    consumed_cap_edges: set = set()

    observed_channel_ids = {edge.source_id for edge in edges} | {edge.target_id for edge in edges}
    for channel_id in sorted(observed_channel_ids):
        suffix = channel_id.split(":", 1)[1] if ":" in channel_id else channel_id
        if suffix not in cap_index:
            continue
        observed_strength = _best_strength_for_channel(channel_id, edges)
        for designed_edge in cap_index[suffix]:
            key = (designed_edge["source_id"], designed_edge["target_id"], suffix)
            if key in consumed_cap_edges:
                continue
            consumed_cap_edges.add(key)
            designed_weight = designed_edge["weight"]
            entries.append(
                CausalGeometryDivergenceEntryV1(
                    source_id=designed_edge["source_id"],
                    target_id=f"{designed_edge['target_id']}#{suffix}",
                    observed_strength=observed_strength,
                    designed_weight=designed_weight,
                    delta=(observed_strength - designed_weight) if observed_strength is not None else None,
                    status="both" if observed_strength is not None else "designed_only",
                )
            )

    # Every remaining YAML edge/channel entry with no observed alias at all.
    for cap_channel, designed_edge_list in cap_index.items():
        for designed_edge in designed_edge_list:
            key = (designed_edge["source_id"], designed_edge["target_id"], cap_channel)
            if key in consumed_cap_edges:
                continue
            consumed_cap_edges.add(key)
            entries.append(
                CausalGeometryDivergenceEntryV1(
                    source_id=designed_edge["source_id"],
                    target_id=f"{designed_edge['target_id']}#{cap_channel}",
                    observed_strength=None,
                    designed_weight=designed_edge["weight"],
                    delta=None,
                    status="designed_only",
                )
            )

    # Significant observed edges whose endpoints don't alias to any capability
    # channel at all.
    aliased_channel_ids = {
        cid
        for cid in observed_channel_ids
        if (cid.split(":", 1)[1] if ":" in cid else cid) in cap_index
    }
    for edge in edges:
        if edge.source_id in aliased_channel_ids or edge.target_id in aliased_channel_ids:
            continue
        entries.append(
            CausalGeometryDivergenceEntryV1(
                source_id=edge.source_id,
                target_id=edge.target_id,
                observed_strength=edge.strength,
                designed_weight=None,
                delta=None,
                status="observed_only",
            )
        )

    # Pairs that never had enough co-sampled data to test at all.
    for result in pair_results:
        if result.n_lag0 < min_samples:
            entries.append(
                CausalGeometryDivergenceEntryV1(
                    source_id=result.channel_a,
                    target_id=result.channel_b,
                    observed_strength=None,
                    designed_weight=None,
                    delta=None,
                    status="insufficient_data",
                )
            )

    return entries


# --- snapshot assembly -------------------------------------------------------


def build_snapshot(
    channels: ChannelSeries,
    topology: Dict[str, Any],
    *,
    window_start: datetime,
    window_end: datetime,
    bucket_seconds: int = BUCKET_SECONDS,
    lag_grid_seconds: Tuple[int, ...] = LAG_GRID_SECONDS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    table_row_counts: Optional[Dict[str, int]] = None,
) -> Tuple[CausalGeometrySnapshotV1, List[PairResult]]:
    channel_buckets = {
        channel_id: bucketize(points, bucket_seconds) for channel_id, points in channels.items()
    }

    pair_results = compute_pairwise_results(
        channel_buckets,
        min_samples=min_samples,
        n_surrogates=n_surrogates,
        alpha=alpha,
        lag_grid_seconds=lag_grid_seconds,
        bucket_seconds=bucket_seconds,
        seed=seed,
    )
    edges = build_edges(
        pair_results, window_start=window_start, window_end=window_end, bucket_seconds=bucket_seconds
    )
    divergence = build_divergence(edges, pair_results, topology, min_samples=min_samples)

    notes: List[str] = []
    if table_row_counts:
        for table_name, row_count in table_row_counts.items():
            if row_count == 0:
                notes.append(
                    f"source table {table_name!r} returned 0 rows in the requested window -- "
                    "every channel it would have contributed is absent from this snapshot, "
                    "not silently zero-valued."
                )
    n_pairs_tested = len(pair_results)
    n_insufficient = sum(1 for r in pair_results if r.n_lag0 < min_samples)
    n_pairs_with_data = n_pairs_tested - n_insufficient
    notes.append(
        f"{len(channels)} channels, {n_pairs_tested} channel pairs considered, "
        f"{n_insufficient} below --min-samples={min_samples}, "
        f"{n_pairs_with_data} tested for significance, {len(edges)} significant edges found."
    )
    insufficient_data = len(edges) == 0
    if insufficient_data:
        if n_pairs_with_data == 0:
            notes.append(
                "No channel pair had enough co-sampled data to test at all -- widen "
                "--window-hours, lower --min-samples, or check that the source tables "
                "actually have rows in the requested window."
            )
        else:
            notes.append(
                f"{n_pairs_with_data} pair(s) had enough data to test but none cleared "
                f"--alpha={alpha} after {n_surrogates} circular-shift surrogates each -- "
                "this is reported honestly as insufficient_data rather than a fabricated graph."
            )

    snapshot = CausalGeometrySnapshotV1(
        snapshot_id=str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc),
        window_start=window_start,
        window_end=window_end,
        edges=edges,
        designed_topology_version=topology.get("schema_version"),
        divergence=divergence,
        insufficient_data=insufficient_data,
        notes=notes,
    )
    return snapshot, pair_results


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
    raise SystemExit(main())
