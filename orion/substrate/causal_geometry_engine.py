"""Causal Geometry v1, Phase A core engine: observed-geometry measurement.

Extracted from `scripts/causal_geometry_report.py` (which now imports from here
and keeps only its CLI wrapper) so the new Phase B producer
(`orion/substrate/causal_geometry_producer.py`, called periodically from
`services/orion-field-digester/app/worker.py`) can call the exact same
measurement logic the standalone report script uses, instead of duplicating it.

See docs/superpowers/specs/2026-07-16-causal-geometry-v1-design.md for the full
three-phase design, and `scripts/causal_geometry_report.py`'s own (still
present) module docstring for the detailed rationale behind the fixed
constants, the four source tables, resampling, lagged cross-correlation, the
Bonferroni-corrected surrogate significance test, and the designed-vs-observed
divergence matching rule (including the `#<capability_channel>` target_id
disambiguation suffix -- see `orion/substrate/field_topology_plasticity.py`'s
`_base_target_id()` for the corresponding join-side fix).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from orion.schemas.causal_geometry import (
    CausalGeometryDivergenceEntryV1,
    CausalGeometryEdgeV1,
    CausalGeometrySnapshotV1,
)

_REPO_ROOT = str(Path(__file__).resolve().parents[2])

# --- documented, fixed knobs (see scripts/causal_geometry_report.py's module
# docstring for the full rationale behind each of these) -------------------

BUCKET_SECONDS = 300
LAG_GRID_SECONDS: Tuple[int, ...] = (0, 300, 600, 900, 1800)

DEFAULT_WINDOW_HOURS = 168.0
DEFAULT_MIN_SAMPLES = 30
DEFAULT_N_SURROGATES = 200
DEFAULT_ALPHA = 0.05
DEFAULT_BIOMETRICS_NODE = "athena"
DEFAULT_SEED = 1337

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

    Uses psycopg2 (already a dependency of several services; no new dependency
    added when called from a service that already has it, e.g. orion-hub,
    orion-field-digester). Each row's jsonb column is flattened into one
    channel per key (`drive:<key>`, `biometrics:<key>`, `self_state:<key>`);
    `attention_salience_trace.salience` has no sub-keys and becomes the single
    scalar channel `attention:salience`.

    Returns `(channels, table_row_counts)`. `table_row_counts` maps each source
    table name to the raw row count pulled in the window -- a table with 0 rows
    in the window contributes zero channels and would otherwise be silently
    invisible from the rest of the report.
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
    simply absent from the returned dict -- this never interpolates or
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

    Eligibility gate: zero-lag co-sampled overlap size (the maximum possible n
    for a pair, since positive lags only shrink alignment further). Pairs
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
            # both directions x the full lag grid (see scripts/causal_geometry_report.py's
            # module docstring's "Significance: circular time-shift surrogates" section).
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

    See `scripts/causal_geometry_report.py`'s module docstring's "Designed-vs-
    observed divergence" section for the full matching rule. Summary: an
    observed channel's suffix (after the first ':') is checked against every
    capability channel name used in the YAML's edge channel_maps; an exact
    string match is treated as a plausible alias. Divergence entries for YAML
    edges get `#<capability_channel>` appended to target_id to disambiguate the
    case where two different capability channels point at the same
    (source_id, target_id) YAML edge -- callers matching against the physical
    lattice edge (e.g. `orion/substrate/field_topology_plasticity.py`) must
    strip that suffix before joining.
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
