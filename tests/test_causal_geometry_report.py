from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import causal_geometry_report as cgr  # noqa: E402
from orion.schemas.causal_geometry import CausalGeometrySnapshotV1  # noqa: E402

BASE_TS = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _points_for(values: np.ndarray, bucket_seconds: int = cgr.BUCKET_SECONDS) -> List[Tuple[datetime, float]]:
    """One point per bucket, so bucketize() produces exactly one value per
    bucket index and every channel built this way shares the same bucket
    index domain (0..len(values)-1)."""
    return [
        (BASE_TS + timedelta(seconds=i * bucket_seconds), float(v)) for i, v in enumerate(values)
    ]


def _fixture_channels(n_buckets: int = 200, seed: int = 42) -> Dict[str, List[Tuple[datetime, float]]]:
    rng = np.random.default_rng(seed)

    # Pair 1: strongly correlated, aliases to a designed capability channel
    # (reasoning_pressure) via self_state:reasoning_pressure.
    x = rng.normal(size=n_buckets)
    y = x * 0.85 + rng.normal(scale=0.15, size=n_buckets)

    # Pair 2: strongly correlated, but neither channel aliases to any
    # capability channel name -> should surface as observed_only.
    u = rng.normal(size=n_buckets)
    w = u * 0.8 + rng.normal(scale=0.2, size=n_buckets)

    # Independent, uncorrelated pair (used to check no spurious edge).
    p = rng.normal(size=n_buckets)
    q = rng.normal(size=n_buckets)

    # Too few points to ever clear --min-samples against anything.
    scarce = rng.normal(size=5)

    return {
        "self_state:reasoning_pressure": _points_for(x),
        "drive:autonomy": _points_for(y),
        "biometrics:cpu": _points_for(u),
        "attention:salience": _points_for(w),
        "self_state:coherence": _points_for(p),
        "drive:continuity": _points_for(q),
        "self_state:uncertainty": _points_for(scarce),
    }


def _fixture_topology() -> Dict:
    return {
        "schema_version": "field_lattice.v1.test",
        "edges": [
            {
                "source_id": "node:athena",
                "target_id": "capability:orchestration",
                "edge_type": "node_capability",
                "weight": 0.9,
                "channel_map": {
                    "reasoning_load": "reasoning_pressure",
                    "execution_load": "execution_pressure",
                },
            },
            {
                "source_id": "capability:transport",
                "target_id": "capability:orchestration",
                "edge_type": "capability_capability",
                "weight": 0.7,
                "channel_map": {"transport_pressure": "transport_pressure"},
            },
        ],
    }


_UNSET = object()


def _build(**overrides):
    channels = overrides.pop("channels", _UNSET)
    if channels is _UNSET:
        channels = _fixture_channels()
    topology = overrides.pop("topology", None) or _fixture_topology()
    window_start = BASE_TS
    window_end = BASE_TS + timedelta(seconds=250 * cgr.BUCKET_SECONDS)
    kwargs = dict(
        window_start=window_start,
        window_end=window_end,
        min_samples=30,
        n_surrogates=200,
        alpha=0.05,
        seed=1337,
    )
    kwargs.update(overrides)
    return cgr.build_snapshot(channels, topology, **kwargs)


# --- bucketize / pearson unit tests -----------------------------------------


def test_bucketize_groups_by_bucket_index_and_means():
    points = [
        (BASE_TS, 1.0),
        (BASE_TS + timedelta(seconds=10), 3.0),  # same bucket as above
        (BASE_TS + timedelta(seconds=cgr.BUCKET_SECONDS + 5), 10.0),  # next bucket
    ]
    buckets = cgr.bucketize(points, bucket_seconds=cgr.BUCKET_SECONDS)
    assert len(buckets) == 2
    idxs = sorted(buckets.keys())
    assert buckets[idxs[0]] == pytest.approx(2.0)  # mean(1.0, 3.0)
    assert buckets[idxs[1]] == pytest.approx(10.0)


def test_pearson_symmetric_and_handles_zero_variance():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    r_xy = cgr._pearson(x, y)
    r_yx = cgr._pearson(y, x)
    assert r_xy == pytest.approx(1.0)
    assert r_yx == pytest.approx(r_xy)

    flat = np.array([5.0, 5.0, 5.0, 5.0])
    assert cgr._pearson(x, flat) == 0.0


# --- significance detection --------------------------------------------------


def test_correlated_series_produce_significant_edge_with_plausible_pvalue():
    snapshot, pair_results = _build()
    assert not snapshot.insufficient_data

    matches = [
        e
        for e in snapshot.edges
        if {e.source_id, e.target_id} == {"self_state:reasoning_pressure", "drive:autonomy"}
    ]
    assert len(matches) == 1, "expected exactly one significant edge for the strongly-correlated pair"
    edge = matches[0]
    assert abs(edge.strength) > 0.5
    assert 0.0 <= edge.significance < 0.05
    assert edge.n_samples >= 30

    matches2 = [
        e for e in snapshot.edges if {e.source_id, e.target_id} == {"biometrics:cpu", "attention:salience"}
    ]
    assert len(matches2) == 1
    assert abs(matches2[0].strength) > 0.5


def test_uncorrelated_series_not_reported_as_significant():
    snapshot, _ = _build()
    matches = [
        e
        for e in snapshot.edges
        if {e.source_id, e.target_id} == {"self_state:coherence", "drive:continuity"}
    ]
    assert matches == [], "independent random series must not clear the significance bar"


def test_insufficient_data_pair_never_fabricates_an_edge():
    snapshot, pair_results = _build()

    # Every pair involving the scarce (5-point) channel must be insufficient_data,
    # never produce a fabricated edge.
    scarce_pairs = [r for r in pair_results if "self_state:uncertainty" in (r.channel_a, r.channel_b)]
    assert scarce_pairs, "expected the scarce channel to appear in pairwise results"
    for result in scarce_pairs:
        assert result.n_lag0 < 30
        assert result.winner is None

    scarce_edges = [
        e for e in snapshot.edges if "self_state:uncertainty" in (e.source_id, e.target_id)
    ]
    assert scarce_edges == []

    scarce_divergence = [
        d
        for d in snapshot.divergence
        if d.status == "insufficient_data" and "self_state:uncertainty" in (d.source_id, d.target_id)
    ]
    assert scarce_divergence, "expected at least one insufficient_data divergence entry for the scarce channel"
    for entry in scarce_divergence:
        assert entry.observed_strength is None
        assert entry.designed_weight is None
        assert entry.delta is None


# --- snapshot schema validity -------------------------------------------------


def test_snapshot_round_trips_through_pydantic_validation():
    snapshot, _ = _build()
    assert isinstance(snapshot, CausalGeometrySnapshotV1)
    dumped = snapshot.model_dump(mode="json")
    revalidated = CausalGeometrySnapshotV1.model_validate(dumped)
    assert revalidated.snapshot_id == snapshot.snapshot_id
    assert len(revalidated.edges) == len(snapshot.edges)
    assert len(revalidated.divergence) == len(snapshot.divergence)


# --- divergence categorization -----------------------------------------------


def test_divergence_status_categories_are_all_present_and_correct():
    snapshot, _ = _build()
    by_status = {}
    for entry in snapshot.divergence:
        by_status.setdefault(entry.status, []).append(entry)

    assert set(by_status.keys()) >= {"both", "designed_only", "observed_only", "insufficient_data"}

    # "both": the reasoning_pressure alias matched a significant observed edge.
    both_entries = [e for e in by_status["both"] if e.target_id.endswith("#reasoning_pressure")]
    assert len(both_entries) == 1
    both_entry = both_entries[0]
    assert both_entry.source_id == "node:athena"
    assert both_entry.designed_weight == pytest.approx(0.9)
    assert both_entry.observed_strength is not None
    assert both_entry.delta == pytest.approx(both_entry.observed_strength - both_entry.designed_weight)

    # "designed_only": execution_pressure and transport_pressure channel_map
    # entries have no observed channel aliasing to them in the fixture.
    designed_only_targets = {e.target_id for e in by_status["designed_only"]}
    assert any(t.endswith("#execution_pressure") for t in designed_only_targets)
    assert any(t.endswith("#transport_pressure") for t in designed_only_targets)
    for entry in by_status["designed_only"]:
        assert entry.observed_strength is None
        assert entry.delta is None
        assert entry.designed_weight is not None

    # "observed_only": biometrics:cpu <-> attention:salience is significant but
    # neither "cpu" nor "salience" alias to any capability channel.
    observed_only_pairs = {
        frozenset((e.source_id, e.target_id)) for e in by_status["observed_only"]
    }
    assert frozenset(("biometrics:cpu", "attention:salience")) in observed_only_pairs
    for entry in by_status["observed_only"]:
        assert entry.designed_weight is None
        assert entry.delta is None
        assert entry.observed_strength is not None

    # "insufficient_data": every entry must have both strengths absent.
    for entry in by_status["insufficient_data"]:
        assert entry.observed_strength is None
        assert entry.designed_weight is None
        assert entry.delta is None


def test_empty_channels_report_insufficient_data_not_empty_shell():
    snapshot, _ = _build(channels={}, min_samples=30)
    assert snapshot.edges == []
    assert snapshot.insufficient_data is True
    assert snapshot.notes, "an empty-input snapshot must explain itself in notes, not look like a real graph"


def test_capability_channel_index_disambiguates_same_edge_different_channels():
    topology = _fixture_topology()
    index = cgr.build_capability_channel_index(topology)
    assert "reasoning_pressure" in index
    assert "execution_pressure" in index
    assert "transport_pressure" in index
    # Both reasoning_pressure and execution_pressure point at the *same* YAML edge.
    assert index["reasoning_pressure"][0]["source_id"] == index["execution_pressure"][0]["source_id"]
    assert index["reasoning_pressure"][0]["target_id"] == index["execution_pressure"][0]["target_id"]


def test_circular_shift_pvalue_add_one_smoothing_never_zero():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(0)
    p = cgr.circular_shift_pvalue(x, y, observed_r=1.0, n_surrogates=50, rng=rng)
    assert p > 0.0
    assert p == pytest.approx(1.0 / 51.0)  # only the observed (unshifted-equivalent) config can match |r|=1 rarely
