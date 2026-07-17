"""Service-level proof: the worker's real graph writers work against a
Falkor-primary RoutedSubstrateGraphStore, not just the fakes/mocks used by
test_worker_prediction_error_node.py / test_worker_dynamics_tick.py.

Design spec: docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md
item 3 -- "substrate-runtime graph writers -> Falkor Cypher-native (routed
then primary-only) -- after adapter is proven live." This file is that proof
at the worker level, mirroring the library-level pattern already established
in orion/substrate/tests/test_dynamics_falkor_routed.py.

None of this flips any default -- SUBSTRATE_WRITE_PREDICTION_ERROR_NODES,
SUBSTRATE_DYNAMICS_TICK_ENABLED, and SUBSTRATE_STORE_BACKEND=routed all stay
off/unset in .env_example and .env; these tests set them only in-process via
monkeypatch.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker, _PREDICTION_ERROR_NODE_FLAG

from orion.substrate.falkor_store import (
    FalkorSubstrateStore,
    FalkorSubstrateStoreConfig,
    RecordingFalkorClient,
)
from orion.substrate.routed_store import RoutedSubstrateGraphStore

_NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


def _make_worker_with_store(store) -> BiometricsSubstrateWorker:
    """Mirrors test_worker_prediction_error_node.py's ``_make_worker`` helper:
    build the worker without running __init__, then set the cached store
    directly on ``_substrate_graph_store`` so ``_get_substrate_graph_store``
    reuses it instead of calling ``build_substrate_store_from_env``."""
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._substrate_graph_store = store
    return worker


def _make_falkor_routed_store(*, hydrate_node_rows: list[dict] | None = None) -> tuple[
    RoutedSubstrateGraphStore, RecordingFalkorClient
]:
    client = RecordingFalkorClient(hydrate_node_rows=hydrate_node_rows or [])
    falkor_store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=bool(hydrate_node_rows),
    )
    routed = RoutedSubstrateGraphStore(primary=falkor_store, shadow=None)
    return routed, client


def _prediction_error_row(
    *,
    node_id: str = "concept-pe-alpha",
    observed_at: str,
    dynamic_pressure: float = 0.0,
    dynamic_pressure_reason: str | None = None,
    dormant: bool = False,
    dormancy_updated_at: str | None = None,
    prediction_error: float = 0.7,
    activation: float = 0.1,
    recency_score: float = 0.1,
) -> dict:
    """A native-property Falkor row for a concept node carrying a standing
    prediction_error -- shaped exactly like FalkorSubstrateStore's cold-start
    hydration query result (NATIVE_NODE_RETURN_FIELDS), so it can be fed
    straight into RecordingFalkorClient(hydrate_node_rows=[...]).

    Inlined rather than imported from orion.substrate.tests.test_dynamics_falkor_routed
    -- a cross-package test-file import is unusual in this repo's test layout,
    and this fixture is small enough that duplicating it here is cheaper than
    the coupling.
    """
    return {
        "node_id": node_id,
        "node_kind": "concept",
        "identity_key": f"concept:{node_id}",
        "label": "Prediction error concept",
        "definition": None,
        "taxonomy_path_json": "[]",
        "anchor_scope": "orion",
        "subject_ref": None,
        "promotion_state": "canonical",
        "risk_tier": "low",
        "confidence": 0.7,
        "salience": 0.5,
        "activation": activation,
        "recency_score": recency_score,
        "decay_half_life_seconds": None,
        "decay_floor": 0.0,
        "observed_at": observed_at,
        "valid_from": None,
        "valid_to": None,
        "provenance_authority": "local_inferred",
        "provenance_source_kind": "test",
        "provenance_source_channel": "test:worker_falkor_routed_store",
        "provenance_producer": "test_worker_falkor_routed_store",
        "provenance_model_name": None,
        "provenance_correlation_id": None,
        "provenance_trace_id": None,
        "provenance_tier_rank": None,
        "evidence_refs_json": "[]",
        "dynamic_pressure": dynamic_pressure,
        "dynamic_pressure_reason": dynamic_pressure_reason,
        "dormant": dormant,
        "dormancy_updated_at": dormancy_updated_at,
        "prediction_error": prediction_error,
    }


def _write_calls(client: RecordingFalkorClient) -> list[tuple[str, dict]]:
    return [(c, p) for c, p in client.calls if "MERGE (n:SubstrateNode" in c]


# ---------------------------------------------------------------------------
# 1. _write_prediction_error_node against a Falkor-primary routed store.
# ---------------------------------------------------------------------------


def test_write_prediction_error_node_against_falkor_routed_store(monkeypatch) -> None:
    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    routed, client = _make_falkor_routed_store()
    worker = _make_worker_with_store(routed)

    node_id = "node:substrate.execution"
    error = 0.6
    reducer_key = "execution_trajectory"

    # No exception raised.
    worker._write_prediction_error_node(
        node_id=node_id,
        error=error,
        now=_NOW,
        reducer_key=reducer_key,
    )

    write_calls = _write_calls(client)
    assert write_calls, "expected the worker's upsert to durably write via Falkor"
    cypher, params = write_calls[-1]
    assert "n.prediction_error = $prediction_error" in cypher
    assert "payload_json" not in cypher.replace("REMOVE n.payload_json", "")
    assert params is not None
    assert "payload_json" not in params

    node = routed.get_node_by_id(node_id)
    assert node is not None
    salience = max(0.0, min(1.0, error))
    expected_prediction_error = round(salience, 6)
    assert node.metadata.get("prediction_error") == pytest.approx(expected_prediction_error)
    assert params["prediction_error"] == pytest.approx(expected_prediction_error)


def test_write_prediction_error_node_preserves_dynamics_state_on_rewrite(monkeypatch) -> None:
    """Regression test: _write_prediction_error_node's docstring says it "writes
    a single node under a fixed identity_key so re-writes collapse" -- i.e. the
    same node_id is upserted repeatedly (see worker.py's _execution_tick /
    _transport_tick callers, both using a fixed node_id every cycle). Now that
    falkor_codec.py always includes dynamic_pressure/dormant/etc in every
    upsert's SET clause, a naive re-write that doesn't carry forward the
    dynamics engine's already-computed state would durably reset it to
    0.0/False on every single re-write -- defeating the very durability this
    branch exists to add.
    """
    from orion.substrate.dynamics import SubstrateDynamicsEngine

    monkeypatch.setenv(_PREDICTION_ERROR_NODE_FLAG, "true")
    routed, client = _make_falkor_routed_store()
    worker = _make_worker_with_store(routed)

    node_id = "node:substrate.execution"

    worker._write_prediction_error_node(
        node_id=node_id, error=0.7, now=_NOW, reducer_key="execution_trajectory"
    )

    engine = SubstrateDynamicsEngine(store=routed)
    tick_result = engine.tick(now=_NOW)
    assert tick_result.pressure_updates, "expected the standing prediction_error to seed pressure"

    node_after_tick = routed.get_node_by_id(node_id)
    assert node_after_tick is not None
    dynamic_pressure_after_tick = node_after_tick.metadata.get("dynamic_pressure")
    assert dynamic_pressure_after_tick, "expected dynamics.tick() to have set a non-zero dynamic_pressure"

    # Same fixed node_id re-written by a fresh prediction-error event.
    worker._write_prediction_error_node(
        node_id=node_id, error=0.65, now=_NOW, reducer_key="execution_trajectory"
    )

    node_after_rewrite = routed.get_node_by_id(node_id)
    assert node_after_rewrite is not None
    assert node_after_rewrite.metadata.get("dynamic_pressure") == pytest.approx(
        dynamic_pressure_after_tick
    ), "dynamics-engine-owned dynamic_pressure must survive an unrelated prediction-error rewrite"
    assert node_after_rewrite.metadata.get("dynamic_pressure_reason") == node_after_tick.metadata.get(
        "dynamic_pressure_reason"
    )
    # The writer's own field still updates to the new value.
    assert node_after_rewrite.metadata.get("prediction_error") == pytest.approx(round(0.65, 6))


# ---------------------------------------------------------------------------
# 2. _dynamics_tick end-to-end against the same Falkor-primary routed store.
# ---------------------------------------------------------------------------


def _make_worker_for_dynamics_tick(monkeypatch, store) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("SUBSTRATE_DYNAMICS_TICK_ENABLED", "true")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._substrate_graph_store = store
    return worker


def test_dynamics_tick_against_falkor_routed_store(monkeypatch) -> None:
    # _dynamics_tick hardcodes datetime.now(timezone.utc) internally (no `now`
    # kwarg like SubstrateDynamicsEngine.tick() itself takes), so anchor the
    # seed row's observed_at to real wall-clock time rather than a fixed
    # constant -- avoids needing to freeze/monkeypatch datetime just to keep
    # the prediction-error decay window (see prediction_error_pressure in
    # orion/substrate/pressure.py) from aging the seed out before the tick runs.
    real_now = datetime.now(timezone.utc)
    observed_at = (real_now - timedelta(seconds=60)).isoformat()
    row = _prediction_error_row(observed_at=observed_at)
    routed, client = _make_falkor_routed_store(hydrate_node_rows=[row])
    worker = _make_worker_for_dynamics_tick(monkeypatch, routed)

    seed_node = routed.get_node_by_id(row["node_id"])
    assert seed_node is not None
    seed_pressure = seed_node.metadata.get("dynamic_pressure")

    # No exception raised.
    worker._dynamics_tick()

    updated_node = routed.get_node_by_id(row["node_id"])
    assert updated_node is not None
    new_pressure = updated_node.metadata.get("dynamic_pressure")
    assert new_pressure != seed_pressure, "expected the tick to actually seed pressure, not no-op"
    assert new_pressure > 0.0

    write_calls = _write_calls(client)
    assert write_calls, "expected the dynamics tick to durably persist the pressure update"
    cypher, params = write_calls[-1]
    assert "n.dynamic_pressure = $dynamic_pressure" in cypher
    assert "payload_json" not in cypher.replace("REMOVE n.payload_json", "")
    assert params is not None
    assert "payload_json" not in params
    assert params["dynamic_pressure"] == pytest.approx(new_pressure)


# ---------------------------------------------------------------------------
# 3. build_substrate_store_from_env() wires a Falkor-backed routed store from
#    the runtime's own documented env var names (no live Redis required).
# ---------------------------------------------------------------------------


def test_build_substrate_store_from_env_wires_falkor_routed_store(monkeypatch) -> None:
    # Point FALKORDB_URI at a closed local port so RedisGraphQueryClient's
    # connection attempt fails fast (ECONNREFUSED, no real network/DNS hop)
    # instead of hanging -- confirmed empirically: connecting to a closed
    # port on localhost returns immediately (< 2ms), well under any test
    # timeout. FalkorSubstrateStore._hydrate_from_durable() wraps the whole
    # hydrate call in try/except Exception and only logs a warning on
    # failure, so construction still succeeds with an empty cache.
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "routed")
    monkeypatch.setenv("SUBSTRATE_STORE_PRIMARY", "falkor")
    monkeypatch.setenv("SUBSTRATE_STORE_SHADOW", "")
    monkeypatch.setenv("FALKORDB_URI", "redis://127.0.0.1:1")
    monkeypatch.setenv("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate_test")

    from orion.substrate.graphdb_store import build_substrate_store_from_env

    store = build_substrate_store_from_env()

    assert isinstance(store, RoutedSubstrateGraphStore)
    assert isinstance(store._primary, FalkorSubstrateStore)
