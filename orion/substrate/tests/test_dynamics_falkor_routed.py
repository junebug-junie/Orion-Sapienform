"""Integration proof that SubstrateDynamicsEngine works end-to-end against a
Falkor-backed store reached through the production routing layer
(RoutedSubstrateGraphStore), not just InMemorySubstrateGraphStore.

This is the acceptance check the design spec calls for: "Runtime SPARQL
cutover ... uses Cypher-native adapter, not blob port." It also closes the
restart-durability gap this patch exists to fix -- before promoting
dynamic_pressure/dormant/prediction_error to native Cypher properties,
FalkorSubstrateStore's cold-start hydration (`decode_concept_node`) always
produced `metadata={}`, so a process restart would silently reset all
dynamics state even though the in-process cache made it look fine while the
process stayed up.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.falkor_store import (
    FalkorSubstrateStore,
    FalkorSubstrateStoreConfig,
    RecordingFalkorClient,
)
from orion.substrate.routed_store import RoutedSubstrateGraphStore

_NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


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
    prediction_error and (optionally) prior dynamics state -- shaped exactly
    like what FalkorSubstrateStore's cold-start hydration query would return
    (NATIVE_NODE_RETURN_FIELDS), so it can be fed straight into
    RecordingFalkorClient(hydrate_node_rows=[...]).
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
        "provenance_source_channel": "test:dynamics_falkor_routed",
        "provenance_producer": "test_dynamics_falkor_routed",
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


def test_tick_seeds_pressure_from_prediction_error_through_routed_falkor_store():
    observed_at = (_NOW - timedelta(seconds=60)).isoformat()
    row = _prediction_error_row(observed_at=observed_at)
    client = RecordingFalkorClient(hydrate_node_rows=[row])
    falkor_store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )
    routed = RoutedSubstrateGraphStore(primary=falkor_store, shadow=None)
    engine = SubstrateDynamicsEngine(store=routed)

    result = engine.tick(now=_NOW)

    assert result.pressure_updates, "expected standing prediction_error to seed dynamic_pressure"
    update = result.pressure_updates[0]
    assert update.new_pressure > 0.0

    node = routed.get_node_by_id(row["node_id"])
    assert node is not None
    assert node.metadata.get("dynamic_pressure") == pytest.approx(round(update.new_pressure, 6))

    write_calls = [(c, p) for c, p in client.calls if "MERGE (n:SubstrateNode" in c]
    assert write_calls, "expected dynamics tick to durably persist the pressure update"
    cypher, params = write_calls[-1]
    assert "n.dynamic_pressure = $dynamic_pressure" in cypher
    assert "payload_json" not in cypher.replace("REMOVE n.payload_json", "")
    assert params is not None
    assert "payload_json" not in params
    assert params["dynamic_pressure"] == pytest.approx(round(update.new_pressure, 6))


def test_dynamic_pressure_persists_through_native_properties_across_process_restart():
    """The restart-durability proof: rebuild a *fresh* FalkorSubstrateStore
    from the client's durably-written row between tick 1 and tick 2 (simulating
    a process restart re-hydrating from durable Falkor rows), and confirm
    tick 2 reads tick 1's dynamic_pressure as its prev_pressure -- proving the
    round trip through native Cypher properties (not the in-process cache) is
    what actually carries dynamics state across a restart.

    RecordingFalkorClient does not auto-persist writes into its own
    hydrate_node_rows (graph_query() only records calls and replays whatever
    rows were passed in at construction -- see FalkorSubstrateStore's
    `_hydrate_from_durable`/RecordingFalkorClient.graph_query), so this test
    manually feeds tick 1's actual upsert params back into a new client's
    hydrate rows to script that restart.
    """
    observed_at = (_NOW - timedelta(seconds=60)).isoformat()
    row = _prediction_error_row(observed_at=observed_at)
    client = RecordingFalkorClient(hydrate_node_rows=[row])
    falkor_store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )
    routed = RoutedSubstrateGraphStore(primary=falkor_store, shadow=None)
    engine = SubstrateDynamicsEngine(store=routed)

    tick1 = engine.tick(now=_NOW)
    assert tick1.pressure_updates
    tick1_pressure = round(tick1.pressure_updates[0].new_pressure, 6)

    write_calls = [(c, p) for c, p in client.calls if "MERGE (n:SubstrateNode" in c]
    assert write_calls
    _, durable_params = write_calls[-1]
    assert durable_params["dynamic_pressure"] == pytest.approx(tick1_pressure)

    # "Process restart": a brand-new client/store/routed-store/engine, hydrated
    # only from the row tick 1 actually wrote durably -- no shared in-process
    # cache with the first store at all.
    restart_client = RecordingFalkorClient(hydrate_node_rows=[durable_params])
    restarted_falkor_store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=restart_client,
        hydrate=True,
    )
    restarted_routed = RoutedSubstrateGraphStore(primary=restarted_falkor_store, shadow=None)

    rehydrated_node = restarted_routed.get_node_by_id(row["node_id"])
    assert rehydrated_node is not None
    assert rehydrated_node.metadata.get("dynamic_pressure") == pytest.approx(tick1_pressure)

    restarted_engine = SubstrateDynamicsEngine(store=restarted_routed)
    tick2 = restarted_engine.tick(now=_NOW + timedelta(seconds=30))

    assert tick2.pressure_updates, "expected further decay to still register as a pressure update"
    update2 = tick2.pressure_updates[0]
    # This is the crux of the restart-durability proof: tick 2's prev_pressure
    # (read from node.metadata.get("dynamic_pressure") at the top of tick())
    # equals tick 1's durably-written value -- not 0.0, which is what it would
    # be if decode_concept_node() still produced metadata={} on every hydrate.
    assert update2.previous_pressure == pytest.approx(tick1_pressure)
    # Further decay since tick 1 means tick 2's freshly recomputed pressure is
    # slightly lower, not a reset back to a fresh seed.
    assert update2.new_pressure < update2.previous_pressure
