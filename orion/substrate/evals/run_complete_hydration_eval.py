"""Quiescent-fixture topology/pressure equivalence across capped Falkor pages."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.falkor_codec import decode_node, decode_edge
from orion.substrate.store import InMemorySubstrateGraphStore
from orion.substrate.evals.complete_hydration_fixture import CappedClient, build


def run() -> dict:
    reports = []
    for page_size, cap in [(1, 50), (7, 3), (1000, 17)]:
        client = CappedClient(nodes=43, edges=173, cap=cap)
        for i, row in enumerate(client._hydrate_node_rows):
            row.update(prediction_error=0.8 if i == 0 else 0.0, activation=0.4,
                       observed_at="2026-10-06T00:00:00+00:00")
        for i, row in enumerate(client._hydrate_edge_rows):
            row.update(source_id=f"node{i % 43}", target_id=f"node{(i + 1) % 43}",
                       observed_at="2026-10-06T00:00:00+00:00")
        reference = InMemorySubstrateGraphStore()
        for row in client._hydrate_node_rows:
            reference.upsert_node(identity_key=row["identity_key"], node=decode_node(row))
        for row in client._hydrate_edge_rows:
            reference.upsert_edge(identity_key=row["identity_key"], edge=decode_edge(row))
        paged = build(client, hydration_page_size=page_size)
        before = paged.snapshot()
        expected = reference.snapshot()
        assert before.scan_receipt.complete
        assert before.nodes == expected.nodes and before.edges == expected.edges
        now = datetime(2026, 10, 6, 0, 0, 30, tzinfo=timezone.utc)
        result = SubstrateDynamicsEngine(store=paged).tick(now=now)
        baseline = SubstrateDynamicsEngine(store=reference).tick(now=now)
        assert result == baseline
        assert result.pressure_updates and result.activation_updates
        # Inspect write-through cache: the recording client intentionally does
        # not persist writes and a second durable refresh would replay old rows.
        assert paged._cache.snapshot().nodes == reference.snapshot().nodes
        reports.append(dict(page_size=page_size, server_cap=cap, nodes=len(before.nodes),
                            edges=len(before.edges), pressure_updates=len(result.pressure_updates),
                            activation_updates=len(result.activation_updates), equivalent=True))
    return {"passed": True, "cases": reports, "integrity": run_integrity_cases()}


def run_integrity_cases() -> dict:
    """#3512: the two corruptions seen live 2026-11-24 cost one row each, never
    the whole graph; a scan where most rows are corrupt still fails closed."""
    client = CappedClient(nodes=43, edges=173, cap=17)
    # One edge read back a second time through a phantom adjacency entry that
    # points at an unrelated node of another kind (orion_substrate, edge 66265).
    client._hydrate_edge_rows.append(dict(client._hydrate_edge_rows[0], target_id="node7",
                                          target_kind="evidence"))
    # Two durable rows share one node_id (orion_substrate_aitown 49919/50246).
    client._hydrate_node_rows.append(dict(client._hydrate_node_rows[3],
                                          observed_at="2026-11-24T21:33:43+00:00"))
    state = build(client, hydration_page_size=7).snapshot()
    assert state.scan_receipt.complete, state.scan_receipt.reason
    assert len(state.nodes) == 43 and len(state.edges) == 173
    assert state.scan_receipt.integrity_anomalies == {"bad_endpoint_edge": 1, "duplicate_node_id": 1}

    broken = CappedClient(nodes=43, edges=173, cap=17)
    for row in broken._hydrate_edge_rows:
        row["target_kind"] = "evidence"
    broken_store = build(broken, hydration_page_size=7)
    assert not broken_store.last_hydrate_ok
    assert "integrity anomalies over budget" in broken_store.last_scan_receipt.reason
    return dict(tolerated=state.scan_receipt.integrity_anomalies, nodes=len(state.nodes),
                edges=len(state.edges), broken_scan_rejected=True)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
