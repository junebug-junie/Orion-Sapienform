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
    return {"passed": True, "cases": reports}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
