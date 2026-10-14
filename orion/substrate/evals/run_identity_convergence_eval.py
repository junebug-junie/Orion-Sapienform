"""Identity convergence across repeated topic-foundry runs and restarts.

Live defect (2026-10-14): each topic-foundry run mints run-scoped node_ids,
and identity resolution was cache-only, so every restart or failed hydrate
re-minted every label -- up to 94 copies of one AI Town concept. This eval
replays R runs of the same L labels, starting each run from an EMPTY cache
(the worst case: a restart whose hydrate failed), and measures how many
durable nodes each identity ends up with. Pass = exactly one per identity.

    python -m orion.substrate.evals.run_identity_convergence_eval
"""
from __future__ import annotations

import json

from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1
from orion.substrate.evals.complete_hydration_fixture import CappedClient, build, topic_concept
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.reconcile import SubstrateIdentityResolver


class PersistingClient(CappedClient):
    """CappedClient whose node MERGE writes land in its durable rows."""

    def graph_query(self, cypher, params=None):
        if cypher.startswith("MERGE (n:SubstrateNode") and params:
            row = {k: v for k, v in params.items()}
            for i, existing in enumerate(self._hydrate_node_rows):
                if existing.get("node_id") == row["node_id"]:
                    self._hydrate_node_rows[i] = {**existing, **row}
                    break
            else:
                self._hydrate_node_rows.append(row)
        return super().graph_query(cypher, params)


def run(runs: int = 6, labels: int = 40) -> dict:
    client = PersistingClient(nodes=0, edges=0, cap=7)
    for r in range(runs):
        store = build(client)
        store._cache = type(store._cache)()  # restart with a failed hydrate
        record = SubstrateGraphRecordV1(
            graph_id=f"sub-graph-topicfoundry-run{r}", anchor_scope="world",
            nodes=[topic_concept(f"sub-concept-topicfoundry-run{r}-{t}", f"topic {t}") for t in range(labels)],
            edges=[],
        )
        SubstrateGraphMaterializer(store=store, identity_resolver=SubstrateIdentityResolver(store=store)).apply_record(record)
    per_identity: dict[str, int] = {}
    for row in client._hydrate_node_rows:
        per_identity[row["identity_key"]] = per_identity.get(row["identity_key"], 0) + 1
    worst = max(per_identity.values())
    report = {"runs": runs, "labels": labels, "identities": len(per_identity),
              "durable_nodes": len(client._hydrate_node_rows), "max_copies_per_identity": worst,
              "passed": len(per_identity) == labels and worst == 1}
    assert report["passed"], report
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
