# orion/substrate

The substrate graph store, its codecs, and the bounded reads over it. Read
contracts: `docs/plans/substrate/2026-10-06-neighborhood-read-contract.md`.

## Concepts

Only concepts with a real producer, a real consumer and a test are listed.
Keep this table in sync with the code in the same PR.

| Concept | Plain-English meaning | Producer | Consumer | Test |
|---|---|---|---|---|
| Batched neighbor fetch | A neighborhood read loads all the related nodes it kept in one query, not one query each | `read_neighborhood` in `neighborhood.py`; Falkor `n.node_id IN $node_ids` in `neighborhood_backends.py` | Every `store.read_neighborhood` caller: the query coordinator's `neighborhood` step, `scripts/replay_substrate_neighborhood.py` | `test_neighbors_are_fetched_in_one_batched_node_call`, `test_batched_neighbor_fetch_still_fails_closed_on_a_vanished_endpoint`, `test_neighborhood_falkor_live.py` |
| Index-anchored group read | When reading the edges coming *into* a node, look that node up by id first instead of scanning every node in the graph | `match()` in `read_falkor_neighborhood` | Same callers as above | `test_every_issued_read_seeks_the_node_id_index` (real FalkorDB, EXPLAIN on every issued query) |
| `read_evidence_handles` / `EvidenceHandleRequestV1` / `EvidenceHandleV1` / `EvidenceHandleResultV1` | "Which source items (memories, reveries, topic runs…) back this thing, as of a given time?" Returns pointers to the items, never their text | `evidence_handles.py` (Falkor one query, SPARQL, in-memory reference) via each store's `read_evidence_handles` | `scripts/replay_substrate_neighborhood.py --evidence N` (operator replay). The planned runtime consumer is recall-by-referent (memory Stage 2 PR F); not wired yet | `test_evidence_handles.py`, `test_evidence_handle_parity_over_random_requests` (real FalkorDB) |
| Provenance shapes (`PROVENANCE_SHAPES`) | The two edge shapes that count as "this evidence is about this node": node `observed_in` evidence, or evidence `supports` node | `evidence_handles.py` | `select_handles` and `EVIDENCE_HANDLES_CYPHER` | `test_rule_validity_order_and_shapes` (wrong-direction edges are ignored) |
