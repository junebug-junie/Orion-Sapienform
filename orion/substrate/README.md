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

Not here on purpose: `read_evidence_handles` (evidence pointers for a node, as
of a time). It was built for memory Stage 2 PR E and cut because nothing at
runtime calls it yet. It lands in PR F with recall-by-referent, its consumer.
