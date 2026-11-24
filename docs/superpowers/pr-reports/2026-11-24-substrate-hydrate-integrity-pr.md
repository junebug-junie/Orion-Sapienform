# fix(substrate): hydrate skips corrupt rows; upsert_node never twins a node_id

Fixes #3512

## Summary

- Hub's boot hydrate of the substrate graphs failed on one bad durable row each, so Hub ran on an empty graph. Now one bad row costs one row: hydrate skips it, counts it, and loads the rest.
- A scan where many rows are bad still fails closed (an integrity budget), so a broken scan can never pass as a good, edgeless refresh.
- `upsert_node` no longer creates a second node with an existing `node_id`. It merged on `SubstrateNode:<TypeLabel>`, so any node missing its type label (an edge placeholder, or a node whose label FalkorDB dropped) got a twin. It now merges on `SubstrateNode {node_id}` and then sets the type label.
- The hydrate page query sorts on the bare id before it projects properties. The old form timed out on the AI Town graph (202k edges) against FalkorDB's 1 s read timeout. That was a third failure, hidden behind the duplicate error.
- The legacy payload migration never lets a stale legacy twin overwrite its native node. The twin is only deleted.

## Outcome moved

Read-only replay of this branch's hydrate against the live graphs (`GRAPH.RO_QUERY`, 2026-11-24):

| graph | before (Hub log) | this branch |
|---|---|---|
| `orion_substrate` | failed, 0 of 19,142 nodes cached | ok: 19,146 nodes, 79,975 edges, node gap 0, 1 anomaly (`bad_endpoint_edge`) |
| `orion_substrate_aitown` | failed, 0 of 52,160 nodes cached | ok: 52,159 nodes, 202,718 edges, node gap 1 (the twin), anomalies `duplicate_node_id: 1`, `duplicate_edge_id: 425` |

## Cause, with evidence

Both logged errors come from inconsistencies inside FalkorDB's own storage. No Orion writer produced the bad rows on purpose.

1. **`missing or mismatched endpoint: sub-concept-wp-read-8b7af82ab1ff79e8`** (`orion_substrate`). One edge object comes back twice from one scan.
   - The edge is internal id 66265, `edge-ecdf2802-...`. It is an `observed_in` edge written 11-17 by the memory referent projector.
   - The scan returns it once as `referent-b96fc90e (5096) -> ev-4a53d148 (15934, evidence)`. That is correct: the edge id is `uuid5(referent | episode_memory:0f1a308f-... | observed_in)`.
   - The scan returns it a second time as `5096 -> 16384`. Node 16384 is the unrelated Concept "MFU". This copy is a phantom adjacency entry.
   - `MATCH ... WHERE e.edge_id = 'edge-ecdf2802-...'` returns both rows with the same `id(e)`.
   - A paged audit of all three graphs found exactly one such multi-endpoint edge.
   - Which error Hub logs depends on sort order. If the phantom copy comes first, Hub logs `mismatched endpoint`. If the real copy comes first, the cursor guard sees the repeated object id and Hub logs `nonadvancing or invalid object cursor`. Hub logged that error at 22:50:33Z after its 22:44Z restart.
2. **`duplicate node_id: sub-concept-topicfoundry-b5c57820-...-33`** (`orion_substrate_aitown`). Two rows hold this `node_id`: 49919 (last written 11-18) and 50246 (written at every topic-foundry upsert since).
   - `labels(49919)` returns `[SubstrateNode, Concept]`, but `MATCH (n:Concept)` does not return 49919. This was checked for every label in every graph. 49919 is the only node missing from its label matrix.
   - `upsert_node`'s `MERGE (n:SubstrateNode:Concept {node_id})` runs as an index scan on SubstrateNode and then a Concept label check (`GRAPH.EXPLAIN`). So it missed 49919 and created 50246.
   - The same typed MERGE also creates a twin of any bare placeholder node that an edge MERGE creates. This was reproduced on a throwaway FalkorDB 4.18.11.
3. **`Query timed out`** (`orion_substrate_aitown`, found while verifying). The first edge page ran `RETURN <30 fields> ... ORDER BY object_id LIMIT 1000`. That projects all 202k rows before sorting, which passes the server `TIMEOUT 1000` ms. Hub never reached this scan before, because it died on the duplicate first. The new form `WITH * ORDER BY id(e) LIMIT $page_size RETURN ...` runs in 537 ms. On `orion_substrate` it runs in 534 ms, down from 799 ms.

Other findings that do not block hydrate:
- `orion_substrate` node 17051 is in the SubstrateNode and Concept label scans, but `labels()` returns `[]` for it. It hydrates fine.
- A filtered reverse traversal (`MATCH (s)-[e]->(t) WHERE t.node_id = X`) returns null edge properties for the phantom edge.

## Effect today (before this patch)

- After the 21:28Z restart, Hub logged `falkor_substrate_hydrate_node_gap ... cache_nodes=0 gap=19142` for `orion_substrate`. It logged `cache_nodes=0 gap=52160` for `orion_substrate_aitown`.
- Retries later showed `cache_nodes=78`, then `429`, for `orion_substrate`, and `398` for `orion_substrate_aitown`. Those counts are only Hub's own writes since boot.
- So every Hub reader of the in-memory substrate graph (snapshot, focal or hotspot or concept regions, the identity cache) ran on an empty or near-empty graph.
- Missing identities were resolved by the durable lookup, so they were not duplicated again.
- `FalkorDirectConceptStore._node_row` returns None for the twinned node, so direct reads see it as absent.

## Current architecture

`FalkorSubstrateStore._hydrate_from_durable` pages native nodes, legacy nodes, native edges and legacy edges into a fresh `InMemorySubstrateGraphStore`. It swaps that store in only if every row validates. Any bad row failed the whole scan and kept the last good cache. At boot there is no last good cache, so the cache stayed empty.

## Architecture touched

- `orion/substrate/falkor_store.py`:
  - hydrate integrity tolerance and budget
  - the page query shape
  - the cursor guard (non-decreasing within a page)
  - the untyped MERGE plus type label in `upsert_node`
  - legacy-twin handling
- `orion/substrate/store.py`: `CompleteScanReceipt.integrity_anomalies`.

## Files changed

- `orion/substrate/falkor_store.py`: the fixes above.
- `orion/substrate/store.py`: new receipt field `integrity_anomalies: dict[str, int]`.
- `orion/substrate/falkor_direct.py`: comment only.
- `orion/substrate/tests/test_complete_hydration.py`: regression tests for each live failure, the budget, legacy twins and the identity rule.
- `orion/substrate/tests/test_falkor_node_identity_live.py` (new): real-FalkorDB tests. A placeholder does not twin, a node missing its label is updated in place, and legacy twin/lone migration works.
- `orion/substrate/evals/run_complete_hydration_eval.py`: integrity case.
- `orion/substrate/tests/test_falkor_store.py`, `test_falkor_direct.py`, `services/orion-recall/tests/test_recall_no_full_graph_hydration.py`: MERGE text assertions.
- `.github/workflows/substrate-neighborhood.yml`: runs the new live test. The step fails if the test is skipped.

## Schema / bus / API changes

- Added: `CompleteScanReceipt.integrity_anomalies` (in-process dataclass, default `{}`). Kinds: `duplicate_node_id`, `bad_endpoint_edge`, `duplicate_edge_id`, `legacy_payload_twin`.
- Added log lines: `falkor_substrate_hydrate_integrity_anomalies`, and `integrity_anomalies=` on `falkor_substrate_hydrate_stats`.
- Behavior changed:
  - Hydrate tolerates up to `min(max(50, 1% of rows), rows/2)` corrupt rows.
  - `upsert_node` Cypher is `MERGE (n:SubstrateNode {node_id}) ... SET n:<Label>`.
- No bus, channel or registry changes.

## Env/config changes

None.

## Tests run

```text
pytest orion/substrate/tests/test_complete_hydration.py                  51 passed
ORION_TEST_FALKOR_URI=<throwaway falkordb 4.18.11> pytest orion/substrate/tests/test_falkor_node_identity_live.py   3 passed
pytest orion/substrate/tests + recall + repair tests                       1304 passed, 53 skipped, 1 failed
  (test_felt_state_lived_answers -- also fails on clean main)
pytest services/orion-recall/tests/test_recall_no_full_graph_hydration.py passed
pytest tests/test_repair_substrate_duplicate_identities.py               passed (live part skipped locally; CI runs it)
```

Pre-existing local failures that also fail on clean main: `orion/spark/concept_induction/tests/*` (28) and `services/orion-substrate-runtime/tests/test_worker_falkor_routed_store.py::test_write_prediction_error_node_preserves_dynamics_state_on_rewrite`.

## Evals run

```text
python -m orion.substrate.evals.run_complete_hydration_eval
  integrity: tolerated {duplicate_node_id: 1, bad_endpoint_edge: 1}, nodes 43, edges 173, broken_scan_rejected true
```

## Docker/build/smoke checks

- Read-only live replay of this branch's hydrate against production FalkorDB (`read_only=True`, so `GRAPH.RO_QUERY` only and no migration writes): numbers are in the table above.
- No image build is needed. The change is in the shared `orion/` package, which Hub and the substrate services bake in.

### Live replay

- `orion_substrate`: ok. 19,146 nodes, 79,975 edges, gap 0, `{bad_endpoint_edge: 1}`.
- `orion_substrate_aitown`: ok. 52,159 nodes, 202,718 edges, `{duplicate_node_id: 1, duplicate_edge_id: 425}`, 253 s from the host. The 425 duplicate edge ids are fallout from the twin. `upsert_edge`'s endpoint MERGE matches both twin rows, so each topic-foundry edge was written onto both (50246 has 425 edges). The repair below removes them.
- Before the page-query change, the same replay failed `Query timed out` on the first aitown edge page.

## Review findings fixed

- Finding: a legacy payload twin could overwrite its native node, in both the cache and Falkor.
  - Fix: a legacy row whose `node_id` already has a native row is counted (`legacy_payload_twin`) and only deleted. It is never seeded or upserted.
  - Evidence: `test_a_legacy_payload_twin_never_overwrites_its_native_node`, and the live test asserts that the canonical label survives.
- Finding: a newer twin replacing the kept row bypassed the smallest-`node_id` identity rule.
  - Fix: `set_identity_canonical(min(...))` after the replace.
  - Evidence: `test_a_newer_twin_keeps_the_smallest_node_id_canonical_for_a_shared_identity`.
- Finding: a small graph whose every edge is bad passed the 50-row floor.
  - Fix: the budget is also capped at half the rows seen.
  - Evidence: `test_a_small_graph_whose_every_edge_is_corrupt_still_fails_closed`.
- Finding: existing durable twins stay. Also, a `SET n:Concept` repair of a dropped label bit is unverified on the real corruption.
  - Fix: the repair command for Juniper is below. A `verify-live` issue was filed.
- Not changed (minor, accepted):
  - A `node_id` reused across node kinds now becomes one node with two type labels instead of two nodes. Ids are kind-prefixed (`sub-concept-...`, `ev-...`), and an edge of the other kind is now a counted anomaly, not a failed boot.
  - Native edges on a deleted legacy twin are dropped by `DETACH DELETE`. That was also true before. Live graphs hold 0 legacy payload rows.

## Data repair needing Juniper (production write, NOT run)

**AI Town twin.** Keep 50246: it is in the Concept label scan, and writers land on it. Re-point 49919's edges onto 50246, then delete 49919. Falkor object ids can be reused, so every statement checks the `node_id` too.

1. Snapshot (read-only):

```bash
mkdir -p /tmp/substrate-twin-3512 && for q in "MATCH (n) WHERE id(n) IN [49919,50246] RETURN id(n), properties(n)" "MATCH (n)-[e]-(m) WHERE id(n) = 49919 RETURN id(e), type(e), id(startNode(e)), id(endNode(e)), properties(e)"; do docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_substrate_aitown "$q"; done > /tmp/substrate-twin-3512/snapshot.txt
```

2. Repair. Run these one at a time. Each statement is idempotent. Edge types were checked live: out `co_occurs_with`, in `supports` and `co_occurs_with`.

```bash
N=sub-concept-topicfoundry-b5c57820-212b-4393-a7bf-6bf7a9199c6f-33; G=orion_substrate_aitown; F="docker exec orion-athena-falkordb redis-cli GRAPH.QUERY $G"
$F "MATCH (old) WHERE id(old) = 49919 AND old.node_id = '$N' MATCH (keep) WHERE id(keep) = 50246 AND keep.node_id = '$N' MATCH (old)-[e:co_occurs_with]->(m) WHERE id(m) <> 50246 OPTIONAL MATCH (keep)-[x:co_occurs_with]->(m) WHERE x.edge_id = e.edge_id WITH keep, e, m, x WHERE x IS NULL CREATE (keep)-[e2:co_occurs_with]->(m) SET e2 = properties(e) RETURN count(e2)"
$F "MATCH (old) WHERE id(old) = 49919 AND old.node_id = '$N' MATCH (keep) WHERE id(keep) = 50246 AND keep.node_id = '$N' MATCH (m)-[e:co_occurs_with]->(old) WHERE id(m) <> 50246 OPTIONAL MATCH (m)-[x:co_occurs_with]->(keep) WHERE x.edge_id = e.edge_id WITH keep, e, m, x WHERE x IS NULL CREATE (m)-[e2:co_occurs_with]->(keep) SET e2 = properties(e) RETURN count(e2)"
$F "MATCH (old) WHERE id(old) = 49919 AND old.node_id = '$N' MATCH (keep) WHERE id(keep) = 50246 AND keep.node_id = '$N' MATCH (m)-[e:supports]->(old) WHERE id(m) <> 50246 OPTIONAL MATCH (m)-[x:supports]->(keep) WHERE x.edge_id = e.edge_id WITH keep, e, m, x WHERE x IS NULL CREATE (m)-[e2:supports]->(keep) SET e2 = properties(e) RETURN count(e2)"
$F "MATCH (old) WHERE id(old) = 49919 AND old.node_id = '$N' DETACH DELETE old"
```

These statements were verified on a throwaway FalkorDB 4.18.11 with a simulated twin. Edges were re-pointed, the keeper's own edges won, a re-run created 0 edges, and the delete removed the old node and its edges. Check afterwards: `MATCH (n:SubstrateNode {node_id:'<N>'}) RETURN count(n)` returns 1. The next Hub hydrate should log neither `duplicate_node_id` nor `duplicate_edge_id` for aitown.

**`orion_substrate` phantom edge 66265.** No repair is recommended. The phantom lives in FalkorDB's adjacency matrix, and the real edge shares its id, so deleting by id risks losing the real edge too. Hydrate now costs it one counted anomaly. If it ever matters, the fix is a graph rebuild, which is out of scope here.

## Restart required

Hub (and any service that hydrates a substrate graph) picks this up on its normal post-merge rebuild from clean primary `main`:

```bash
scripts/rebuild_services_from_git_diff.sh
```

## Risks / concerns

- Severity: medium. Concern: rows tolerated under the budget hide real corruption. Mitigation: every refresh logs the per-kind counts, the stats line carries a total, and the counts ride on the scan receipt.
- Severity: low. Concern: the untyped MERGE now updates both rows of an existing twin until the repair runs. Mitigation: both rows get identical data; run the repair above.
- Severity: low. Concern: whether `SET n:Concept` re-adds a label bit that FalkorDB dropped is UNVERIFIED. Mitigation: verify-live issue #3864.
- Issues opened: #3864 (verify-live + needs-juniper: post-deploy hydrate check and the aitown twin repair). Issues closed: #3512.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
