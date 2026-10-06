# perf(substrate): batch neighborhood node fetches + read_evidence_handles

Memory Stage 2, PR E (design: `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md`, PR #2496, section 4.3 and row E of 7.5). Recall-by-referent needs to walk from a thing to its related things inside a 150 ms chat budget. Today that walk takes a quarter of a second for one node and over a second for eight.

## Summary

- **Related-things reads are about 10x faster.** For one node with budgets 4/8/8, p95 goes from about 250–390 ms to about 26 ms. Same answers, byte for byte.
- **The real cost was not what the spec guessed.** The spec blamed one round trip per neighbor. Once the `node_id` index went in (#2513), those cost about 1 ms each. The real cost was the query for edges *coming into* a node: both FalkorDB 4.18 and 6.0 planned it as a scan over every node in the graph, 50–120 ms per query. That query now looks the node up by id first.
- **Neighbors are still batched as the spec asked.** All kept neighbors load in one query instead of one query each. Round trips for one node drop from 13 to 7; for the 8-node hub set, from 49 to 27.
- **New read: `read_evidence_handles`.** It answers "which source items back this thing, as of a given time?", newest first, at most N per thing. It returns pointers (`content_ref`), never text. Falkor answers in one query, p95 7–12 ms against the 30 ms budget.
- **CI runs the real-FalkorDB lane on both engine versions** (v4.18.11, which production runs, and 6.0.1). A new test EXPLAINs every query a real read issues and fails on any full scan.

## Outcome moved

Measured on throwaway FalkorDB containers restored from a read-only `DUMP` of production `orion_substrate` (5,051 nodes, 38,393 edges, `node_id` index present). Restores went only into fresh containers, never with `REPLACE`. 20 runs per request, from the host.

| Request (states incl. `proposed`) | 4.18.11 p95 before → after | 6.0.1 p95 before → after | round trips |
|---|---|---|---|
| circe entity, 4/8/8 (the spec's acceptance case) | 249.6 → **26.5 ms** | 391.6 → **25.2 ms** | 13 → 7 |
| circe entity, 12/16/16 | 224.8 → 22.9 ms | 375.1 → 24.8 ms | 17 → 7 |
| "GPU performance and work" concept, 12/16/16 | 430.8 → 53.7 ms | 330.7 → 71.9 ms | 27 → 12 |
| #2497's 8 hub nodes, 12/16/16 | 1,077.4 → 104.5 ms | 1,326.0 → 127.8 ms | 49 → 27 |

Read-only against **production itself** (`GRAPH.RO_QUERY`, no writes), before → after, p95: circe 4/8/8 **258.6 → 26.1 ms**; circe 12/16/16 281.7 → 23.6 ms; GPU 375.1 → 67.5 ms; hub-8 1,488.7 → 102.4 ms. Ids and receipt flags were identical. Only live activation-decay values moved between runs, because production is being written to.

**Equivalence (the contract did not change):**
- Neighborhood: 305 requests (the 5 above plus 300 random focal sets, budgets, states and directions; 150 had non-empty boundaries, 117 truncated, 108 degraded by filter or missing focal, 0 by exception). Every receipt, with full node and edge models, was identical before and after on both engines.
- Evidence handles: 400 random requests against the in-memory reference rule over a fully hydrated copy of the same graph. 0 differences on both engines; 196 non-empty, 114 truncated. p50/p95: 3.6/12.4 ms (4.18), 2.8/7.3 ms (6.0).

The spec's gate ("4/8/8 single-focal ≤ 150 ms p95 on replay", "evidence-handle read ≤ 30 ms p95") is met on the copy and on read-only production. In-container latency is **UNVERIFIED** (no caller is deployed).

## Current architecture

`read_neighborhood` (#2497) ran one query for the focal nodes, one internal-edge query, predicate discovery per focal node and direction, one edge query per (focal, direction, predicate) group and page, and then **one query per admitted neighbor**. The group queries used `MATCH (source)-[e]->(target) WHERE target.node_id = $focal`. Both engines start that plan with a label scan of every `source`, so each incoming-direction query cost 50–120 ms. Nothing read a node's evidence.

## Architecture touched

- `orion/substrate/neighborhood.py`: the budget loop now selects edges first (that needs only ids), then calls `nodes()` once for all admitted neighbors. A missing, duplicate or ineligible neighbor still fails the whole read closed.
- `orion/substrate/neighborhood_backends.py`:
  - Falkor `nodes()` is one `n.node_id IN $node_ids` query. `LIMIT 2n` keeps duplicates visible, as `LIMIT 2` per id did.
  - Group queries bind the focal endpoint with a `WITH` barrier. The WHERE clause is unchanged.
  - SPARQL keeps per-id node reads, because an endpoint may cap result rows and a capped batch would drop nodes. The existing parity test simulates exactly that.
- `orion/substrate/evidence_handles.py` (new): request, handle and result models; one reference rule (`select_handles`); Falkor (one Cypher query), SPARQL and in-memory backends.
- `read_evidence_handles` on the store Protocol, `InMemorySubstrateGraphStore`, `FalkorSubstrateStore`, `GraphDBSubstrateStore`, and `RoutedSubstrateGraphStore` (primary only).
- `scripts/replay_substrate_neighborhood.py --evidence N`: the operator replay now also reads handles for the focal nodes plus the returned neighbors, which is the recall-by-referent shape.

## Concepts (producer → consumer → test)

| Concept | Producer | Consumer | Test |
|---|---|---|---|
| Batched neighbor fetch | `read_neighborhood`; Falkor `IN $node_ids` | every `read_neighborhood` caller (query coordinator `neighborhood` step, replay script) | `test_neighbors_are_fetched_in_one_batched_node_call`, `test_batched_neighbor_fetch_still_fails_closed_on_a_vanished_endpoint`, live parity |
| Index-anchored group read | `match()` in `read_falkor_neighborhood` | same callers | `test_every_issued_read_seeks_the_node_id_index` (mutation-checked: removing the anchor fails it) |
| `read_evidence_handles` + request/handle/result models | `evidence_handles.py` via each store | `replay_substrate_neighborhood.py --evidence N` (operator replay). Recall-by-referent (PR F) is the planned runtime consumer and is **not wired** | `test_evidence_handles.py` (15), `test_evidence_handle_parity_over_random_requests` (real FalkorDB) |
| `PROVENANCE_SHAPES` (`observed_in` out, `supports` in) | `evidence_handles.py` | `select_handles`, `EVIDENCE_HANDLES_CYPHER` | `test_rule_validity_order_and_shapes` (wrong-direction edges ignored) |

The same table is in `orion/substrate/README.md` (new; Juniper asked for a Concepts section per package).

**Cut from the spec on purpose:**
- **The `voices` filter.** The spec's signature has `voices`, but evidence nodes have no voice property until PR A adds it. A filter on a field with no producer would be a keyword cathedral, so it waits for PR A.
- **Hydration through the provenance resolver.** `EvidenceLineageV1` does not exist yet (#2497 follow-up). Handles stop at `content_ref`, and recall (PR F) reads the text from Postgres.

**Added beyond the spec:** handles also follow `evidence -supports-> node`. That is the provenance shape topic-foundry already writes (3,310 live edges), and it made the read measurable on real data today. Without it, every live read would have returned nothing, because no `observed_in` edges exist yet.

## Files changed

- `orion/substrate/neighborhood.py`: one batched neighbor `nodes()` call.
- `orion/substrate/neighborhood_backends.py`: Falkor `IN` node read, index-anchored group reads, shared `sparql_nodes`.
- `orion/substrate/evidence_handles.py`: new read contract and its three backends.
- `orion/substrate/{store,falkor_store,graphdb_store,routed_store}.py`: `read_evidence_handles` method.
- `scripts/replay_substrate_neighborhood.py`: `--evidence N`.
- `orion/substrate/tests/test_neighborhood_backends.py`: the fake Falkor client asserts the batched node query and the anchored group query; 2 new tests.
- `orion/substrate/tests/test_evidence_handles.py`: new, 15 tests (rule, as-of, truncation, degradation, bounds, routed, SPARQL parity, secret-free failure).
- `orion/substrate/tests/test_neighborhood_falkor_live.py`: new real-FalkorDB lane (parity for both reads, plus the EXPLAIN gate).
- `.github/workflows/substrate-neighborhood.yml`: engine matrix `[v4.18.11, 6.0.1]` (was `latest`); new tests added.
- `docs/plans/substrate/2026-10-06-neighborhood-read-contract.md`, `orion/substrate/README.md`: contract and concepts.

## Schema / bus / API changes

- Added: `SubstrateGraphStore.read_evidence_handles(EvidenceHandleRequestV1) -> EvidenceHandleResultV1` (internal store API; not a bus or registry schema).
- Removed / renamed: none.
- Behavior changed: none observable. Neighborhood receipts are identical; only query count and plans changed.
- Compatibility: any `read_neighborhood` backend callback may now receive several ids in one `nodes()` call. All three in-repo backends handle it.

## Env/config changes

- Added / removed / renamed keys: none. `.env_example` not touched, so no `.env` sync was needed.

## Tests run

```text
CI substrate list (13 files):                                    223 passed
real-FalkorDB lane, v4.18.11 (falkor_direct + neighborhood_live): 29 passed
real-FalkorDB lane, 6.0.1:                                        29 passed
orion/substrate/tests + orion/graph/tests:                        1111 passed, 3 failed, 18 skipped
  the 3 failures (test_felt_state_self_definition_lane.py) fail identically on origin/main 029322db2
mutation check: removing the index anchor -> test_every_issued_read_seeks_the_node_id_index FAILS
static gates (all 25 run steps of orion-static-gates.yml, incl. check_definition_drift --gate): 25/25 PASS
pyflakes on touched files: clean; git diff --check: clean
```

## Evals run

```text
python -m orion.substrate.evals.run_neighborhood_eval  -> passes (unchanged output contract)
prod-copy equivalence, neighborhood: 305 requests x 2 engines, 0 receipt diffs
prod-copy equivalence, evidence handles: 400 requests x 2 engines, 0 diffs vs reference
read-only production replay before/after: table above
```

## Docker/build/smoke checks

```text
Not deployed (instruction). No dependency or runtime config change.
Throwaway containers only: falkordb/falkordb:v4.18.11 (127.0.0.1:16411) and :6.0.1 (127.0.0.1:16412),
each restored from one read-only DUMP of orion_substrate into an empty instance.
Production was only read (DUMP, GRAPH.RO_QUERY, one CALL db.indexes()).
```

## Review findings fixed

Review subagent not run (instruction). Self-found during the work:

- Finding: an evidence edge with no `valid_from` counted as valid at any time in the past. "What did I know on 09-29" would have included things first seen on 10-04.
  - Fix: the lower bound is `valid_from`, else `observed_at` (when Orion first saw it), in both the reference rule and the Cypher.
  - Evidence: `test_as_of_reads_the_past_and_future_windows`.
- Finding: the first plan test EXPLAINed a hand-written query, so it could not catch a regression in the real code.
  - Fix: it now records and EXPLAINs every query a real read issues.
  - Evidence: mutation check above.

## Restart required

```text
No restart required. Library code only. Callers pick it up on their next normal rebuild; nothing calls read_evidence_handles at runtime yet.
```

## Risks / concerns

- Severity: low. Concern: Falkor compares validity times as ISO strings. A stored time with a non-UTC offset would compare wrongly. Mitigation: `at` is normalized to UTC; codec writes come from timezone-aware models; 400-request parity on real data showed 0 differences.
- Severity: low. Concern: the per-node evidence limit is applied after the server collects all of a node's matching edges. That work grows with the node's evidence degree; the response size does not. Mitigation: the worst live node (110 evidence edges) still reads in under 20 ms; FalkorDB's own 1 s query timeout still applies.
- Severity: low. Concern: `read_evidence_handles` has only an operator consumer until PR F. Mitigation: it is listed as such in the README and here; if PR F changes the shape, it is cheap to change (no schema, bus or stored data depends on it).
- UNVERIFIED: in-container latency; any runtime caller of `read_evidence_handles`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2519

🤖 Generated with [Claude Code](https://claude.com/claude-code)
