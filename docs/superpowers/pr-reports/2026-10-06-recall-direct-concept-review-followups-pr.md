## Summary

Follow-ups from the code review of PR #2505, the change that made recall's concept lookups read FalkorDB directly instead of loading the whole graph. This PR also adds the node_id index that Juniper approved.

- **One read per reinforced concept.** Each concept that gets an activation bump used to cost two identical queries (one for the node, one for its identity key). Now one query returns both.
- **Deadline respected mid-loop.** If the recall stops waiting partway through the reinforcement loop, no further concept is read or written. Before, the check ran only once, before the loop.
- **A hung FalkorDB is bounded.** The read timeout drops from 5 s to 1.5 s per query. A small circuit breaker skips concept_region for 60 s after 3 timeouts in a row. Each skip is logged and counted, and nothing falls back to loading the graph.
- **The comparison script cannot report a false match.** It refuses to compare when the cached graph load was incomplete. It now reports matched and empty turns separately, with the true maximum over every sample. A `--reinforce` mode times the real live unit (reads plus reinforcement writes), and it only runs against a throwaway copy of the graph.
- **node_id index (approved 2026-10-06).** Every substrate store now creates `CREATE INDEX FOR (n:SubstrateNode) ON (n.node_id)` when it starts. The call is idempotent, has a bounded time and never blocks startup. EXPLAIN confirms the queries use it on FalkorDB 4.18 (production) and 6.0 (CI).

## Outcome moved

Measured on a real copy of `orion_substrate`: a read-only `DUMP` from production, restored into a throwaway FalkorDB 4.18 with 5,041 nodes and 38,206 edges. The 152 queries are real recall queries taken from `recall_telemetry`. Every number below is computed over all samples (3 per query), not over per-query medians.

**The live reinforcing unit** (`fetch_concept_region_fragment_and_reinforce`, what `worker.py:2076-2081` runs), with this PR's single read:

| | no index | with node_id index |
|---|---|---|
| matched turns (52 queries, 156 samples) | median 105.4 ms, p90 140.6, p99 170.7, **max 175.3** | median 99.8 ms, p90 137.2, p99 163.8, **max 175.7** |
| empty turns (100 queries, 300 samples) | median 8.9 ms, p90 13.1, **max 18.7** | median 8.9 ms, p90 14.1, **max 19.3** |
| equivalence with the old cached path | 152/152 identical | 152/152 identical |

**Read-only against production** (host to `orion-athena-falkordb`, no reinforcement, no writes):
- matched turns: median 119.9 ms, p90 207.7, p99 340.7, **max 743.3 ms**;
- empty turns: median 9.5 ms, p90 20.1, max 188.5;
- 152/152 identical.

The production maximum is much higher than on the copy. Production FalkorDB is busy with live writers; the copy is idle.

## Correction to the PR #2505 report

The PR #2505 report has three errors:

1. **It compared numbers that measure different things.** It set the direct path's timings (142 ms median, 217 ms p90) against `pcr_concept_region` from telemetry (37 ms p50, 69 ms p90). The direct timings exclude reinforcement and were measured from the host. The telemetry figure includes reinforcement and is measured in the container. They are not the same measurement.
2. **Its "max" was the wrong number.** The "max 297 ms" was the largest per-query median, not the largest sample. The true worst sample on production, read-only, is 743 ms (above).
3. **Its matched-turn number left out reinforcement.** The comparable figure for the real reinforcing unit is in the table above: about 100–105 ms median and 175 ms max on an idle copy of production. Production adds load on top of that, and the in-container number is still UNVERIFIED.

The orchestrator also measured a full 500/500 read at 673 ms in the container. That read pulls the whole concept slice, which the live path never does, so it is not the hot path.

## Current architecture

As shipped in #2505:
- `reinforce_matched_concepts` called `get_node_by_id` and then `get_identity_key_by_node_id`. Each one issued `NODE_BY_ID_CYPHER` against the same row.
- The `abandoned` check ran once, before reinforcement started.
- Socket reads had a 5 s timeout and there was no backoff, so a hung FalkorDB could cost up to 5 s per query on every turn.
- The comparison script trusted the cached graph load without checking it, and reported the max of per-query medians.
- `orion_substrate` had no indexes: `CALL db.indexes()` came back empty. Every node_id MERGE or lookup scanned the whole label.

## Architecture touched

- `orion/substrate/falkor_direct.py`:
  - new `get_node_and_identity_key()` that does both reads in one query;
  - the builder now runs the index bootstrap on its write client.
- `orion/substrate/falkor_store.py`:
  - `SUBSTRATE_INDEXES`, `ensure_substrate_indexes()` and a `FalkorSubstrateStoreConfig.ensure_indexes` flag (default True);
  - the bootstrap runs only when the store builds its own client, so injected test and replay clients are untouched.
- `services/orion-recall/app/collectors/concept_region.py`: per-node abandon check, and uses the combined read when the store has it.
- `services/orion-recall/app/substrate_store.py`:
  - read timeout 1.5 s, connect timeout 1.0 s;
  - `ConceptRegionBreaker` (3 timeouts, 60 s cooldown, then one probe turn);
  - `_BreakerGuardedStore`, which refuses calls immediately while the breaker is open so one turn cannot wait out several timeouts;
  - `breaker_stats()`.
- `services/orion-recall/scripts/compare_concept_region_direct_vs_cache.py`: hydrate gate, matched/empty split, true maxima, guarded `--reinforce` mode.

## Files changed

- `orion/substrate/falkor_direct.py`: combined read and the index bootstrap in the builder.
- `orion/substrate/falkor_store.py`: index bootstrap.
- `orion/substrate/tests/test_falkor_direct.py`:
  - combined-read unit and live checks;
  - index bootstrap unit tests;
  - EXPLAIN tests on a real FalkorDB for NODE_BY_ID, node MERGE and edge MERGE.
- `orion/substrate/tests/test_falkor_store_hydrate_signal.py`: the timeout-threading test now accounts for the separate, bounded index client.
- `services/orion-recall/app/collectors/concept_region.py`: fixes 1 and 2.
- `services/orion-recall/app/substrate_store.py`: fix 3.
- `services/orion-recall/scripts/compare_concept_region_direct_vs_cache.py`: fix 4.
- `services/orion-recall/tests/test_recall_no_full_graph_hydration.py`: regression tests for fixes 1–3, including a hung-server test.
- `services/orion-recall/tests/test_compare_concept_region_script.py`: regression tests for fix 4.
- `services/orion-recall/tests/test_recall_pcr_block_timing.py`: stub signature gains `abandoned`.
- `services/orion-recall/README.md`, `app/collectors/CONCEPT_REINFORCEMENT_DESIGN.md`: docs.

## Schema / bus / API changes

- Added:
  - FalkorDB index `SubstrateNode(node_id)`, created by every substrate store at startup;
  - `FalkorDirectConceptStore.get_node_and_identity_key`;
  - `ensure_substrate_indexes`;
  - `reinforce_matched_concepts(..., abandoned=)`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - recall's Falkor read timeout is 1.5 s;
  - concept_region is skipped for 60 s after 3 consecutive timeouts.
- Compatibility notes:
  - no bus, schema or env change;
  - the index changes query plans only, never results (152/152 identical with it).

## Env/config changes

- Added keys: none. The timeouts and breaker settings are code constants.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
# baseline (origin/main 74803018a), services/orion-recall
2 failed, 350 passed  (known: recall_policy_harness diagnostic, recall_vector_amputation)
# after
2 failed, 361 passed  (same 2 known failures; +11 new)

# mutation: remove the per-node abandoned check -> test_deadline_mid_reinforcement_loop_stops_further_writes FAILS

# substrate CI list (substrate-neighborhood.yml) + test_falkor_direct.py, no FalkorDB:
190 passed, 15 skipped (real-FalkorDB lane)
# real-FalkorDB lane, both engine versions:
ORION_TEST_FALKOR_URI=...:16401 (4.18.11, prod)  -> 23 passed
ORION_TEST_FALKOR_URI=...:16402 (6.0.1, CI)      -> 23 passed

# blast radius (every test touching FalkorSubstrateStore / the env builders):
orion/spark/concept_induction + orion/substrate/relational + codec tests: 105 passed, 2 failed
orion-substrate-runtime: 85 passed, 3 failed; orion-hub: 57 passed; orion-cortex-orch: 64 passed
  The 5 failures fail identically on origin/main (checked in a detached worktree of main):
  test_falkor_materialization x2, test_worker_attention_self_model_tick x2,
  test_worker_falkor_routed_store x1 -- pre-existing, unrelated.

# static gates (all 25 run steps of orion-static-gates.yml incl. check_definition_drift --gate): 25/25 PASS
check_env_template_parity.py: PASS; git diff --check: clean; pyflakes on new code: clean
```

## Evals run

```text
compare_concept_region_direct_vs_cache.py (152 real recall queries):
  read-only vs production                         -> 152/152 identical (table above)
  --reinforce vs throwaway copy, no index          -> 152/152 identical
  --reinforce vs throwaway copy, node_id index     -> 152/152 identical
  --reinforce --confirm-throwaway at port 6380     -> refused, exit 2
```

## Docker/build/smoke checks

```text
Not deployed (instruction). No new dependencies. Throwaway FalkorDB containers only
(4.18.11 and 6.0.1); production FalkorDB was only read (GRAPH.RO_QUERY and one DUMP).
```

## Index proposal (approved by Juniper; included in this PR)

**The index:** `CREATE INDEX FOR (n:SubstrateNode) ON (n.node_id)`.

**Why only node_id.** Every substrate MERGE and every single-node read filters on it. No Cypher anywhere filters on `identity_key`, because identity lookups go through the in-process cache, so it is not indexed. `edge_id` is matched inside `MERGE (source)-[e {edge_id}]->(target)`, which the planner resolves by expanding from the already-indexed endpoints. EXPLAIN confirms `Expand Into` there.

**Who benefits:**
- Writers:
  - `FalkorSubstrateStore.upsert_node` and `upsert_edge` (two endpoint MERGEs each);
  - these run in orion-cortex-exec, orion-hub (concept atlas, decay scheduler), orion-substrate-runtime (dynamics, prediction-error writers), Spark concept induction (`falkor_materialization`, `bus_worker`) and orion-recall's reinforcement.
- Readers:
  - recall `NODE_BY_ID_CYPHER` (falkor_direct);
  - `neighborhood_backends.read_falkor_neighborhood` node fetch (`n.node_id = $node_id`) and its `source.node_id IN $ids` / `target.node_id = $focal` filters;
  - Hub `attention_organ_routes` (`n.node_id STARTS WITH $prefix`), whose comment already notes the missing index.

**Measured on a throwaway FalkorDB 4.18 holding a real copy of production (5,041 nodes, 38,206 edges):**

| query | before (median / p90 / max) | after |
|---|---|---|
| NODE_BY_ID, 200 random ids | 4.14 / 4.69 / 7.65 ms | 0.94 / 1.01 / 4.74 ms |
| node + identity (reinforcement read) | 4.15 / 4.55 / 7.42 ms | 1.00 / 1.26 / 4.53 ms |
| upsert MERGE (reinforcement write) | 1.59 / 1.95 / 5.45 ms | 1.05 / 1.31 / 4.35 ms |
| reinforcing unit, matched turn | 105.4 ms median, max 175.3 | 99.8 ms median, max 175.7 |

- Matched turns barely move, because the edge-cut query still dominates them.
- The node read gets about 4x faster.
- The gain grows with graph size, since an unindexed lookup scans every SubstrateNode.

**Build time and locking at production size:**
- `CREATE INDEX` took **2.54 ms**.
- A concurrent reader issued 369 NODE_BY_ID queries while the index was being built: median 0.92 ms, max 9.85 ms. No stall was seen.
- Both 4.18 and 6.0 answer a repeat create with "Attribute 'node_id' is already indexed", which the bootstrap treats as success. Neither version accepts `IF NOT EXISTS`.

**Startup safety:**
- The bootstrap uses its own client with a 5 s read and 2 s connect timeout. In recall it uses the store's 1.5 s / 1.0 s timeouts.
- It catches every exception and logs `falkor_substrate_index_create_failed`.
- It returns, never raises, so a down FalkorDB delays construction by at most a few seconds and never fails it.

**Apply to production now** (one line, for the orchestrator to run):

```bash
docker exec orion-athena-falkordb redis-cli GRAPH.QUERY orion_substrate "CREATE INDEX FOR (n:SubstrateNode) ON (n.node_id)"
```

Verify: `docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_substrate "CALL db.indexes()"`

**Rollback:**

```bash
docker exec orion-athena-falkordb redis-cli GRAPH.QUERY orion_substrate "DROP INDEX ON :SubstrateNode(node_id)"
```

Once this PR is deployed, every substrate service re-creates the index when it starts. To keep it dropped, the code must be reverted as well, or `ensure_indexes=False` set in `FalkorSubstrateStoreConfig`. The flag is code-only; there is no env key for it.

The bootstrap also creates the same index on the AI Town and Self graphs (`orion_substrate_aitown`, `orion_substrate_self`) the next time their stores start.

## Review findings fixed

- Finding: reinforcement issued two `NODE_BY_ID` queries per node.
  - Fix: `get_node_and_identity_key()` makes one query; the collector uses it when the store has it.
  - Evidence: `test_reinforcement_reads_each_node_once`; the recall spy test now sees exactly one `NODE_BY_ID` per reinforced node; the live fixture test checks the result equals the cache.
- Finding: the abandoned check ran only before the loop.
  - Fix: the event is passed into `reinforce_matched_concepts` and checked before every node's read and write.
  - Evidence: `test_deadline_mid_reinforcement_loop_stops_further_writes` (writes `["c1"]` only; it fails when the check is removed); `test_abandoned_is_forwarded_from_the_live_entry_point`.
- Finding: a hung Falkor could cost up to 5 s on every turn, with no backoff.
  - Fix: 1.5 s read timeout and the breaker.
  - Evidence: `test_hung_falkor_is_bounded_per_turn_then_skipped`. It uses a real socket server that accepts connections and never answers: each turn stays under 1.5 s, the breaker opens after 3 timeouts, and the next turn returns in under 50 ms with no new connection. Also `test_breaker_opens_after_consecutive_timeouts_and_reopens_after_cooldown`, `test_success_resets_the_consecutive_count` and `test_non_timeout_errors_do_not_trip_the_breaker`.
- Finding: the comparison script did not check that the cached graph load completed, and mixed matched and empty turns together.
  - Fix: exits 2 before comparing when the load is incomplete; separate matched/empty summaries with true maxima.
  - Evidence: `test_compare_concept_region_script.py` (4 tests).
- Code review of this follow-up: not run here. The orchestrator runs it.

## Restart required

After merge, from the primary checkout on main. Every service that builds a substrate store picks up the index bootstrap; recall picks up fixes 1–3.

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-recall up -d --build
```

The other substrate services (cortex-exec, hub, substrate-runtime, concept induction) get the bootstrap on their next normal rebuild. The one-line `CREATE INDEX` above covers production until then.

## Risks / concerns

- Severity: medium
  - Concern: matched turns still cost about 100 ms median on the copy and up to 743 ms on busy production (read-only). The edge-cut query dominates, and the node_id index does not help it.
  - Mitigation: concept_region runs at the same time as active_packet and under the recall deadline. The breaker caps the damage from a hung Falkor. The faster semantic option (rank only the matched concepts' own edges) still needs Juniper's decision.
- Severity: low
  - Concern: the breaker counts only timeouts. A FalkorDB that refuses connections fails fast on every turn and does not trip it.
  - Mitigation: refusals cost milliseconds, not seconds. Tested in `test_non_timeout_errors_do_not_trip_the_breaker`.
- Severity: low (test-infra observation)
  - Concern: on FalkorDB 4.18, `RESTORE ... REPLACE` over an existing graph key hung the whole throwaway server (0% CPU, PING timed out; the log shows the graph counted twice, "10082/5041 nodes").
  - Mitigation: benchmarks restore only into a fresh container. Production never runs RESTORE, and production was unaffected (PONG throughout).
- UNVERIFIED: none of this has run in the recall container yet; all latencies are from the host. The production index has not been created by this branch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2513

🤖 Generated with [Claude Code](https://claude.com/claude-code)
