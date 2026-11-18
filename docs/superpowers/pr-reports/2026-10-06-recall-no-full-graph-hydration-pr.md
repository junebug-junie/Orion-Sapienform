## Summary

- Recall no longer loads Orion's whole concept graph. Before, it built a complete in-memory copy of the substrate graph (17-25 s since PR #2500) just to look up a few concepts per turn. Now every lookup is a small, bounded read straight from FalkorDB.
- The results are the same. On 152 real recall queries from `recall_telemetry`, the new reads gave exactly the same fragments in the same order as the old copy, 152 out of 152 (3,506 fragments across the 52 turns that matched a concept).
- Deleted the machinery that only existed to hide the slow load: the boot warm-up task, its lifespan hook, the lock timeouts, the retry backoff and the `last_hydrate_ok` checks.
- Reinforcement (the small activation bump a concept gets when it comes up in a turn) now reads the concept's current value from FalkorDB, not a copy frozen at boot. It still skips the bump when the recall has already given up on the call.
- A new test fails if any recall path ever hydrates or snapshots the graph. A mutation check confirmed it: swapping the old store back in failed 5 of its 8 tests.

## Outcome moved

- **Boot / cold start:** no 17-25 s load any more, and no near-30 s timeout after which concept_region silently returned nothing.
- **Freshness:** the old copy was built once and never refreshed on the recall path, so concept_region read a graph frozen at boot. Now it reads the live graph on every turn.
- **Per-turn cost, measured live (read-only, host to `orion-athena-falkordb`, 152 real queries):**
  - Turns where nothing matches (about two thirds): **10.6 ms median** (p90 16 ms). This is one light query that returns only labels.
  - Turns where a concept matches: **142 ms median, 217 ms p90, 297 ms max**.
  - For comparison, the old cached path (`pcr_concept_region` in telemetry, last 7 days): 37 ms p50, 69 ms p90, 439 ms max. That excludes the 17-25 s first-load cost, which landed on boot or on the first turn.
- **The 100 ms target is NOT met on turns that match.** See Risks below: the cost comes from keeping the old edge-selection rule exactly as it was.

## Current architecture

- `services/orion-recall/app/substrate_store.py` cached a `FalkorSubstrateStore`. Its `read_concept_region`, `get_node_by_id` and `get_identity_key_by_node_id` all read from `self._cache`, a complete copy of the graph hydrated when the store is built.
- `app/main.py` started a warm-up task at boot (`WARMUP_TIMEOUT_S=30`). Request threads waited up to `REQUEST_LOCK_TIMEOUT_S=2`. A failed or empty hydrate triggered a 5 s to 300 s backoff.
- concept_region asks for the top 500 concepts by (salience, confidence), matches their labels against the turn text, and keeps the edges that touch a matched concept, out of the top 500 edges touching any of those 500 concepts.

## Architecture touched

- New shared module `orion/substrate/falkor_direct.py` with `FalkorDirectConceptStore`. It has no `snapshot()` and no cache. A turn runs at most three bounded `GRAPH.RO_QUERY` reads:
  1. Rank the top N concepts in Cypher and return only their object ids and labels. The labels are matched in Python, with the same `_label_matches` as before.
  2. Only on a match: fetch the full rows for the matched concepts by object id.
  3. Only on a match: compute the edge cut in Cypher (the top M edges by salience and confidence among all edges touching the N ranked concepts) and return full rows only for the cut edges that touch a matched concept. This query recomputes the concept ranking itself instead of taking object ids as a parameter, because FalkorDB 6.0 has a planner bug with the id-parameter form (see Review findings fixed).
- The ranking keys copy the codec's decode defaults: a NULL salience counts as 0.0, and a NULL or 0 confidence counts as 0.5. Ties break in hydration order (ascending Falkor object id). So the order is identical to `InMemorySubstrateGraphStore._read_by_node_predicate`.
- Reinforcement reads look up one node by `node_id`. The write still goes through `FalkorSubstrateStore(hydrate=False).upsert_node` (one `MERGE ... SET`, with the reducer-owned metadata keys skipped) on a separate client that is allowed to write.
- `fetch_concept_region_fragment` calls `read_concept_region_matching(keep_label=...)` when the store has it. The full-slice path stays for the in-memory store. Applying the collector's own filter afterwards changes nothing, because the slice is already filtered.
- Recall refuses the `routed`, `graphdb` and `sparql` backends (each brings its own cached graph): it logs `recall_substrate_store_unsupported_backend` and concept_region returns nothing. `in_memory` and unset behave as before.

## Files changed

- `orion/substrate/falkor_direct.py`: new hydration-free store and its builder.
- `orion/substrate/tests/test_falkor_direct.py`: unit lane (query count, no snapshot surface, single-node edge cases, write path, builder). Equivalence and latency lane against a real throwaway FalkorDB.
- `.github/workflows/substrate-neighborhood.yml`: adds a `falkordb/falkordb` service container and runs the equivalence lane in CI.
- `services/orion-recall/app/substrate_store.py`: rewritten. Builds the direct store and never hydrates. Socket timeouts kept, now 5 s read and 2 s connect, against the 30 s and 5 s that were sized for hydration.
- `services/orion-recall/app/main.py`: warm-up task, lifespan hook and cancel removed.
- `services/orion-recall/app/collectors/concept_region.py`: uses the matching read when available. Docstring updated.
- `services/orion-recall/app/worker.py`: comments only.
- `services/orion-recall/tests/test_recall_no_full_graph_hydration.py`: new spy tests plus boot, abandon, Falkor-down and backend-refusal tests.
- `services/orion-recall/tests/test_recall_pcr_block_timing.py`: removed 14 warm-up, backoff and hydrate tests whose code is gone. The deadline and abandon tests are kept.
- `services/orion-recall/tests/test_concept_region_wiring.py`: singleton tests now patch `_build_store`.
- `services/orion-recall/scripts/compare_concept_region_direct_vs_cache.py`: read-only live equivalence and latency check.
- `services/orion-recall/.env_example`, `README.md`, `app/collectors/CONCEPT_REINFORCEMENT_DESIGN.md`: docs, and the dead key removed.

## Schema / bus / API changes

- Added: `FalkorDirectConceptStore` (library class), with `read_concept_region`, `read_concept_region_matching`, `get_node_by_id`, `get_identity_key_by_node_id` and `upsert_node`.
- Removed: `app.substrate_store.warm_substrate_store`, `last_failure_reason`, `REQUEST_LOCK_TIMEOUT_S`, `WARMUP_TIMEOUT_S`, `RETRY_BACKOFF_*`.
- Renamed: none.
- Behavior changed: recall's concept reads and reinforcement reads hit the live FalkorDB on every turn. The `recall_substrate_store_warmed` and `_warmup_failed` boot log lines are gone. Non-Falkor graph backends are refused.
- Compatibility notes: no bus, schema or channel change. The fragment shape and the ids are unchanged.

## Env/config changes

- Added keys: none.
- Removed keys: `SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC`, from `services/orion-recall/.env_example` only. Recall no longer snapshots, and compose never passed this key to the container anyway.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-recall`: yes, "No changes needed". The sync script only adds keys, so the removed key is still in the local `services/orion-recall/.env`. It is harmless (nothing reads it and compose does not pass it), and the operator can delete it by hand.
- Skipped keys requiring operator action: none.

## Tests run

```text
# baseline (main @ f9309426b), services/orion-recall
.venv/bin/python -m pytest tests -q -p no:cacheprovider
  2 failed, 356 passed   (known: recall_policy_harness diagnostic, recall_vector_amputation)

# after
.venv/bin/python -m pytest tests -q -p no:cacheprovider
  2 failed, 350 passed   (same 2 known failures; -14 obsolete warm-up/hydrate tests, +8 new)

# mutation check: swap recall's builder back to build_falkor_substrate_store_from_env
pytest tests/test_recall_no_full_graph_hydration.py  -> 5 failed, 3 passed (restored -> 8 passed)

# substrate CI lane (substrate-neighborhood.yml list + new file, no FalkorDB)
188 passed, 12 skipped (the 12 real-FalkorDB tests)

# real throwaway FalkorDB, both graph-module versions
ORION_TEST_FALKOR_URI=redis://127.0.0.1:16399 (4.18.11, prod) pytest orion/substrate/tests/test_falkor_direct.py
ORION_TEST_FALKOR_URI=redis://127.0.0.1:16400 (6.0.1, CI)     pytest orion/substrate/tests/test_falkor_direct.py
  18 passed on each: full slice == hydrated cache for (500,500),(32,64),(40,200),(1,1),(120,1500);
  matching slice == collector filter over cache for 5 needles; single-node reads == cache.
  Fixture includes repeated salience values (ties), stored 0 and NULL confidence, a NULL
  salience, self-loops, evidence/entity endpoints.

# static gates (.github/workflows/orion-static-gates.yml, every run step): 25/25 PASS,
# incl. check_definition_drift.py --gate, check_metric_lineage.py --gate
python scripts/check_env_template_parity.py -> PASS (94 services)
pyflakes on all new/changed Python -> clean
```

## Evals run

```text
# Live, read-only equivalence + latency (orion_substrate: 5,004 nodes, 37,921 edges)
python services/orion-recall/scripts/compare_concept_region_direct_vs_cache.py \
  --uri redis://127.0.0.1:6380 --queries-file <152 distinct recall_telemetry queries>
  final query shape, graph at 5,041 nodes / 38,206 edges:
  hydrate (old path) 13.1 s, complete=True
  152/152 identical (ordered), mean Jaccard 1.0; 52 matched turns, 3,506/3,506 fragments
  direct latency (median of 3 per query): unmatched median 10.6 ms, p90 16 ms;
  matched median 142 ms, p90 217 ms, max 297 ms
  (first shape, same day: also 152/152 identical; unmatched 11.8 ms, matched 151 ms median)

# Realistic fixture in the throwaway FalkorDB (870 concepts, 37k edges)
  FalkorDB 4.18 (prod version): median no-match 8.3 ms, match 123 ms
  FalkorDB 6.0  (CI image):      median no-match 7.1 ms, match 62 ms

python services/orion-recall/evals/run_recall_bounded_retrieval_eval.py -> runs clean (unaffected)
```

## Docker/build/smoke checks

```text
Not built or deployed (explicit instruction: do not deploy). No new dependencies:
redis-py and the codec are already in the recall image.
```

## Review findings fixed

- Finding (from CI, before review): the equivalence lane failed 8 of 18 on GitHub, which pulls FalkorDB 6.0.1. Locally (4.18.11, the production version) all 18 passed. Reproduced on 6.0.1: for `MATCH (n) WHERE id(n) = $x MATCH (n)-[e]-()`, the 6.0 planner drops the id filter and walks every edge in the graph (2,973 rows instead of 11 for one node), so the edge cut came out wrong.
  - Fix: the edge-cut query recomputes the concept ranking and passes `n` through `WITH ... LIMIT`, so it never seeks by an id parameter. That plan is correct and bounded on both versions.
  - Evidence: 18/18 on 4.18.11 and 18/18 on 6.0.1. Live comparison re-run with the final query: 152/152 identical.
- Code review: not run in this branch. The orchestrator runs it.

## Restart required

After merge, from the primary checkout on main (production deploys from main, never from a worktree):

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-recall up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: turns that match a concept take about 150 ms median (p90 199 ms, one outlier at 631 ms), against the 100 ms target. Almost all of that is FalkorDB walking the ~32k edges that touch the top 500 concepts, to rank them for the old "top 500 edges" cut. Profiled: the traverse alone takes 60-80 ms inside the engine, and there is no index on edge salience.
  - Mitigation: concept_region runs at the same time as active_packet (p50 602 ms), so it is not on the recall's critical path, and it stays under the recall deadline. If the target matters more than exact equivalence, there are two options: rank only the matched concepts' own edges (a few ms, but that changes which edges show up and needs Juniper's approval), or add a Falkor range index on edge salience (a schema change).
- Severity: low
  - Concern: the reader does not serve legacy `payload_json` rows. Hydration decoded them; this reader only sees native rows.
  - Mitigation: the live count is 0 nodes and 0 edges (checked 2026-10-06), and writers rewrite legacy rows on their own hydrate.
- Severity: low (observation, existing behavior kept as-is)
  - Concern: a broad match such as the seed concept "Orion" brings back about 341 edge fragments in one turn ("Hi Orion..." returns 342 fragments). The equivalence requirement preserved this.
  - Mitigation: none in this PR.
- Severity: low
  - Concern: production FalkorDB is configured with `TIMEOUT 1000` (ms), so a read slower than 1 s is killed. The slowest live read measured was 297 ms.
  - Mitigation: a killed read raises, the collector catches it and returns nothing for that turn, and nothing falls back to hydration.
- UNVERIFIED: the change has not run inside the deployed recall container. The latencies above were measured from the host to the published Falkor port, not over the container bridge network.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2505

🤖 Generated with [Claude Code](https://claude.com/claude-code)
