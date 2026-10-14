## Summary

- Each topic-foundry run was re-creating concepts and entities that already existed in the AI Town and Self Atlas graphs. The concept graph only checked its in-memory copy before creating a node, and that copy was often incomplete. It now asks the database when the copy has no match (`FalkorSubstrateStore.get_node_id_by_identity`, plus a new `SubstrateNode(identity_key)` index).
- Loading the graph on startup no longer fails outright on the first duplicate. It keeps every node, points the shared name at one of them (the smallest `node_id`), and logs and counts the duplicates.
- Added a dry-run repair that merges the duplicates (`scripts/repair_substrate_duplicate_identities.py`). Its snapshot and report are written under `/tmp/substrate-dedupe-identity/`. Nothing has been applied.
- Hub startup no longer re-runs schema setup that is already in place. The setup used to wait on the nightly backup's table locks and then fail (`orion/core/storage/schema_fingerprint.py`).
- Hub startup now runs the memory-cards and crystallization schema files independently and retries a failed one in the background. Before, one failure skipped the other file silently.

## Outcome moved

- **Duplicates per name.** Replaying 6 topic-foundry runs × 40 labels, each run starting from an empty cache: the old code ends with 240 nodes (6 copies of every label), the new code with 40 (`run_identity_convergence_eval`).
- **Graph load on startup.** It used to fail on every refresh for `orion_substrate_self` and `orion_substrate_aitown` (30 journald warnings over 10-12 to 10-14). With the fix it succeeds and reports the duplicate count.
- **Schema setup during a backup.** Tested against a real Postgres with a pg_dump-style ACCESS SHARE lock held: the old raw DDL failed with `LockNotAvailable` after 2.0s. The new apply skips in 0.01s.

## Current architecture

- `SubstrateGraphMaterializer.apply_record` resolves each incoming node's identity (`SubstrateIdentityResolver.canonical_node_key`, e.g. `concept|world||label:<label>`) with `store.get_node_id_by_identity`. A miss means create.
- Topic-foundry mints run-scoped node_ids (`sub-concept-topicfoundry-<run_uuid>-<topic_id>`). Only identity resolution stops a new run from duplicating an old one.
- `FalkorSubstrateStore.get_node_id_by_identity` read only the in-process cache. Durable writes MERGE on `node_id`, and nothing in the graph enforced one node per identity.
- `_hydrate_from_durable` raised on a duplicate identity. A failure keeps the last good cache, and at boot that cache is empty.

## Root cause (live evidence, read-only)

- **Which nodes collide.** FalkorDB, grouped by `identity_key` and checked on 2026-10-14:
  - `orion_substrate_aitown`: 1,237 identities on 14,022 nodes (43,873 nodes total; up to 94 copies of one concept).
  - `orion_substrate_self`: 1,109 identities on 4,640 nodes (16,988 total).
  - `orion_substrate`: 0 duplicates (5,921 nodes).
  - Every duplicate has `provenance_producer=topic_foundry_adapter`.
- **How the cache went incomplete, twice.**
  - Before PR 2026-10-06 (`fb9e03ea0`), hydrate issued one unpaged query. FalkorDB's `RESULTSET_SIZE` is 10000 on this host, so a graph over 10k nodes loaded partially. AI Town passed about 10.8k nodes around 09-14, which is when its duplicates start. Self had about 10.4k nodes when its duplicates start on 10-06.
  - After paging fixed that, the first duplicate made every hydrate fail. The hub then booted with an empty cache, and every new run after a restart duplicated every label.
- **Ruled out.**
  - PR #2621 (ON CREATE-only pressure/dormancy) left the MERGE key on `node_id`, and duplicates predate it by four weeks.
  - PR #2649 (person-model) is dry-run only and wrote no substrate nodes: the producer census lists no person-model producer.
  - Reading concept minting and concept induction write to `orion_substrate`, which has no duplicates.
- **What "accelerating" means.** The warning fires once per ingest tick per affected graph, and ticks were about every 10 minutes on 10-14. So the rate tracks ingest ticks and hub restarts. The duplicate count does not grow in step with the warnings.
- **Schema lock.** The nightly `orion-backup-databases.service` ran 03:45:00 to 04:41:38 UTC on 10-14. Three of the four lock timeouts (03:45:42, 04:20:39, 04:30:18) fall inside that window. The 00:29:17 one was on `memory_crystallizations.sql`, and its lock holder is UNVERIFIED (`log_lock_waits` is off).

## Blast radius

- Only `orion-athena-hub` logs `falkor_substrate_hydrate_failed`. A 72h docker-logs sweep of every running container found it nowhere else.
- No other service reads the self or AI Town graphs. Recall, memory-consolidation, graph-compression and substrate-runtime read `orion_substrate`, which is clean.
- Inside the hub, the Concept Atlas graph switcher for `aitown` and `self` (`concept_atlas_routes._substrate_store_for_graph`) served a cache holding only what this process had written since boot. That is the latest run's few hundred nodes, out of 17k and 44k.
- The schema failure had no data effect: the schema was already current. But `memory_crystallizations.sql` was skipped on those boots.

## Architecture touched

- `orion/substrate/falkor_store.py`, `orion/substrate/store.py`: identity resolution, hydrate, index set.
- `orion/core/storage/`, `orion/memory/crystallization/repository.py`: schema apply.
- `services/orion-hub/scripts/main.py`: startup schema bootstrap.
- New FalkorDB index `SubstrateNode(identity_key)`. Every reader bootstrap creates it on all substrate graphs. Rollback: `DROP INDEX ON :SubstrateNode(identity_key)`.
- New Postgres table `orion_schema_fingerprints(name, sha256, applied_at)`, created on the first successful apply.

## Files changed

- `orion/substrate/falkor_store.py`: durable identity fallback (`NODE_BY_IDENTITY_CYPHER`), hydrate degrades per node, `identity_key` index, counters.
- `orion/substrate/store.py`: `CompleteScanReceipt.duplicate_node_identities`; `InMemorySubstrateGraphStore.set_identity_canonical`.
- `orion/substrate/tests/test_falkor_identity_dedupe.py`: regressions (hydrate degrade, durable fallback, materializer merge, index).
- `orion/substrate/tests/test_complete_hydration.py`: a duplicate identity is no longer a reason to reject the scan.
- `orion/substrate/tests/test_falkor_direct.py`: index bootstrap now creates two indexes.
- `orion/substrate/evals/run_identity_convergence_eval.py`, `complete_hydration_fixture.py`: convergence eval.
- `scripts/repair_substrate_duplicate_identities.py`, `tests/test_repair_substrate_duplicate_identities.py`: dry-run repair. The tests cover the plan rules and a real-FalkorDB apply on a throwaway graph.
- `orion/core/storage/schema_fingerprint.py`, `memory_cards.py`, `orion/memory/crystallization/repository.py`, `tests/test_schema_apply_lock_timeout.py`: fingerprint skip.
- `services/orion-hub/scripts/memory_schema_bootstrap.py`, `services/orion-hub/tests/test_memory_schema_bootstrap.py`, `services/orion-hub/scripts/main.py`: per-schema apply, background retry, cancelled on shutdown.
- `.github/workflows/substrate-neighborhood.yml`, `.github/workflows/memory-schema-bootstrap.yml`: CI lanes.

## Schema / bus / API changes

- Added:
  - FalkorDB index `SubstrateNode(identity_key)`.
  - Postgres table `orion_schema_fingerprints`.
  - Receipt field `duplicate_node_identities`.
  - Log lines `falkor_substrate_hydrate_duplicate_identities`, `falkor_substrate_identity_lookup_failed`, `schema_apply_skipped`, `memory_schema_ready`, `memory_schema_apply_failed`, `memory_schema_apply_gave_up`.
- Removed: log lines `memory_cards_schema_applied`, `memory_cards_schema_apply_failed` and `memory_crystallizations_schema_applied` (the last one is hub only; the crystallizer keeps its own).
- Behavior changed:
  - A cache miss in `FalkorSubstrateStore.get_node_id_by_identity` now costs one indexed Falkor read.
  - Hydrate succeeds when identities are duplicated.
  - `apply_memory_cards_schema` and `apply_memory_crystallizations_schema` return a bool and accept `force=`.
- Compatibility: no bus or schema-registry change.

## Env/config changes

- None. No `.env_example` changed, so no `.env` sync was needed.

## Tests run

```text
orion/substrate/tests + schema + repair tests: 1141 passed, 53 skipped, 3 failed
  (the 3 are test_felt_state_self_definition_lane.py; the same 3 fail on main)
orion/substrate/tests/test_falkor_identity_dedupe.py on the pre-fix falkor_store.py: 6 failed (reproduces)
tests/test_schema_apply_lock_timeout.py against a throwaway postgres:15: 10 passed
tests/test_repair_substrate_duplicate_identities.py against throwaway falkordb 6.0.1 and v4.18.11: 8 passed each
services/orion-hub: tests/test_memory_schema_bootstrap.py 4 passed
services/orion-hub: concept atlas + topic foundry scheduler suites: 152 passed
services/orion-hub: startup-adjacent suites: 323 passed, 9 failed
  (all 9 are test_memory_consolidation_draft_routes.py / test_memory_graph_consolidation_draft_approve.py; they fail identically on main)
```

## Evals run

```text
python -m orion.substrate.evals.run_identity_convergence_eval
  -> 40 labels, 6 runs, 40 durable nodes, max 1 copy (pre-fix code: 240 nodes, max 6)
python -m orion.substrate.evals.run_complete_hydration_eval -> passed
```

## Docker/build/smoke checks

```text
Throwaway falkordb:6.0.1, real FalkorSubstrateStore:
  graph seeded with 2 duplicates -> hydrate_ok=True, duplicate_node_identities=1,
  canonical=...run1-53; empty cache + new run3 node -> created=0 merged=1;
  nodes after = [run1-53, run2-53]; indexes = [node_id, identity_key]
Throwaway postgres:15 holding ACCESS SHARE on memory_cards (pg_dump shape):
  old raw DDL -> LockNotAvailable after 2.0s; new apply -> skipped in 0.01s
Live (read-only) repair dry run: /tmp/substrate-dedupe-identity/report.md
No hub image build was run: the change is Python-only under orion/ and services/orion-hub/scripts/.
```

## Repair plan (dry run, awaiting approval, NOT applied)

Rule:
- The keeper is the smallest `node_id` in each group.
- Edges on duplicates are re-pointed to the keeper. A re-pointed edge that becomes a self-loop is dropped. One that matches an edge the keeper already has (same type and endpoints) is also dropped, and the keeper's edge stays as it is.
- Keeper properties win, except `evidence_refs_json` (union of all copies) and `observed_at` (latest).
- The duplicates are then DETACH DELETEd.

| graph | identities | nodes deleted | edges re-pointed | dropped as parallel | dropped as self-loop | snapshot |
|---|---|---|---|---|---|---|
| orion_substrate_self | 1,109 | 3,531 (1,778 concept, 1,753 entity) | 9,767 | 14,877 | 47 | written, 29,331 rows |
| orion_substrate_aitown | 1,237 | 12,785 (10,733 concept, 2,052 entity) | 19,310 | 68,577 | 2 | **not written: 101,911 rows, over the 100k section-14 limit** |

Juniper decides:
- Approve self.
- For AI Town, choose one:
  - (a) allow an oversized snapshot;
  - (b) use the nightly FalkorDB RDB backup at `/mnt/storage-warm/backups/athena/db/falkordb/snapshots/` as the snapshot;
  - (c) drop and rebuild AI Town's graph from topic-foundry. It is interpretability-only.

Re-run the dry run right before applying.

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`. No blockers.

- Finding: a write to a larger-id duplicate moved the cached identity onto that copy, so new runs could merge into whichever copy was written last.
  - Fix: `FalkorSubstrateStore._cache_upsert_keep_canonical` keeps the smallest node_id on every cache write.
  - Evidence: `test_a_write_to_the_larger_duplicate_does_not_move_the_identity`.
- Finding: a durable-lookup hit cached during an in-flight hydrate was lost at the cache swap, so the materializer could still create a duplicate.
  - Fix: the hit is journaled when a scan is in flight.
  - Evidence: `test_durable_hit_during_an_inflight_scan_survives_the_swap`.
- Finding: the repair dropped "parallel" edges by (type, source, target). That could delete a semantic-projection edge sitting next to a legacy one.
  - Fix: dedupe on the mapped edge identity, falling back to (type, source, target, role, assertion).
  - Evidence: `test_a_projection_edge_is_not_dropped_as_parallel_to_a_legacy_edge`.
- Finding: `--apply` deleted duplicates even after a rewire error, which would leave the snapshot as the only copy of an edge.
  - Fix: deletes are aborted when any error occurred, the script exits 4, and the report records the result.
  - Evidence: `test_apply_aborts_before_deleting_when_a_rewire_failed`.
- Finding: the rewire created the new edge and deleted the old one in separate steps. In between, two relationships shared one edge_id, which hydrate rejects.
  - Fix: one query now does CREATE and DELETE of the old edge together.
  - Evidence: the live apply test passes on throwaway FalkorDB 6.0.1 and v4.18.11 (production's engine).
- Finding: the edge snapshot only covered SubstrateNode neighbours. The edge type was interpolated without escaping.
  - Fix: the snapshot covers edges of every label, and the type is backtick-escaped.
- Finding: the new counters had no consumer.
  - Fix: they are now in the `falkor_substrate_hydrate_stats` log line.
- Finding: the run order matters. A process that still caches a deleted duplicate re-creates it as a blank node.
  - Fix: `report.md` states the order (deploy the fix and restart, stop the hub, dry-run, apply, start).
- Accepted, not fixed:
  - The durable lookup ignores legacy `payload_json` rows. The live census shows 0 such rows on every substrate graph.
  - Parallel co_occurs_with edges keep only the keeper's properties, with no weight merge. This is disclosed in the report.
  - The fingerprint skip does not re-create objects after a database restore or a manual DROP. Use `force=True` (noted in the hub README).

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

The `identity_key` index is created by every substrate reader as it bootstraps. orion-hub creates it for all three graphs on boot; no other service needs a restart for this fix.

## Risks / concerns

- Severity: low. Two processes ingesting the same new identity at the same moment can still both create it, because the MERGE key is `node_id`, not identity. Today only the hub writes the self and AI Town graphs, so this cannot happen there.
- Severity: low. The fingerprint skip will not re-create a schema object dropped by hand while the .sql file is unchanged. `force=True` re-applies it.
- Severity: medium. The existing duplicates stay until the repair is approved and applied. After this deploy they stop growing. Startup loading works with them, and names resolve to one canonical node.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
