# Complete Falkor hydration — reading property graph, patch 2

Implements section 2 of [the reading property graph design](2026-10-06-reading-property-graph-design.md), after #2497. This patch repairs complete reads for existing consumers. It does not implement reading assertions or migrate local consumers.

## Current architecture

- Service: shared `orion/substrate` store, used by substrate-runtime, Hub and Recall.
- Entry points: `FalkorSubstrateStore.__init__`, `snapshot`; routed snapshots retain the primary's receipt.
- Config path: existing Falkor store config/environment builder. `hydration_page_size` is a constructor parameter, default 1,000, bounded 1–10,000. No env key changes.
- Bus channels / schema registry entries: none; no wire event changes. Scan receipts are local dataclasses attached to snapshots.
- Docker compose: consumers use `services/orion-substrate-runtime/docker-compose.yml`, `services/orion-hub/docker-compose.yml`, `services/orion-recall/docker-compose.yml`. No compose changes.
- Tests: existing Falkor/store/dynamics tests plus `test_complete_hydration.py`.
- Evals: `orion.substrate.evals.run_complete_hydration_eval`.
- Gap: four unpaged queries were subject to the deployed result cap; row decode failures were skipped and refresh failures still advanced the success cursor.

## Read and failure contract

Scan native nodes, legacy nodes, native substrate edges and legacy edges using `id(n)` / `id(e)` cursors and ordered bounded pages. Only an empty page ends a scan. Object IDs are temporary database cursors, never durable business identity. The default 1,000-row request continues correctly even when the server returns fewer rows.

Stage the next cache separately. Reject nonadvancing/invalid cursors, malformed rows, unsupported node kinds, invalid models, duplicate stable node/edge IDs, duplicate node identities, incompatible edge identities, and missing/mismatched endpoints. Any failure preserves the prior cache and last successful timestamp/generation. Failed boot remains unready; the next snapshot retries even when the refresh ceiling is disabled.

Existing live edges often share the same lookup `identity_key` while having distinct `edge_id`s. Preserve every edge. Compatible aliases must have identical typed endpoints and predicate; the historical single-edge identity index points deterministically to the smallest edge ID. The receipt counts additional aliases, explicitly disclosing that this index is not a uniqueness constraint. This changes which parallel edge an ambiguous lookup may return; it does not deduplicate edges or change topology. Incompatible aliases fail the scan.

`MaterializedSubstrateGraphState.scan_receipt` is optional for other backends. Falkor returns scan start/end, `complete`, `stale`, scanned node/edge counts, completed page requests, last successful refresh, failure reason and alias count. These are operational diagnostics, not cognition metrics or model inputs. On failure the counts describe the rejected attempt, not the returned old cache. `last_hydrate_node_count` continues to describe the last successful cache. The receipt is available directly as `store.last_scan_receipt` for cold-start diagnostics.

Coverage is **non-atomic keyset**, not snapshot isolation. Local write-generation changes during a scan reject it; changes after the scan invalidate the next snapshot. Detected external topology inconsistencies fail. Undetectable external changes, including object ID reuse below the cursor, remain a documented consistency limit. A strict equivalence job must use quiescent data or an export with snapshot isolation.

Legacy payloads are validated into the staging cache before publication. Existing native rewrite/cleanup remains best-effort after a valid scan; read-only clients skip all rewrite calls. Invalid/duplicate legacy records fail closed rather than clobber a native record. No data migration, deletion or repair is performed by this patch's diagnostic command.

## Acceptance and evidence

- Regression tests cover a 10,007-edge fixture, smaller server caps, terminal empty pages, stale-cache retention, failed-boot retry, duplicate IDs/identities, bad rows, changing topology, same-process writes and read-only legacy handling.
- Dynamics eval compares complete topology and every pressure/activation/dormancy result against an in-memory reference over a quiescent 43-node/173-edge fixture, across three page/cap combinations. It produces nonempty pressure and activation updates.
- Read-only live command: `python -m scripts.replay_substrate_complete_scan --uri redis://localhost:6380`. Explicit client `read_only=True`; no bus connection or graph write.
- Initial complete live scan: 2026-10-06 04:13:35–04:14:00 UTC, 4,973 nodes, 37,630 edges, 47 page queries, 16,269 additional compatible lookup aliases; complete, non-stale, about 25.54 seconds. Local evidence: `/tmp/substrate-complete-hydration/live-scan.json`.
- This proves the new reader against live durable data. New code in deployed service loops remains **UNVERIFIED** until deployed. No deployment is part of this patch.

Falkor documents that its [result cap limits returned records](https://docs.falkordb.com/getting-started/configuration) and that [internal IDs are not immutable](https://docs.falkordb.com/cypher/functions). The live probe confirmed cursor syntax on this deployment.

## Limits and next patch

Complete scans cost more than silently truncated reads: about 25.54 seconds on this live graph with 1,000-row pages. Existing read/write loops invalidate the cache on their next snapshot; this patch does not claim they are efficient. Migrate curiosity to the bounded neighborhood contract and node-only inputs, preserving persisted candidate semantics, next. Full consumer migration, SPARQL complete-read repair, cognitive provenance isolation, retained reading evidence, assertion review/projection and shared provenance surfaces remain subsequent patches.

No env templates, schemas, dependencies, bus settings or new cognition signals change. Roll back the code to revert behavior; durable data needs no reversal. Restart/rebuild the affected reader services from a merged worktree when deploying; no restart was run for this patch.
