# fix(bus-mirror): stop the 20-minute crash loop

## Summary

- The bus mirror had restarted **581 times** (about every 20 min) since around 2026-09-24. Redis was disconnecting it for falling behind (`client_output_buffer_limit_disconnections:581`, the same count as the restarts).
- **Root cause:** every message MERGEs into the FalkorDB bus-synapse graph, and that graph had 118,112 Channel nodes and no index, so each MERGE did a full scan (about 45 ms). The mirror could handle about 19 msgs/s against a bus rate of about 77 msgs/s.
- **Why the graph got that big:** the running image dated from 2026-08-30 and predated the GPU pool's `orion:gpu_pool:reply:*` / `orion:gpu_pool:state:reply:*` catalog wildcards, so every lease reply became a new node.
- **Fix:** the writer now creates indexes on the MERGE keys (`Channel.channel`, `Organ.organ_id`, `Verb.verb_name`) at startup. SQLite moves to WAL with `synchronous=NORMAL`, which ends the per-message fsync and stops outside readers from crashing the mirror. FalkorDB socket timeouts are now bounded.
- **Retention fix:** the loop slept before its first prune, so a process that never lived 3600 s never pruned. The file reached 30.5 GB with rows dating back 8 days. It now prunes first, in rowid-ordered batches, with a one-row stop test.
- **Cleanup script fix:** `stale_channel_node_cleanup.py` read only 10,000 of 118,112 nodes, because FalkorDB's `RESULTSET_SIZE` silently truncates results. It now pages.

## Outcome moved

- Graph MERGE lookup went from 45 ms to 0.03 ms, by profile (`Node By Label Scan` became `Node By Index Scan`).
- Mirror throughput went from about 19 to about 77+ msgs/s.

## Current architecture

`mirror_bus()` psubscribes `orion:*` and handles each message serially: SQLite INSERT+commit, then FalkorDB read and write per edge. Redis enforces `client-output-buffer-limit pubsub 64mb 32mb 120`.

## Files changed

- `services/orion-bus-mirror/app/graph_writer.py`: `INDEXED_PROPERTIES` and `ensure_indexes()`.
- `services/orion-bus-mirror/app/main.py`: WAL pragmas, batched prune with an early stop, prune-first loop, bounded Falkor client, `ensure_indexes()` at build.
- `services/orion-bus-mirror/scripts/stale_channel_node_cleanup.py`: paged fetch.
- Tests: `test_graph_writer.py`, `test_sqlite_retention.py`, `test_stale_channel_node_cleanup.py`.

## Schema / bus / API changes

None.

## Env/config changes

None. No `.env_example` change.

## Tests run

```text
cd services/orion-bus-mirror && PYTHONPATH=<worktree>:. .venv/bin/python -m pytest tests -q
96 passed
```

New regression tests:

- Every MERGE key in the writer must be in `INDEXED_PROPERTIES`.
- The prune stop must not issue a `WHERE timestamp <` scan.
- Pruning happens on start, before the first interval.
- The fetch pages past the result cap.

## Evals run

None. The service has no eval harness. This is a throughput and stability fix, and the live checks below are the evidence.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-bus-mirror up -d --build   -> started 03:30:30Z, restarts=0
container: ensure_indexes present, channels.yaml has orion:gpu_pool:reply:* (catalog_channels=332, was 276)
bus_mirror.sqlite-wal present (WAL active)
```

## Live operations done (2026-10-02)

- Created FalkorDB indexes `Channel(channel)` and `Organ(organ_id)` on `orion_bus_synapse` by hand before deploy. Both are OPERATIONAL, and the MERGE plan uses an index scan.

## Review findings fixed

- **Finding:** the final prune batch did a full table scan, because there is no timestamp index, on the connection the inserts share.
  - **Fix:** a one-row oldest-timestamp stop test.
  - **Evidence:** `test_does_not_scan_recent_rows_once_old_ones_are_gone`.
- **Finding:** `ensure_indexes()` could block startup forever because the Falkor client had no socket timeout.
  - **Fix:** `socket_timeout=10`, `socket_connect_timeout=5`. Writes are fail-open.
- **Finding:** cleanup paging is unstable under concurrent writes.
  - **Fix:** documented. A missed name is caught on the next run.
- **Finding:** the cancelled test task was never awaited.
  - **Fix:** it is now awaited.

## Restart required

Already deployed from this worktree. After merge, redeploy from main so compose is not pinned to the worktree:

```bash
scripts/safe_docker_build.sh orion-bus-mirror up -d --build
```

## Risks / concerns

- **Medium: FalkorDB has not persisted to disk since 2026-09-30.**
  - Its BGSAVE child (`redis-rdb-bgsave`, container pid 11301) has been deadlocked on a futex for about 38 h.
  - A FalkorDB restart now would lose about 38 h of every graph.
  - **Mitigation:** an operator kills that child and runs `BGSAVE`. The auto-mode guard blocked this from the session.
- **Low:** 117,874 stale per-request Channel nodes are still in the graph. They are harmless now that lookups are indexed. Remove them with the cleanup script `--execute` after a fresh RDB exists. The guard blocked this too.
- **Low:** the SQLite file stays about 30 GB on disk until a one-off `VACUUM`. Freed pages are reused, so it won't grow further.
- **Low:** `synchronous=NORMAL` can lose the last few rows of the 24 h mirror log on power loss.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2468
