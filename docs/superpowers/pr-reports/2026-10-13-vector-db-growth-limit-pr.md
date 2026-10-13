## Summary

- Orion's vector database (Chroma, `orion-athena-vector-db`, data at `/mnt/graphdb/chroma`) now has a size gauge and a growth alarm. Every 5 minutes orion-biometrics measures the folder size and the row count of every collection and puts it on its heartbeat (`details.vector_db`).
- Alarm: when the folder reaches 4 GiB, `over_alert` flips true and one `vector_db_over_alert` WARNING is logged per crossing.
- One table now records, for each collection, who writes it, whether it can be rebuilt, whether recall reads it, and why it is not pruned (`orion/memory/vector_collection_policy.py`). A collection with no entry shows up in `unpoliced_collections` and in a WARNING log, so a new writer can't go unnoticed.
- **Nothing is deleted.** No collection qualifies for a retention limit. The recall-backing ones are protected, and the derived ones are upserts keyed by their source (one row per letter part, per concept, per doc), so their size follows the source, not the clock. A test enforces that rule for future policies.
- Finding, not fixed here: about 40% of the folder is Chroma 0.4.24's write-ahead log (`embeddings_queue`, 79 MB, 13,093 rows since 10-11). That version never cleans it up. Chroma 0.5.6+ does it automatically. Deleting those rows by hand while the server is running could lose vectors that haven't been flushed to disk yet, so that is left as a follow-up.

## Outcome moved

Before this patch, nobody could see how big the vector DB was or which collection was growing. Now that is visible live on the bus, and there is an alarm on the folder size.

### Growth measured 2026-10-13 (live, data dir since 2026-10-11 04:54 UTC)

| collection | rows | rows/day | class |
| --- | --- | --- | --- |
| orion_main_store | 9,174 | ~3,000-3,900 (10-11: 2,240, 10-12: 3,890) | primary, recall |
| orion_atlas_concepts | 1,018 | one-off index pass 10-13 (upsert per concept) | derived |
| orion_memory_crystallizations | 762 | 0 since 10-11 | derived, recall |
| orion_day_letter_parts | 451 | ~20 | derived index, reread |
| orion_curiosity | 444 | ~15 | primary, recall |
| orion_dreams | 236 | ~4 | primary, recall |
| orion_reading_results | 58 | ~4 | derived index |
| doc_semantic_drift | 36 | ~7 | derived |
| orion_chat | 21 | ~4 | primary, recall |

`orion_main_store` accounts for about 99% of the row growth. vector-host stores every text that llm-gateway sends for embedding there (`role=embedding_request`, 9,013 of 9,034 rows have `requester_service=llm-gateway`). 7,647 of the 9,034 texts are distinct.

Disk: the folder is 206 MB now (sqlite file 136 MB, of which `embeddings_queue` is 79 MB, plus ~70 MB of HNSW vector-index files). It got there in ~2.6 days, about 80 MB/day, and that figure includes the 10-11 re-population. The graphdb drive is 825 GB with 782 GB free, so at 80 MB/day it fills in about 27 years, and even at 10x that rate it still takes about 2.7 years. The 4 GiB alarm is about 7 weeks away at the current rate. It is there to force the Chroma upgrade / log-cleanup decision, not to warn that the drive is full.

## Current architecture

orion-biometrics already published disk percent-full per mount (`details.disk_usage_pct.graphdb`) through a read-only `/host_mnt/graphdb` mount. The vector DB had no per-folder size, no per-collection counts, and no written-down retention rule. No code anywhere deletes from Chroma.

## Architecture touched

- orion-biometrics: new background poller (same pattern as `IloPoller`). The heartbeat only reads its cache, so the walk and HTTP calls stay off the heartbeat's time budget. Nodes without the Chroma folder (atlas, circe) report `error=not_present` and make no HTTP calls.
- Shared policy module `orion/memory/vector_collection_policy.py` (already in the image via `COPY orion /app/orion`).
- Reads use Chroma's documented HTTP API, not its sqlite file.

## Files changed

- `orion/memory/vector_collection_policy.py`: per-collection policy and the rule for when retention is allowed.
- `services/orion-biometrics/app/vector_db_footprint.py`: size/count gauge, alarm, poller.
- `services/orion-biometrics/app/main.py`: poller start/stop, `_heartbeat_details` merge.
- `services/orion-biometrics/app/settings.py`, `.env_example`, `docker-compose.yml`: six `VECTOR_DB_*` keys, shipped ON.
- `services/orion-biometrics/README.md`: `details.vector_db` contract.
- `services/orion-biometrics/tests/test_vector_db_footprint.py`: gauge tests against a real local HTTP server.
- `tests/test_vector_collection_policy.py`: no-silent-memory-loss gate.
- `.github/workflows/vector-db-footprint-tests.yml`: CI lane for both (biometrics had none).

## Schema / bus / API changes

- Added: `SystemHealthV1.details.vector_db` (free-form `details` dict, same convention as `disk_usage_pct`). No new channel or schema.
- Removed / Renamed: none.
- Behavior changed: none for existing keys.
- Compatibility notes: additive key in an existing free-form dict.

## Env/config changes

- Added keys (orion-biometrics): `VECTOR_DB_FOOTPRINT_ENABLED=true`, `VECTOR_DB_DATA_PATH=/host_mnt/graphdb/chroma`, `VECTOR_DB_URL=http://orion-athena-vector-db:8000`, `VECTOR_DB_ALERT_BYTES=4294967296`, `VECTOR_DB_POLL_INTERVAL_SEC=300.0`, `VECTOR_DB_REQUEST_TIMEOUT_SEC=5.0`
- Removed / renamed: none
- `.env_example` updated: yes
- local `.env` synced: yes. The default run skips orion-biometrics (it isn't in the script's default service list), so I ran `python scripts/sync_local_env_from_example.py --all-keys orion-biometrics`. It added exactly these 6 keys (`services/orion-biometrics/.env` lines 103-108) and nothing else.
- skipped keys requiring operator action: none

## Metric quality gate

1. Provenance: `dir_size_bytes()` is a `scandir` walk over the persist folder. Counts come from Chroma's `/api/v1/collections/{id}/count`.
2. Independence: `disk_usage_pct.graphdb` is the whole drive's fill level. This is one folder on that drive, so at 206 MB of 825 GB it adds information the drive figure can't show. It is not fed into any pressure channel or cognition loop.
3. Theory anchor: bytes on disk are the resource that actually runs out, and row counts tell you which writer is causing the growth.
4. Live sanity: live smoke below. Counts moved between samples (main_store 9,119 -> 9,174 within minutes). Its "rest" state is a flat size, which is a real reachable state.
5. Existing mechanism: reused biometrics' heartbeat and its `/host_mnt/graphdb` mount. Nothing else measured this.
6. Reversibility: one key in a free-form dict, one flag (`VECTOR_DB_FOOTPRINT_ENABLED=false`).

## Tests run

```text
services/orion-biometrics: pytest tests/test_vector_db_footprint.py -> 13 passed
pytest tests/test_vector_collection_policy.py -> 5 passed
services/orion-biometrics: pytest tests -> 181 passed, 2 failed (test_node_catalog::test_circe_expected_offline
  and test_biometrics_grammar_emit::test_circe_node_availability_reflects_expected_offline: they fail
  identically on main; config/biometrics/node_catalog.yaml is untouched here)
scripts/check_env_template_parity.py -> PASS (93 services)
scripts/check_env_key_single_source.py -> OK
```

## Evals run

```text
No eval harness for orion-biometrics. The live smoke below is the behavioral check: the gauge
run against the real Chroma folder and API.
```

## Docker/build/smoke checks

```text
sample_footprint('/mnt/graphdb/chroma', 'http://localhost:8500', 4 GiB):
  dir_bytes=206063210 sqlite_bytes=135966720 over_alert=false unpoliced_collections=[]
  collection_counts: main_store 9174, atlas_concepts 1018, crystallizations 762, day_letter_parts 451,
  curiosity 444, dreams 236, reading_results 58, doc_semantic_drift 36, chat 21
In the running orion-athena-biometrics container: `getent hosts orion-athena-vector-db` resolves and
/host_mnt/graphdb/chroma is readable, so the deployed path will reach both.
Not deployed. Container not rebuilt (per task). UNVERIFIED: a real athena heartbeat carrying `details.vector_db` has not been observed yet. Check after restart with `docker logs orion-athena-biometrics | grep vector_db` or by reading one `orion:system:health` message.
```

## Review findings fixed

Review by a code-review subagent on commit 9046b858c. Nothing blocking; all three "should" findings and most nits fixed:

- Finding: if a segment folder was deleted between being queued and being scanned, the whole size sample failed.
  - Fix: `os.scandir` of a vanished dir is now skipped.
  - Evidence: `test_dir_removed_mid_walk_is_skipped_not_fatal`.
- Finding: before the first poll, the heartbeat carried all-null fields that looked like a real reading.
  - Fix: it now carries `error: not_sampled_yet` until then.
  - Evidence: `test_before_first_poll_says_not_sampled`.
- Finding: the report did not say that nothing consumes the gauge, and the live heartbeat path was unverified.
  - Fix: both are now stated (Risks, Docker/smoke, README).
- Nit: the unpoliced-collection WARNING repeated every 5 minutes.
  - Fix: it now logs only when the set of unpoliced collections changes.
  - Evidence: `test_unpoliced_logged_only_when_set_changes`.
- Nit: these cases had no tests: setting the alarm to 0, a failed sample keeping the alarm state, start/poll/stop, and the API-down test depending on port 9.
  - Fix: `.env_example` now documents that 0 disables the alarm. Added tests `test_failed_size_sample_keeps_previous_alert_state` and `test_lifecycle_start_poll_stop`. The API-down test now uses a port taken from a closed socket.
- Nit: the README's "derived ones are bounded" claim was wrong for two collections, the 40% WAL figure had no cited source, a point-in-time "flat since 10-11" was baked into code, and the policy test didn't say it only checks a snapshot.
  - Fix: the README is reworded and cites the measurement (sqlite `dbstat` on a copy of the file). The date-stamped wording is gone from the code. The policy test's docstring now names the runtime `unpoliced_collections` as the live guard.
- Not changed: the poller's `stop()` can't interrupt a blocked HTTP thread. Same as `IloPoller`. The worst case is a slower container shutdown.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-biometrics up -d --build
```

## Risks / concerns

- Severity: low. Concern: nothing consumes `details.vector_db` yet. The 4 GiB alarm is a log line plus a flag, so nothing pages and no reducer or UI reads it. Mitigation: the data is on the bus, ready for a consumer to pick up.

- Severity: medium. Concern: Chroma's write-ahead log is never pruned on 0.4.24 and is the biggest single consumer of space. Mitigation: the alarm fires well before it matters. Follow-up: upgrade Chroma to >=0.5.6, which prunes it automatically, as its own migration PR.
- Severity: low. Concern: `orion_main_store` stores every embedding request, including duplicates (7,647 distinct of 9,034). Mitigation: monitored only. Deduplicating or narrowing what vector-host stores is a writer-side change that needs Juniper's call.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2630

🤖 Generated with [Claude Code](https://claude.com/claude-code)
