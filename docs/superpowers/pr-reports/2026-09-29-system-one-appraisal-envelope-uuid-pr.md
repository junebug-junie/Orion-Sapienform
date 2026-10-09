# fix(substrate-runtime): publish System One frame with a UUID correlation id

## Summary

- Orion's fast "System One" gut-check (a small model scoring how much the current focus pulls toward curiosity, reverie, deliberation, or interruption) was computed and saved every ~40 s, but never made it onto the bus. Every publish attempt crashed on envelope validation and the error was swallowed.
- Cause: the publisher used the frame's own id (`system-one-appraisal-<24 hex>`) as the envelope's correlation id, which must be a UUID.
- Fix: derive a deterministic UUID from the frame id (`uuid5`), so a retry of the same frame keeps the same correlation id. The frame id itself is unchanged and still travels in the payload.
- Regression test builds a real frame through the real producer and publishes it through the real worker method onto a fake bus.

## Outcome moved

`orion:system_one:appraisal` goes from 0 messages ever to ~1.5 per minute. Live before the fix: 40 `substrate_system_one_appraisal_completed` vs 39 `substrate_system_one_appraisal_publish_failed` in 25 minutes on `orion-athena-substrate-runtime`. Every failure was `1 validation error for BaseEnvelope correlation_id Input should be a valid UUID`.

## Current architecture

- `services/orion-substrate-runtime/app/worker.py::_system_one_appraisal_tick` runs `orion.substrate.system_one_appraisal.run_system_one_appraisal`, saves the frame to Postgres, and queues it and a grammar shadow.
- `_publish_pending_system_one_appraisal` publishes the typed frame on `orion:system_one:appraisal`. It has been broken since the channel shipped on 2026-09-23.
- The grammar shadow on `orion:grammar:event` and the persisted frame (used by the same-service curiosity admission gate) were unaffected. Both use string ids, not envelope correlation ids.

## Architecture touched

- `orion-substrate-runtime` publish path only. There is no schema or payload change.

## Files changed

- `orion/schemas/system_one_appraisal.py`: adds `system_one_appraisal_correlation_id(frame_id) -> UUID`, next to the channel/kind constants.
- `services/orion-substrate-runtime/app/worker.py`: uses that function for `correlation_id`.
- `services/orion-substrate-runtime/tests/test_worker_system_one_appraisal.py`: regression test.
- `docs/superpowers/pr-reports/2026-09-29-system-one-appraisal-envelope-uuid-pr.md`: this report.

## Schema / bus / API changes

- Added: none. There is a new helper function, but no new schema.
- Removed: none.
- Renamed: none.
- Behavior changed: `orion:system_one:appraisal` now actually carries frames. The envelope `correlation_id` is `uuid5(NAMESPACE_URL, "orion:system_one:appraisal:<frame_id>")`.
- Compatibility notes: the channel had zero messages before, so no consumer depends on the old (never-valid) shape.

### Who starts receiving frames (checked, not assumed)

- **No code subscribes to this channel by name.** A repo-wide search for `SYSTEM_ONE_APPRAISAL_CHANNEL` / `orion:system_one:appraisal` finds only the producer, the catalog, the registry, and tests. `channels.yaml` lists `orion-substrate-runtime` as a consumer, but the curiosity admission gate reads the persisted frame from the store, not the bus (see the design spec). That catalog entry is aspirational.
- **`orion-bus-mirror`** (live, `MIRROR_PATTERN=orion:*`) will record the traffic:
  - It records every message into its 24 h-retained SQLite log.
  - It adds one new `PUBLISHES` edge (`orion-substrate-runtime -> orion:system_one:appraisal`) to the FalkorDB synaptic graph. Every frame has its own fresh correlation id, so no `CAUSALLY_FOLLOWED_BY` hop is created.
- **Live-behaviour flag:** once that edge passes the cold-start count floor, its inter-arrival gap z-score joins `bus_synaptic_prediction_error()`, the fraction of live bus edges currently anomalous. That value feeds autonomy/field consumers. It is one more edge in the denominator. Its gaps follow the System One LLM call latency, so a slow or unreachable Kev endpoint could briefly mark it anomalous. The effect is small, but it is a real input change.
- `orion-bus-tap` (also `orion:*`) is not running on this host.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
# regression test, pre-fix (main code): FAILED "typed frame was not published"
#   underlying: ValidationError ... BaseEnvelope ... Input should be a valid UUID,
#   input_value='system-one-appraisal-6ff4c412766207fa561d1d1e'
# post-fix:
cd services/orion-substrate-runtime && PYTHONPATH=<worktree> .venv/bin/python -m pytest tests/test_worker_system_one_appraisal.py -q
5 passed

# full service suite (excluding Postgres-only integration file), before vs after fix:
# identical pre-existing FAILED/ERROR set (env/DB-dependent), no new failures
pytest tests/test_system_one_appraisal.py tests/test_system_one_appraisal_bus_catalog.py -q
9 passed
```

## Evals run

```text
No new eval: this is a transport bug fix and does not change what the frame means.
The existing post-live calibration script (scripts/smoke_system_one_appraisal.py)
still applies unchanged.
```

## Docker/build/smoke checks

```text
Not deployed from this branch (production deploys run from the primary checkout on main).
Pre-fix channel baseline: `timeout 90 redis-cli -u $ORION_BUS_URL SUBSCRIBE orion:system_one:appraisal`
  -> subscribe confirmed, 0 frames in 90 s (while ~2 frames completed and failed publish).
Live evidence of the bug (pre-fix): 40 completed / 39 publish_failed in 25 min.
Post-deploy verification: UNVERIFIED until deployed. See the commands below.
```

## Review findings fixed

Code review ran in a subagent against commit `eacabceb0`: no must-fix findings. It verified the fix, that the test fails on the pre-fix code, and that no other `BaseEnvelope` site in orion-substrate-runtime passes a non-UUID correlation id (`worker.py` `substrate-c-tick:<hex>` goes into a str field on an embodiment intent, not an envelope).

- Finding (nit): the test used try/except/else to assert frame_id is not a UUID.
  - Fix: switched to `pytest.raises(ValueError)`.
  - Evidence: `tests/test_worker_system_one_appraisal.py` 5 passed.
- Finding (should, deferred): the 4 grammar-shadow events for a frame each get a random `uuid4()` correlation id (`orion/grammar/publish.py`), so the frame and its shadow are not linked as one chain.
  - Fix: not in this PR. The grammar path already works, and changing it alters `orion:grammar:event` ledger correlation ids. Follow-up: set `correlation_id=system_one_appraisal_correlation_id(frame_id)` on those events.
  - Evidence: n/a (deferred).
- Finding (nit, deferred): the publish `except Exception` swallow is what hid this bug for 6 days. A publish-failure counter on the health surface would expose a permanently failing publish.
  - Fix: not in this PR (follow-up).

## Restart required

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Post-deploy check:

```bash
docker logs --since 10m orion-athena-substrate-runtime 2>&1 | grep -c "validation error for BaseEnvelope"   # expect 0
docker logs --since 10m orion-athena-substrate-runtime 2>&1 | grep -c substrate_system_one_appraisal_publish_failed  # expect 0
# channel actually receives messages (expect ~1 per 40 s):
(timeout 90 redis-cli -u redis://100.92.216.81:6379/0 SUBSCRIBE orion:system_one:appraisal > /tmp/s1.txt; true); grep -c system_one.appraisal.frame.v1 /tmp/s1.txt
```

## Risks / concerns

- Severity: low
  - Concern: a new live edge enters `bus_synaptic_prediction_error()`'s population (see above).
  - Mitigation: it is one edge among many, it must pass the existing cold-start floor first, and it can be reverted by reverting this commit.
- Severity: low
  - Concern: `channels.yaml` names `orion-substrate-runtime` as a consumer that does not subscribe.
  - Mitigation: none in this PR. This is documentation drift, not a runtime fault.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2406

🤖 Generated with [Claude Code](https://claude.com/claude-code)
