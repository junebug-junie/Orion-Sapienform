# The vision recovery notification must survive a restart

## Summary

- vision-host has emitted **8 `vision_blind` records since 2026-08-21 and zero `vision_recovered`, ever.**
- Recovery fires only on the `alerting -> clear` transition, and `_alerting` was in-memory, so a restart between alert and clear lost it permanently.
- Adds an atomic JSON state file on the volume vision-host already mounts.
- 38 tests. 7 mutations against the real files, all caught — one only after adding coverage the first pass missed.

## Outcome moved

An outage now closes. Before this, every vision incident stayed open forever in the attention store and in anything reading it.

## The diagnosis, and how it was pinned

Three alerts on 2026-08-29 — 20:25, 21:00, 22:13. All three bodies are near-identical:

```text
Orion cannot see. 100% of vision tasks failing (gpu_hard_floor) on athena
for 3m (88 recent tasks).
```

**"for 3m" is exactly `sustain_sec`, on all three.** That is a fresh sustain clock every time. Two facts make restart the only consistent reading:

1. A live process **cannot** re-alert — `if self._alerting or failing_for < self._sustain_sec: return` blocks it.
2. `_alerting` only clears via a genuine recovery (which would have left a record — none exists) or a delivery rollback (`post_attention_request` returns True on 2xx, and the records exist, so delivery succeeded).

So the process kept starting over. `RestartCount=0` on the current container with `StartedAt=2026-08-29T22:59:12` confirms the instance that fired those alerts is gone.

## Current architecture

`VisionLivenessWatcher` (`app/liveness.py`) is a rolling-window failure tracker with hysteresis. `_alerting`, `_failing_since`, `_last_alert_at` were plain instance attributes. No persistence anywhere in the service.

## Architecture touched

One service, one new module. No bus, no schema, no new dependency — vision-host deliberately avoids HTTP libs (it uses `urllib` for its one POST), and this uses only stdlib.

## Files changed

- `services/orion-vision-host/app/liveness_state.py`: new. Atomic load/save, staleness and future-date guards.
- `services/orion-vision-host/app/liveness.py`: `state_store` param, `_restore_state`, `_persist_state`, persist on all three transitions, `_last_now` clock reference.
- `services/orion-vision-host/app/settings.py` + `.env_example`: `VISION_LIVENESS_STATE_PATH`.
- `services/orion-vision-host/tests/test_liveness_state_survives_restart.py`: new, 17 tests.

## Design notes

**Wall clock on disk, monotonic in memory.** `time.monotonic()` is process-relative; persisting it raw restores a cooldown deadline from another process's epoch.

**One clock reference.** `_persist_state` converts against `_last_now` (the value `record()` actually used), not a fresh `time.monotonic()`. Mixing an injected test clock with the real one produced wall timestamps **days** off — harmless in production, which never injects, but it made the persisted value untestable and would have rotted silently. Caught by a test asserting the stored value *is* wall-clock.

**Stale and future-dated files are refused.** >24h old, or dated ahead (a clock step), starts clean — coming back after days down must not resurrect a forgotten incident or park a cooldown permanently ahead.

**Samples are not persisted.** The deque is a rolling `window_sec` view; after a restart it is stale by definition.

**Restoring `alerting=True` is the point.** If the service returns and traffic is healthy, the first records clear it and emit exactly the recovery that was being lost.

**Nothing here may stop the service from seeing.** Missing, corrupt, unwritable, or exploding store all degrade to in-memory with a log line — the policy `build_watcher_or_default` already states. The watcher is constructed at module import, so a raise here would crashloop the service.

## Schema / bus / API changes

None.

## Env/config changes

- Added: `VISION_LIVENESS_STATE_PATH` (default `/mnt/telemetry/orion-vision-host/liveness_state.json`, on the volume already mounted at `docker-compose.yml:83`). Empty string disables persistence and restores prior behaviour.
- `.env_example` updated: yes.
- local `.env` synced: **by hand**, deliberately — `sync_local_env_from_example.py` reads `.env_example` from the *primary* checkout, so a worktree-added key is invisible to it. Verified present at line 62.

## Tests run

```text
pytest test_liveness_state_survives_restart.py test_liveness_alert.py -q --noconftest
  -> 38 passed

Full service suite: 9 collection errors, all `No module named 'torch'`.
Verified IDENTICAL on origin/main (stash / run / pop) -- pre-existing, unrelated.
--noconftest is this service's existing convention for its torch-free tests.
```

### Mutation tests (real files)

| mutation | result |
|---|---|
| restore never called | CAUGHT |
| delivered alert not persisted | CAUGHT |
| persist monotonic raw (no wall conversion) | CAUGHT |
| no stale-state guard | CAUGHT |
| no future-dated guard | CAUGHT |
| clear-threshold branch not persisted | **initially NOT caught** → coverage added → CAUGHT |

That last one is worth naming: a restored watcher has an empty sample deque, so the first success always takes the *below-min_samples* recovery branch and the clear-threshold branch was never reached by any test. Reaching it needs one failure first (keeping `rate != 0`) then successes past the sample floor.

## Evals run

```text
None. services/orion-vision-host has no evals/ directory.
```

## Review findings fixed

_Pending — review dispatched; findings land as a follow-up commit._

## Restart required

```bash
sudo docker compose --env-file .env --env-file services/orion-vision-host/.env \
  -f services/orion-vision-host/docker-compose.yml up -d --build
```

## Risks / concerns

- **Severity: low.** First deploy starts with no state file, so the currently-open incident (8 unclosed alerts) is not retroactively closed. Only gaps from this point forward close correctly.
- **Severity: low.** If the volume is unwritable in some environment, persistence silently degrades to the old in-memory behaviour. Logged, not fatal — deliberate, since the alternative is a crashloop in the seeing path.

## Related

- #1977 corrects the rationale in the capability-gap journal seed, which had assumed a re-arm implied recovery.

## PR link

<!-- filled after gh pr create -->
