# The vision recovery notification must survive a restart

## Summary

- vision-host has emitted **8 `vision_blind` records since 2026-08-21 and zero `vision_recovered`, ever.**
- Recovery fires only on the `alerting -> clear` transition, and `_alerting` was in-memory, so a restart between alert and clear lost it permanently.
- Adds an atomic JSON state file on the volume vision-host already mounts.
- 50 tests. 11 mutations against the real files, all caught.
- Review returned one **HIGH regression** in the first cut; reproduced, fixed, and pinned by a test.

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
- `services/orion-vision-host/tests/test_liveness_state_survives_restart.py`: new, 29 tests.
- `services/orion-vision-host/docker-compose.yml`: the new key in the explicit `environment:` list, matching its seven siblings.

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
  -> 50 passed

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

## Docker/build/smoke checks

Live, inside the running `orion-athena-vision-host` container, on the real volume:

```text
state path: /mnt/telemetry/orion-vision-host/liveness_state.smoke.json
save -> True
load -> alerting: True | failing_since_wall set: True
on-disk: {'version': 1, 'alerting': True, 'failing_since_wall': 1788054150.895,
          'last_alert_at_wall': 1788054150.895, 'saved_at_wall': 1788054150.896}
cleaned up: True
```

Rendered compose config resolves the key:

```text
VISION_LIVENESS_STATE_PATH: /mnt/telemetry/orion-vision-host/liveness_state.json
```

(`docker compose config` cannot run from a worktree unaided — the compose file's own `env_file:` is a relative path and the gitignored `.env` lives only in the primary checkout. Verified via temporary symlinks, removed afterwards.)

Not deployed. Bringing vision-host down is Juniper's call, not a smoke I run unasked.

## Evals run

```text
None. services/orion-vision-host has no evals/ directory.
```

## Review findings fixed

Review found a **HIGH regression in my own fix**, plus a second live cause of the bug I was fixing. Both reproduced before fixing.

- **HIGH — the patch made a restart during a partial outage strictly worse than main.**
  A restored watcher has an empty deque, so the first `record(ok=True)` hit the thin-sample clear branch and declared recovery **off one sample** — posting *"Vision is working again ... 100% of recent tasks succeeding"* while the GPU was still broken. The restored `_last_alert_at` then held the full cooldown against the re-alert. Measured with a single fake clock driving both `time.monotonic()` and `time.time()`:

  ```text
  OLD (no persistence)   false recovery=False   blind-and-silent 225s
  NEW (as committed)     false recovery=True    blind-and-silent 3590s
  ```

  Two fixes, both needed: a `_restored_alerting` flag gates the thin-sample clear until the sample floor is met, and `_last_alert_at` is no longer restored at all — it bought nothing, because `_alerting=True` already short-circuits the arm path long before the cooldown check.

- **MEDIUM — the recovery notification had no delivery confirmation.** The arm path got a rollback in PR #1805 for exactly this reason; the clear path never did. A recovery whose POST failed was consumed, never retried — and this patch was about to make that loss *durable*. That is a **second live route to "zero `vision_recovered`, ever"** which the original fix did not close. Recovery branches now set `_alert_unconfirmed` with a `_pending_kind`, and an undelivered clear rolls back to alerting so the next healthy sample re-emits it.

- **MEDIUM — the restore path had zero effective coverage.** Three mutations of it survived the whole suite, because every test injected `now=` while `_restore_state` read the real clocks. Added a `clock` fixture that drives one fake clock through both modules; the restore path is now actually exercised.

- **LOW — `load()` could raise, contradicting its own "Never raises" docstring** (`int(raw.get("version"))` sat outside the `try`; a list version raised `TypeError`). Moved inside.

- **LOW — the version-guard test was vacuous.** Its fixture had no `saved_at_wall`, so the *staleness* guard rejected it and the version guard was never exercised; deleting the version guard survived the suite.

- **LOW — the atomicity test tested neither atomicity nor cleanup.** Non-atomic `copyfile`, deleted `fsync`, and deleted temp-cleanup all survived. Now spies on `os.replace`, and reaches the cleanup path by failing the rename rather than `makedirs`.

- **LOW — a persistence failure was invisible.** `save()`'s return was discarded. `snapshot()` (the `/health` surface) now carries `state_path`, `state_write_ok`, `arm_restored_from_disk`.

- **LOW — the circe-qwen lane inherited a path it has no volume for.** It has no `env_file:` and a strict `environment:` allowlist, so a non-empty *default* made it write a state file into an unmounted path. The setting default is now `""` (opt-in); the real path comes from `.env`/`.env_example`.

- **Pattern break** — the key was absent from `docker-compose.yml`'s explicit `environment:` list, unlike all seven sibling `VISION_LIVENESS_*` keys. Added; verified it renders.

### Mutations after the fixes — 11/11 caught

| mutation | result |
|---|---|
| restored arm may clear on thin samples | CAUGHT |
| restore `_last_alert_at` again (the cooldown trap) | CAUGHT |
| recovery has no delivery confirmation | CAUGHT |
| undelivered recovery does not roll back | CAUGHT |
| restore never called | CAUGHT |
| drop the future clamp | CAUGHT |
| snapshot hides write failures | CAUGHT |
| no VERSION guard | CAUGHT |
| non-atomic write | CAUGHT |
| no temp cleanup on failure | CAUGHT |
| no stale guard | CAUGHT |

## Restart required

```bash
scripts/safe_docker_build.sh orion-vision-host up -d --build
```

## Risks / concerns

- **Severity: low.** First deploy starts with no state file, so the currently-open incident (8 unclosed alerts) is not retroactively closed. Only gaps from this point forward close correctly.
- **Severity: low.** If the volume is unwritable in some environment, persistence silently degrades to the old in-memory behaviour. Logged, not fatal — deliberate, since the alternative is a crashloop in the seeing path.

## Related

- #1977 corrects the rationale in the capability-gap journal seed, which had assumed a re-arm implied recovery.

## PR link

<!-- filled after gh pr create -->
