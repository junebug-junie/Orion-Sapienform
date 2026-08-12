# doc-semantic-drift: durable baseline sha (fixes redeploy-swallow bug)

## Summary

- Fixes a real live bug: `doc_semantic_drift_loop`'s `last_sha` baseline lived only in process memory, and every redeploy re-seeded it at whatever HEAD was current at that moment.
- Since `orion-cocreation-signals` gets redeployed far more often than a real `*.md` doc commit lands, a redeploy landing right after a doc merge (the normal real workflow) silently swallowed that doc's score forever.
- Confirmed live against two real PRs (#1571, then again #1577) before this fix — Juniper caught the pattern directly.
- Fix: persist `last_sha` to Redis after every real advance; resume from it on startup instead of always cold-starting.

## Outcome moved

`doc_semantic_drift` now survives redeploys without losing real doc-drift events — the producer's core value proposition (scoring real doc changes) was structurally broken under this repo's actual operating rhythm (frequent redeploys) until this patch.

## Current architecture

Before this patch: `last_sha: str | None = None` was a plain local variable inside `doc_semantic_drift_loop`. A container restart always reset it to `None`, triggering a fresh cold-start seed at current HEAD — silently discarding whatever real doc changes happened between the last successful tick and the restart, *and* (the actual bug) discarding any doc change that had already landed by the time the restart's cold-start scan ran.

## Architecture touched

- `orion-cocreation-signals`'s `doc_semantic_drift` producer only. No schema, channel, or contract changes.

## Files changed

- `services/orion-cocreation-signals/app/producers/doc_semantic_drift.py`: added `_load_last_sha(bus, state_key)` / `_save_last_sha(bus, state_key, sha)` (both fail soft via `bus.redis.get`/`bus.redis.set`, the real `aioredis.Redis` client `OrionBusAsync` already exposes). `doc_semantic_drift_loop` gained a required `state_key: str` param, loads the baseline on startup, persists after every real advance (cold-start seed included), and does not persist a failed publish's sha (unchanged contract, now durable).
- `services/orion-cocreation-signals/app/settings.py`: new `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_STATE_KEY` (default `orion:cocreation_signals:state:doc_semantic_drift:last_sha`).
- `services/orion-cocreation-signals/app/main.py`: wired `state_key` into the `doc_semantic_drift_loop` call.
- `services/orion-cocreation-signals/docker-compose.yml`, `.env_example`: new env var passthrough.
- `services/orion-cocreation-signals/tests/conftest.py`: `FakeRedis` (in-memory `get`/`set`) + `FakeBus.redis` property backed by a `redis_store: dict` field — two separate `FakeBus` instances can share a `redis_store` to simulate state surviving a real process restart.
- `services/orion-cocreation-signals/tests/test_doc_semantic_drift_producer.py`: `state_key=` added to every existing loop call (now required); 4 new regression tests — cold-start persists its seed, a real change persists the new baseline, a failed publish does not persist, and (the real regression test) a fresh `FakeBus` sharing an earlier one's `redis_store` resumes correctly and scores a change instead of cold-starting past it.

## Schema / bus / API changes

None — this is a durability fix to existing producer state, not a contract change.

## Env/config changes

- Added keys: `COCREATION_SIGNALS_DOC_SEMANTIC_DRIFT_STATE_KEY` (default `orion:cocreation_signals:state:doc_semantic_drift:last_sha`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: yes (primary checkout + worktree).
- skipped keys requiring operator action: none.

## Tests run

```text
.venv/bin/python -m pytest services/orion-cocreation-signals/tests/ orion/structural_mass/tests/ -q
107 passed, 15 warnings in ~6s
```

## Evals run

No dedicated eval harness for this fix — it's a durability/state-persistence correction, not a scoring-quality change.

## Docker/build/smoke checks

Deployed live:

```text
scripts/safe_docker_build.sh orion-cocreation-signals up -d --build
```

Live verification:
- Container log shows a clean cold start (`cocreation_doc_semantic_drift_cold_start head_sha=3cddd4b7...`) — expected, since this was the first-ever run of the new code (no prior state to resume from).
- `redis-cli -h 100.92.216.81 -p 6379 -n 0 GET orion:cocreation_signals:state:doc_semantic_drift:last_sha` against the real live Redis instance returns `3cddd4b7...`, matching the cold-start HEAD — confirms the baseline is now durably persisted, not just held in process memory. From this point forward, any redeploy resumes from this real key instead of resetting.

## Review findings fixed

- Finding: none material — code review traced the fix by hand against the exact real failure scenario (doc merge → immediate redeploy) and confirmed it correct: the persisted `last_sha` is loaded before the cold-start branch, so a redeploy after a real doc merge now diffs against the pre-merge baseline and scores the doc, instead of re-seeding past it.
  - Fix: none needed.
  - Evidence: `test_restart_resumes_from_durable_state_instead_of_cold_starting_at_new_head` passing; independently re-run by the review agent (`107 passed`); live redis-cli confirmation above.
- Finding (informational, not blocking): a stale persisted `last_sha` after a hypothetical force-push/history-rewrite of `main` could cause `git diff` to compare non-ancestor commits — traced to `changed_doc_files`'s `check=False` git subprocess call, confirmed this produces either a plausible-but-wrong diff range or a silent no-op tick (returncode != 0 → `[]`), never a crash. Same pre-existing property `git_delta_loop` already has; not worsened by this patch (a stale sha now persists across a restart instead of resetting, but this repo isn't force-pushed).
  - Fix: not addressed — accepted as a pre-existing, documented risk, consistent with `git_delta_loop`'s own docstring.

## Restart required

```text
No restart required -- orion-cocreation-signals was already redeployed live as part of this patch.
```

## Risks / concerns

- Severity: low
- Concern: a Redis outage at startup silently falls back to the old cold-start behavior (fail-soft `_load_last_sha`) — an operator watching only the "resumed from durable state" log line might not immediately realize persistence itself is degraded, just that a cold start happened.
- Mitigation: `_load_last_sha`/`_save_last_sha` both `logger.warning` on failure with `exc_info=True` — a real, greppable signal exists, just not surfaced as a metric/alert yet. Low severity since the behavior degrades to exactly the pre-patch baseline, never worse.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/doc-semantic-drift-durable-state
