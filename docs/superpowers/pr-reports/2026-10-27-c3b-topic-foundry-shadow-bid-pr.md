# Body economy Patch C3b (topic-foundry): shadow bid at training run start

## Summary

- Each topic-foundry training run now asks the body economy "would you fund this run?" at run start, and ignores the answer. The answer is only logged.
- The bid's `want_id` is the run act's correlation id (`act_emit.run_correlation_id(run_id)`).
- The run is synchronous (a BackgroundTasks worker thread), so the bid runs on a daemon thread. The run never waits on it, and its act events and status writes are the same whatever the economy answers.
- The env sync now visits orion-topic-foundry. Before, the default run never looked at this service at all.
- Switch `TOPIC_FOUNDRY_PHYSIOLOGY_SHADOW_BID_ENABLED`, shipped on.

## Outcome moved

Topic-foundry runs are discretionary GPU work (Tier 2) with no gate of their own. Each run now leaves a `physiology_shadow_bid ... shadow=<funded|deferred|none> real=ran` line, and durable-runs writes the decision to `want_ledger`. That is the first evidence of how often the economy would have deferred a run that ran.

## Current architecture

Patch C1 built the market and client. `_run_training` marks the run `running` and opens a `maintain` act (actor `topic_foundry.run`) keyed on the run id. Nothing gates the run at that point.

## Architecture touched

- orion-topic-foundry only, plus one contract registration and the env sync script. No new bus channel.

## Files changed

- `services/orion-topic-foundry/app/services/physiology_shadow.py`: new, synchronous `ShadowBid` (daemon thread and a lock; whichever of the bid answer and the settle comes second logs once; releases a funded-but-blocked reservation; swallows every failure).
- `services/orion-topic-foundry/app/services/training.py`: `_shadow_bid_run_start` (never raises) right after the run act opens; settles `ran`.
- `services/orion-topic-foundry/app/settings.py`, `.env_example`, `docker-compose.yml`: the switch (compose lists this service's env explicitly). The client's default URL resolves on app-net (checked from the running container).
- `scripts/sync_local_env_from_example.py`: `TOPIC_FOUNDRY_PHYSIOLOGY_SHADOW_BID_ENABLED` in `SYNC_EXACT`; `orion-topic-foundry` added to `DEFAULT_SERVICES`.
- `config/physiology/contract.yaml`: the helper registered as `display`.
- `services/orion-topic-foundry/tests/conftest.py`: new; autouse fixture so tests never POST to the live market.
- `services/orion-topic-foundry/tests/test_physiology_shadow_bid.py`: new.
- `services/orion-topic-foundry/README.md`: section.

## Correlation join

| family | want_id | act correlation id | verified by |
|---|---|---|---|
| `topic_foundry.run` (Tier 2) | `act_emit.run_correlation_id(run_id)` | `RunActTrace.correlation_id` (same function) | `test_a_run_is_identical_for_every_answer_and_joins_its_act` asserts want_id == the emitted act_opened correlation id |

## Real-gate field

Always `ran` (admitted, not completed): the run has no gate at run start (the `/runs/train` route already created it).

## Schema / bus / API changes

- Added / Removed / Renamed: none. Behavior changed: none (shadow only).

## Env/config changes

- Added keys: `TOPIC_FOUNDRY_PHYSIOLOGY_SHADOW_BID_ENABLED=true` (orion-topic-foundry).
- `.env_example` updated: yes. `docker-compose.yml` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (primary checkout `services/orion-topic-foundry/.env`; the sync added only this key).
- skipped keys requiring operator action: none.

## Tests run

```text
In the service image (host venv lacks joblib), worktree mounted read-only, no network:
  pytest tests/test_physiology_shadow_bid.py tests/test_training_run_act.py -> 12 passed (also 13 repeated runs by the reviewer, no flakes)
  pytest tests -> 158 passed, 5 failed; the same 5 fail on origin/main in that image (no asyncio
  plugin there; chat corpus / drift / heartbeat tests), unrelated.
orion-static-gates.yml python steps (38 gates, incl. check_body_economy_contract) -> all OK
```

## Evals run

```text
None. The shadow-vs-real comparison needs live decisions (scripts/analysis/body_economy_shadow_vs_real.py, another agent).
```

## Docker/build/smoke checks

```text
docker exec orion-athena-topic-foundry: GET http://orion-athena-durable-runs:8121/health -> 200.
The live bid from a real run is UNVERIFIED until deploy.
```

## Review findings fixed

- Finding: when a blocked settle came second, `release()` ran on the producer's thread (up to 0.5 s plus DNS).
  - Fix: the release POST runs on its own daemon thread.
  - Evidence: `test_helper_blocked_funded_releases_off_the_caller_thread` (settle returns in under 0.2 s while a 0.3 s release runs).
- Finding: `real=ran` reads as "completed".
  - Fix: README and module docstring say `ran` means admitted, not completed.
- Nit: the release test swapped module attributes by hand. Fix: monkeypatch.
- Not changed: the helper is copied per service (hub, thought, dream, actions, topic-foundry), not shared. `orion/physiology/` belongs to another agent in this patch; moving it there is a follow-up.
- Reviewer also confirmed: `--dry-run` of the sync for this service selects only `LOG_LEVEL` and the new key, so adding the service to `DEFAULT_SERVICES` syncs nothing unintended; durable-runs is reachable from inside the container.

## Restart required

```bash
scripts/safe_docker_build.sh orion-topic-foundry up -d --build
```

Run it from the clean primary `main` checkout after merge.

## Risks / concerns

- Severity: low. Concern: orion-topic-foundry has no CI test workflow, so these tests run only locally. Mitigation: run in the service image as above; follow-up to add a workflow.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
