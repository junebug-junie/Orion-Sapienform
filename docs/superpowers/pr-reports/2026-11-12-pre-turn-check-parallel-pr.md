## Summary

- cortex-exec now runs pre-turn checks ("repair pressure" appraisals) side by side, not one at a time (`concurrent_handlers=True` on `pre_turn_appraisal_svc`, `services/orion-cortex-exec/app/main.py`).
- New regression test: a slow check must not block a later fast check. It failed before the fix and passes after it.
- New CI job `cortex-exec-concurrency-tests` runs this test and the older `test_exec_concurrent_handlers.py`. No CI job ran either file before.

## Outcome moved

Every Hub turn waits for its pre-turn check, with a 180 s timeout. The worker ran checks one at a time. On 2026-11-12 between 02:50Z and 03:45Z, six turns failed with `appraisal_unavailable` at 180 s. Most of them started while another check was still running. Juniper's check (corr `aaee51e3-9f6b-484f-9ac7-6136b5d827a7`) waited 180 s behind world-pulse checks, and then ran in 2.4 s. A check at 03:44:28 took 33 s and finished right after the check ahead of it timed out. That is serial queueing.

After this patch, a turn's check does not wait for other turns' checks. GPU work still queues in the GPU pool.

## Current architecture

The Hub is the only sender on `orion:cortex:pre_turn_appraisal:request`. Only the `orion-athena-cortex-exec` container handles it (`ENABLE_PRE_TURN_APPRAISAL_HANDLER=true`). The handler keeps all state per request, and every reply goes to a per-turn reply channel.

## Architecture touched

- One `Rabbit` flag in cortex-exec. No contract, schema, bus or env change.

## Files changed

- `services/orion-cortex-exec/app/main.py`: `concurrent_handlers=True` and a comment on why.
- `services/orion-cortex-exec/tests/test_pre_turn_appraisal_concurrency.py`: flag pin plus a slow-then-fast dispatch test.
- `.github/workflows/cortex-exec-concurrency-tests.yml`: CI gate for both concurrency test files.
- `docs/superpowers/pr-reports/2026-11-12-pre-turn-check-parallel-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: pre-turn checks run concurrently in cortex-exec.
- Compatibility notes: none. The request and reply shapes are unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
cd services/orion-cortex-exec
PYTHONPATH=<worktree>:. pytest -q -p no:cacheprovider tests/test_exec_concurrent_handlers.py tests/test_pre_turn_appraisal_concurrency.py
4 passed in 0.39s   (the 2 new tests failed before the main.py edit)

Full suite, branch vs main (same command, --continue-on-collection-errors):
branch: 137 failed, 1032 passed, 14 errors
main:   133 failed, 1036 passed, 13 errors
The extra collection error is the new file. The full-suite run hits a "Verb already
registered" error for every file that imports app.main. That error is on main too.
The other 5 differences (test_chat_reply_logprobs_gate, test_harness_finalize_max_tokens)
depend on test order. Both files pass in isolation on branch and on main (13 passed each).
CI never runs the full cortex-exec suite. It runs single files, which this PR matches.
```

## Evals run

```text
No eval harness for this seam. The live check below is the eval: overlapping
pre_turn_appraisal sightings in cockpit_turn_sighting must finish without the 180 s timeout.
```

## Docker/build/smoke checks

```text
After merge: scripts/safe_docker_build.sh orion-cortex-exec up -d --build cortex-exec (from primary main).
```

## Review findings fixed

- Finding: the CI path filter did not include the handler file `app/pre_turn_appraisal.py`, so new shared state there would not trigger the gate.
  - Fix: added the file to both path lists.
  - Evidence: `.github/workflows/cortex-exec-concurrency-tests.yml`.
- Finding (minor, not fixed): there is no cap on concurrent checks. A world-pulse burst now sends its GPU probes to the gateway together.
  - Decision: each probe is bounded by its own timeout, and the GPU pool queues the work. Add a semaphore only if live data shows probes being shed.
- Reviewer clean-env run: a fresh 3.12 venv with the workflow's installs gave 4 passed. With the flag off, the new test fails with TimeoutError after 2 s (no hang).

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build cortex-exec
```

## Risks / concerns

- Severity: low
- Concern: more GPU probes in flight at once during world-pulse bursts.
- Mitigation: the GPU pool queues them, and each probe has a timeout. Watch `cockpit_turn_sighting` for slow `appraisal · ok` times.

## PR link

See the PR on GitHub.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
