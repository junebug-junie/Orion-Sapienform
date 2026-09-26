# GPU pool: the visual-baseline guard stopped blocking every load

## Summary

- The pool's visual-baseline guard compared each reading's timestamp against a "now" taken **before** the HTTP request. The activity endpoint stamps `observed_at` while it answers, a few milliseconds later. So every fresh reading looked slightly in the future and failed the guard with `visual_activity_unavailable`, and the pool could never load gpu2.
- The age is now measured when the answer arrives, and a clock difference of up to 2 s between processes is tolerated. A reading stamped further in the future still blocks.

## Outcome moved

Live on 2026-09-26, after the stage-4 deploy, `GET /v1/pool` showed `swap_guards.visual_baseline = "visual_activity_unavailable"` even though the endpoint answered 200 with `history_status=ok` from inside the pool container. The guard fails closed, so nothing unsafe happened. But the second 27B seat could never be opened by the pool.

## Current architecture

`services/orion-gpu-pool/app/main.py` `_guards_forever` passes `datetime.now()`, taken before the reads, into `GuardReader.read`. `_read_visual` then required `0 <= now - observed_at`.

## Architecture touched

`services/orion-gpu-pool/app/guards.py`: an optional `clock` parameter, age measured on receipt, and `VISUAL_CLOCK_SKEW_SEC = 2.0`. `main.py` passes a clock.

## Files changed

- `services/orion-gpu-pool/app/guards.py`
- `services/orion-gpu-pool/app/main.py`
- `services/orion-gpu-pool/tests/test_guards.py`: 2 regression tests. The first fails on the old code.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
services/orion-gpu-pool: 87 passed, 7 skipped
the new "stamped just after the read began" test fails on the old guards.py (1 failed, 5 passed)
```

## Evals run

```text
N/A: a guard's timing, not scheduling.
```

## Docker/build/smoke checks

```text
Live diagnosis: from inside the pool container, the endpoint returns 200 with history_status=ok, yet the guard reads visual_activity_unavailable.
After deploy (UNVERIFIED until then): GET /v1/pool swap_guards.visual_baseline should be null or visual_baseline_urgent/visual_attempt_running, never visual_activity_unavailable while the endpoint is healthy.
```

## Review findings fixed

- Small, self-reviewed diff.

## Restart required

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: 2 s of tolerated future skew.
  - Mitigation: the endpoint runs on the same host clock, and anything larger still blocks.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
