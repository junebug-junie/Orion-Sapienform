# Measure the FCC leg, the budget that actually binds

Status: **DONE** — producer, contract, and consumer are all wired and tested;
no restart-blocking concerns beyond the deploy itself.

## Summary

- `HarnessRunV1` carried `step_count`, `exit_code`, `grounding_status` and
  `fcc_served_model` — and **no duration at all**.
- `HARNESS_FCC_TIMEOUT_SEC` (1600s) governs the FCC motor leg and is what
  decides `grounding_status == "fcc_timeout"`. Nothing measured that leg.
- PR #2017 could only record whole-turn wall time, which also spans the stance
  leg (≤400s), the governor queue and the finalize chain (≤485s).
- Now measured in the motor, passed through untouched, and rendered beside the
  whole-turn number so the difference is the overhead.

## Outcome moved

"Is the budget too small, or does the turn never converge?" becomes answerable.
Before: for a timed-out run the motor leg is pinned at 1600s by construction,
so every bit of variance Hub could see was overhead. After: a grounded run's
distance from 1600s is real headroom.

## Architecture touched

```text
HarnessRunner.run  -> HarnessMotorResult.fcc_elapsed_sec   (producer)
bus_listener       -> HarnessRunV1.fcc_elapsed_sec         (5 sites)
turn_orchestrator  -> frame["harness_fcc_elapsed_sec"]     (governor/Hub seam)
curiosity journal  -> "of which harness 1598s"             (consumer)
```

The clock starts before **any** work in `run()` — the served-model probe and
the concurrent bus reads are part of the leg the deadline governs, and
excluding them would understate it in the direction that hides a budget problem.

## Files changed

- `orion/harness/runner.py`: `fcc_elapsed_sec` on `HarnessMotorResult`, set at
  both return sites.
- `orion/schemas/harness_finalize.py`: optional field on `HarnessRunV1`.
- `services/orion-harness-governor/app/bus_listener.py`: forwarded at all 5
  construction sites.
- `orion/hub/turn_orchestrator.py`: onto the final frame.
- `services/orion-hub/scripts/curiosity_investigation.py`: read off the frame
  and rendered.
- Tests in all three suites.

## Schema / bus / API changes

- Added: `HarnessRunV1.fcc_elapsed_sec: float | None = None`;
  frame key `harness_fcc_elapsed_sec`.
- Compatibility: optional with a `None` default, so an older governor omitting
  it still validates and Hub renders nothing rather than a guess. Absent means
  **"no motor leg happened"** (the refusal/validation paths never run the
  motor), not "it took no time".
- Registry: `HarnessRunV1` is registered by class reference, so no registry
  edit was needed. Verified through `resolve("HarnessRunV1")` rather than the
  class — the registry is what consumers bind to, and a field can exist on the
  class while the registered model is a different one.

## What the tests caught

A string anchor matched the four early-return `HarnessRunV1` sites at one
indent level and **missed the success path**, which sits at another. The
governor pass-through test failed on exactly that, so the one path a normal
completed turn takes had been silently unwired. Fixed with a line-exact edit.

The two governor path families are covered separately and mutate red
independently:
- success path → `test_harness_run_carries_the_fcc_leg_duration_from_the_motor`
- no-draft early return → `test_a_motor_that_produced_no_draft_still_reports_its_leg_duration`

The second is the case the number matters most for: a turn that burned the
whole budget and salvaged nothing never gets a journal entry, so the run
artifact is the only record it happened.

## Tests run

```text
orion/harness/tests ........... 3 failed, 238 passed
    ^ the 3 failures are PRE-EXISTING on main -- verified identical set AND
      identical assertion text (grounding_status carries the message, not the
      code). Not introduced here; see Risks.
services/orion-harness-governor/tests ..... 20 passed
services/orion-hub  (curiosity + ws frames)  142 passed
9 mutations verified red (each anchor asserted to match an exact line count)
10/10 CI static gates PASS
```

Mutation coverage by layer: runner sets it (2), schema carries it (1),
governor forwards it on both path families (2), Hub frame carries it (2),
consumer reads and renders it (2).

## Env/config changes

None. No new keys. `HARNESS_FCC_TIMEOUT_SEC` and
`HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC` are both unchanged — this patch
measures, it does not tune.

## Restart required

```bash
# Both, and the governor first -- it is the producer.
cd /mnt/scripts/Orion-Sapienform
docker compose --env-file .env --env-file services/orion-harness-governor/.env \
  -f services/orion-harness-governor/docker-compose.yml \
  up -d --force-recreate orion-athena-harness-governor
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --force-recreate orion-athena-hub
```

Hub alone is harmless but pointless — it would render nothing, because no
frame would carry the field.

## Risks / concerns

- Severity: low. Concern: `orion/harness/tests` has 3 failures on main,
  including `test_harness_runner_surfaces_fcc_error_code`, which asserts
  `grounding_status == "fcc_timeout"` but receives the human message
  `"fcc turn timed out after 120.0s"`. That is a real defect in the same area
  this patch touches — the code is being written into the status field
  somewhere — but it is pre-existing and fixing it would change behaviour
  downstream consumers key on. Left alone, flagged.
- Severity: low. Concern: `spark_meta` in `turn_orchestrator.py:812` carries
  `harness_step_count`/`harness_grounding_status` but not the duration. That is
  a different consumer with its own telemetry contract; not extended here.

## Next

With a few runs recorded, the grounded-run distribution against 1600s answers
the original question directly. If grounded runs cluster near the ceiling the
budget is genuinely too small; if they cluster far below it, the timeouts are
non-convergence and a larger budget just moves the wall.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2019
