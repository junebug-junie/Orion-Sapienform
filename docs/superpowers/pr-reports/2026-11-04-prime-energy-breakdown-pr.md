# fix(prime): show Prime what idling costs vs what activity adds; never invent per-organ cost

## Summary
- Every Prime thought now gets an "ENERGY MONEY" section, read from durable-runs `GET /physiology/report`:
  - each machine's idle cost per month, and its draw above idle right now;
  - a pointer to the BODY energy line, which already carries the month's totals: the always-on body per month and the activity pace;
  - one plain line saying per-organ cost is not measured, so slowing one organ saves an unknown, probably small amount, and idling is untouched by slowing anything.

## Outcome moved
Prime has never sent a "slow" signal, though it is switched on and curiosity and dream both receive it. Its one worry is energy. The body's own energy line says, live: always-on body about $109 a month (more than the $100 limit by itself); activity pace about $29 a month. So declining to slow curiosity or dream was reasonable. But Prime blamed the activity on "attention:juniper", which is Juniper's message cap. This section gives Prime the per-machine picture and says plainly that per-organ cost is unknown.

## Why no per-organ numbers
Over the last 30 days, the act ledger's per-act dollar costs add up to well under $1, against about $25 a month of activity cost:
- curiosity: $0.17, recorded on only 23 of its 81 acts;
- painting, reading and Prime itself: no cost recorded at all.

Per-organ figures would be about 50 times too low, a worse false belief than the one this fixes. Measuring per-organ cost properly is body-economy work, not a frame change.

## Live preview (2026-11-04 report)
```text
- circe: idling costs about $50 a month; right now it runs 95 W above idle, about $8.7 a month at this rate
- athena: idling costs about $31 a month; right now it runs 87 W above idle, about $8.0 a month at this rate
the month's totals (always-on body vs activity pace) are in the energy line under BODY
per-organ cost is not measured: acts carry only a few percent of the activity cost, so slowing one organ saves an unknown, probably small amount; the idle floor is not touched by slowing anything
```

## Review findings fixed
- **The first version misread the month's money.**
  - It said "idling alone costs $111, activity adds $25", computed from `money.floor_usd` and `forecast_usd`.
  - `floor_usd` is month-to-date spend plus idling for the hours left, not a month of idling (`orion/physiology/budget.py`). So "forecast − floor" counted only the activity still to come, and would read about $0 near month end.
  - Fix: removed. The BODY energy line already states the correct split, computed by the producer.
- **A "rest of the floor" line compared mismatched bases:** a full-month p05 host rest against a month-to-date-plus-remaining p10 floor.
  - Fix: removed.
- **Checked and fine:**
  - A slow or failed report read happens before the model call, never on Juniper's turn path, and becomes a named gap.
  - The golden re-record hid nothing: with the section stripped, it is identical to main's.

## Files changed
- `orion/prime/thought.py`: `FrameInputs.energy` and `energy_error`, `_energy_lines` (per-machine lines only), `NO_PER_ORGAN_COST`, and the new frame section. A failed read shows as a named gap.
- `services/orion-mind/app/prime_loop.py`: reads the report once per thought, never per tick, because the report takes about 1.5 s to build.
- `orion/prime/evals/fixtures/energy_report_live_2026-11-04.json`: the live money and host fields only.
- Tests: `orion/prime/tests/test_prime_core.py`.
- Replay golden re-recorded. The only change is the new section in each model call's prompt; with it stripped, all 12 changed parts match the previous golden.

## Tests run
```text
orion/prime + orion/mindloop   87 passed
services/orion-mind/tests     226 passed
frame eval                    PASS (3096 chars, cap 12000)
no-personal-terms             7 passed
body-economy contract         OK
```

## Restart required
```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns
- **Severity: low.** The report read adds about 1.5 s to each thought, and Prime thinks about 5 times a day.
- **Overlap:** this touches `orion/prime/thought.py` at the same time as the curiosity replica patch 4, which adds an "ORGAN REPLIES" section. Whichever merges second needs a small rebase in the frame list.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
