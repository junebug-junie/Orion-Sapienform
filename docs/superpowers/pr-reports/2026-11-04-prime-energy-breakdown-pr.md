# fix(prime): show Prime what idling costs vs what activity adds; never invent per-organ cost

## Summary
- Every Prime thought now gets an "ENERGY MONEY" section, read from durable-runs `GET /physiology/report`. It shows:
  - this month's forecast against the limit;
  - how much of that is idling (the always-on floor) and how much is activity on top;
  - each machine's idle cost and its draw above idle right now;
  - the rest of the floor that the report doesn't split by machine.
- One plain line says per-organ cost is not measured, so slowing one organ saves an unknown, probably small amount, and the floor is untouched by slowing anything.

## Outcome moved
Prime has never sent a "slow" signal, though it is switched on and curiosity and dream both receive it. Its one worry is energy. Live today:
- **Idle floor:** $111 a month, already over the $100 limit with nothing running. No slow signal can reach this.
- **Activity on top:** about $25 a month.

So declining to slow curiosity or dream was a reasonable call. But Prime blamed the activity cost on "attention:juniper", which is Juniper's message cap. Prime had nothing better to read, so it guessed. This section gives it the real split.

## Why no per-organ numbers
Over the last 30 days, the act ledger's per-act dollar costs add up to well under $1, against about $25 a month of activity cost:
- curiosity: $0.17, recorded on only 23 of its 81 acts;
- painting, reading and Prime itself: no cost recorded at all.

Per-organ figures would be about 50 times too low, a worse false belief than the one this fixes. Measuring per-organ cost properly is body-economy work, not a frame change.

## Live preview (2026-11-04 report)
```text
this month: forecast $136 against a $100 limit. Idling alone (the always-on floor) costs $111; activity on top of idling adds about $25
- circe: idling costs $50/month; right now it runs 95 W above idle, about $8.7/month at this rate
- athena: idling costs $31/month; right now it runs 87 W above idle, about $8.0/month at this rate
- the rest of the floor, about $30/month, is always-on draw not split by machine in this report
per-organ cost is not measured: acts carry only a few percent of the activity cost, so slowing one organ saves an unknown, probably small amount; the idle floor is not touched by slowing anything
```

## Files changed
- `orion/prime/thought.py`: `FrameInputs.energy` and `energy_error`, `_energy_lines`, `NO_PER_ORGAN_COST`, and the new frame section. A failed read shows as a named gap.
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
