# PR: a dispatch budget that can actually run out

Branch: `feat/exogenous-dispatch-budget`
Step 2 of 3 on the decision-budget arc. Follows #1813 (per-action cost).

## Summary

- The existing daily ceiling was not a budget, for two independent reasons.
- Replaced with **motor-seconds against an operator-set allowance** — real,
  finite, and not under Orion's control.
- **Advisory first**, with the exit criterion written into `settings.py`, and
  a would-refuse count published every tick so the flip has evidence behind it.
- The old risk cap stays live until this enforces. Then it goes entirely.

## Why the old ceiling could never bind

**Self-sized.** `_derive_daily_risk_cap` sets the ceiling to `ewma + 3*sd` of
Orion's own past *demand* — three standard deviations above what it already
wanted. An allowance that tracks what you asked for is a mirror, not a
constraint. Its live drift across this arc:

```
17 -> 29 -> 347 -> 554 -> 475 -> 3,475 -> 1,787
```

**Fake-denominated.** In `risk_score` — five hand-typed constants in a YAML
file, with 67% of dispatches carrying exactly `0.05`.

Either alone would be fatal. Together they mean the number was never going to
stop anything.

## The replacement

Motor-seconds: the wall-clock an action occupies on the dispatch path,
measured at the send (live since #1813). The day is 24 hours long whatever
Orion would prefer.

Denominating the budget in the scarce resource itself also sidesteps the
exchange-rate problem raised earlier in this arc: allocation becomes a
knapsack rather than a conversion, so nobody has to invent how many nats a
GPU-second is worth.

**Sized from measurement, not a guess.** Live on the day it shipped: p50 5.0s
per action, p95 6.5s, 1.7x concurrency, **~40 motor-hours consumed per day**.
Default allowance is **36 motor-hours** — deliberately ~10% *below* current
draw so the ceiling is exercised rather than decorative.

## Outcome moved

First live tick after deploy:

```
motor_budget mode=advisory spent_sec=2366.5 allowance_sec=129600.0
             remaining_sec=127233.5 pace=1.08x projected_day_h=39.0
             pending=0 would_refuse=0
```

Pace 1.08x, projecting **39.0 motor-hours against a 36-hour allowance** — on
track to overrun by 8%. The ceiling will bind, which is the design intent.

## Advisory is a stage, not a hedge

CLAUDE.md 0A bans a switch that reports success while changing nothing, and an
advisory budget whose only output is a reassurance is exactly that. So:

- Every tick publishes `spent / allowance / remaining / pace /
  projected_day_h / pending / **would_refuse**`.
- The exit criterion is in `settings.py`: flip once a full day of would-refuse
  counts exists and the refused set has been inspected and judged droppable.
- Explicitly written down: *if nobody has looked in a week, that is the answer
  — either flip it or delete it.*

`pace` is the number worth watching, not `exhausted`. By the time a budget is
exhausted the interesting decision is hours past; pace says at 06:00 what the
day ends at.

## Files changed

- `orion/autonomy/budget.py`: new. Pure arithmetic — `BudgetState`, `pace`,
  `projected_day_sec`, `would_refuse`.
- `services/orion-execution-dispatch-runtime/app/store.py`:
  `sum_motor_seconds_for_day`.
- `services/orion-execution-dispatch-runtime/app/worker.py`:
  `_derive_motor_budget`, per-tick reporting, enforcement behind the flag.
- `app/settings.py`, `.env_example`, `docker-compose.yml`: three keys.
- `tests/test_motor_budget.py`: new, 12 tests.

## Design decisions worth arguing with

**Unconfigured is not exhausted.** `budget_state()` returns `None` for no
allowance, never a zero-allowance budget. One is unconfigured and the other is
a real ceiling reached; a caller that collapses them refuses everything or
nothing, and both look like a broken dispatcher rather than a policy.

**Absent cost contributes nothing, not zero — and warns.** Those are the same
number and not the same claim. Counting an unmeasured action as free makes a
ceiling systematically over-permit, which is backwards.

**Spend is read from the result table, not the frame table.** The risk cap
re-derives its state by scanning `substrate_execution_dispatch_frames`, and
that single pattern is 49.8% of this database's entire buffer traffic
(pg_stat_statements, 2026-08-20). This sums a narrow time-indexed column on a
much smaller table.

**A read failure returns no budget rather than an empty one.** A budget that
cannot read its own spend must not behave as though nothing has been spent —
that reads as a full allowance and permits everything.

## Env/config changes

- Added: `ORION_DISPATCH_MOTOR_BUDGET_SEC_PER_DAY` (129600.0),
  `ORION_DISPATCH_MOTOR_BUDGET_ENFORCE` (false),
  `ORION_DISPATCH_MOTOR_TYPICAL_COST_SEC` (5.0).
- `.env_example`, `docker-compose.yml` updated; **local `.env` hand-synced**.
  `scripts/sync_local_env_from_example.py` reads `.env_example` from the
  PRIMARY checkout, so keys added in a worktree are invisible to it — verified
  again here, it reported no new keys.
- `check_service_env_compose_parity` clean for the new keys (the 2 it reports
  missing, `EXECUTION_DISPATCH_STALENESS_{MIN,MAX}_SEC`, are pre-existing).

## Tests run

```text
pytest tests/test_motor_budget.py -q                          12 passed
pytest tests/test_execution_dispatch_runtime_{worker,store}.py
       tests/test_per_action_cost.py -q                       107 passed
```

Note: `tests/test_per_action_cost.py` must be split across two PYTHONPATHs —
the dispatch and feedback runtimes both expose a package named `app`, so a
single run silently tests one service's module twice.

## Docker/build/smoke checks

```text
safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
live: motor_budget line emitting every tick, no errors
```

## Restart required

```text
Already applied.
```

## Risks / concerns

- Severity: MEDIUM — **advisory means nothing is enforced yet.** The old,
  useless cap is still the only live ceiling. That is deliberate (removing it
  now would leave none) but it means this PR changes no behaviour, and the
  value is entirely in the evidence it produces. It must not sit here.
- Severity: LOW — `would_refuse` uses a flat p50 cost for not-yet-run actions.
  A real allocator will use each action's own measured history; this is a
  placeholder for the advisory count and is not used for anything enforced.
- Severity: LOW — the budget day is UTC, matching the existing risk cap.
  Juniper is in MDT, so the reset lands at 18:00 local.

## PR link

<filled in on push>
