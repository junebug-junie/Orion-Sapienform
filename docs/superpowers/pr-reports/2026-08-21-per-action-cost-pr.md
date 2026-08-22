# PR: per-action cost — the denominator a decision budget divides by

Branch: `feat/per-action-cost`
Step 1 of 3 on the decision-budget arc.

## Summary

- **Orion had no per-action cost.** Not "a rough one" — none. `latency_ms` was
  a field on `ActionOutcomeRecordV1`, a column on `substrate_action_outcomes`,
  and a `_latencies()` reader in the feedback runtime, and it was populated on
  **0 of 5,739 rows over 6 hours**.
- It was dropped **twice, in series**, which is why it looked implemented.
- This measures it at the send, carries it through both drops, and proves it
  landing live.

## Outcome moved

First per-action cost in the system, measured live within two minutes of the
fixed deploy:

```
status    n   with_cost   min_ms   avg_ms   max_ms
success   5           5      895     3887     5089
```

At ~5,400 dispatches/day and ~3.9s each, that is roughly **5.8 hours of motor
time per day** — a real, finite, exogenous quantity, and the first thing Orion
has that can actually run out. It is both the budget's denominator and a
candidate for its size.

## The two drops

1. **Nothing wrote it.** cortex-exec measures a latency internally
   (`bound_capability_exec.py:204`, `perf_counter`) and nothing carries it out
   of the service. The dispatch worker never recorded one.
2. **Nothing could read it.** `load_cortex_result_evidence()` built its
   evidence dict from four hardcoded keys — `result_id`, `dispatch_id`,
   `status`, `evidence_refs`. `_latencies()` scans those entries for
   `latency_ms` / `duration_ms` / `elapsed_ms`, all three of which were
   filtered out one layer earlier. **The reader was unreachable regardless of
   what any producer wrote.**

A schema field, a column and a reader, none of which could ever carry a value.
That is what stopped the budget being buildable: value with no denominator is
a ranking, and a ranking cannot say *"none of these were worth it."*

## Why measured at the send

Wall-clock around `client.dispatch`, not read off the verb's own report.

- **Right quantity.** It is the time the action occupied the motor path,
  queueing and transport included, which is what spending it costs.
- **Reliable.** It does not depend on a verb choosing to report anything, and
  `skills.runtime.*` verbs report nothing.
- **Every exit path carries it, failures included.** A failed send still
  consumed real time — usually the entire RPC timeout, the most expensive
  outcome there is. Recording it as absent would make failure look free and
  bias any cost-weighted comparison toward whatever fails fastest.
- **Absent stays absent.** Never coerced to `0.0`, which reads as "this action
  was free."

## Files changed

- `services/orion-execution-dispatch-runtime/app/worker.py`: `perf_counter`
  around the send; all three save paths carry the cost.
- `services/orion-execution-dispatch-runtime/app/store.py`: `latency_ms`
  parameter, INSERT, and `ON CONFLICT DO UPDATE` branch.
- `services/orion-feedback-runtime/app/store.py`: evidence dict carries
  latency; `.get()` rather than `[...]`.
- `services/orion-sql-db/manual_migration_dispatch_latency.sql`: new,
  **applied**.
- `tests/test_per_action_cost.py`: new, 6 tests.

## A regression I shipped, and how it was caught

The first deploy **broke dispatch-result writes.** `:latency_ms` went into the
INSERT and never into the parameter dict, so SQLAlchemy raised `A value is
required for bind parameter 'latency_ms'` on every write.

Containers came up green. Tests passed. Health was fine. It was found only by
asking whether the number actually landed — it was still 0 — and then reading
the service log rather than trusting the deploy.

Fixed, and there is now a guard test that parses every `:bind` out of that
INSERT and asserts each one is supplied. It fails on exactly that defect.

Also caught by the tests: `row["latency_ms"]` on a mapping without the key
raised `KeyError` and aborted the **entire** evidence load, taking feedback
scoring down for the sake of one absent optional measurement. `.get()` now.

## Tests run

```text
pytest tests/test_per_action_cost.py -q                       6 passed
pytest tests/test_execution_dispatch_runtime_{worker,store}.py
       tests/test_execution_dispatch_result_extraction.py -q  91 passed
pytest tests/test_feedback_runtime_store.py
       tests/test_action_outcome_resolution.py
       tests/test_feedback_builder.py -q                      61 passed
```

## Docker/build/smoke checks

```text
safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
safe_docker_build.sh orion-feedback-runtime up -d --build

live: 5/5 dispatch results carrying latency_ms, 895–5089ms
```

## Restart required

```text
Already applied — both runtimes rebuilt and verified.
```

## Risks / concerns

- Severity: LOW — cost is wall-clock, which includes queueing behind other
  dispatches. That is correct for "what did spending this cost me" and wrong
  for "how expensive is this verb intrinsically". A budget wants the former.
- Severity: LOW — no cost for actions that never reach the send (blocked,
  deferred). Correct: they consumed nothing.
- Not addressed: GPU-seconds and dollars are separate denominators. Dollars
  already exist in `dev_economics_ledger_log` at aggregate granularity.

## PR link

<filled in on push>
