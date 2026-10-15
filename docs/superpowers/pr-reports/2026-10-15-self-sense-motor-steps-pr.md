## Summary

- Self-sense eval rows now carry `motor_steps=N` in `notes`: how many harness steps the answering turn took.
- Both producers pass it (durable-runs `self_sense_graph.publish`, Hub in-process `tick_self_sense_eval`); `build_row` is the single writer.

## Outcome moved

Orion's Day 10-13: "the unfiltered mirror where motor tool text gets graded as my answer." Checked live 2026-10-15: not a leak. Every probe runs as a full tool-using agent turn (avg 8.9 / 60.4 / 7.6 / 14.9 steps for who_matters / last_day_unasked / what_are_you / cannot_do_now across ~90 rows each since 10-01), and `answer_text` equals the trace's `final_text` on every joined row. So an answer can be a research report (e.g. "I've now reviewed the `link_accepted_seeds` module"). The scorer cannot tell, and the trend mixes both.

## Current architecture

`build_row` picks trace text over HTTP text, scores `self_label_score` / `grounded_record_score`. Both live producers pass the turn text as `http_text` (so `answer_source` reads `http` for 458 of 476 recent rows).

## Architecture touched

`orion/evals/self_sense_runner.py` (additive kwarg), `services/orion-durable-runs/app/self_sense_graph.py`, `services/orion-hub/scripts/curiosity_investigation.py`. No schema, bus or env change (`notes` is free text).

## Files changed

- `orion/evals/self_sense_runner.py`: `motor_steps` kwarg + note
- the two producers: pass `harness_step_count` from the turn's debug
- `orion/evals/tests/test_self_sense_runner_motor_steps.py`: new

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
orion/evals/tests/test_self_sense_runner_motor_steps.py  3 passed
services/orion-hub/tests/test_curiosity_self_sense_eval.py + test_self_sense_eval_producer.py  26 passed
services/orion-durable-runs/tests/test_self_sense_graph.py  6 passed
```

## Evals run

No new eval; this change adds a field the existing eval will emit.

## Docker/build/smoke checks

Not run. Live check after deploy: next scheduled self-sense rows should show `motor_steps=` in `notes`.

## Review findings fixed

Small additive change; no separate review pass. Deliberately NOT built: a narration/work-report detector (a word list would be a keyword cathedral). The step count is deterministic and lets the trend be split.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Low: rows before deploy have no step count; the join on `harness_turn_trace.step_count` recovers it for history.
- Open question for Juniper: should the three non-research probes (who_matters, what_are_you, cannot_do_now) run with tools off, so they measure the chat self rather than an agent? That changes the eval's meaning; not done here.

## PR link

(see PR)
