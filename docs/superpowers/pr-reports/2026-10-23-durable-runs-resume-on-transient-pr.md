## Summary

- Admitted durable runs now pick up again after a timeout or crash instead of failing on the first error. One shared rule decides this: `failed_attempt` in `services/orion-durable-runs/app/admitted_graph.py`.
- The rule sorts errors by exception type, not by error text:
  - Timeouts, transport errors and failed turns retry with backoff, within the attempt budget.
  - Programming errors and contract-validation errors (`ValueError`, pydantic `ValidationError`, `TypeError`, `LookupError`, ...) fail at once. They do not loop.
- Four workflows use it: `curiosity.investigate` (same behavior as before, plus the new fail-fast on deterministic errors), `self_sense_eval`, `self_study.reflect` and `curiosity.compaction`.
- `self_sense_eval` asks one question per node run, so each answer is saved to the checkpoint as soon as it lands. A crash or timeout resumes at the next unanswered question.
- Each question gets its share of the time budget (`timeout_sec / n`). Before this patch, one question could use the whole budget.
- Each question's turn id is a uuid5 built from values in the checkpoint. A restarted worker can therefore name and cancel a turn the dead worker left running in Hub. The id is also written on `run.started` before the turn starts, and `last_turn` is saved with every outcome.
- A question that uses up its retries, or fails deterministically, is recorded as an empty answer with its error, and the run continues. Near the deadline, the questions left over are recorded as skipped, and the answers already in hand are still published.

## Outcome moved

Failure mode: an admitted run went terminal `failed` on its first exception and threw away its progress. In the live incident, self-sense run `20261023T180730Z-0842e5` held agent-gpu2 for its whole 8840 s budget on one hung question. The `TimeoutError` then failed the run, and every answer was lost. Three more runs failed the same way at 08:37:32 on 2026-10-23, each after holding a GPU for 4 to 5.5 hours.

What happens now in the same case:
- That question's timer is about 2210 s, not 8840 s.
- Hub's own turn limit is 30 s shorter than the timer, so Hub returns first.
- If the node timer still fires, the run goes to `retrying` and keeps the answers it already has.
- If the question keeps failing, it becomes an error answer, and the other three answers are still published.

## Current architecture

- Only `curiosity.investigate` retried, through inline code in `admitted_graph.py`.
- `self_sense_eval` sent every unexpected exception straight to `failed`. It retried `SelfSenseAskFailed` through `waiting_resource` with no limit and no attempt count.
- `self_study.reflect` and `curiosity.compaction` retried every exception, including deterministic ones, up to `max_attempts`.
- `AdmissionRuntime.execute` ran a whole node under `asyncio.timeout(brief.timeout_sec)`.

## Architecture touched

All changes are inside `services/orion-durable-runs`. No change to the bus, the schemas or Hub.

- `admitted_graph.py`: `DETERMINISTIC_ERRORS`, `is_deterministic`, `failed_attempt(..., backoff=)`, `retry_wait_node`, and an `attempt_sec` argument on `retry_cannot_finish`.
- `admission_runtime.py`:
  - `execute` honors an optional `node_timeout_sec` state key.
  - `run.started` now carries `step_correlation_id` for self-sense too.
  - `_cancel_harness` also cancels the derived id of the question in flight.
- `admitted_self_sense_graph.py`: the per-question loop, a `retry_wait` node, `_give_up`, and skip-rest near the deadline.
- `self_sense_graph.py`: `ask_one`, `next_unanswered`, `question_share_sec`, `question_turn_id`, `last_turn` in the state, and `finish_detail.attempts` = the most attempts any one question took.
- `admitted_reflect_graph.py` and `curiosity_compaction_graph.py`: call `failed_attempt(backoff=False)`, because these graphs re-queue at once and have no backoff node.

## Files changed

- `services/orion-durable-runs/app/admitted_graph.py`: the shared policy and the classifier.
- `services/orion-durable-runs/app/admitted_self_sense_graph.py`: self-sense becomes durable.
- `services/orion-durable-runs/app/self_sense_graph.py`: the per-question helpers and the derivable turn id.
- `services/orion-durable-runs/app/admission_runtime.py`: the per-node timer slice, the step id on `run.started`, and cancel on recovery.
- `services/orion-durable-runs/app/admitted_reflect_graph.py`, `curiosity_compaction_graph.py`: use the shared policy.
- `services/orion-durable-runs/README.md`: the policy, the graphs left off it and why, and the self-sense durability notes.
- `services/orion-durable-runs/tests/test_admitted_graph.py`, `test_admitted_self_sense_graph.py`, `test_urgent_preemption.py`: regression tests.

## Schema / bus / API changes

- Added: graph-state keys `last_turn` (self-sense) and `node_timeout_sec` (the state passed to `execute`, never persisted). These are not bus contracts.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - `finish_detail.attempts` for self-sense is now the most attempts any one question took. Before, it was the pass count.
  - The turn `attempt` sent to Hub is now per question.
  - Self-sense question turn ids are uuid5, not uuid4. They are still valid UUIDs for `is_uuid`.
  - `last_error` gets the prefix `deterministic:` when a run fails fast.
- Compatibility notes: an in-flight self-sense checkpoint from before this patch resumes correctly. `next_unanswered` skips the answers already recorded.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Existing knobs used: `DURABLE_RUNS_RETRY_MAX_ATTEMPTS` (3), `DURABLE_RUNS_RETRY_BASE_SEC` (30) and `DURABLE_RUNS_RETRY_MAX_SEC` (300).

## Proposal-mode fields (autonomy-adjacent: runs re-acquire GPU holds on their own)

- **Capability change:**
  - A failed admitted run tries again on its own, up to 3 attempts per run, or per question for self-sense.
  - The waits between attempts back off: 30 s, then 60 s, capped at 300 s.
  - Each new attempt asks for a fresh GPU pool hold.
- **Data touched:** durable-run graph checkpoints only. Self-sense publishes partial answer rows when some questions failed; this was already the per-question contract.
- **Privacy boundary:** unchanged.
- **Trace that proves it worked:**
  - `run.retrying` events appear on the run.
  - `last_turn` and `last_error` are saved in the checkpoint.
  - `run.started.step_correlation_id` is recorded for each question.
  - `finish_detail.turns[*].error` appears for questions that were given up.
- **Dangerous failure mode:** a transient error that keeps happening makes a run hold and release GPU cards repeatedly. Two limits bound this: `max_attempts` with backoff, and `retry_cannot_finish`, which refuses a retry that cannot finish before the deadline.
- **Kill switch / rollback:**
  - Set `DURABLE_RUNS_RETRY_MAX_ATTEMPTS=1` and restart. Every transient error then fails on the first attempt, as before.
  - Or revert this PR. The checkpoints remain compatible.

## Tests run

```text
pytest services/orion-durable-runs/tests -q  (with a disposable postgres:16 container as ORION_ADMISSION_TEST_DSN)
590 passed, 1 skipped   (after the first review round)
pytest services/orion-durable-runs/tests -q  (no DSN, final head)
495 passed, 97 skipped
```

New regression tests:
- A node `TimeoutError` puts the run in `retrying` and keeps the answered questions. The resume asks only the questions not yet answered.
- Each question gets `timeout_sec / n`, and Hub's limit is lower than that.
- A deterministic `ValueError` gives up that one question after a single execute. If every question hits it, the run fails with no retry loop.
- A question that uses up its retries is recorded, and the other answers are published.
- Near the deadline, the remaining questions are skipped and the answers in hand are published.
- `question_turn_id` comes from the checkpoint and changes on every kind of replay.
- Recovery cancels the derived id of the question in flight.
- `execute` honors `node_timeout_sec` and logs the step id before the turn.
- The shared policy: transient errors retry, deterministic ones fail, budget-spent fails, and `backoff=False` does not refuse an immediate retry that fits.
- Curiosity fails a deterministic harness error without retrying.

## Evals run

```text
ORION_ADMISSION_TEST_DSN=<disposable pg> python services/orion-durable-runs/evals/hold_fairness.py      -> verdict PASS
ORION_ADMISSION_TEST_DSN=<disposable pg> python services/orion-durable-runs/evals/deploy_order_skew.py -> verdict PASS
```

## Docker/build/smoke checks

```text
Not run: no new dependencies, no Dockerfile change. Live path UNVERIFIED until deploy (not deployed, per instructions).
```

## Review findings fixed

The code-review subagent ran twice. These findings were fixed:

- Finding: the self-sense deadline check reserved time for only one question.
  - Fix: `attempt_sec = share * remaining`.
  - Evidence: `admitted_self_sense_graph.py`.
- Finding: a terminal failure threw away the answers already saved.
  - Fix: a question that uses up its budget, or fails deterministically, becomes an error answer and the run continues. Near the deadline, the rest are recorded as skipped and the answers in hand are published.
  - Evidence: `test_a_question_that_spends_its_retries_...` and `test_near_the_deadline_...`.
- Finding: reflect and compaction refused immediate retries that would fit, because the deadline check counted a backoff they never wait.
  - Fix: `failed_attempt(backoff=False)`.
  - Evidence: `test_an_immediate_retry_is_not_refused_...`.
- Finding: a worker crash during a question could not cancel that question's Hub turn.
  - Fix: the uuid5 `question_turn_id` comes from the checkpoint, and `_cancel_harness` cancels it.
  - Evidence: `test_recovery_cancels_the_self_sense_question_...`.
- Finding: Hub's turn limit lost its 30 s grace under the deadline clamp.
  - Fix: clamp the node window first, then subtract the grace.
- Finding: `attempts` in `finish_detail` changed meaning.
  - Fix: each answer records its own `attempts`, and `finish_detail` reports the most any question took.
- Finding: `retry_at` was left stale after a success, and a completed run carried a given-up question's `last_error`.
  - Fix: both are cleared.
- Finding: the README said bus transport errors retry for self-sense.
  - Fix: corrected. The runner returns those as `ok=False` results, and they are recorded as empty answers.
- Finding: the compaction evidence count was wrong.
  - Fix: recounted live. 24 runs, 0 completed, 22 ended on ValueError/ValidationError, 2 on RuntimeError, and none were JSON-parse errors.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build   # from the clean primary main checkout, after merge
```

## Risks / concerns

- Severity: medium.
  - Concern: `ValueError` now fails fast in compaction. That includes LLM-answer validation (body too long, bad refs, over budget). A fresh sample could in principle pass.
  - Mitigation: live data shows 0 of 22 such runs recovered over 3 attempts. `journal.compose` and `memory.episode_distill`, where resampling does recover, keep their own policy. The real compaction fix (the prompt or the validator) is a separate follow-up.
- Severity: low.
  - Concern: a `WorkflowDeadline` raised during a self-sense question still fails the run without publishing the partial answers. Publishing after the deadline would be refused by the guard anyway.
  - Mitigation: the deadline check now stops early, while there is still time to publish what the run has.
- Severity: low.
  - Concern: the self-sense runner turns a bus RPC timeout into a recorded empty answer, not a retry. This is the existing per-question contract and is unchanged.
- Not on this policy, on purpose: `reading.turn` (Hub's seed queue owns its retries), `journal.compose`, `memory.episode_distill`, `compactor.digest`, `orion_day.letter`, `reverie.visual` and `dream.carry`. Each already retries with backoff under its own policy. The README says why for each one.

## PR link

See the PR this file ships in.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
