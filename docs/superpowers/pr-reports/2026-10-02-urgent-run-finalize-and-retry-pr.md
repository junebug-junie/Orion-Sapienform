## Summary

- When an urgent run hits its time limit, Orion's answer is no longer thrown away. The investigation stops early enough for the harness to finish the answer while the run still holds its GPU. If finishing still fails or runs long, Hub sends Juniper Orion's working draft, labelled UNFINISHED.
- A retry that cannot finish before the run's deadline is skipped. The run fails at once instead of restarting from zero and being killed partway through.
- Every held curiosity turn (urgent and ordinary) now keeps `HUB_CURIOSITY_HELD_TURN_FINALIZE_RESERVE_SEC` (330 s) out of the motor's budget. It also tells the harness when Hub stops waiting (`HarnessRunRequestV1.reply_budget_sec`).
- harness-governor caps finalize at the time Hub has left. If finalize runs over, the governor replies with the draft rather than finishing after nobody is listening.

## Outcome moved

Incident: urgent run `a153451fe423`, 2026-10-01. Attempt 1 ran Orion's motor for 841 s across 107 steps and produced a 7,506-char draft. At 21:41:03 durable-runs stopped the attempt and released the GPU hold, because the 900 s turn limit was up. The harness had been given the same 900 s, so its finalize calls only started after the hold was gone. Both failed with `gpu_pool_unavailable:hold_not_granted:released` and the draft was discarded. Attempt 2 started from zero with about 290 s left and was killed by `workflow_deadline`. Juniper received only "investigation failed".

After this patch the same timeline goes like this:
- The motor stops at about 570 s of the turn.
- Finalize runs under the hold.
- If finalize still fails, the urgent report carries Orion's draft, marked UNFINISHED.
- No doomed retry is started.

## Current architecture

Who cut the turn at 900 s: durable-runs' `admission_runtime.execute`, using `asyncio.timeout(brief.timeout_sec)`. It cancels the harness with reason `durable_attempt_stopped` and then releases the hold with `attempt_failed`. Three timers were set to the same 900 s:
- the durable-runs attempt timer,
- Hub's `asyncio.wait_for`,
- the harness motor's `inference_timeout_sec`.

That left no time for anything after the motor. Finalize normally takes 91 s at the median and 261 s at the 90th percentile (63 turns, 2026-09-30..10-02). Stance and recall before the motor took about 60 s.

Why the resume found no prior hops (`curiosity_resume_no_prior_hops`): the urgent prompt (`orion/curiosity/urgent_prompt.py`) never asks Orion to write `:Hop` nodes. Resume therefore has nothing to read on an urgent retry, by design. Fixing the finalize ordering is the root-cause fix. Carrying the draft forward between attempts was not built.

## Architecture touched

- **Hub** (`curiosity_investigation._generate` / `_turn_result_for`):
  - gives the motor's budget and the reply deadline,
  - counts the turn limit from when the request arrives,
  - salvages `partial_draft` for urgent turns.
- **Hub** (`orion/hub/turn_orchestrator.py`):
  - turns the reply deadline into `reply_budget_sec` at the moment the harness request is built,
  - clamps the motor budget to it,
  - lets urgent error frames carry up to 8,000 chars of draft.
- **Schema**: added optional `HarnessRunRequestV1.reply_budget_sec`.
- **harness-governor** (`bus_listener`): `run_bounded_finalize` caps the finalize stage.
- **durable-runs**:
  - `attempt_timeout_sec` clamps the turn limit sent to Hub to the run's deadline, the same clamp its own timer uses,
  - `retry_cannot_finish` skips doomed retries,
  - a salvaged draft is not journaled,
  - `finish_detail` carries `draft_salvaged`.
- **urgent_report**: an UNFINISHED line is added next to Orion's words, so a real verdict still leads the report.

## Files changed

- `services/orion-hub/scripts/curiosity_investigation.py`: motor budget, reply deadline, limit counted from receipt, urgent draft salvage.
- `orion/hub/turn_orchestrator.py`: `_held_turn_budgets`, urgent partial-draft cap.
- `orion/schemas/harness_finalize.py`: `reply_budget_sec`.
- `services/orion-harness-governor/app/bus_listener.py`: bounded finalize.
- `services/orion-durable-runs/app/graph.py`: clamped turn limit, journal skip, `draft_salvaged` in the finish detail.
- `services/orion-durable-runs/app/admitted_graph.py`: retry guard.
- `services/orion-hub/scripts/urgent_report.py`: UNFINISHED line.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`, `scripts/main.py`: new setting.
- Tests:
  - `services/orion-hub/tests/test_curiosity_held_turn_finalize.py` (new)
  - `services/orion-harness-governor/tests/test_finalize_reply_budget.py` (new)
  - `services/orion-durable-runs/tests/test_urgent_runs.py` (extended)
  - two Hub tests updated because the limit is now counted from receipt
- Eval: `services/orion-hub/evals/run_urgent_report_eval.py` gained two salvaged-draft cases.

## Schema / bus / API changes

- Added: `HarnessRunRequestV1.reply_budget_sec` (optional, default None). The urgent finish detail also gains `draft_salvaged` and `salvaged_from_error`, present only when a draft was salvaged.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - the motor budget for held turns is smaller,
  - the turn limit sent to Hub is clamped to the run's deadline,
  - retries that cannot fit before the deadline are skipped.
- Compatibility notes: the model has no `extra="forbid"`, so it can be deployed in either order.
  - New Hub with old governor: the new field is ignored, and the motor still gets its reserve.
  - Old Hub with new governor: the field is None, so finalize is unbounded, as before.

## Env/config changes

- Added keys: `HUB_CURIOSITY_HELD_TURN_FINALIZE_RESERVE_SEC=330`.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`services/orion-hub/.env` now has the key at 330).
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-durable-runs: PYTHONPATH=. pytest tests -q          -> 253 passed, 71 skipped
services/orion-harness-governor: pytest tests -q                    -> 62 passed
services/orion-hub: pytest tests -k "curiosity or urgent or turn_orchestrator"
                                                                    -> 1 failed (pre-existing), rest passed
  test_turn_orchestrator_utterance_origin::...mind_appraisal... fails on HEAD as well (role-teach splice), unrelated
orion/harness/tests                                                 -> 396 passed
orion/hub/tests                                                     -> 22 passed
scripts/check_env_template_parity.py                                -> PASS
New regression tests checked against HEAD code: all fail without the fix (durable 2, governor 4, hub 8).
```

## Evals run

```text
python services/orion-hub/evals/run_urgent_report_eval.py -> 10 cases, VERDICT: PASS
  (new: salvaged_draft_no_report, salvaged_draft_with_report)
```

## Docker/build/smoke checks

```text
Not run: per instructions, no deploy or restart. The live path is UNVERIFIED until the next urgent run.
```

## Review findings fixed

- Finding: a salvaged draft was journaled as if it were a finished investigation, even though it never went through response repair.
  - Fix: the `journal` node skips when `debug.draft_salvaged` is set.
  - Evidence: `test_salvaged_draft_is_not_journaled_as_a_finished_investigation`.
- Finding: Hub's deadline was based on `brief.timeout_sec`, while durable-runs clamps the attempt to the run's deadline.
  - Fix: `attempt_timeout_sec` sends Hub the clamped limit.
  - Evidence: `test_turn_limit_sent_to_hub_is_clamped_to_the_run_deadline`.
- Finding: durable-runs' timer starts before Hub's `_generate` does (hold fence, prompt read).
  - Fix: Hub counts the turn limit from when the request arrives.
  - Evidence: `test_held_turn_limit_counts_from_receipt`.
- Finding: a context-overflow draft could be salvaged.
  - Fix: `salvage_urgent_draft` rejects overflow frames and overflow text.
  - Evidence: `test_context_overflow_draft_is_never_salvaged`.
- Finding: the `.env_example` text "0 = old behaviour" was wrong.
  - Fix: the comment now says what 0 actually does.
- Finding: a reserve larger than the turn failed silently.
  - Fix: a warning is logged.
- Finding: the failure path for a finalize cut by the reply budget was undocumented.
  - Fix: docstring added. That path does not emit the chain's own failure artifacts; the log line and `grounding_status` are the trace.
- Finding: a stray blank line.
  - Fix: removed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Any order works.

## Risks / concerns

- Severity: medium.
  - Concern: the urgent retry now rarely fires. A 900 s turn inside a 1,200 s deadline means any attempt that fails after about 290 s gets no retry. Retries now only cover failures that happen early.
  - Mitigation: intended. A retry starts from zero (urgent runs write no hops) and could not finish anyway.
- Severity: medium.
  - Concern: the motor loses 330 s on every held turn. On urgent runs that is about 37% of the turn. On ordinary runs (8,840 s) it hardly matters.
  - Mitigation: the reserve is a setting. A smaller value buys more investigation time at the cost of more UNFINISHED drafts.
- Severity: low.
  - Concern: a finalize cut by the reply budget skips the chain's system_error, closure and outcome artifacts.
  - Mitigation: the log line and `grounding_status=finalize_reply_deadline...` remain.
- Severity: low.
  - Concern: attempt 2 of an urgent run still cannot see an IncidentReport that attempt 1 already wrote.
  - Mitigation: none in this patch; possible follow-up.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2469

🤖 Generated with [Claude Code](https://claude.com/claude-code)
