## Summary

Reading turns kept failing because the model answered in prose, or wrote broken JSON, instead of the JSON object the reading asks for. Now the motor catches that answer before it leaves and asks the model once more, on the same session, for only the JSON object.

- A reading turn whose final answer would fail finalize's JSON check is held back. The motor resumes the Claude session once (`--resume <claude_session_id>`) with a short note: reply with only the JSON object the task asked for, nothing else.
- This reuses the undeclared-tool-call resume from PR #3025. A turn gets at most one resume in total, for either reason. The same deadline floor (`RESUME_MIN_BUDGET_SEC`) applies.
- The JSON check is the one finalize already applies. It moved to `orion/harness/structured_output.py`, so the motor and finalize share one parser and cannot disagree.
- Output-cap stitcher fix: a continuation that re-opens a code fence in the middle of the answer no longer puts the fence inside a JSON string (reading `a17a5397`).

## Outcome moved

Readings that failed with `turn_error:Could not parse JSON object from LLM text` get one retry for the JSON object inside the same turn. The run no longer fails outright and does not spend a durable retry. A metadata stamp (`fcc_reading_json_resume`) records each resume.

### Failure classification (Postgres, 2026-10-31 08:00 UTC to 2026-11-03)

There were 43 failed reading runs. 19 of them were `Could not parse JSON object`. The other 24: 17 `RpcNoListener`, 3 `empty_imperative`, 2 `finalize_reply_deadline`, 2 `stance_react_failed` (a 502 from the stance model). For 15 of the 19 parse failures I read the full final text from the governor's CLI transcripts. For the other 4 I used the 200-character preview stored in the database.

| Class | Count | Runs (turn correlation prefix) | Handled by |
|---|---|---|---|
| Pure prose, no JSON object at all | 10 | a392a064, ca7e0835, 39588ec6 (stage 1), ed0adaa9, a1d13681, d7b4197d ("Now writing the Stage 2 output." then stopped), 55223448 (prose + mangled `<functionfunction=WebFetch>` call), 80683a92, 69c60f9d (stage 1, preview only), 0354890d ("Stance pass —", preview only) | this PR: JSON resume |
| JSON object with a syntax error written by the model | 5 | 572a5ff5 (single-quoted string in a list), cea9591d (`why_note":` with no opening quote), 2b9900a9 (string with no closing quote after a continuation), 15b25539 (continuation that called its own fragment "malformed" and rewrote part of it), cbcd030b (fenced, preview only, cause unknown) | this PR: JSON resume |
| Head of the answer lost to the output cap | 3 | c5df42da, 2b8585f3, 832767bd (preview only, begins mid-object) | already fixed by PR #3029 (stitcher). Both transcripts parse when replayed through it. |
| Stitcher bug: continuation re-opened a code fence | 1 | a17a5397 | this PR: stitcher fix (replay now parses) |
| Parser bug in `orion/core/llm_json.py` (a valid object it missed) | 0 | I scanned every draft for any decodable JSON object. The parser misses none. | n/a |

All 19 failures hit Stage 2 except 2 stage-1 prose answers.

### Where the check lives

The parse contract is in the governor's finalize. It sets `preserve_structured_output=bool(request.reading_only)` (services/orion-harness-governor/app/bus_listener.py), then calls `canonicalize_structured_output`, which raises on non-JSON. Every `reading_only` turn carries this contract. Both producers (`services/orion-hub/scripts/reading_turn_listener.py` for stage 1 and stage 2, and `orion/evals/model_replay/extract.py`) parse the result as JSON. So the motor gates on `reading_only` and imports the same function. It does not have its own copy of the rule.

## Current architecture

`reading_turn_listener` runs a unified turn with `reading_only=True`. The governor's FCC motor spawns `claude -p` and yields the final answer. Finalize then canonicalizes the answer as JSON or raises, and the durable run fails with `turn_error:Could not parse ...`. PR #3025 already held a final that ended on an undeclared tool call and resumed it once. Prose and broken JSON went straight through to the failure.

## Architecture touched

- `orion/harness/fcc_motor.py`: `reading_resume_need()` decides why a resume is needed: an undeclared tool first, then not-JSON. The spawn path holds the final for either reason and resumes once with the matching note.
- `orion/harness/structured_output.py` (new): `canonicalize_structured_output` (moved here from finalize) and `is_structured_output`.
- `orion/fcc/output_cap_continuation.py`: strips a fence opener at the top of a continuation while the head is still inside its own fence.

## Files changed

- `orion/harness/structured_output.py`: the one JSON check for reading output, shared by the motor and finalize.
- `orion/harness/finalize.py`: re-exports `canonicalize_structured_output` from the new module.
- `orion/harness/fcc_motor.py`: JSON-only resume, `JSON_ONLY_NOTE`, `reading_resume_need`, metadata `fcc_reading_json_resume`.
- `orion/fcc/output_cap_continuation.py`: drops a re-opened fence in a continuation.
- `orion/harness/tests/test_fcc_reading_json_resume.py` (new): regression tests built from trimmed real failure texts.
- `orion/harness/tests/test_fcc_motor_output_cap_continuation.py`: re-opened-fence regression tests (a17a5397).
- `.github/workflows/orion-fcc-warm-pool-tests.yml`: runs the new test file; path filter for `structured_output.py`.
- `.github/workflows/orion-durable-runs-tests.yml`, `.github/workflows/orion-gpu-pool-tests.yml`: path filter for `structured_output.py` (next to `finalize.py`).

## Schema / bus / API changes

- Added: final-frame metadata key `fcc_reading_json_resume` = `{session_id, held_text}` (first 500 chars of the held answer).
- Removed: none.
- Renamed: none. `canonicalize_structured_output` is still importable from `orion.harness.finalize`.
- Behavior changed: a reading-only final that fails the JSON check is resumed once instead of being returned.
- Compatibility notes: `fcc_undeclared_tool_resume` is unchanged for the tool case.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest orion/harness/tests orion/world_pulse_read orion/fcc -q
  1 failed, 987 passed, 1 skipped
  (the failure is test_no_undiscovered_root_callers_of_claude_permission_argv, which already fails on main)
.venv/bin/python -m pytest services/orion-harness-governor/tests -q      96 passed
.venv/bin/python -m pytest services/orion-hub/evals/test_reading_handoff_eval.py -q   11 passed
```

Mutation checks:
- I removed the not-JSON branch of `reading_resume_need` and the stitcher's fence strip. 15 tests failed. I restored both.
- I widened the fence regex back to `[a-zA-Z]*` and removed the middle-piece strip. The closing-fence test and the three-piece test failed. I restored both.

Replay check: I ran the 15 real transcripts through the current stitcher plus the parser. Before the stitcher fix, 2 parsed (both PR #3029 cases). After it, 3 parsed (+a17a5397). The remaining 12 are the prose and broken-JSON cases that the resume now covers.

## Evals run

```text
services/orion-hub/evals/test_reading_handoff_eval.py   11 passed (finalize's structured-output canonicalization, through the re-export)
```

No live eval has run on the resume. It needs a deployed governor and a failing reading.

## Docker/build/smoke checks

```text
Not run. Pure Python change in the governor's motor and finalize. No deploy (as instructed).
```

## Review findings fixed

A subagent ran the code review on commit 9f8132061.

- Finding: the stitcher's fence regex also matched a bare ```. That bare line is often the answer's own CLOSING fence, when the cut lands just before it. The bug would hit chat code blocks too.
  - Fix: strip only an opener that has a language tag (`[a-zA-Z]+`).
  - Evidence: `test_a_bare_closing_fence_at_the_top_of_a_continuation_is_kept`
- Finding: with two or more continuations, a re-opened fence in a middle piece stayed in the draft and threw off the fence count.
  - Fix: `_drop_reopened_fence` also runs when a middle piece joins the head.
  - Evidence: `test_a_reopened_fence_in_a_middle_piece_is_dropped_too`
- Finding: the warm-pool workflow did not trigger on changes to `orion/core/llm_json.py`, which now decides whether a resume runs.
  - Fix: added that path to the workflow's PR and push path lists.
  - Evidence: workflow diff.
- Finding: a skipped resume (no session id, or too close to the deadline) left no trace.
  - Fix: the yielded final now carries `fcc_reading_resume_skipped = {reason, need}`, and a log line is written.
  - Evidence: `test_a_skipped_json_resume_is_stamped`, `test_no_json_resume_without_a_session_id`
- Finding: when the MCP config render failed on a JSON resume, the log line still named the tool cause.
  - Fix: renamed it to `fcc_reading_resume_mcp_failed reason=...`.
  - Evidence: code.
- Finding (test gap): no test covered an error frame from the resumed run.
  - Fix: added `test_an_error_from_the_json_resume_run_carries_the_record`.
  - Evidence: the test passes.

Not fixed. These are logged as concerns below.
- A text tool call to a DECLARED tool wrapped in JSON-looking braces passes the JSON check.
- The parser accepts any JSON object, not just the reader's schema.
- Older #3025 issues: the resumed run's metadata carries only the second run's usage, and `_record_session_end` is not called on the resumed run.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

The post-pull rebuild redeploys orion-harness-governor, the service that runs the motor.

## Risks / concerns

- Severity: low. Concern: each resume costs one more model call on a turn that would otherwise have failed. Mitigation: at most one resume per turn, and only when the deadline has at least `RESUME_MIN_BUDGET_SEC` left.
- Severity: low. Concern: the JSON check accepts any JSON object, the same as finalize. A final that is a text call to a declared tool, or prose quoting a small `{...}`, passes with no resume and then fails in the reading consumer. Mitigation: none in this PR. A follow-up could require a reader key such as `summary`.
- Severity: low. Concern (from #3025, now also reached on the JSON path): the resumed final's metadata carries only the second run's usage, so the first run's token/GPU use is missing from accounting. On a durable retry with `session_plan`, the resumed run does not record its own session end. Mitigation: follow-up. Merge the first run's usage into the metadata, and call `_record_session_end` on the resumed final frame.
- Severity: low. Concern: the model may answer the JSON note with prose again. Mitigation: that answer is final and the reading fails as it does today. Watch `fcc_reading_json_resume` and the `fcc_reading_json_resume` log line to measure the success rate. UNVERIFIED live.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
