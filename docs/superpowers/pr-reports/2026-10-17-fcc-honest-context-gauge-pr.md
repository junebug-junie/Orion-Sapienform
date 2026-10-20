## Summary

- The context gauge in the FCC trace now uses the provider's own token count. It used to count only what streamed past it, so it never saw the CLI's ~24k tokens of system prompt and tool schemas. A live turn on 2026-10-17 showed "18% fill" and then compacted.
- The warn and critical lines now sit below the point where the CLI really compacts, so they fire before compaction, not after.
- A compaction whose message has no `Summary:` section is flagged: a log line, a "⚠ compaction lost the task" trace step, and `fcc_degenerate_compactions` in the turn's metadata. This happened in 9 of 22 stored compactions, and in 7 of the last 9.
- The compaction message is never taken as Orion's draft answer. Stored turns that died after compacting had drafts that began "This session is being continued...".
- More working room: `CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384` and an autocompact percent of 90 (was 70). On the 64k lane, compaction moves from about 32k to about 36k tokens. Working room goes from about 8k to about 12k tokens.

## Outcome moved

- The trace can no longer show "18% fill" on a turn the CLI is about to compact.
- A compaction that lost the task is now visible, not silent. It was common (9 of 22) and nobody had seen it.
- Each turn on the 64k lane gets about 50% more room before compaction.

## Current architecture

`run_fcc_turn` (governor) and `run_turn` (hub) each kept `budget_chars` = prompt length + every streamed step's chars. On `compact_boundary` it was rebased. The gauge, the warn/critical labels and the pressure nudge all read it against the full window. The CLI itself compacts at a lower line, set by its output reserve and the percent setting. The compaction message was treated like any other text.

## Architecture touched

- `orion/fcc/context_budget.py`: new `observed_context_chars`, `autocompact_trigger_tokens`/`_chars`, `fcc_max_output_tokens`, `compact_summary_text`, `is_degenerate_compact_summary`, `is_compact_summary_draft` and `build_degenerate_compaction_step`. The pressure and risk lines are capped below the compaction line. `CLAUDE_CODE_MAX_OUTPUT_TOKENS` is set for the subprocess.
- `orion/harness/fcc_motor.py`, `services/orion-hub/scripts/fcc_claude_bridge.py`: the same loop change in both. Re-anchor on usage, flag a degenerate compaction once per compaction, and keep the summary out of the draft.
- `services/orion-hub/static/js/agent-claude-trace.js`: renders the new step.

## Files changed

- `orion/fcc/context_budget.py`: the helpers above.
- `orion/harness/fcc_motor.py`: wiring, plus the `fcc_degenerate_compactions` metadata field. The pct default in code goes from 70 to 90.
- `services/orion-hub/scripts/fcc_claude_bridge.py`: the same wiring. Passes `HUB_AGENT_CLAUDE_MAX_OUTPUT_TOKENS`. The pct default goes from 70 to 90.
- `services/orion-hub/static/js/agent-claude-trace.js`: the "⚠ compaction lost the task" label.
- `services/orion-hub/app/settings.py`: new `HUB_AGENT_CLAUDE_MAX_OUTPUT_TOKENS`. The pct default goes from 70 to 90.
- `services/{orion-hub,orion-harness-governor}/{.env_example,docker-compose.yml}`: the new key and the new pct.
- `services/orion-harness-governor/README.md`: documents the gauge, the compaction line and the flag.
- Tests: `orion/fcc/tests/test_context_budget.py`, `orion/harness/tests/test_fcc_context_ceiling_compaction.py`, `services/orion-hub/tests/test_fcc_claude_bridge_run.py`, `services/orion-hub/tests/test_agent_claude_trace_js.py`.

## Schema / bus / API changes

- Added: the trace step type `context_compaction_degenerate` and the turn metadata field `fcc_degenerate_compactions`. Both are additive and live in the stream-step dicts. No registered schema changes.
- Removed / Renamed: none.
- Behavior changed: the `context_obs.fill_pct` and `risk` values now include the CLI overhead when usage is present. A compaction message is never the draft.
- Compatibility notes: none. These are new optional keys.

## Env/config changes

- Added keys: `HARNESS_FCC_MAX_OUTPUT_TOKENS=16384` (governor) and `HUB_AGENT_CLAUDE_MAX_OUTPUT_TOKENS=16384` (hub).
- Changed values: `HARNESS_FCC_AUTOCOMPACT_PCT_OVERRIDE` and `HUB_AGENT_CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` go from 70 to 90.
- `.env_example` updated: yes, both services.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes for the new keys. The two pct values were updated in local `.env` by hand, because the sync script does not overwrite keys whose values differ.
- Skipped keys requiring operator action: none.

## Metric gate (context fill)

1. Provenance: the assistant event's `message.usage`, which the CLI streams from the FCC proxy's `message_start`. That is the same usage the CLI compacts on.
2. Independence: it replaces the char estimate for the same quantity (live context). It is not a new signal. `max()` with the old estimate keeps it from reading lower than before.
3. Theory anchor: the CLI decides to compact from the provider's reported usage, so this is the number that predicts compaction.
4. Live data: **UNVERIFIED**. No stored trace keeps per-step `usage` (`harness_turn_trace` stores no raw events), and reading the governor's CLI transcripts was refused. If usage is absent, the gauge behaves exactly as before.
5. Existing mechanism: `extract_result_output_tokens` reads the end-of-turn `result.usage` (confirmed live on 2026-07-24). Nothing read the per-step usage.
6. Reversibility: cheap. It is one function and two call sites, with no schema or storage.

The compaction-line model `min((window - out) * pct, window - out - 13000)` fits both live clusters of 154 auto-compactions (about 32k on 64k, about 76k on 128k). It has not been probed at the new values (16384 output, 90%). It is only trusted where it lands at 40% or more of the window. Below that, for example the 32768 lane, the old percent line applies.

## Tests run

```text
PYTHONPATH=<wt> .venv/bin/python -m pytest -q orion/fcc orion/harness
  725 passed, 1 skipped, 1 failed (test_no_undiscovered_root_callers_of_claude_permission_argv: also fails on main, unrelated files)
cd services/orion-hub && pytest -q tests/test_fcc_claude_bridge_run.py tests/test_agent_claude_trace_js.py
  13 passed
hub -k "fcc or claude or agent or settings": 2 failures, both also fail on main
```

## Evals run

```text
No eval harness covers FCC context accounting. The replay test
test_replay_2026_10_17_gauge_is_honest_and_lost_task_is_flagged plays the live incident's shape
(hidden overhead, 12k-char tool result, compaction, refusal summary, result echo).
The degenerate-summary test uses openings copied from stored harness_turn_trace rows.
Follow-up: a periodic eval counting fcc_degenerate_compactions per day once this is live.
```

## Docker/build/smoke checks

```text
check_env_key_single_source: OK
check_settings_defaults orion-hub --example-drift: OK
check_definition_drift --gate: PASS
check_metric_lineage --gate: PASS
check_service_env_compose_parity: N/A (env_file services)
git diff --check: clean
No docker build: no dependency or Dockerfile change.
```

## Review findings fixed

- Finding: on the 32768 lane the modelled compaction line (3384 tokens) is below the CLI overhead, so every step read "critical".
  - Fix: the line is trusted only at 40% or more of the window. Otherwise the code falls back to the percent line.
  - Evidence: `test_shipped_defaults_give_the_64k_lane_more_room` asserts `autocompact_trigger_tokens(32768) is None`.
- Finding: `fcc_degenerate_compactions` counted the model's echo as a second loss.
  - Fix: at most once per compaction, in both loops.
  - Evidence: the replay test expects 1 flag for 1 compaction.
- Finding: a `result` event that repeats the summary could still become the draft.
  - Fix: `is_compact_summary_draft` checks the extracted text on every event type.
  - Evidence: the replay test now ends with a `result` echo, and the draft is not the summary.
- Finding: summing input and cache tokens could double-count under proxy semantics, then trip the ceiling kill.
  - Fix: the prompt side is now `max(input, cache_read + cache_creation)`.
  - Evidence: `test_observed_context_reads_provider_usage` covers both shapes.
- Not fixed (nits): thinking inside one API message is counted at the next call (one call of lag, then it self-corrects). The hub gauge reads env while its CLI reads settings; compose sets both.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the clean primary `main` checkout after merge.

## Risks / concerns

- Severity: medium.
  - Concern: with a 16384 output cap, any single step that wants more than 16k output tokens (mostly long Qwen thinking) is cut off. Earlier data: 577 of 18,088 steps since 10-01 were over 8,192. How many were over 16,384 was not measured.
  - Mitigation: set `HARNESS_FCC_MAX_OUTPUT_TOKENS` back to 32000 to undo it.
- Severity: low.
  - Concern: percent 90 compacts later. The 13k margin and the output reserve still keep it below the worker's hard limit.
  - Mitigation: set the percent back to 70.
- Severity: low.
  - Concern: per-step usage is UNVERIFIED live (see the metric gate).
  - Mitigation: if usage is absent, the gauge falls back to the old estimate.
- Not done: trimming the ~24k fixed overhead. It needs a `claude -p /context` probe inside the governor.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2744

🤖 Generated with [Claude Code](https://claude.com/claude-code)
