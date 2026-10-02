## Summary

- The motor's "context full" guard was counting everything a turn had ever seen, and never noticed when the claude CLI compacted the conversation. Long investigations that had already compacted were still killed. It now resets to the CLI's own post-compaction size on every compaction event.
- The kill that remains (context really did outgrow the lane window, i.e. the CLI failed to compact in time) is renamed from `fcc_draft_length_ceiling_exceeded` (it was never about the draft) to `fcc_context_ceiling_exceeded`. Every consumer accepts both names.
- When the motor stops a still-working turn for any budget reason (context ceiling, time budget, stalled step, oversized line), the draft is now built from what the turn actually recorded (tool results paired with their calls, plus Orion's interim notes) under a `[Cut short - not a finished answer.]` marker, instead of the last line ("Let me check X:"). A cut-short turn with nothing recorded fails instead of shipping an empty shell.
- Finalize skips response repair for cut-short drafts, so the findings are passed through as-is instead of being rewritten into something that reads finished; the outcome molecule and the run record carry the same marked text. The governor re-attaches the marker as a backstop.
- Two cases keep the old behavior: a timeout after the CLI already reported the turn complete (the process hung on exit, so the text is the finished answer), and reading-only machine turns (they need JSON and retry on a clean `turn_error:<code>`).
- Hub's own claude bridge gets the same rebase for its "context nearly full" nudge.

## Outcome moved

Failure mode: long autonomous turns (curiosity investigations, self-sense evals, 140-253 steps, 20-47 min) killed after they had already compacted, then shipped as empty-shell answers.

Live evidence (2026-10-02, read-only prod queries):

- `analytics_source.curiosity_run_journals`, `grounding_status = fcc_draft_length_ceiling_exceeded` per day: 09-22: 8, 09-24: 12, 09-30: 1 (plus 10-01 runs in `journal_entries`; 10-01 journal rows have null status). Burst ratios from the investigation: 09-22 8/39, 09-24 15/79, 10-01 8/78.
- Drafts were 74-546 chars, so "draft length" was never the cause.
- Killed turns that had already compacted: journal bodies beginning "This session is being continued from a previous conversation that ran out of context" with grounding `fcc_draft_length_ceiling_exceeded` (e.g. turns abb0d619, 788a8625; 4 such journals 09-22/09-24).
- Empty-shell finals shipped from this path on 10-01: corr 2987c31e ("...Let me check what `outcome_from_followup` produces:") and corr 6fd11826 ("I cannot complete this investigation from the draft. The previous turn hit a context wall...").

Compaction signal evidence (not a guess):

- The CLI actually shipped in `orion-athena-harness-governor` (`/usr/local/bin/claude`, Claude Code 2.1.287): its SDK output schema declares `{"type":"system","subtype":"compact_boundary","compact_metadata":{"trigger":"manual"|"auto","pre_tokens":int,"post_tokens":int?}}` and its print-mode stream serializer emits exactly that (`subtype:"compact_boundary",session_id:q(),uuid:...,compact_metadata:...`).
- A captured live turn carries it: `orion/fcc/tests/fixtures/fcc_repeat_failure_a153451fe423.jsonl` line 41.

Why rebase rather than delete the guard: the CLI autocompacts at `HARNESS_FCC_AUTOCOMPACT_PCT_OVERRIDE` (70%) of the lane window, and provider overflow is already caught as `fcc_context_overflow`. Rebasing keeps a cheap runaway backstop for the case where compaction does not land in time, while removing every false kill on turns that did compact.

## Metric gate (context estimate + `fcc_compactions`)

1. Provenance: `budget_chars` in `orion/harness/fcc_motor.py::run_fcc_turn` = `len(prompt)` + `measure_step_payload_chars(step)` per stream event (`orion/fcc/context_budget.py`), rebased by `post_compaction_context_chars` on `compact_boundary`. Surfaced as `context_obs.accumulated_chars` / `fill_pct` on each step via `annotate_harness_step`. `fcc_compactions` = count of `compact_boundary` events in the turn.
2. Independence: not a new signal. It is the same counter with its definition changed from "lifetime total" to "estimated live context"; `fcc_compactions` is a direct count from the CLI stream. Neither feeds a model or cognition loop; they gate the kill and the pressure nudge and appear on operator step frames.
3. Theory anchor: the model's window holds only post-compaction content; the CLI's own `compact_boundary` event is the authoritative "transcript was replaced" signal (verified in the shipped CLI binary).
4. Live data: UNVERIFIED post-deploy. Pre-deploy, the lifetime total was demonstrably degenerate as a context measure: it kept growing past compactions and killed turns whose context had already been compacted (counts above). Whether live events carry `post_tokens` is UNVERIFIED: the schema marks it optional and the captured fixture is field-stripped. Post-deploy check: `docker logs orion-athena-harness-governor 2>&1 | grep fcc_context_compacted` shows `budget_chars=<before>-><after>`.
5. Existing mechanism: the CLI's autocompact (`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`) is the real context manager; this only stops the motor's guard from contradicting it. `scripts/context_mode_hooks_smoke.py::has_compact_event` already matched the same event shape.
6. Reversibility: cheap. No schema, registry or stored column; revert the commit.

## Privacy boundary

A cut-short draft is the first path that puts excerpts of raw tool output (Bash output, file reads, HTTP bodies) into user-visible text, chat history and curiosity journals. Before, tool output reached only the finalize LLM via step summaries. Each excerpt is capped at 300 chars and scrubbed of credential-shaped values (`orion/harness/cut_short.py::scrub_secrets`: `KEY=value` with credential-like names, `Bearer` tokens, URL passwords, `sk-`/`ghp_`/`AKIA`-style keys). Best effort, not a DLP. Disable: revert, or drop the tool-result entries in `TurnFindings.observe`.

## Current architecture

`run_fcc_turn` started `budget_chars = len(prompt)`, added `measure_step_payload_chars(step)` for every stream event, and killed the subprocess once it reached `max_context_chars(lane_n_ctx)`. Nothing ever subtracted. On error, `HarnessRunner` set `draft_text` to the error frame's `llm_response` (the last assistant text block) with `compliance_verdict="partial"`, and finalize/repair turned that into Orion's answer.

## Architecture touched

- Motor context accounting (`orion/harness/fcc_motor.py`, `orion/fcc/context_budget.py`).
- Runner error branch (`orion/harness/runner.py`) via new `orion/harness/cut_short.py`.
- Governor success path (`services/orion-harness-governor/app/bus_listener.py`).
- Hub claude bridge pressure nudge (`services/orion-hub/scripts/fcc_claude_bridge.py`).
- Analytics grounding-status allow-lists.

## Files changed

- `orion/fcc/context_budget.py`: `is_compact_boundary_event`, `post_compaction_context_chars`; `compact_boundary` measures 0 chars.
- `orion/harness/fcc_motor.py`: rebase on compaction, re-arm pressure nudge, `fcc_compactions` in metadata, renamed error code.
- `orion/harness/cut_short.py` (new): `TurnFindings`, `build_cut_short_draft`, `ensure_cut_short_marked`, `scrub_secrets`, `CUT_SHORT_CODES`, error-code constants.
- `orion/harness/finalize.py`: `run_harness_finalize_chain(cut_short=...)` skips response repair and passes the marked draft through.
- `orion/harness/runner.py`: records findings per step; cut-short error branch; `HarnessMotorResult.cut_short_reason`.
- `services/orion-harness-governor/app/bus_listener.py`: both codes in `_FCC_SELF_KILL_CODES`; marker re-attached after finalize.
- `services/orion-hub/scripts/fcc_claude_bridge.py`: same rebase for the nudge.
- `orion/harness/fcc_motor.py` also flags `fcc_result_seen` on timeout/stall/line-limit errors raised after the CLI's `result` event.
- `services/orion-analytics/scripts/bootstrap_analytics_roles.sql`, `services/orion-analytics/tests/*.sql`: accept `fcc_context_ceiling_exceeded` (old name kept for history).
- `services/orion-harness-governor/README.md`, `orion/curiosity/README.md`: docs.
- Tests: `orion/harness/tests/test_fcc_context_ceiling_compaction.py` (new), `test_fcc_motor_mcp.py`, `test_harness_runner.py`, `test_response_repair_gate.py`, governor `test_harness_governor_rpc.py`, `test_rpc_health_publish.py`, hub `test_fcc_claude_bridge_run.py`.

## Schema / bus / API changes

- Added: `metadata.fcc_compactions` on motor final/ceiling-error frames, `metadata.fcc_result_seen` on timeout/stall/line-limit error frames (in-process dicts, not registered schemas); `cut_short` kwarg on `run_harness_finalize_chain`; `HarnessMotorResult.cut_short_reason`.
- Removed: none.
- Renamed: motor error code / `grounding_status` `fcc_draft_length_ceiling_exceeded` -> `fcc_context_ceiling_exceeded`.
- Behavior changed: budget-cut turns carry a findings draft marked cut short, or fail if nothing was recorded.
- Compatibility notes: all consumers accept both names. `HarnessMotorResult` is an in-process dataclass. No bus channel or registry change.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=<wt> pytest orion/harness/tests orion/fcc/tests services/orion-harness-governor/tests orion/curiosity -q
  698 passed, 1 skipped, 1 failed -- test_claude_spawn::test_no_undiscovered_root_callers_of_claude_permission_argv,
  fails identically on main (services/orion-room-companion caller), unrelated
pytest orion/harness/tests/test_fcc_context_ceiling_compaction.py -q    20 passed
services/orion-hub: pytest tests -k "fcc_claude_bridge or harness"     71 passed, 1 failed --
  test_turn_orchestrator_utterance_origin::test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness,
  fails identically on main, unrelated
pytest services/orion-analytics/tests -q                                 16 passed
static gates: check_metric_lineage --gate PASS, check_definition_drift --gate PASS,
  check_scripts_dir_no_stdlib_shadow, check_async_routes_not_blocking, check_control_surface_store_parity clean
```

New regression tests (each fails on the pre-fix code):

- A stream that compacts mid-turn no longer trips (lifetime total over the ceiling, live context under it).
- A genuine runaway with no compaction still stops as `fcc_context_ceiling_exceeded`; a compaction that frees little (CLI `post_tokens`) still stops.
- The live fixture carries the real `compact_boundary` shape.
- A cut-short turn's draft carries the tool results (paired with their calls) and interim notes under the marker, not the "Let me check" line; every budget code takes this path; no findings -> failed; non-budget errors keep the old path.
- A hang after the CLI's `result` keeps the finished answer unmarked; reading-only turns keep the clean `turn_error` path.
- Finalize skips repair for cut-short drafts and the outcome molecule carries the marked text; the governor backstop keeps the marker.
- Tool output is scrubbed of credentials; long last text kept once with its paragraphs; marker match tolerates re-rendering.
- Hub bridge: compaction rebases the total and re-arms the pressure nudge.

## Evals run

```text
No eval harness for the harness motor's budget path. Gap: a replay eval over recorded live streams (count false ceiling kills before/after) would be the right one; the post-deploy SQL below is the live check.
```

## Docker/build/smoke checks

```text
Not deployed (task scope: no deploy/restart). No compose/env/Dockerfile change.
```

## Review findings fixed

Review subagent ran against `origin/main...HEAD`. All should-fix and nit findings addressed:

- Finding: a finished turn whose CLI hangs on exit would ship as "cut short".
  - Fix: motor sets `fcc_result_seen` after the `result` event; runner keeps the old partial path when set.
  - Evidence: `test_motor_flags_a_hang_after_the_cli_result`, `test_hang_after_result_keeps_the_answer_unmarked`.
- Finding: reading-only (world-pulse) turns would hit a finalize JSON crash and lose the `fcc_timeout` code.
  - Fix: cut-short path skipped for `reading_only`.
  - Evidence: `test_reading_only_turn_keeps_the_clean_turn_error_path`.
- Finding: repair could rewrite findings into a synthesized answer; outcome molecule carried unmarked text.
  - Fix: `run_harness_finalize_chain(cut_short=True)` skips repair and passes the marked draft through; governor marker kept as backstop.
  - Evidence: `test_cut_short_chain_skips_repair_and_keeps_marked_findings`.
- Finding: raw tool output (possible secrets) now reaches journals/chat.
  - Fix: 300-char excerpts + `scrub_secrets`; boundary named above.
  - Evidence: `test_tool_output_is_scrubbed_of_credentials`.
- Finding: hub bridge rebase untested.
  - Fix: `test_run_turn_rebases_context_on_cli_compaction`.
- Finding: PR report untracked; no metric-gate notes; `post_tokens` path unverified.
  - Fix: report committed with gate notes; UNVERIFIED stated.
- Nits: post_tokens asymmetry documented; long last text no longer duplicated and truncates with an ellipsis; assistant notes keep line breaks; marker matched by tolerant regex; legacy-code constant now used by `cut_short.py`/`bus_listener.py`; analytics re-apply command listed below.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

Hub (`orion-hub`) picks up the bridge change on its next rebuild; not urgent (nudge only). Analytics (so the new code is not bucketed as `other`), from repo root, per `services/orion-analytics/README.md`:

```bash
set -a; . services/orion-analytics/.env; set +a; docker exec -i orion-athena-sql-db psql -U postgres -d "$ORION_ANALYTICS_DATABASE" -v analytics_transformer_password="$ORION_ANALYTICS_DBT_PASSWORD" -v analytics_reader_password="$ORION_ANALYTICS_READER_PASSWORD" < services/orion-analytics/scripts/bootstrap_analytics_roles.sql
```

Verify after deploy:

```sql
-- Ceiling kills per day: should drop to ~0, and none on turns that compacted.
select journaled_at::date, grounding_status, count(*)
from analytics_source.curiosity_run_journals
where grounding_status in ('fcc_draft_length_ceiling_exceeded','fcc_context_ceiling_exceeded')
group by 1,2 order by 1;

select created_at::date, count(*) as killed,
       sum((body ilike '%continued from a previous conversation%')::int) as killed_after_compacting
from journal_entries
where body ~ 'grounding: fcc_(draft_length|context)_ceiling_exceeded'
group by 1 order by 1;

-- Cut-short finals must carry the marker and real findings, never a lone lead-in.
select created_at, left(correlation_id::text,8), length(body),
       body like '[Cut short - not a finished answer.]%' as marked,
       left(regexp_replace(body,'\s+',' ','g'),200)
from journal_entries
where created_at > now() - interval '7 days'
  and body ~ 'grounding: fcc_(context_ceiling_exceeded|timeout|stream_stalled|stream_line_limit)'
order by created_at desc;
```

Also: `docker logs orion-athena-harness-governor 2>&1 | grep fcc_context_compacted` shows rebases happening live.

## Risks / concerns

- Severity: low. Concern: without `post_tokens` the rebase falls back to the prompt size and slightly undercounts (the CLI summary may not be streamed). Mitigation: CLI autocompact still runs at 70%; provider overflow is still caught as `fcc_context_overflow`.
- Severity: low. Concern: chat turns that hit `fcc_timeout` now get a marked findings draft instead of the bare last text, and skip response repair. Mitigation: a long in-progress write-up is kept whole (up to 4000 chars) at the end; a hang after the CLI's `result` keeps the old path.
- Severity: medium. Concern: credential scrub is best-effort regex; unusual secret shapes in tool output could reach a journal. Mitigation: 300-char excerpts; boundary documented; easy to drop tool-result entries if it proves risky.
- Severity: low. Concern: analytics buckets the new code as `other` until the role bootstrap SQL is re-applied. Mitigation: listed under restart.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
