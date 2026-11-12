## Summary

- Reading-queue calls (`mcp__orion-reading__*`) hold back a turn's working notes only when their result could carry outside text. A bare queue receipt for a chat-made request no longer does.
- "Clean receipt" is an allowlist: known fields only, empty `summary`/`error`, and a `request` that a chat turn made (no `title`, not `world_pulse`, not a re-entry). Non-JSON, unknown fields, or an unpaired call hold back.
- Note lines carry no model-written time. Orion guessed the clock 1 h off live; a written time is removed and each carried batch keeps the harness's own "turn ending" stamp.
- `dig turn-steps` hides reading results with the same check.
- The baseline eval reports `live_*` held-back counts from the `working_notes` cockpit rows (its replay cannot check receipts cut to 400 chars).

## Outcome moved

Live 2026-11-11 20:30Z: 15 clean note lines were held back only because Orion queued a spec with `recommend_reading` (receipt `summary: ""`). That receipt now reads clean (checked against the stored live receipt). Before deploy, `recommend_reading` caused 6 of 16 held-back turns in the 14-day baseline. Live 2026-11-12 03:10Z: notes said 02:09Z for steps run 03:05-03:10Z; lines now carry no time.

## Current architecture

`orion/harness/working_notes.py` tainted a turn at call time for any MCP tool outside orion-introspect/gitnexus, so every reading call held notes back. Note lines required `- <HH:MMZ> | ...`.

## Architecture touched

Harness governor only (`orion/harness`), plus `orion/dig/turn_steps.py` and the baseline eval. No bus, schema, or env change.

## Files changed

- `orion/harness/working_notes.py`: `OutsideContent` tracker (pairs tool_use with tool_result), `outside_result` + receipt allowlist, time-optional `ENTRY_RE`, time stripped in `parse_entries`, prompt shape without time, `_blocks` reads content on raw too.
- `orion/harness/runner.py`: uses the tracker.
- `orion/dig/turn_steps.py`: hides reading results that are not clean; `result_text` moved to working_notes.
- `orion/harness/evals/turn_repeat_baseline.py`: `measure_live` + `LIVE_SQL`.
- Tests: `orion/harness/tests/test_working_notes.py`, `orion/dig/tests/test_turn_steps.py`.
- Docs: governor README "Turn working notes", spec implementation notes.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: reading calls taint by result; notes have no time field. Old carried lines with times still render (regex accepts both) and age out in 24 h.
- Compatibility notes: none.

## Env/config changes

- None. `.env_example` not changed; no local `.env` sync needed.

## Tests run

```text
pytest orion/harness/tests orion/dig/tests/test_turn_steps.py orion/hub/tests/test_cockpit_working_notes_hop.py
793 passed
pytest orion/dig/tests  -> 2 failures, same on main (dig help 42 chars over 3250 cap after #3193 + #3187 merged together; fixed by open PR #3212)
```

## Evals run

```text
python -m orion.harness.evals.turn_repeat_baseline --days 2
tool_using_turns: 3, tainted_share: 0.333 (replay)
live_turns_with_notes: 3, live_held_back: 1 (recommend_reading, pre-fix), live_carried: 2
```

## Docker/build/smoke checks

```text
Not run: pure Python in the governor image path; live check is the next working_notes row after deploy.
```

## Review findings fixed

- Finding: a `reading_status` receipt with empty summary/error could carry a world-pulse feed headline in `request.title` (or a page-chosen re-entry link).
  - Fix: allowlist receipt check (known fields, chat-made request, no title, no re-entry).
  - Evidence: `test_reading_calls_taint_by_result` world-pulse/title/re-entry/unknown-field cases.
- Finding: the baseline replay cannot check receipts cut to 400 chars, so it cannot show this fix.
  - Fix: `live_*` counts from cockpit `working_notes` rows; docstring says the replay over-counts.
  - Evidence: `test_baseline_live_counts_what_the_harness_decided`; live run above.
- Finding: a reading call without an id, or content on `raw` without `message`, failed open.
  - Fix: no-id reading call holds back at once; `_blocks` mirrors `reading_receipts._content_blocks`.
  - Evidence: `test_reading_result_is_matched_by_tool_use_id`.

## Restart required

```bash
git pull   # in the primary main checkout; the post-pull rebuild restarts orion-harness-governor
```

## Risks / concerns

- Severity: low. Concern: `mcp__orion-introspect__reading_results` returns read summaries and is still on the inside list, which disagrees with this rule. Mitigation: open policy question for Juniper; unchanged here.
- Severity: low. Concern: dig-kit CI on this PR fails until #3212 merges (pre-existing on main). Mitigation: merge #3212 first.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
