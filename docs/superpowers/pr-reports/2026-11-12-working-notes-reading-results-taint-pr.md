## Summary

- Orion's `reading_results` tool now counts as outside content when it returns any read page.
- Its server (orion-introspect) is still allowlisted as inside; only this one tool is judged by its result.
- Clean only when the result is `ok`, `operation=reading_result`, no items, and only known fields. Items, an error, an unknown field, or non-JSON hold the notes back.
- `orion_day` results that hold a page-read record (`reading:`, `reading_journal:`, `world_pulse_digest:` ref: readings or world-news section, or carry citations) hold back too (review finding).
- `dig turn-steps` hides those results too (it shares `outside_result`).

## Outcome moved

Closes a gap in the carry rule: page text could reach the next turn's prompt through `reading_results` while a bare `recommend_reading` receipt could not. Live cost is small. In the 30 days to 2026-11-12: `reading_results` 1 call (10-05), now tainted; `orion_day` 27 calls, 3 now tainted.

## Current architecture

`orion/harness/working_notes.py` taints a turn from tool calls (`outside_reason`), and judges `mcp__orion-reading__*` by result (`outside_result`, #3215). Every `mcp__orion-introspect__*` tool was treated as inside.

## Architecture touched

- `orion/harness/working_notes.py`: `_result_checked(name)`, `_empty_reading_result(payload)`, `_PAGE_READ_REF`; `reading_results` and `orion_day` join the result-checked tools.

## Files changed

- `orion/harness/working_notes.py`: the rule.
- `orion/harness/tests/test_working_notes.py`: result cases (live 10-05 shape), other introspect tools stay inside, runner-level carry vs hold-back.
- `services/orion-harness-governor/README.md`, `docs/superpowers/specs/2026-11-05-turn-working-notes-design.md`: rule text.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: `tainted_by` can now list `mcp__orion-introspect__reading_results`.
- Compatibility notes: none.

## Env/config changes

None. No `.env_example` changed.

## Tests run

```text
pytest orion/harness/tests/test_working_notes.py orion/dig/tests/test_turn_steps.py -> 68 passed
same tests against the old working_notes.py -> 7 failed (new cases), 53 passed
pytest orion/harness/tests/test_working_notes.py orion/dig/tests/test_turn_steps.py -> 76 passed (after review fix)
pytest orion/harness orion/dig orion/cockpit -> 1044 passed
live 10-05 reading_results tool_result through outside_result -> True (tainted)
live orion_day tool_results, 30 d -> 27 calls, 3 tainted
```

## Evals run

```text
No eval change needed: turn_repeat_baseline counts tainted_by names, so live_held_back_reasons
will list reading_results on its own. Readout due ~2026-11-18.
```

## Docker/build/smoke checks

```text
None in this PR. Governor rebuild after merge picks it up (post-pull rebuild).
```

## Review findings fixed

- Finding: `orion_day` (same inside server) returns page-read records in its readings / world_news sections and in carry citations, and read as clean.
  - Fix: judge `orion_day` by result; any `reading:` / `reading_journal:` / `world_pulse_digest:` ref holds back. `walkway_day` stays inside (camera rollup).
  - Evidence: 8 new cases in `test_orion_day_taints_on_page_read_records`; live 3/27 calls tainted.
- Finding (low): the baseline replay re-scores old turns with the new rule, so before/after comparisons shift slightly.
  - Fix: noted here; the 11-18 readout uses `live_*` counts from real rows.
  - Evidence: `orion/harness/evals/turn_repeat_baseline.py` replay path.
- Checked, no bug: MCP errors are non-JSON and fail closed; `ok=false` taints; not-found returns no items and stays clean; key allowlist matches the `extra="forbid"` schema.

## Restart required

```text
No manual restart. The post-pull rebuild rebuilds orion-harness-governor and syncs the FCC sandbox.
```

## Risks / concerns

- Severity: low
- Concern: turns where Orion discusses a page they read lose the free carry.
- Mitigation: Orion can re-run `reading_results` or `dig turn-steps`; data is not lost.
- Severity: low
- Concern: curiosity results may also quote page text; not checked in this PR.
- Mitigation: open question for Juniper.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
