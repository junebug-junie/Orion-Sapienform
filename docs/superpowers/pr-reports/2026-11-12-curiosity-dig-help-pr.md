## Summary

- Curiosity runs are now told they have more dig tools than the few listed, and to run `dig help` before they pick one.
- Both curiosity prompts now name the week's PR record: self-inquiry gets `dig field-notes 7`; the investigation prompt gets `dig field-notes 7` and `dig history 7`.
- The "Ask Orion" modal in Hub gets a "Read one PR on GitHub" entry. Its drift test now reads the GitHub tool brief, so a new GitHub tool with no entry fails CI.
- The `dig help` length test was red on main: turn-steps (#3193) pushed the text to 3,292 characters against a 3,250 cap. The cap is now 3,500.

## Outcome moved

Before this patch, curiosity runs never read Orion's own notes on recent PRs (`dig field-notes` was used 0 times in 14 days). The only pointer was "`dig help` lists the rest" at the end of a list.

Baseline, 14 days to 2026-11-12 (`docker exec -i orion-athena-harness-governor python3 - 14 < scripts/report_dig_kit_usage.py`):
- Self-inquiry: dig used in 26 of 94 runs.
- Investigation: dig used in 17 of 115 runs.
- Subcommands across both: body-economy 44, body 27, help 13, inventory 9, act 6, arc 3, history 3, reminisce 3, field-notes 0.

Re-run the same command about 7 days after deploy and compare the `field-notes`, `history` and `help` counts.

## Current architecture

- Self-inquiry (`orion/curiosity/self_inquiry_prompt.py`): the records section lists 5 dig commands and ends with "`dig help` lists the rest."
- Investigation (`orion/curiosity/kickoff_prompt.py`): the access section names SQL, the graphs, MCP tools and Bash. It does not name dig. The every-turn harness brief (`orion/dig/brief.py`) names dig in one line.
- Ask Orion catalog: covers every dig command and every introspect, reading, gitnexus and AI Town tool. It has no GitHub entry and no drift check for GitHub tools.

## Architecture touched

Prompt text only. No bus, schema or env changes.

## Files changed

- `orion/curiosity/kickoff_prompt.py`: adds `DIG_HELP_FIRST` and `DIG_RECENT_WORK_LINES`. `_access_section(dig_kit=True)` adds them before the read_recall line.
- `orion/curiosity/self_inquiry_prompt.py`: the dig list gains `field-notes 7` and the help line. Self-inquiry and the self-naming sitting pass `dig_kit=False`.
- `services/orion-hub/static/catalog/orion-abilities.json`: adds the `pr-read` entry and a hidden reason for `search_pull_requests`.
- `services/orion-hub/tests/test_orion_abilities_catalog.py`: `MCP_BRIEFS` now includes `orion/fcc/github_repo_context.py`.
- `orion/dig/tests/test_dig.py`, `test_capacities_cmd.py`: help cap 3250 -> 3500.
- Tests: `orion/curiosity/tests/test_kickoff_prompt_dig_kit.py` (new); new cases in `test_self_inquiry_prompt_dig_body.py` and `services/orion-hub/tests/test_curiosity_self_naming.py`.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: curiosity prompt text; Ask Orion catalog content
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/curiosity/tests orion/dig/tests services/orion-hub/tests/test_orion_abilities_catalog.py
  -> 414 passed, 29 skipped
PYTHONPATH=.:services/orion-hub pytest services/orion-hub/tests/test_curiosity_self_naming.py -> 17 passed
node --test services/orion-hub/static/js/orion-abilities.test.js -> fail 0
```

On main, 2 dig help tests failed before this patch (cap). They pass now.

## Evals run

```text
Live: dig field-notes 7 and dig history 7 run in the harness-governor container -> real output
(68 notes, 72 merged PRs for 11-05..11-11).
Baseline usage report above. Post-deploy re-run is the eval.
```

The browser smoke (`services/orion-hub/evals/orion_abilities_browser.py`) was NOT run: Playwright is not installed on this host. The modal builds its cards from the JSON, so a new entry needs no JS change.

## Docker/build/smoke checks

```text
None. Prompt text and static JSON only.
```

## Review findings fixed

- Finding: "run dig help first" competed with the "YOUR FIRST TOOL CALL: write your answer" step in self-inquiry and in the self-naming sitting. The patch had also given the naming sitting a dig block it never had.
  - Fix: the wording is now "before you pick a dig command, run ... help". The naming sitting passes `dig_kit=False`.
  - Evidence: `test_naming_sitting_gets_no_dig_kit`; `test_access_section_carries_the_dig_kit_lines_by_default` asserts no bare " first".
- Finding: the dig block split the access section from its boundary paragraph, and "run first" contradicted the section's "none of these is expected" header.
  - Fix: the block now sits before the read_recall line, with the conditional wording.
  - Evidence: an ordering assertion in the same test.

## Restart required

Hub builds the curiosity prompts. The Hub image picks up this code on the next post-pull rebuild from main. The FCC sandbox sync updates `orion/` for Orion's own CLIs.

```text
No manual restart required (post-pull rebuild covers orion-hub).
```

## Risks / concerns

- Severity: low
- Concern: about 400 more characters in the investigation prompt. No size cap applies.
- Mitigation: none needed.
- Severity: low
- Concern: a live override of `GITHUB_TOOLSETS` would make the catalog's GitHub entry point at tools that are not there.
- Mitigation: checked 2026-11-12. No service `.env` sets it, and neither does the harness-governor container env. The default `repos,pull_requests` (read-only) covers both tools.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
