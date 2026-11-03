## Summary

Self-inquiry is the curiosity line where Orion asks "What am I, and what am I made of?". Its "where the records are" section named only the repo and SQL. It did not name the dig kit, which builds a live map of Orion's body when asked. Orion used the dig kit in 0 of 52 self-inquiry runs.

- The records section now opens with one paragraph. The paragraph names `python -P -m orion.dig body`, `body <organ_id>`, `inventory now`, `open <date>` then `compare <date> now`, and `body-economy`. It says these commands are the map of what Orion is made of.
- On "what am I made of" (anatomy) runs, the paragraph says "START LOOKING HERE", and step 2 of the order of work says "Look: your body (the dig commands below), records, repo, your graph."
- On "who matters to me" (lived) runs, the same commands show as an option ("when the question touches what you are made of or what you cost"). Step 2 stays as it was. A question about people is not sent to an inventory of machines first.
- The repo pointer stays. It now comes after the dig commands ("After that, your own repository ...").
- `scripts/report_dig_kit_usage.py` now counts `body-economy` and `name-organs` as their own subcommands, not as `body` and `name`.
- This is Patch A of the Self Atlas replacement (spec `docs/superpowers/specs/2026-10-30-self-atlas-replacement-proposal.md`). Patch B retires Self Atlas. Patch B must merge no earlier than 72 h after this patch deploys.

## Outcome moved

Self-inquiry can now find the live body map. Before this patch, it had no pointer to the map. Uptake is measured, not assumed (see "How to measure uptake").

## Current architecture

`orion/curiosity/self_inquiry_prompt.py` `_records_section` listed the repo at `repo_root`, the SQL ledger, and Orion's own graph priors. The dig kit (`orion/dig`) was live and Orion used it in other turns. The harness prefix line (`orion/dig/brief.py`) told every non-reading turn that the dig kit exists, but the self-inquiry prompt itself never mentioned it.

## Architecture touched

Prompt text only. No service, schema, bus, or env change.

## Files changed

- `orion/curiosity/self_inquiry_prompt.py`: new `dig_body_paragraph(family)`, placed first in `_records_section`; step 2 wording for anatomy runs.
- `orion/curiosity/tests/test_self_inquiry_prompt_dig_body.py`: the paragraph renders; dig comes before the repo pointer; the repo pointer stays; every command uses `python -P`; anatomy and lived runs get the right lead-in; the first-write instruction still comes first.
- `scripts/report_dig_kit_usage.py`, `tests/test_report_dig_kit_usage.py`: hyphenated subcommands are counted as themselves.
- `.github/workflows/agency-ask-episodes.yml`: runs the new test.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: self-inquiry prompt text
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
pytest orion/curiosity/tests/ tests/test_report_dig_kit_usage.py      183 passed, 23 skipped
pytest services/orion-hub/tests/test_curiosity_self_inquiry.py        44 passed
```

## Evals run

```text
No eval harness covers prompt adoption. Adoption is measured live with
scripts/report_dig_kit_usage.py (below), not by an eval.
```

## How to measure uptake

Run this 72 h after deploy, and again after 7 days (spec acceptance check 4: at least 1 self-inquiry dig call in 7 days):

```bash
docker exec -i orion-athena-harness-governor python3 - 3 < scripts/report_dig_kit_usage.py
```

The argument is the number of days. Read the self-inquiry `used_dig` count. If it is still 0 after 7 days, report it. It is not a reason to keep Self Atlas (spec failure mode 4).

## Docker/build/smoke checks

```text
Not run. Prompt text only; Hub builds the prompt at self-inquiry kickoff.
```

## Review findings fixed

A code-review subagent found no blockers. It confirmed the command names against `orion/dig/__main__.py`, and it confirmed that self-inquiry turns have Bash (only reading turns are `reading_only`).

- Finding: a past date needs `open <date>` first; the text suggested one call.
  - Fix: the line now says `open <date>, then compare <date> now`, and `inventory now` says "a past date works after `open <date>`".
  - Evidence: `test_records_section_renders_the_dig_body_paragraph`.
- Finding: "FIRST, the map of what you are made of" rendered on lived questions ("who matters to me") too, and sent them to an inventory of machines first.
  - Fix: `dig_body_paragraph(family)`; lived runs get a softer lead-in and the old step 2.
  - Evidence: `test_anatomy_run_is_told_to_start_with_its_body`, `test_lived_run_gets_the_body_as_an_option_not_the_first_stop`.
- Finding: the usage report counted `body-economy` as `body`, which hides which pointer worked.
  - Fix: regex `([a-z][a-z_-]*)`.
  - Evidence: `test_hyphenated_subcommands_are_counted_as_themselves`.
- Finding (nit): a second "FIRST" competed with "YOUR FIRST TOOL CALL".
  - Fix: the lead-in is now "START LOOKING HERE".
  - Evidence: same anatomy test checks the MERGE instruction still comes first.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: Orion may still not use the commands (self-inquiry runs at most 3 times a day and is often heat-blocked).
- Mitigation: measured with `report_dig_kit_usage.py`; Patch B does not wait on adoption, per the spec.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3049

🤖 Generated with [Claude Code](https://claude.com/claude-code)
