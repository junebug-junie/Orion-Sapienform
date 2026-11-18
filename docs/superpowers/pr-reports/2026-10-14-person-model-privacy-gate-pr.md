## Summary

- Main was red on `tests/test_no_personal_terms.py`: PR #2649 (person model P0-P2) merged after the privacy gate (#2651) with 85 personal-term hits across 11 files.
- Real stated answers and eval labels moved to gitignored `*.local.*` files (real copies in the primary checkout); committed `*.example.*` files are redacted or fake.
- New `orion.person_model.person.local_first()` makes the dry-run CLI and both evals read the local file first, the example otherwise (same idea as `.env` / `.env_example`).
- Tests and code comments use fake names; the #2649 PR report uses role placeholders. Bake-off numbers are unchanged.

## Outcome moved

Privacy gate green again: `scripts/check_personal_terms.py` reports no personal terms in tracked files. The live dry run, run from the primary checkout, still reads the same real answers and labels (byte-identical, checked below).

## Current architecture

`scripts/run_person_linkage.py --answers` defaulted to the tracked `config/person_model/stated_answers.yaml`; `run_linkage_eval.py` / `run_extractor_bakeoff.py` defaulted to tracked label files holding real names and snippets.

## Architecture touched

- `orion/person_model/person.py`: `DEFAULT_ANSWERS_PATH`, `local_first()`.
- `scripts/run_person_linkage.py`, `orion/person_model/evals/run_linkage_eval.py`, `run_extractor_bakeoff.py`: defaults go through `local_first`.
- No service, container, bus, or schema reads these files.

## Files changed

- `.gitignore`: `config/person_model/*.local.yaml`, `orion/person_model/evals/*.local.jsonl`.
- `config/person_model/stated_answers.yaml` -> `stated_answers.example.yaml` (placeholder home).
- `orion/person_model/evals/{cleanup,mention}_labels.jsonl` -> `*.example.jsonl` (small fake-name fixtures).
- `orion/person_model/tests/test_local_first_data.py`: resolver, gitignore, fixture-shape tests.
- `orion/person_model/tests/*.py`, `linkage.py`, `scripts/run_person_linkage.py`: fake names (same lengths, same rules exercised).
- `orion/person_model/README.md`: local vs example table.
- `docs/superpowers/pr-reports/2026-10-14-person-model-p0-p2-pr.md`: placeholders; corrected a stale risk line claiming the review HTML is committed (it is not).

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: tracked data files above. Behavior changed: default input paths resolve local first. Compatibility: explicit `--answers` / `--labels` unchanged.

## Env/config changes

- Added/removed/renamed keys: none. `.env_example` updated: no. Sync: not needed. Skipped keys: none.

## Tests run

```text
check_personal_terms.py                       -> no personal terms in tracked files
pytest orion/person_model/tests tests/test_interlocutor_pronouns_in_prompts.py \
  tests/test_person_model_schemas_registered.py tests/test_no_personal_terms.py -> 72 passed
local_first(primary checkout example paths) vs origin/main originals:
  stated_answers parsed YAML identical: True; cleanup/mention labels bytes identical: True
```

## Evals run

```text
Not run: both evals need live data (decisions.jsonl / chat_history_log). Fixture shape is covered by test_local_first_data.py.
```

## Docker/build/smoke checks

```text
None: no compose file or Dockerfile references orion/person_model or these data files.
```

## Review findings fixed

- Finding: `run_extractor_bakeoff.py` used `local_first` without importing it (`--help` raised NameError).
  - Fix: import added; `test_eval_scripts_start` runs `--help` on both eval CLIs.
  - Evidence: 72 passed.
- Finding: a fixture string still held a real-looking contact name, and a fake fixture name was paired with the interlocutor's real surname.
  - Fix: both swapped for fake names (tests, CLI docstring, #2649 report).
  - Evidence: `check_personal_terms.py` clean; 72 passed.
- Finding: removed values remain in git history.
  - Fix: not fixable without a force push; listed under risks for Juniper.
  - Evidence: n/a.
- Note (not changed): the interlocutor's own name stays in fixtures and `people.yaml`; the denylist allows it.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: the removed values remain in git history (#2649 commits). Mitigation: none here; a history rewrite needs a force push and Juniper's explicit approval.
- Severity: low. Concern: a fresh worktree has no local files, so a dry run there uses the placeholder home. Mitigation: README says to copy the local files or pass `--answers`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2653

🤖 Generated with [Claude Code](https://claude.com/claude-code)
