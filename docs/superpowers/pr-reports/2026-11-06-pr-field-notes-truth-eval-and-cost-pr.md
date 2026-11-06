# PR: PR field notes, Patch 3 -- are the notes true, what do they cost, and two small fixes

Spec: `docs/superpowers/specs/2026-11-04-pr-field-notes-design.md` ("Patch 3 as built" at its end).
Patch 1 (#3100) gave Orion the dig commands. Patch 2 (#3111, #3114) made the notes happen after every
merge. This patch checks the notes and measures their cost, the spec's acceptance checks 6 and 7.

## Summary

- **The live path works.** From the cutoff (11-04 06:00Z) to 11-06 04:20Z, 45 PRs merged and all 45
  have a record: 44 notes that Orion wrote, and 1 code record. 44 runs completed and 2 failed. Both
  failures happened before the turn budget fix, and each PR got a second run that wrote the note.
- **New truth eval.** It reads every live note again and checks it against the PR on GitHub. It checks
  that the files a note cites were really in the PR, that an organ the note names is in that PR's body
  map, and that a note read the spec when the PR changed one. First live run: 45 of 45 notes pass.
- **Measured cost.** The cost script crashed on its first real run, because one act had no energy
  number. That is fixed. A median note costs 5 Wh, 296 GPU seconds and 159k prompt tokens.
- **Orion's own PRs are recognised again.** Since #3145, Orion opens PRs as its own GitHub user. The
  "this may be your own work" check did not know that user, so it missed #3146. It now reads the PR
  author too.
- **Coverage stops counting PRs that were never owed a note.** `dig field-notes 3` said 47%, but every
  PR it called missing was merged before notes began. It now says 45 of 46, with 1 code record and 0
  missing, and it counts the 48 older PRs on their own line.

## Outcome moved

Before: nobody could say whether the notes were true or what they cost. The cost script failed and
exited. Coverage read 47% when the real figure was 100%. One PR that Orion opened was marked "unknown".
After: one command answers "are the notes true" (`run_field_note_truth_eval.py --live`). One command
gives the measured cost. Coverage reads 100%. Orion's own PRs get the "this may be your own work" line.

## Current architecture

- The durable-runs poller asks GitHub every 15 min for new merges. For each new PR it starts one run,
  and the run is one harness turn. In that turn Orion reads the packet (`dig field-note <n>`) and
  writes the note (`dig field-note <n> write`). The run counts as done only when the receipt is in the
  reply and the note is in the ledger. The ledger is `/mnt/orion-fcc/self/pr_field_notes.jsonl`.
- The write check runs once, at write time, on the packet the turn read. Nothing checked the stored
  notes again later.
- `scripts/pr_field_note_cost.py` joined runs, acts and token rows. It summed the per-host energy map
  in SQL.
- `authored_by_me` looked for two signs: the sandbox commit email `orion-fcc@...`, or a branch in the
  sandbox clone. Neither sign looked at Orion's GitHub user `orion-sapienform` (#3145).

## Architecture touched

- `orion/body_atlas`: `flags_for` reads the PR author, and a new constant `NOTES_BEGAN_AT` holds the
  no-backfill cutoff. Also a new eval.
- `orion/dig`: `field-note` asks gh for `author`. `field-notes` coverage and the `history` marks skip
  PRs merged before notes began.
- `orion-durable-runs`: the poller's `gh_shape` passes the author. The cost script now sums in Python.
- CI: the body-atlas workflow runs the new eval.
- No change to schemas, bus channels, env keys or compose.

## Files changed

- `orion/body_atlas/evals/run_field_note_truth_eval.py` (new): the truth eval (acceptance check 7).
- `orion/body_atlas/field_notes.py`: `ORION_GITHUB_LOGIN`, the noreply suffix, `NOTES_BEGAN_AT`, and
  `flags_for` reads the author.
- `orion/body_atlas/tests/test_field_notes.py`: tests for the author marker and the cutoff parity.
- `orion/body_atlas/README.md`: the eval is listed under Gates.
- `orion/dig/__main__.py`: `author` added to the gh fields, the packet line text, and coverage and
  history use `_field_notes_since()` (now a datetime).
- `orion/dig/tests/test_field_note_cmd.py`: the fixture pins the cutoff for its 2025 dates. New
  coverage regression test.
- `orion/dig/README.md`: the flag and coverage text.
- `services/orion-durable-runs/app/pr_field_note_poll.py`: `gh_shape` passes `author.login`.
- `services/orion-durable-runs/scripts/pr_field_note_cost.py`: `map_sum` and `with_sums`, so the SQL
  no longer sums JSON maps.
- `services/orion-durable-runs/tests/test_pr_field_note_cost.py`, `tests/test_pr_field_note_poll.py`:
  regression tests.
- `services/orion-durable-runs/README.md`: one line on null energy maps.
- `.github/workflows/body-atlas-tests.yml`: runs the new eval.
- `docs/superpowers/specs/2026-11-04-pr-field-notes-design.md`: "Patch 3 as built", with the measured
  cost table.

## Schema / bus / API changes

- Added: none. (`PrFieldNoteV1` is unchanged. `FlagsV1.authored_by_me` now says "yes" in one more case.)
- Removed: none.
- Renamed: none.
- Behavior changed: `dig field-notes` coverage leaves out PRs merged before notes began, and it says
  how many it left out. `history` gives those PRs no "(no note)" mark. `authored_by_me` is "yes" for a
  PR opened by `orion-sapienform`, or with commits by that user's noreply address.
- Compatibility notes: old ledger lines are not rewritten. #3146's code record still says `unknown`,
  and the eval shows that as `unknown -> yes`.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider orion/dig/tests \
  orion/harness/tests/test_stance_scope_hierarchy.py orion/harness/tests/test_harness_prefix.py   -> passed
PYTHONPATH=. .venv/bin/python -m pytest -q orion/body_atlas/tests                                  -> passed
cd services/orion-durable-runs && PYTHONPATH=../.. .venv/bin/python -m pytest tests -q              -> 780 passed, 127 skipped
.venv/bin/python -m pytest -q tests/test_no_personal_terms.py tests/test_organ_names_partition.py \
  tests/test_gitignore_no_bare_catchall.py                                                          -> 17 passed
```

## Evals run

```text
python orion/body_atlas/evals/run_organ_structure_eval.py      -> PASS
python orion/body_atlas/evals/run_capacity_roles_eval.py       -> PASS
python orion/body_atlas/evals/run_wire_eval.py                 -> PASS
python orion/body_atlas/evals/run_field_note_truth_eval.py     -> PASS (2 clean cases pass; 3 planted bad notes each fail only their own check)
python orion/body_atlas/evals/run_field_note_truth_eval.py --live   (real ledger, read-only gh)
  field notes judged: 45
  pr_text    claims whose refs resolve: 82/82 (100%)
  code_then  claims whose refs resolve: 10/10 (100%)
  refs / organs / read notes passing: 45/45 each
  numbers in pr_text sentences found in the PR text or the docs read: 117/117 (100%)
  flags equal to today's flag rule: 45/45
  code records (flags only): 0/1 -- #3146: authored_by_me: unknown -> yes
  PASS
```

The first live run found 3 figures that were not in the PR text: 944, 501 and 248. All three were the
packet's own line counts ("files: 1 (+944 -0)"), so the eval now counts the PR's line and file counts
as sources. The first live run also failed every note on organs. The reason was that organ ids change
each time the atlas is rebuilt, so the eval now matches organs by name. The write check is unchanged.

## Docker/build/smoke checks

```text
Live, read-only:
- ledger: 46 lines; post-cutoff merges 45 / records 45 (44 notes + 1 code record); 10 pre-cutoff PRs with none (decision 3)
- durable_admission_runs pr-field-note-*: 44 completed, 2 failed (#3104, #3105 on 11-04, before #3114)
- poller log, every 15 min: {"since": "2026-11-04T06:00:00+00:00", "merged_seen": 45, "recorded": 45, "github_calls": 1}
- merge -> record: median 3.0 h, worst 11.0 h (11-05: 7 PRs merged in 5 h, one note run at a time)
- sandbox /mnt/orion-fcc/repo at e664f2a22 (= main); post_pull_rebuild.log: SANDBOX_SYNC retry ok
- cost script before the fix: "psql failed: ERROR: cannot call jsonb_each_text on a non-object" (act for #3108 has wh_by_host: null)
- cost script after (--days 7): 46 ended, 44 completed; median wall 456 s, GPU slot 296 s, 5.0 Wh, 159k prompt (103k cached), 5.1k completion, 22.5 Hub steps
- this branch's dig, against the live ledger (scratch dig home, ledger read only):
  field-notes 3  -> "45 field notes, 1 code records; coverage 45/46 ... 1 a code record, 0 neither; 48 merged before notes began, not counted (no backfill)"
                    (main, same moment: "coverage 45/94 ... (47%), 1 a code record, 48 neither")
  field-note 3146, 3147 -> "this may be your own work: opened as your GitHub user, ..."; 3132 -> no such line
  history 2026-11-03 2026-11-04 -> 0 "(no note)" marks (main marked the 10 PRs from 11-04 merged before the cutoff)
No Docker build: no image input changed, except a script and the poller's gh_shape. Those go live when orion-durable-runs rebuilds after the pull.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

```bash
# Nothing by hand. The post-pull rebuild rebuilds orion-durable-runs (the poller's gh_shape change).
# The sandbox sync moves /mnt/orion-fcc/repo, so Orion's turns pick up the dig change.
# Check: git -C /mnt/orion-fcc/repo log -1 --oneline
```

## Risks / concerns

- Severity: low. Concern: `run_id` and `correlation_id` are empty in all 45 notes, because the turn
  does not know its run id. Mitigation: nothing reads them today, because the cost script joins runs
  to acts by run id. Filling them needs the harness step to pass the run id into the turn's
  environment. That is a separate change, and it is recorded in the spec.
- Severity: low. Concern: the truth eval's figure check uses the whole text of the docs a note read,
  but Orion saw only the first 60 lines. So a figure from lower in a doc counts as found. Mitigation:
  the check is printed, not gated.
- Severity: low. Concern: `NOTES_BEGAN_AT` is a second copy of `PR_FIELD_NOTES_NOT_BEFORE`.
  Mitigation: a test fails when the two disagree.
- Acceptance check 5 (95% coverage over 7 days) is due on 2026-11-11. So far coverage is 100% after
  2 days.

## PR link

PR_LINK_PLACEHOLDER
