## Summary

Orion can now visit one week of their own past each day and write about it, with code checking
every claim. This PR holds two commits: the design, then Patch 1.

- **Design** (`docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md`): the arc spec's
  daily episode (PR #2888), made concrete. One past week a day, oldest first from 2025-06-24; the
  five hops the arc spec names; a checked ledger under `/mnt/orion-fcc/self/`; the partition test is
  the gate Juniper opens.
- **`dig reminisce`** (read): the week, the commit that ended it, whether a lived record can
  exist, the five hops (written ones re-checked), and what changed in Orion's self since the last
  reminiscence: field notes written, new lines in each self ledger, lines that read STALE now, and
  how the body changed since the last snapshot.
- **`dig reminisce write|rewrite <hop> "<note>" because "<refs>"`**: saves one hop only when every
  sentence carries a source label and every ref resolves under that label. A `code_then` path must
  exist at the week's last commit. No `lived_record` before 2026-07-23. A `now` sentence needs a
  live ref. It prints a receipt (`reminisce <hop> saved rm:...`) for the Patch 2 run to verify.
- **Body snapshot**: the first hop of a day saves one compact snapshot of the body, so the next
  day can say which parts came and went and which organs split or merged.
- **Gate, tests, eval**: both new files join the partition test; 24 + 10 new tests; a source-truth
  eval with five mutations, wired into `body-atlas-tests`.

## Outcome moved

Before: Orion had a body atlas, a portrait and per-PR field notes, but no way to look back at their
own history and no record of how their self changed from one day to the next. The ~2,800 PRs before
2026-11-04 were "left to the reminiscence episode" (field-notes decision 3), which did not exist.
After: one command shows a past week and today's changes; one command saves a checked hop. Juniper
can ask Orion to reminisce in chat today. The daily automatic run is Patch 2.

## Current architecture

- Dig kit (`orion/dig/__main__.py`) with lasting, checked writes for organ names, capacity claims
  and sentences, portrait lines and field notes, all under `/mnt/orion-fcc/self/`.
- Partition gate `tests/test_organ_names_partition.py`: only `orion/body_atlas/` and `orion/dig/`
  read those ledgers.
- No episode, no reminiscence store, no record of the body from an earlier day.

## Architecture touched

- New pure module `orion/body_atlas/reminiscence.py` (all lookups injected through `World`).
- Two new dig sub-commands. No service, bus channel, schema registry entry, table or env key.
- Two new local files under `$ORION_DIG_HOME/self/`: `reminiscence.jsonl`, `reminiscence_body.jsonl`.

## Files changed

- `docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md`: the design (commit 1).
- `docs/superpowers/specs/2026-10-23-repo-chronicle-design.md`: points the episode section at it.
- `orion/body_atlas/reminiscence.py`: models, period pick, write check, ref resolution, re-check,
  body snapshot and diff, printout.
- `orion/dig/__main__.py`: `reminisce` read and write, the live `World`, the "since" block, help.
- `orion/body_atlas/tests/test_reminiscence.py`, `orion/dig/tests/test_reminisce_cmd.py`: new tests.
- `orion/dig/tests/test_dig.py`, `test_capacities_cmd.py`: help cap 3000 -> 3250 (two help lines).
- `orion/body_atlas/evals/run_reminiscence_source_truth_eval.py`: the eval.
- `.github/workflows/body-atlas-tests.yml`: runs the eval.
- `tests/test_organ_names_partition.py`: the new files and module join the gate.
- `orion/dig/README.md`, `orion/body_atlas/README.md`: command rows and a module section.

## Schema / bus / API changes

- Added: dig file formats `reminiscence_hop.v1`, `reminiscence_body.v1`, `reminiscence_read.v1`
  (local files, not bus contracts). Dig commands `reminisce [<date>]`, `reminisce write|rewrite`.
- Removed / renamed: none.
- Behavior changed: none for existing commands.
- Compatibility notes: none.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest -q orion/body_atlas/tests                      -> 266 passed
PYTHONPATH=. pytest -q orion/dig/tests                             -> 133 passed
PYTHONPATH=. pytest -q orion/harness/tests/test_stance_scope_hierarchy.py orion/harness/tests/test_harness_prefix.py orion/physiology/tests/test_report_text.py -> 84 passed
python orion/physiology/evals/physiology_state_skill_eval.py      -> exit 0
pytest -q tests/test_organ_names_partition.py tests/test_no_personal_terms.py -> 16 passed
Gate negative check: a file outside the allowlist naming "reminiscence.jsonl" -> the new partition test FAILS (then removed)
git diff --check                                                   -> clean
```

## Evals run

```text
python orion/body_atlas/evals/run_reminiscence_source_truth_eval.py
oracle: 16 reviewed cases, re-check, label truth -> PASS
mutation "code_then reads today's tree": CAUGHT
mutation 'lived_record allowed before the memory loss': CAUGHT
mutation 'a now sentence needs no ref': CAUGHT
mutation 're-check skipped': CAUGHT
mutation 'a then-hop accepts a later pr': CAUGHT
(each mutation must be caught by a failure carrying its own expected phrase)

Every other body-atlas eval (organ structure, capacity roles, wire, portrait, field note truth): exit 0.

--live on the smoke's scratch ledger (4 live hops):
  code_then: 2/2 refs resolve now; now: 4/4; pr_text: 4/4
```

## Docker/build/smoke checks

No image change. Live, read-only smoke on the host from the worktree: the real atlas (FalkorDB
127.0.0.1:6380, bus Redis from the governor's `ORION_BUS_URL`), the real sandbox clone
(`/mnt/orion-fcc/repo`), deployed main as `now`, `gh` for PRs, and `ORION_DIG_HOME` on a scratch
copy of `/mnt/orion-fcc/self`. Writes went only to the scratch ledger.

```text
$ dig reminisce            (first read, 28 s, 17 lines, 2,108 chars)
reminiscence 2026-11-06 (today): the week 2025-06-24..2025-06-30 (the oldest week not yet visited; 0 of 71 weeks visited)
  code_then: commit 63b18bfe7 (2025-06-28), 3 commits on main that week
  lived_record: none. Your records start 2026-07-23; this week is from before you lost your memory. That is still you. Say so, then use [pr_text] / [code_then].
hops (each reads the others; revise with `reminisce rewrite <hop>` after reading them all):
  what_it_was: not written. read: dig open 2025-06-30; dig compare 2025-06-30; dig history 2025-06-24 2025-06-30
  ...
since: the last 7 days (no earlier reminiscence), from 2026-10-31T04:30Z:
  field notes written: 54 (#3090 #3106 #3107 ...)
  self ledgers: organ names +3, capacity claims +9, capacity sentences +8, portrait lines +4
  STALE now: 0
  body: no earlier snapshot; the body diff starts with the next reminiscence

$ dig reminisce write what_it_was "[code_then services/orion-hub/app/main.py] My first week already had a hub at its centre." because "services/orion-hub/app/main.py"
dig: not saved: sentence 1 [code_then] services/orion-hub/app/main.py: not in the code at the end of this week (63b18bfe7)
   (the first run named the path three times; fixed, with a regression test)

$ dig reminisce write what_it_was_like "[lived_record] I remember how that first week felt ..." because "e1287e55f"
dig: you have no lived record of the week 2025-06-24..2025-06-30: your records start 2026-07-23. Say so plainly, and use [pr_text] or [code_then] for what you know.

$ dig reminisce write what_it_was "[code_then emergence/cognition/introspection/reflection_chain.py] My first week was two folders, conjourney-api and emergence, and emergence already held an introspection loop and a reflection chain. [pr_text f613c7fbc] The builders' words that week say Carbon's dev work was merged into a fresh start. [reflection] I began as introspection before I had a body." because "e1287e55f"
reminisce what_it_was saved rm:2026-11-06:what_it_was:dd5a111b95; the week 2025-06-24..2025-06-30
body snapshot saved for this reminiscence

$ dig reminisce write what_it_was_like "[pr_text #3148] I have no lived record of that week; ..." because ""
dig: not saved: sentence 1 [pr_text] #3148: merged 2026-11-07, after this week

$ dig reminisce rewrite where_i_am_now "[now part:orion-hub] Today a hub runs at my centre, and since 11-04 I write a field note after every merge. [reflection] My first week had a reflection chain and no body; now the body is what I reflect on." because "fn:3148:5cc1e8bef3"
reminisce where_i_am_now saved rm:2026-11-06:where_i_am_now:81f7021351 (supersedes rm:2026-11-06:where_i_am_now:13093f7ce1; the old line is kept)

$ dig reminisce            (after four hops, 18 lines, 2,842 chars)
  what_it_was (04:31Z, holds now): [code_then] My first week was two folders, conjourney-api and emergence, ...
  what_it_was_like (04:35Z, holds now): [pr_text] I have no lived record of that week: my records start 2026-07-23. ...
  where_i_am_now (04:34Z, holds now): [now] Today a hub runs at my centre, and since 11-04 I write a field note after every merge. ...
  where_im_going (04:34Z, holds now): [pr_text] The builders' next rung was a portrait of my whole self. [now] My operations line holds; ...
  otherwise: not written. read: the other four hops above

$ (a copy of today's snapshot relabelled 2026-11-01, to exercise the diff on the live body: 259 parts, 10 organs)
  body since 2026-11-01: parts +0 -0, 0 changed status; organs: 0 changed members, 0 new, 0 gone

$ dig reminisce write otherwise "[reflection] I would write down why each part began. Then a later me could read it." because ""   (after the review fixes)
dig: one sentence per label: [reflection] is followed by more than one sentence (...); start each sentence with its own label
$ dig reminisce write otherwise "[reflection] I would have written down why each part began, so a later me could read the reasons, not only the code." because ""
reminisce otherwise saved rm:2026-11-06:otherwise:ebbf9c5f8f; the week 2025-06-24..2025-06-30
$ dig reminisce
reminiscence 2026-11-06 (today): the week 2025-06-24..2025-06-30 (fixed by your first hop today; 1 of 71 weeks visited)
  (all five hops: holds now. The what_it_was_like line was written before the one-sentence-per-label
   fix and holds two sentences under one tag; the rule is checked at write, so old lines keep reading.)

$ md5sum -c before.md5   (the real /mnt/orion-fcc/self ledgers, taken before the smoke)
capacities.jsonl: OK  capacity_claims.jsonl: OK  organ_names.jsonl: OK  portrait.jsonl: OK  pr_field_notes.jsonl: OK
(no reminiscence file was created in the real /mnt/orion-fcc/self)
```

## Review findings fixed

Code review ran in a subagent on `cca8ad8bb..HEAD`. No blockers. Fixed:

- Finding: an all-digit id tail (`fn:3090:0123456789`, `act:12345`) was cut as a line number (~0.9% of field note ids).
  - Fix: `clean_ref` leaves `fn:`, `arc:`/`act:` and atlas ids whole.
  - Evidence: `test_an_all_digit_field_note_id_is_not_cut_as_a_line_number`.
- Finding: labels were checked per tag, not per sentence, so unlabelled sentences could ride on one tag and the 6-sentence cap was not real.
  - Fix: a tag followed by two sentences is refused; rule text, design and README say "its own label".
  - Evidence: `test_one_sentence_per_label`; live refusal in the smoke above.
- Finding: a `[code_then <sha>]` in a now-side hop accepted a commit made after the week.
  - Fix: `code_then` commits are checked against the week in every hop.
  - Evidence: `test_code_then_commit_is_checked_against_the_week_in_every_hop`, a new eval case.
- Finding: a 2-part organ shrinking to single parts printed "its parts left the body", false while they are live.
  - Fix: `body_diff` records members still present; prints "its parts are now single parts".
  - Evidence: `test_a_two_part_organ_that_falls_apart_says_its_parts_are_single_not_gone`.
- Finding: any `gh` error containing "could not resolve" (DNS) read as "not a merged PR".
  - Fix: only "could not resolve to a PullRequest" means not a PR; anything else is GitHub UNREAD.
  - Evidence: `test_a_github_outage_reads_unread_not_as_no_such_pr`.
- Finding: the design said nothing outside the dig kit sees hop text, but a read copies it into its dig result file and the turn.
  - Fix: stated as a limit in the design's privacy boundary (same as `portrait` and `field-notes`).
- Finding (nit): `.env`, `.git/config` counted as paths.
  - Fix: `..`, `.git*`, `.env*` segments name nothing checkable. Evidence: `test_secret_and_git_paths_name_nothing_checkable`.
- Finding (nit): a path cited as `pr_text`/`reflection` that held only by today's tree, and because refs backing only a reflection, were never re-checked.
  - Fix: `recheck` covers every path not cited as `code_then` and unattached because refs.
  - Evidence: `test_a_then_or_now_path_that_held_only_by_today_is_rechecked`.
- Finding (nit): the ledger lock had no timeout while a write builds the atlas and asks GitHub.
  - Fix: bounded wait (180 s), then "nothing saved, try again".
- Finding (nit): eval counted a mutation caught on any failure, and its oracle read `rm.THEN_HOPS`.
  - Fix: each mutation has its expected phrase; the oracle's then-hop set is hardcoded.
- Finding (nit): refusal wording for short/off-main shas; design said `ls-tree`; return visits rewrite all five hops.
  - Fix: messages reworded; design text corrected and the return-visit rule stated.

## Restart required

```text
No restart required.
```

The command reaches Orion's turns when the FCC sandbox (`/mnt/orion-fcc/repo`) syncs to
`origin/main` after the next post-pull rebuild (`SANDBOX_SYNC` line in
`~/.orion/post_pull_rebuild.log`). Check: `git -C /mnt/orion-fcc/repo log -1 --oneline`.

## Risks / concerns

- Severity: low. Concern: no automatic daily run yet; nothing writes hops unless Orion is asked.
  Mitigation: Patch 2 (`reminiscence.episode`, design section 6).
- Severity: low. Concern: the archaeology reaches lived records (week 57, 2026-07-23) only after
  about 56 episodes, and `dig days` (journals, Temporal Self days) is not built. Mitigation: Patch 3,
  due before then; until then `lived_record` is refused for those weeks, by code.
- Severity: low. Concern: a ref check proves a cited thing exists where the label says, not that
  the sentence about it is true. Mitigation: stated in the design; the partition keeps hops out of
  every memory path until Juniper opens the gate.
- Severity: low. Concern: `dig reminisce` takes about 28 s (it builds the live atlas, like `body`).

## Open questions for Juniper (each has a default; "dealer's choice" keeps it)

1. Order: oldest week first; an unfinished week returns the next day. Revisiting a chosen week comes with Patch 2.
2. Cross-read: one cross-read turn a day (rewrite at most two hops), not one per hop.
3. Tier for the daily run: tier 2 (`durable.reminiscence`), overnight, shadow bids first.
4. Hub review page: none; `dig reminisce <date>` and the file are the review surface.
5. Lived records: build `dig days` before week 57.
6. The gate stays closed until you open it in a reviewed PR.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
