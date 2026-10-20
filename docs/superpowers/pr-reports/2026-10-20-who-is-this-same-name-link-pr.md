## Summary

- Every name Juniper types now links that person to someone she already gave the same name. Before this, only names typed on walkway story cards did that (PR #2753). Now plain "Who is this?" answers and the name boxes on "same person?" cards do it too. Names that come back from older pair answers through the heal path do it as well.
- One function handles every case. `_apply_typed_name` calls `_link_same_name` with the same rules as before: no link over an existing pair verdict, a copied name is not a target, the link is idempotent, and it logs `[VISION_SAME_NAME_LINK]`.
- The cannot-link rule is now stricter. A link is refused when it would join this person to a group that contains someone she said is a different person. Before, only a direct verdict between the two people blocked the link.
- Correcting a typed name to a different name removes the links the old name made (`[VISION_SAME_NAME_UNLINK]`). The UI has no card for removing an automatic link, so renaming is how Juniper undoes one. Her own "same person?" answers are never removed.
- If two typed names differ only in letter case, they count as one name. This no longer logs a false `[VISION_GROUP_NAME_CONFLICT]`.

## Outcome moved

Typing "Maya" on a "Who is this?" card now joins that person to the Maya Juniper already named, the same way a story card does. One person no longer splits into several unlinked individuals depending on which card she used.

## Current architecture

`services/orion-sql-writer/app/vision_individuals.py` `_apply_answers` applies answered `orion_ask` rows each tick. A same-name link was added only when `answer_detail.story_id` was set. The split (`app/vision_split.py`) has its own same-name link for the new individuals it creates.

## Architecture touched

sql-writer only. No schema change, no bus change, no env change, no migration.

## Files changed

- `services/orion-sql-writer/app/vision_individuals.py`: the link moves into `_apply_typed_name`. Adds a group-level cannot-link check in `same_name_target`, adds `_unlink_old_name`, and makes the conflict check ignore letter case.
- `services/orion-sql-writer/tests/test_vision_split.py`: the old "plain answers do not link" test is inverted. Adds tests on real Postgres for: who-is-this links plus the log line plus idempotency; a copied name is skipped; 'different' through a group; pair-card names, one link per group and never across 'different'; a rename unlinks but keeps her own pair answer; a case-only change; the heal path; the split cannot-link.
- `services/orion-sql-writer/README.md`: documents the same-name link rules.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: who-is-this answers and pair-card names now write `vision_individual_pair_verdict` 'same' rows (ask_id = the answering ask, face_cosine NULL). A typed rename deletes the 'same' rows involving that individual whose ask is a `vision_individual` ask.
- Compatibility notes: forward-only. Answers already applied are not re-linked.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
throwaway postgres:16 container, CI's migration list applied, ORION_TEST_PG_URI set:
pytest test_vision_confirm_asks test_vision_split test_vision_osnet_shadow test_vision_body_match
       test_vision_reanchor test_vision_motion test_vision_story_strip test_vision_rollup
  -> 202 passed, 0 skipped (before the review fixes); 100 passed on the vision ask/split/story files after them
Mutation checks: revert the app file -> 3 new tests fail; remove the group cannot-link -> 2 fail;
  remove the unlink -> 1 fails; case-sensitive conflict -> 1 fails; unlink without the source_kind filter -> 1 fails
Full sql-writer suite without PG: the 14 failures are identical to origin/main's (env-sensitive, already there before this change)
```

## Evals run

```text
None. sql-writer has no eval harness for vision asks. The real-Postgres tests are the behavioural check.
```

## Docker/build/smoke checks

```text
No Docker build (pure Python logic change, no deps/config). Live path UNVERIFIED until the sql-writer rebuild after merge;
look for [VISION_SAME_NAME_LINK] on the next who-is-this answer that repeats an existing typed name.
```

## Review findings fixed

- Finding (must-fix): an automatic link outlived a rename or typo fix, and no card could remove it.
  - Fix: `_unlink_old_name` runs before re-linking. It deletes 'same' rows involving the individual whose ask is a `vision_individual` ask (a name-made link), and never touches her own pair answers.
  - Evidence: `test_a_corrected_name_drops_the_old_names_links_but_not_her_pair_answers`, plus mutation checks.
- Finding (should-fix): a case-only difference logged a false group-name conflict.
  - Fix: `lower(label) <> lower(:l)` in the conflict check.
  - Evidence: `test_a_case_only_difference_is_one_name_not_a_conflict`.
- Finding (should-fix): no tests for the heal path or the split.
  - Fix: added `test_healed_pair_names_link_once` and `test_a_name_never_links_across_a_split`.
- Not changed (nits): vision_split keeps its own `_same_name_target`. The reviewer confirmed it is equivalent today. Heal timestamps on copies and stored-label whitespace are left as low-risk.

## Restart required

```bash
# none by hand: the post-pull rebuild rebuilds orion-sql-writer from the clean main checkout after merge
```

## Risks / concerns

- Severity: low
- Concern: more automatic 'same' links mean a wrongly typed name merges two people until Juniper corrects the name.
- Mitigation: the rename now removes the link. 'different' verdicts are respected across the whole group. Every link is logged.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
