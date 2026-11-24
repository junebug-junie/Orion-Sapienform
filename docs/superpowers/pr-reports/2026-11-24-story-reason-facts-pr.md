## Summary

- Each "How could you tell?" reason from a walkway story now becomes a fact on that person's People page (category "How you could tell").
- It lands on the person that camera cluster is linked to. For "Someone else", that is the person Orion had guessed, and the fact starts "A sighting that was NOT them: ...".
- "Not sure" reasons are skipped: they say nothing about the person. Dogs and other non-person clusters are skipped too.
- The category is never asked as a question.
- Patch 1 of the clue-evidence design (#3811, ticket #3768). No guess changes yet.

## Outcome moved

Juniper's reasons were stored on the answer and read by nothing. Now every reader of person facts sees them: the People page, recall (`orion/memory/lived_memory.py`), core pages, mesh edges. Patch 2 learns its clue list from them.

## Current architecture

`run_person_record` (orion-sql-writer, inside the camera individuals tick) links named clusters to `person` rows, then turns answered person questions into `person_fact` rows. Story reasons sat in `orion_ask.answer_detail.why`.

## Architecture touched

- `services/orion-sql-writer/app/person_record.py`: new `apply_story_reasons`, in its own savepoint after `apply_fact_answers`.
- `orion/person_model/facts.py`, `orion/person_model/categories.yaml`: new voice `story_reason`, category `how_you_could_tell` (both question templates null).

## Files changed

- `services/orion-sql-writer/app/person_record.py`: reasons -> facts; summary keys `story_reasons_stored|waiting|skipped`.
- `orion/person_model/facts.py`: constants.
- `orion/person_model/categories.yaml`: the category.
- `services/orion-sql-writer/tests/test_person_record_pg.py`: lands on the right person, "not them" marking, unsure skipped, dog skipped, unlinked retried, removed not re-added, failure never rolls back links.

## Schema / bus / API changes

- Added: `person_fact` category `how_you_could_tell`, voice `story_reason`, fact_id `why-<ask_id>`. No DDL (category and voice are free text).
- Removed / Renamed: none.
- Behavior changed: the Hub add-fact list also offers "How you could tell" (all categories are offered).
- Compatibility notes: additive.

## Env/config changes

- None.

## Tests run

```text
pytest services/orion-sql-writer/tests/test_person_record_pg.py   (throwaway Postgres 16, person migrations)  12 passed
pytest services/orion-sql-writer/tests/test_vision_person_model.py                                            48 passed, 27 skipped
pytest tests/test_no_personal_terms.py                                                                         7 passed
orion/person_model/tests and hub people tests: same 20 / 8 failures on clean origin/main (rapidfuzz missing locally); not from this change.
```

## Evals run

```text
Read-only dry run on live data (same SQL as the writer): of 7 saved reasons, 6 land on the right person; 1 is about a dog (skipped).
```

## Docker/build/smoke checks

```text
Not run. Pure sql-writer Python; no dependency, env or compose change. Post-pull rebuild covers orion-sql-writer.
```

## Review findings fixed

- Finding: the step ran outside a savepoint, so an error would roll back that tick's person links and answers.
  - Fix: own `begin_nested()`; error logged as `story_reasons_failed`, summary `story_reasons_error`.
  - Evidence: `test_a_failing_story_reason_never_rolls_back_the_links`.
- Finding: the fact dropped what her answer was, so "not Jason because X" would read as a clue for Jason in recall and core pages.
  - Fix: "someone else" reasons are prefixed "A sighting that was NOT them: "; "not sure" reasons are skipped.
  - Evidence: test asserts the prefix and that the unsure reason stores nothing.
- Finding: a deleted cluster was counted as "not a person".
  - Fix: counter renamed `story_reasons_skipped` (non-person or deleted cluster).
- Accepted, documented in the module doc: a fact is written once to the person linked at that moment; a later relink does not move it.
- Left as is: the Hub add-fact list now offers "How you could tell" for hand-written reasons.

## Restart required

```text
No manual restart: the post-pull rebuild rebuilds orion-sql-writer after merge.
```

## Risks / concerns

- Issues opened: none. Refs #3768.
- Severity: low. A reason given on a mis-clicked "yes" lands on the wrong person (one live case, 2026-11-24 05:28 UTC). She can remove it on the People page.
- Live path UNVERIFIED until merged and one tick runs: check `SELECT count(*) FROM person_fact WHERE voice='story_reason'` = 6.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3824

🤖 Generated with [Claude Code](https://claude.com/claude-code)
