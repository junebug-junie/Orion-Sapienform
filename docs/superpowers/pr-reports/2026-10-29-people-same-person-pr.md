## Summary

- Orion matches names exactly. Juniper named 23 camera groups with a first name and 1 with the full name, so Orion made two people. She answered "How do you know <full name>?" and was then asked the same about the first name.
- Each card in "People Orion knows" now has **Same person as… → Join**. The joined name shows as "Also called <name>" on the person kept, with an Undo button.
- On a join:
  - the joined name's facts move to the person kept;
  - its open question closes;
  - orion-sql-writer moves its camera groups to the person kept on its next tick.
- Nothing is joined automatically. Two real people can share a first name.
- An answer, or a fact typed, for a joined name lands on the person kept.

## Outcome moved

- Live 10-29: one person had two records, and the relationship question was asked twice. After she joins them, it is one record and one question.
- This also covers nicknames (two names of one child).

## Current architecture

- `record.plan` links each camera name to the person whose name folds the same.
- A first name and a full name are different keys, so they became two `person` rows. Each row got its own questions.

## Architecture touched

- New column `person.merged_into`. The migration is additive with no foreign key, so it passes the post-pull auto-apply allowlist.
- `orion/person_model/record.py`: a joined name resolves to the person kept. Following the chain is safe against loops.
- orion-sql-writer `person_record.py`:
  - never asks about a joined person;
  - writes late answers to the person kept.
- orion-hub `people_routes.py`:
  - `POST /api/people/{id}/same-as`;
  - `also_called` on `GET /api/people`;
  - joined names are hidden from the list.
- orion-hub `people.js`: the Join control and "Also called" with Undo.
- `orion/person_model/pictures.py`: a picture picked for a joined name counts for the person kept.

## Files changed

- `services/orion-sql-db/manual_migration_person_merge_v1.sql` (+ rollback): `person.merged_into`.
- `orion/person_model/record.py`, `tests/test_record.py`.
- `orion/person_model/pictures.py`.
- `services/orion-sql-writer/app/person_record.py`, `tests/test_person_record_pg.py`, `README.md`.
- `services/orion-hub/scripts/people_routes.py`, `static/js/people.js`, `tests/test_people_routes.py`, `tests/test_people_panel_browser_smoke.py`, `README.md`.

## Schema / bus / API changes

- Added:
  - `person.merged_into`;
  - `POST /api/people/{id}/same-as` with body `{into: <person_id>|null}`;
  - `also_called: [{person_id, name}]` on each person in `GET /api/people`.
- Removed: none. Renamed: none.
- Behavior changed:
  - joined names leave the People list and are not asked about;
  - facts and answers for a joined name go to the person kept.
- Compatibility notes:
  - No bus change.
  - The graph loop and the walkway rollups read through `person_link`, which moves to the person kept. The joined row's graph node retires on the next graph tick.
  - The Hub needs the migration (`REQUIRED-BY: orion-sql-writer, orion-hub`). It auto-applies on pull.

## Env/config changes

- None.

## Tests run

```text
orion/person_model/tests (record, facts, special-cases gate), test_no_personal_terms, migration gates   passed
orion-sql-writer (real postgres:15): test_person_record_pg 10, test_vision_person_model, test_vision_individuals,
  test_vision_confirm_asks                                                                   122 passed
orion-hub: test_people_routes (real Postgres) 7; test_ask_routes + test_vision_asks_js 78 passed, 1 skipped
orion-hub Chromium: test_people_panel_browser_smoke (join, 360 px, undo)                      2 passed
```

## Evals run

```text
Live read-only diagnosis 2026-10-29: first name 23 links + full name 1 link = two person rows; the
relationship was answered on the full-name row and asked again on the first-name row.
A second open relationship question dates from 10-26 and was never answered (no fact exists): not a repeat.
```

## Docker/build/smoke checks

```text
Not built here. No Dockerfile or dependency change. The live path is UNVERIFIED until deploy.
```

## Review findings fixed

- Finding (medium): an answer already in, but not yet applied, was written to the hidden joined row.
  - Fix: `apply_fact_answers` and the Hub `add_fact` write to `coalesce(merged_into, person_id)`.
  - Evidence: the `ask-late` case in the writer test and the fact case in the Hub test.
- Finding (medium): the interlocutor's own card could be joined away.
  - Fix: 422 `interlocutor_is_kept`, and the panel shows no Join control on that card.
- Finding (medium): Undo could not be reached from the UI.
  - Fix: each "Also called" line has an Undo button. The dropdown says which way the join goes.
  - Evidence: the Chromium smoke.
- Finding (low): two joins at once could leave a two-step chain.
  - Fix: the route locks the source row and its joined rows with `FOR UPDATE`.
- Finding (low): she could join into a "not a person" row.
  - Fix: 422 `into_is_not_a_person`.
- Not changed:
  - A people.yaml referent with exactly the joined name still wins that name. No such row exists today.
  - A "different people" verdict between a joined cluster and the kept person's cluster leaves that cluster unlinked and logs the conflict. That is the intended guard.
  - Undo does not move facts back to the restored name.

## Restart required

From the clean primary main checkout after merge (the migration auto-applies on pull):

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: a wrong join moves camera groups to the wrong person.
  - Mitigation: Undo is one click, and the next tick moves the links back.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2947

🤖 Generated with [Claude Code](https://claude.com/claude-code)
