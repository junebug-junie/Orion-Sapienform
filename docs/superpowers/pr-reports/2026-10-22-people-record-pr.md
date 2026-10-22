## Summary

- Orion now keeps **one record per person**. Every camera cluster Juniper named links to one person per name. A name in `config/person_model/people.yaml` (Juniper) joins its existing referent id. Named bikes and cars link as that person's belongings. Junk names like "no" are ignored.
- Orion **asks Juniper about people**, which the 10-14 design promised but never built. The order is: "Who is X to you?", then pronouns, age, gender, sexuality, likes, routines, anything else. She gets "you" wording about herself. Each question shows the person's best camera picture. Her answers are stored in her own words, never inferred.
- The questions have **their own daily cap** (3), shared with nothing, so the starvation fixed in #2841 cannot happen here.
- A Hub panel, **People Orion knows**, sits under "Orion is asking". It shows each person's picture, days seen, what she said, and Orion's open question. She can add or remove a fact, or mark a name as "not a person".
- PR #2847's graph guards are adopted: a "different people" verdict blocks a merge, and only person kinds become people. #2847's graph slice reads `person_link` as its seam.

## Outcome moved

- Orion had asked 0 questions about anyone's life. After deploy it asks up to 3 a day, about the 11 people found on live data.
- Camera names were scattered across 41 clusters. They now resolve to 11 people with one id each.

## Current architecture

Orion kept three lists of people that nothing connected:
- camera clusters with text labels (`vision_individual.label`);
- one graph node per cluster;
- the 10-14 person model, which held 2 people, had no facts table, and asked no questions.

`orion_ask` had only camera and memory question kinds.

## Architecture touched

- orion-sql-writer: the individuals tick runs a person-record step after answers apply. It has its own transaction, and a savepoint around opening questions.
- orion-hub: new routes and a new panel.
- orion/person_model: the link planner, the question list, the dry-run eval.
- New migration.

## Files changed

- `services/orion-sql-db/manual_migration_person_record_v1.sql` (+ rollback): `person`, `person_link`, `person_fact`.
- `orion/person_model/record.py`: pure link planner. Links come only from typed names. The "different" guard applies. people.yaml owns its name. Renames revoke the old link. Belongings link to their owner.
- `orion/person_model/facts.py`, `categories.yaml`: which question comes next, its wording, and verbatim answers.
- `orion/person_model/evals/run_person_record_backfill_eval.py`: read-only live dry run.
- `services/orion-sql-writer/app/person_record.py`: writes links, seeds pronoun facts, applies answers, opens questions.
- `services/orion-sql-writer/app/vision_individuals.py`, `vision_individuals_loop.py`, `settings.py`, `.env_example`, `Dockerfile` (copies people.yaml), `README.md`.
- `services/orion-hub/scripts/people_routes.py`, `static/js/people.js`, `templates/index.html`, `scripts/main.py`, `README.md`.
- `orion/schemas/ask.py`: documents the `person_fact` kind.
- Tests: `orion/person_model/tests/test_record.py`, `test_facts.py`; `services/orion-sql-writer/tests/test_person_record_pg.py`; `services/orion-hub/tests/test_people_routes.py`, `test_people_panel_browser_smoke.py`. CI: `.github/workflows/schedule-browser-smoke.yml`.
- `docs/superpowers/specs/2026-10-22-people-one-record-design.md`: the spec, rev 2 as built. It supersedes the design-only PR #2845.

## Schema / bus / API changes

- Added: tables `person`, `person_link`, `person_fact`; ask kind `person_fact` (`source_ref` `<person_id>|<category>`); Hub `GET /api/people`, `POST /api/people/{id}/facts`, `POST /api/people/{id}/facts/{fact_id}/remove`, `POST /api/people/{id}/status`.
- Removed: none.
- Renamed: none.
- Behavior changed: the individuals tick also maintains the person record and opens person questions.
- Compatibility notes:
  - No new bus channel or payload field.
  - Existing ask consumers ignore the new kind; the review checked each one.
  - The migration is additive and passes the post-pull auto-apply allowlist (`auto_apply_refusals` = []).

## Env/config changes

- Added keys (orion-sql-writer): `PERSON_RECORD_ENABLED=true`, `PERSON_FACT_ASK_DAILY_CAP=3`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced: yes. The keys were appended to the primary checkout's `services/orion-sql-writer/.env`. `check_env_template_parity.py`: PASS.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/person_model/tests/test_record.py test_facts.py test_no_person_special_cases.py   21 passed
services/orion-sql-writer/tests/test_person_record_pg.py (real postgres:15, all migrations)  8 passed
existing camera suites, same database: test_vision_individuals 27, test_vision_person_model 38,
  test_vision_confirm_asks 44, test_vision_split 45                                       154 passed
services/orion-hub/tests/test_people_routes.py (real Postgres)                             6 passed
hub regression: test_ask_routes, test_vision_asks_js, test_motion_story_routes           96 passed, 1 skipped
test_people_panel_browser_smoke.py + test_orion_is_asking_browser_smoke.py (Chromium)     2 passed
tests/test_no_personal_terms.py, test_sql_migration_drift_gate.py, test_deploy_migration_gate.py  149 passed
scripts/check_async_routes_not_blocking.py: no blocking calls
```

The other `orion/person_model/tests` failures (20) also fail on main. They come from `rapidfuzz` missing in the local venv.

## Evals run

```text
orion/person_model/evals/run_person_record_backfill_eval.py --live <READ ONLY>
11 people from 38 links; 0 conflicts; 3 named things ignored (2 junk vehicle names,
1 bicycle labelled with a full name that does not match people.yaml's first name).
Two entries are groups rather than people (a delivery company; two children named together);
the Hub "Not a person" button is for those.
```

## Docker/build/smoke checks

```text
Not built here. The image change is one COPY of config/person_model/people.yaml (the build context
is the repo root; the review confirmed the path resolves to /app/config/person_model/people.yaml and
that PyYAML is in the live image). The live path is UNVERIFIED until deploy.
```

## Review findings fixed

- Finding (slice 1, medium): a people.yaml row added after the camera had already created that name split the person. New clusters then stayed in conflict forever, and the belongings were revoked.
  - Fix: a people.yaml person owns its name, so camera-created duplicates stop being candidates and their links move over.
  - Evidence: `test_a_people_yaml_row_added_later_takes_over_a_camera_minted_person`.
- Finding (slice 1, medium/low): after a rename, the old name stayed an active person, and Orion asked about it first.
  - Fix: questions go only to people with a live link, plus Juniper. The Hub hides names with no link and no facts.
  - Evidence: `test_a_corrected_typo_leaves_no_person_to_ask_about`.
- Finding (low): the cooldown default was not longer than the expiry, so an expired question could come straight back.
  - Fix: the default is now 30 days. Production already passed 30.
- Finding (low): an answer about a missing person was dropped silently.
  - Fix: it is now logged as `person_fact_answer_not_stored`.
- Finding (low): a failing question rolled back the link sync.
  - Fix: opening questions runs inside a savepoint.
- Not changed, by design: names must match exactly. Live data has one first-name versus full-name pair that becomes two people; renaming on a camera card merges them.
- Not changed: junk names on person clusters are not filtered. Live data has none, and "Not a person" handles them.
- Finding (Hub, medium): a failed save silently threw away what she typed.
  - Fix: on error the panel says "That did not save" and keeps the text.
  - Evidence: the browser smoke covers a 503.
- Finding (Hub, low-medium): removing a fact that had no question behind it (people.yaml, or a Hub edit) made Orion ask that category again.
  - Fix: any fact, removed or not, marks its category as done.
  - Evidence: `test_a_removed_fact_is_not_asked_about_again`.
- Finding (Hub, low): the minute-by-minute refresh could wipe unsaved text.
  - Fix: the poll is skipped while any box holds text.
- Finding (Hub, low): a double tap on Save created two facts.
  - Fix: the card's buttons lock while a request is in flight.
- Not changed: "days seen" sums days across a person's clusters, so it can overcount a day two clusters share. Question ordering uses the same sum, so the two agree.

## Restart required

From the clean primary main checkout after merge (the migration auto-applies on pull):

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Then check the live path:

```sql
SELECT count(*) FROM person_link WHERE revoked_at IS NULL;   -- expect 38
SELECT question FROM orion_ask WHERE source_kind='person_fact';   -- expect rows within a tick
```

## Risks / concerns

- Severity: low
  - Concern: two real people with the same name merge, unless Juniper said "different people" on a camera card.
  - Mitigation: links are revocable, the guard is tested, and the dry run shows 0 conflicts.
- Severity: low
  - Concern: up to 3 more Hub cards a day.
  - Mitigation: `PERSON_FACT_ASK_DAILY_CAP` (0 = off).
- Severity: info
  - Concern: chat does not read the facts yet, and the graph has no person node yet.
  - Mitigation: these are the next slices. Chat needs a reader-first schema field, because the street line is clipped at 520 characters. The graph is PR #2847's slice, built on `person_link`.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
