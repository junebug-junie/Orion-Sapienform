## Summary

- "Who is X to you?" questions showed three different people. They took a camera cluster's newest pictures, and a cluster can hold more than one person. They now show only pictures Juniper tied to that name herself: the picture she named on a "who is this?" card, ones she marked "the same person", a face she named on a "same person?" card, and a picture she picked on an "Is this X?" card.
- Questions already open get their pictures corrected on every pass.
- "Is this X?" cards with several pictures put a choice under each picture: Not sure / X / Not X. "One of the three is Emerald" can now be said. A mixed answer never names the whole cluster; each marked picture teaches Orion about that one crop.
- The relationship question now reads "How do you know X? For example: a family friend, a neighbor, a classmate of your kid." The old "Who is X to you?" was not clear.
- The People panel uses the same confirmed pictures.

## Outcome moved

- Live, read-only: 9 of 11 people have confirmed pictures (38 rows; Maeve 6, Emerald 4). The query takes 9 ms.
- Before: a person question could show anyone in a mixed cluster. After: only pictures she tied to that name, or none.

## Current architecture

- `person_record._pictures` took the newest 3 sighting thumbnails of the person's most-seen linked cluster.
- An "Is this X?" card took one yes/no for all its pictures. Yes named the whole cluster.

## Architecture touched

- orion/person_model: new `pictures.py`, one SQL string used by orion-sql-writer and orion-hub.
- orion-sql-writer:
  - `person_record.py` uses confirmed pictures and refreshes open asks.
  - `vision_individuals._apply_person_confirm` reads per-picture verdicts.
- orion-hub:
  - `ask_routes.confirm_picture_detail` validates the per-picture answer.
  - `vision-asks.js` renders the per-picture card.
  - `people_routes` uses confirmed pictures.

## Files changed

- `orion/person_model/pictures.py`: the confirmed-pictures query.
- `orion/person_model/categories.yaml`, `tests/test_facts.py`: the reworded relationship question.
- `services/orion-sql-writer/app/person_record.py`: confirmed pictures, refresh of open asks, unused parameter dropped.
- `services/orion-sql-writer/app/vision_individuals.py`:
  - `confirm_picture_verdicts`;
  - `_crops_for_refs` (moved out of `_record_pictures`);
  - per-picture exemplars.
- `services/orion-sql-writer/README.md`, `services/orion-hub/README.md`.
- `services/orion-hub/scripts/ask_routes.py`, `people_routes.py`, `static/js/vision-asks.js`.
- Tests:
  - `test_person_record_pg.py`, `test_vision_person_model.py` (real Postgres);
  - `test_ask_routes.py`, `test_vision_asks_js.py`, `test_people_routes.py` (real Postgres);
  - `test_orion_is_asking_browser_smoke.py` (Chromium).

## Schema / bus / API changes

- Added:
  - `answer_detail.pictures` on `vision_person_confirm` answers: `[{image_ref, index, verdict}]`, where verdict is same, different, or unsure;
  - exemplar sources `confirm_pick_yes` (+1) and `confirm_pick_no` (-1).
- Removed: none. Renamed: none.
- Behavior changed:
  - person-fact question pictures;
  - the relationship question wording;
  - mixed "Is this X?" answers do not name the cluster.
- Compatibility notes: no migration and no bus change. Single-picture cards and old answers work as before.

## Env/config changes

- None. No `.env_example` change.

## Tests run

```text
orion-sql-writer (real postgres:15): test_person_record_pg, test_vision_person_model, test_vision_individuals,
  test_vision_confirm_asks, test_vision_split                                            all passed (49 + 116)
orion-hub: test_people_routes (real Postgres), test_ask_routes, test_vision_asks_js   79 passed, 1 skipped
orion-hub Chromium: test_orion_is_asking_browser_smoke, test_people_panel_browser_smoke  2 passed
orion/person_model tests, test_no_person_special_cases, test_no_personal_terms, test_schema_skew_discovery  passed
```

## Evals run

```text
Live read-only run of the confirmed-pictures query: 38 picture rows, 9 of 11 people, 9.0 ms (EXPLAIN ANALYZE).
```

## Docker/build/smoke checks

```text
Not built here. No Dockerfile or dependency change. The live path is UNVERIFIED until deploy.
```

## Review findings fixed

- Finding (medium): Send with nothing chosen dismissed the card, and a dismissed pair is never asked again.
  - Fix: Send is refused with a message, and "Can't tell" is its own button.
  - Evidence: the JS unit test and the Chromium smoke.
- Finding (low): a picture the card could not show made the answer fail with 422.
  - Fix: a picture left out counts as "not sure".
  - Evidence: `test_a_picture_the_card_could_not_show_counts_as_not_sure`.
- Finding (low): a picture she later marked "someone else" or "not X" could still show.
  - Fix: both are excluded in the query.
  - Evidence: the extended `test_a_person_question_shows_only_pictures_she_tied_to_that_name`.
- Not changed: after a mixed answer with no "not X" picture, the album may guess X for that cluster again. The card is not asked again. The guess only feeds the handoff reader, as before.
- Not changed: the People panel now also needs the album tables. They exist live.

## Restart required

From the clean primary main checkout after merge:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: a person with no confirmed picture gets a question with no picture.
  - Mitigation: the question names them. Live, 2 of 11 people have none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2859

🤖 Generated with [Claude Code](https://claude.com/claude-code)
