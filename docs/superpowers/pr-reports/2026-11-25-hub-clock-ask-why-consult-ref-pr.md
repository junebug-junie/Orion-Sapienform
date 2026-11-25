## Summary

- The Hub header shows the time now in UTC and Mountain time, under the UI version label: `12:07am UTC / 5:07pm MST` (MDT in summer).
- "Orion is asking" cards that say who someone is ("Who is this?", "Is this Sam?") get the same optional **How could you tell?** box as the walkway who-card.
- Her reason becomes a fact on the person, the same kind the walkway reasons already make. For an "Is this Sam? No" answer, the fact says it was a time it was NOT Sam.
- Every Orion & Claude proposal card shows `ref consult <id>`. Orion opens it in chat with a new lookup command, `python -P -m orion.dig consult <id>`.

## Outcome moved

- Juniper can read UTC log times against her own clock at a glance.
- Reasons typed on picture cards reach the person record. Before, only walkway story reasons did.
- Juniper can name a proposal in chat and Orion can read it. Before, consults were never in chat history, so Orion had no way to see what she meant.

## Current architecture

- Header: `#hubUiBuildLabel` showed only the UI version.
- Picture cards posted `/api/asks/{id}/answer` with an answer, names and per-picture choices. They had no reason field. The walkway who-card stored `answer_detail.why`. `person_record.apply_story_reasons` read only `via=motion_story` rows.
- Consult cards (`static/js/consult.js`) showed no id. The dig kit had no consult reader, and `orion_readonly` could not read `orion_consult`.

## Architecture touched

- orion-hub: header template, `hub-clock.js`, `vision-asks.js`, `consult.js`, `ask_routes.py` (the `why` field and the shared `clean_why`), abilities catalog.
- orion-sql-writer: `person_record.apply_card_reasons` runs in the same savepoint as the story reasons.
- orion/dig: the `consult` command, one pointer in the always-on brief line (cap 740 to 800, dated reason in `brief.py`), README.
- orion/person_model: the `VOICE_CARD_REASON` constant and a categories comment.

## Files changed

- `services/orion-hub/static/js/hub-clock.js`, `hub-clock.test.js`: the clock and its MST/MDT test.
- `services/orion-hub/templates/index.html`: the clock element and its script tag.
- `services/orion-hub/static/js/vision-asks.js`, `vision-asks.test.js`: the reason box on who-is-this and is-this cards, sent only with an answer.
- `services/orion-hub/scripts/ask_routes.py`: `AskAnswerBody.why`, `WHY_KINDS`, 422 on other cards, and `clean_why` (moved here).
- `services/orion-hub/scripts/motion_story_routes.py`: imports `clean_why` and `WHY_MAX_CHARS` from ask_routes, so there is one copy.
- `services/orion-hub/tests/test_ask_routes.py`: tests for the reason on the route.
- `services/orion-sql-writer/app/person_record.py`, `tests/test_person_record_pg.py`: `card_reason_plan`, `apply_card_reasons`, and the tests for them.
- `orion/person_model/facts.py`, `categories.yaml`: the `card_reason` voice.
- `services/orion-hub/static/js/consult.js`, `tests/test_consult_js.py`: the ref line on proposal cards and summary lines.
- `orion/dig/consults.py`, `orion/dig/__main__.py`, `orion/dig/tests/test_consults.py`, `orion/dig/brief.py`, `orion/dig/README.md`: the `dig consult` command.
- `services/orion-hub/static/catalog/orion-abilities.json`: the catalog entry that the dig-coverage gate requires.
- `scripts/sql/2026-11-25_grant_orion_readonly_consult.sql`: read-only grant for the two consult tables.

## Schema / bus / API changes

- Added: optional `why` (max 1000 chars) on `POST /api/asks/{id}/answer`. It is stored as `answer_detail.why`. Only `vision_individual` and `vision_person_confirm` cards accept it; other cards return 422 `why_only_on_who_is_this_and_is_this_cards`.
- Added: `person_fact` rows with voice `card_reason`, category `how_you_could_tell`, fact id `why-<ask_id>`.
- Removed / renamed: none.
- Compatibility: additive. An old client that sends no `why` behaves as before. No bus or schema-registry change.

## Env/config changes

- None. No `.env_example` changed, so no sync was needed.

## Tests run

```text
node --test services/orion-hub/static/js/*.test.js           all pass (hub-clock 2/2, vision-asks 41/41)
pytest orion/dig/tests services/orion-hub/tests/test_orion_abilities_catalog.py    391 passed
pytest services/orion-hub/tests -k "index or template or static or ask or consult or motion_story or walkway or people or ui"
                                                               898 passed; 4 errors in test_consult_belief_store_sql.py (needs Postgres; same on main)
ORION_TEST_PG_URI=<throwaway pg16> pytest services/orion-sql-writer/tests/test_person_record_pg.py   14 passed (2 new)
scripts/check_integration_seam.py, check_personal_terms.py, check_migration_required_by.py, check_async_routes_not_blocking.py: ok
```

## Evals run

```text
orion/physiology/evals/physiology_state_skill_eval.py   passed (checks the dig brief length against its cap)
```

No eval measures how often reasons are typed or turned into facts. The PG test covers the full write path.

## Docker/build/smoke checks

```text
Browser smoke tests/test_orion_is_asking_browser_smoke.py: SKIPPED locally (no playwright); runs in CI.
No Docker build run: the change has no new dependency and no compose change.
```

## Review findings fixed

- Finding: on a "Who is this?" card where another picture was marked "someone else" (with its own name), the reason was saved on the first person. It could be about the other one.
  - Fix: `card_reason_plan` and `_CARD_REASONS_SQL` skip a who-card with any picture marked "different", the same way they skip an "Is this" card answered both ways.
  - Evidence: `c-two` in `test_card_reasons_land_on_the_named_person_and_a_no_says_not_them`, plus a pure-plan assert.
- Finding: rows that can never become a fact (vehicle clusters, mixed answers) were read again on every pass. The skipped count grew with each pass.
  - Fix: the SQL filters them out (`i.kind = ANY(person kinds)`, plus the mixed-picture checks). Python now skips only a malformed "Is this" ref.
  - Evidence: the PG test adds a vehicle row and a two-person row. Neither is counted. `c-bad` is the one skip.
- Finding (nit): `check_id(None)` crashed while building its own error message.
  - Fix: `str(consult_id or "")`.

## Restart required

Rebuild Hub and sql-writer after merge (the post-pull rebuild does this). Then apply the grant once:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < scripts/sql/2026-11-25_grant_orion_readonly_consult.sql
```

The FCC sandbox sync after deploy makes `dig consult` live for Orion.

## Risks / concerns

- Severity: low. Concern: `dig consult` fails until the grant is applied. Mitigation: it names the missing GRANT and never returns an empty answer.
- Severity: low. Concern: an "Is this <name>?" reason waits if two active people share that name. Mitigation: it is counted as `card_reasons_waiting` and stored once the tie clears.
- Severity: low. Concern: the reason box also shows on "whose vehicle is this?" cards. A reason typed there stays on the ask row and becomes no fact. The ask carries no field for the cluster kind, and matching on question words is banned. Mitigation: those rows are filtered out, so they cost nothing.
- Severity: low. Concern: a "Who is this?" reason goes to the cluster's newest person link at that moment. If a cluster that already had a link is renamed, the reason could land on the old person, and it is written once. Plain who-cards are only opened for clusters with no name, so this should not happen. It is not checked live.
- Severity: low. Concern: `UNVERIFIED` live: no reason has been typed on a picture card yet, and no consult has been opened from chat yet.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3878
