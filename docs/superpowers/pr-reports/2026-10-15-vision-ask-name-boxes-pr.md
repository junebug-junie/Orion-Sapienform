## Summary

Juniper can now type who someone is on the card she is already answering, instead of waiting for a separate "Who is this?" card.

- **"Are these the same person?" cards** are now a choice (Same person / Different people / Not sure) plus one Send button. Same person opens one optional name box for both faces. Different people opens one optional name box under each face. Not sure has no names.
- **"Who is this?" cards with several pictures** now ask, under every picture after the first, whether it is the same person (the default), someone else (with an optional name), or not sure. The main box still names the first picture, which is the face the question is about. This fixes "there is only one text box so I can't answer" when one card mixes two children.
- A typed name is applied exactly like a "Who is this?" answer: Orion's record of that person gets the name, the name reaches everyone Juniper said is the same person, and the substrate gets the same named-person entity.
- If a named person still has an open "Who is this?" card, that card closes at once, so Juniper is not asked again.
- Per-picture answers are **recorded only**. Nothing splits or rewrites an individual in this PR.

## Outcome moved

- Before: on 2026-10-15 at 22:35 UTC Juniper answered three pair cards, then had to wait for separate "Who is this?" cards to type the names. At 22:45 UTC she could not answer a card with three pictures of two different children.
- After: one card takes the verdict and the names. A mixed card can say "picture 3 is someone else".
- New learning data: the eval now reports how often a "Who is this?" card shows faces from more than one person. Live, read-only, 2026-10-15: **1 of the 2 open multi-picture cards shows faces from more than one person.** Card `ask-64490457…` (individual `f5242d40…`) has three faces, and the least-alike pair has a face similarity of 0.125. That is far below the low edge of the measured same-person band (0.392 = p10 of "the same face a few seconds apart"). The other card (`ask-9dd6a5ad…`, `62cb5fcc…`) has only one picture with a face, so the eval cannot tell. This matches what Juniper saw. n=2, so this is a first count, not a rate.

## Current architecture

- The Hub card (`static/js/vision-asks.js`) posted `{"answer": "yes"|"no"}` for a pair card, and `{"answer": "<text>"}` for a "Who is this?" card. The Hub (`scripts/ask_routes.py`) wrote `orion_ask.status/answer` and published `OrionAskAnsweredV1`.
- sql-writer (`app/vision_individuals.py`) applied "Who is this?" answers as a label, then copied the label to unnamed members of the confirmed group. It applied pair answers separately, after all "Who is this?" answers, in no fixed order.
- No column said whether a label was typed by Juniper or copied from a group member.
- The substrate listener wrote an entity only for `vision_individual` answers.

## Architecture touched

- **Contract:** `OrionAskAnsweredV1.names` (additive, `dict[str, str]`, default empty). The Hub leaves the key out of the bus payload when it is empty.
- **Postgres:** new migration `manual_migration_vision_ask_names_v1.sql` (`orion_ask.answer_detail`, `vision_individual.label_copied`, `vision_ask_picture_verdict`).
- **Hub:** request body, validation, one transaction for the pair answer plus closing "Who is this?" cards, UI.
- **sql-writer:** answers are applied oldest first. There is one typed-name path for both card kinds. Typed and copied labels are kept apart. Per-picture verdicts are recorded.
- **substrate-runtime:** the listener writes one entity per name on a pair answer.

## Files changed

- `orion/schemas/ask.py`: additive `names` on `OrionAskAnsweredV1`.
- `services/orion-sql-db/manual_migration_vision_ask_names_v1.sql`: new, additive, idempotent. The `label_copied` backfill runs only when the column is first created.
- `services/orion-hub/scripts/ask_routes.py`: `names`/`pictures` in the answer body, `pair_answer_detail`, `picture_answer_detail`, `CLOSE_NAMED_WHO_SQL`, one transaction per answer, payload omits empty `names`. A missing column now returns 503 `ask_schema_missing`, not 500.
- `services/orion-hub/static/js/vision-asks.js`: pair card is now radio choice + Send + name boxes. Multi-picture card has a select per picture plus an optional name box. A picked choice blocks the 60 s poll from rebuilding the card.
- `services/orion-sql-writer/app/vision_individuals.py`: `_apply_answers` (oldest first, both kinds), `_set_typed_label`, `_propagate_label(replace_copied=)`, `_close_who_is_this`, `_apply_pair_answer`, `pair_names`, `picture_rows`, `_record_pictures`, `picture_face_split`, `_check_tables` V4.
- `services/orion-substrate-runtime/app/ask_answered_listener.py`: pair names become entities through the same writer as "Who is this?".
- `services/orion-sql-writer/evals/run_vision_confirm_asks_replay_eval.py`: picture purity, plus pair verdicts with their face similarity.
- Tests: `services/orion-hub/static/js/vision-asks.test.js`, `services/orion-hub/tests/test_ask_routes.py`, `services/orion-hub/tests/test_ask_names_pg.py` (new), `services/orion-hub/tests/test_ask_resolve_pg.py`, `services/orion-hub/tests/test_orion_is_asking_browser_smoke.py`, `services/orion-sql-writer/tests/test_vision_confirm_asks.py`, `services/orion-substrate-runtime/tests/test_ask_answered_listener.py`.
- CI: `.github/workflows/orion-sql-writer-tests.yml` (new step: the sql-writer visual-ask tests against real Postgres, and they must not skip). `.github/workflows/orion-memory-episode-tests.yml` (adds `test_ask_names_pg.py` and the migration path).
- Docs: `services/orion-hub/README.md`, `services/orion-sql-writer/README.md`, this report.

## Schema / bus / API changes

- Added: `OrionAskAnsweredV1.names` (individual_id -> name). `POST /api/asks/{id}/answer` accepts optional `names` (pair cards only) and `pictures` (multi-picture "Who is this?" cards only). The response gains `closed_asks`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Pair cards answer through radio + Send, not three buttons. The bodies for no-name answers are unchanged (`{"answer":"yes"}`, `{"answer":"no"}`, dismiss).
  - A "Who is this?" answer now replaces a *copied* name in the confirmed group (before, it only filled empty names).
  - Answers apply in answer-time order.
  - Long labels are cut to 120 characters in sql-writer (the substrate adapter already cut them there).
- Compatibility notes: `OrionAskAnsweredV1` is `extra="forbid"`. An event with no names leaves out the key, so an old consumer accepts it. An event **with** names is rejected by an old orion-substrate-runtime, which drops it with a log line; the Postgres label is still applied by sql-writer. So deploy orion-substrate-runtime before orion-hub. Registry entry unchanged (same model, same kind).

### Decision: typed vs copied names ("last typed answer wins")

A name Juniper types for a person, on either card kind, is a typed label. The newest typed name wins, by answer time, and replacing an older typed name is logged as `[VISION_LABEL_REPLACED]` with the old and new name and both ask ids. `label_ask_id` also records which answer set the name. A typed name replaces a copied one. A copied name (carried across a "same" verdict) never replaces a typed one.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Tests run

```text
node --test services/orion-hub/static/js/vision-asks.test.js                      33 pass, 0 fail
pytest services/orion-hub/tests/test_ask_routes.py                                59 passed, 1 skipped (real-PG test, run separately below)
ORION_TEST_PG_URI=<throwaway pg16> pytest services/orion-hub/tests/test_ask_routes.py -k real_postgres    1 passed
ORION_MEMORY_EPISODE_TEST_DATABASE_URL=<throwaway> pytest services/orion-hub/tests/test_ask_names_pg.py services/orion-hub/tests/test_ask_resolve_pg.py   10 passed
ORION_TEST_PG_URI=<throwaway pg16, 4 walkway migrations> pytest services/orion-sql-writer/tests/test_vision_confirm_asks.py tests/test_vision_individuals.py tests/test_vision_walkway_faces_stationary.py   all passed, 0 skipped
PYTHONPATH=.:services/orion-substrate-runtime pytest services/orion-substrate-runtime/tests/test_ask_answered_listener.py   22 passed
pytest services/orion-hub/tests/test_orion_is_asking_browser_smoke.py (Chromium)  1 passed
scripts/check_migration_required_by.py, check_async_routes_not_blocking.py, check_personal_terms.py, check_env_template_parity.py   OK
Migration: applied twice to a fresh pg16 (idempotent); backfill checked on synthetic rows and re-run (not re-applied)
```

Pre-existing, unrelated failure on main: `tests/test_field_channel_glossary.py` (59 != 58 channels). This diff does not touch it.

## Evals run

```text
python services/orion-sql-writer/evals/run_vision_confirm_asks_replay_eval.py --live <conjourney, READ ONLY> --scratch <throwaway>
OK: 0 ask(s) would open
who_is_this_picture_purity.open_now: band 0.392, cards 2, multi_picture 2, impure 1, unknown 1
  ask-64490457… (f5242d40…): 3 pictures, 3 faces, min face cosine 0.125 -> impure
  ask-9dd6a5ad… (62cb5fcc…): 3 pictures, 1 face -> unknown
pair_verdicts: 4354c877/9bea16d5 same @0.533; 62cb5fcc/f5242d40 different @0.758; 4354c877/f5242d40 different @0.405
```

**Threshold note (recorded, not changed):** Juniper said `62cb5fcc…`/`f5242d40…` are **different people**. Their faces were 0.758 alike when her answer was applied. That is well above the live `VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD=0.60`, which decides whether two faces become one individual. So 0.60 is too loose: it can merge two different children into one individual, which is how mixed "Who is this?" cards come about. This PR does not change it. It is one data point, and the per-picture verdicts this PR records are the data to recalibrate from.

## Docker/build/smoke checks

```text
Not built. Hub change is a static asset + route change; browser smoke ran in Chromium against the real template panel and real JS.
No deploy/restart performed (per instructions).
```

## Review findings fixed

A review subagent read the whole diff and ran the new SQL against a throwaway pg16. It found no blockers.

- Finding: two names typed for one confirmed group (a typed "Alice", b typed "Bob") were silent. Copied members follow the newest typed name, with no signal.
  - Fix: `_propagate_label` now logs `[VISION_GROUP_NAME_CONFLICT]` when another typed name exists in the group.
  - Evidence: `test_two_typed_names_in_one_group_are_logged` (real Postgres).
- Finding: a picture marked "same" with a different name dropped the client's text without telling anyone.
  - Fix: the Hub returns 422 `same_picture_takes_the_answer`.
  - Evidence: `test_bad_pictures_are_refused[...]`.
- Finding: the memory-episode workflow ran the Hub real-PG tests only on pull requests, not on main pushes that touch the migration or test files.
  - Fix: added the migrations and both Hub PG test paths to the `push:` paths.
  - Evidence: workflow YAML diff.
- Finding (nit): the pair card's legend relied on Tailwind's `sr-only` class to stay hidden.
  - Fix: inline visually-hidden styles.
  - Evidence: browser smoke still passes.
- Finding (nit): `names` values had no length cap at the request-model layer.
  - Fix: each value is capped at 500 characters in `AskAnswerBody`. `clean_name` still enforces 120.
  - Evidence: route tests pass.

A second review pass on the updated branch found:

- Finding: if the new Hub runs next to an old sql-writer, the old writer applies the verdict and marks the pair answer applied, but it ignores the names. Those names would be lost for good.
  - Fix: sql-writer now has a heal pass (`_heal_pair_names`). Each tick it applies any typed pair name whose individual's label is older than the answer (logged as `[VISION_PAIR_NAMES_HEALED]`). A newer name is never overwritten. The Risks section also says to deploy sql-writer before the Hub.
  - Evidence: `test_names_marked_applied_by_an_old_writer_are_healed` (real Postgres).
- Finding: the one-time `label_copied` backfill had no test.
  - Fix: `test_label_copied_backfill` drops the column in a transaction, seeds labels typed on the person's own card, set by a pair answer, copied from someone else's card, entered with no ask, and absent, then runs the real migration file twice.
  - Evidence: false / true / true / false / false; the second run does not re-backfill.
- Finding: a repeated picture ref was indexed at its last position, but the card shows the first.
  - Fix: the first position wins, and refs are trimmed.
  - Evidence: `test_picture_index_is_the_first_position_shown`.
- Finding: names sent to a card that is already closed returned 422, not 409.
  - Fix: the locked SELECT checks status and expiry first.
  - Evidence: `test_names_on_a_closed_card_are_409_not_422`.
- Finding: `CLOSE_NAMED_WHO_SQL` closed expired-but-still-open cards.
  - Fix: it now checks expiry.
  - Evidence: real-PG SQL test passes.
- Not changed, recorded:
  - The substrate writes the entity twice for a named face that also had an open "Who is this?" card. The second write lands on the same node id and logs "updated" (`test_same_answer_twice_is_idempotent`).
  - Cards closed for people who only got a *copied* name publish no event, so they get no substrate entity. This is now written in the README.
  - Four pictures at 360 px would be tight. The opener caps "Who is this?" cards at 3 pictures, and the browser smoke covers 3.
  - `vision_crop_observation` has no index on the thumbnail refs. Each multi-picture answer scans about 13k rows; the table is kept to 7 days.

## Restart required

Order matters: migration first, then the consumer, then the producer.

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_ask_names_v1.sql
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Deploy note: the post-pull hook currently recreates `orion-sql-db` when a new migration file arrives. The fix for that is on branch `fix/rebuild-skip-sql-db-migrations`, not merged yet. Expect a short `orion-sql-db` restart on pull until that lands.

Post-deploy check:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select individual_id, label, label_copied from vision_individual where label is not null"
# expect 9bea16d5…|<name>|f (typed) and 4354c877…|<name>|t (copied by the 22:35 "same" answer) -- the one-time backfill
```

Then answer one card with a name and check:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select ask_id, status, answer, answer_detail, applied_at from orion_ask where answered_at > now() - interval '10 minutes'"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select * from vision_ask_picture_verdict order by created_at desc limit 5"
```

Expect `applied_at` set within about 60 s (one individuals tick). Expect `[VISION_PAIR_VERDICT] ... names=N` in the orion-sql-writer logs and `ask_answered entity_` in the orion-substrate-runtime logs.

## Risks / concerns

- Severity: medium. Concern: if orion-hub deploys before orion-substrate-runtime, the substrate drops named pair answers (`extra="forbid"`). Mitigation: the deploy order above. The Postgres labels are unaffected, and empty `names` are left out of the payload.
- Severity: low. Concern: if orion-hub deploys before orion-sql-writer, the old writer marks named pair answers applied without applying the names. Mitigation: the new writer's heal pass applies them on its first tick. Deploy sql-writer before Hub anyway.
- Severity: medium. Concern: if orion-hub deploys before the migration, every Answer and Dismiss fails. Mitigation: it now returns 503 `ask_schema_missing`, not 500. `safe_docker_build.sh` refuses `up` while a required migration is unapplied (header `ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub`).
- Severity: low. Concern: the per-picture verdicts are recorded but not used. Mitigation: this is intended (record only). Splitting impure individuals is a follow-up that needs a proposal (identity/memory change).
- Severity: low. Concern: the purity count is n=2. Mitigation: the eval reruns on demand, and every new multi-picture answer adds labelled rows.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2701

🤖 Generated with [Claude Code](https://claude.com/claude-code)
