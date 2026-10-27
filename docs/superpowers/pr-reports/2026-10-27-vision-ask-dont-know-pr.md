## Summary

- "Who is this?" and "Do you know whose vehicle this is?" cards now have an **I don't know** button next to Answer and Dismiss.
- Pressing it closes the card and stores **no name**. Before this change, typing was the only way to say "I don't know". The typed text ("no", "unsure", "I do not know whose vehicle this is.") was saved as the vehicle's name.
- Orion may ask again after the normal re-ask wait (`VISION_ASK_COOLDOWN_DAYS`, 30 days). The wait starts when she presses the button. A plain Dismiss still means "never ask again".
- Typed answers are not filtered by any word list. The button is the fix.
- Production: the 4 existing non-answer vehicle names were cleared. The rows were snapshotted to the operator's `/tmp` first.

## Outcome moved

A non-answer can no longer become a name. Before this change, 4 named vehicles were called "no", "unsure" (twice) and "I do not know whose vehicle this is.". These non-answers also appeared in the walkway day line that Orion's day letter reads (for example "named: <name> (10:24), no (19:15)"). The button stores `answer` NULL. sql-writer never applies a label from it, because `_apply_answers` reads only `status='answered'` rows.

## Current architecture

The Hub card (`static/js/vision-asks.js`) posts to `POST /api/asks/{id}/answer` or `/dismiss` (`scripts/ask_routes.py`). Dismiss set `status='dismissed'`. sql-writer's opener (`_NOT_ASKED_SQL`) treated that as blocked for good. Answer set `status='answered'` with the typed text. sql-writer applied that text as `vision_individual.label`.

## Architecture touched

- orion-hub: `/dismiss` takes an optional body `{"reason": "dont_know"}`, on who-is-this cards only. It uses the existing close path (`_close_ask` -> `_answer_detail` row lock -> `CLOSE_ASK_SQL`). It stores `answer_detail={"dont_know": true}`.
- orion-sql-writer: `_NOT_ASKED_SQL` treats a dont_know dismissal like an expired ask. The individual is blocked only for the cooldown, counted from `answered_at`.
- No bus, schema or registry change. `OrionAskAnsweredV1` is unchanged: the status is `dismissed` and the answer is null. The substrate listener ignores events that are not `answered`.

## Files changed

- `services/orion-hub/static/js/vision-asks.js`: the I don't know button, the dismissal body, the empty-answer hint, and the error text.
- `services/orion-hub/scripts/ask_routes.py`: `AskDismissBody`, and validation and storage of dont_know.
- `services/orion-sql-writer/app/vision_individuals.py`: `DONT_KNOW_SQL`, so a dont_know dismissal is asked again after the cooldown.
- `services/orion-hub/static/js/vision-asks.test.js`, `services/orion-hub/tests/test_ask_routes.py`, `services/orion-hub/tests/test_orion_is_asking_browser_smoke.py`, `services/orion-sql-writer/tests/test_vision_confirm_asks.py`: tests.
- `services/orion-hub/README.md`, `services/orion-sql-writer/README.md`, `services/orion-sql-writer/.env_example` (comment only): docs.

## Schema / bus / API changes

- Added: an optional `/api/asks/{id}/dismiss` body, `{"reason": "dont_know"}`. With no body or an empty body, the route behaves as before.
- Removed: none.
- Renamed: none.
- Behavior changed: a who-is-this dismissal with dont_know is asked again after `VISION_ASK_COOLDOWN_DAYS`. A plain dismissal does not change.
- Compatibility notes: deploy order does not matter. If the Hub ships first, an old sql-writer treats a dont_know row as a plain dismissal, so that individual is never asked again. An old Hub never writes dont_know. `orion_ask.answer_detail` already exists, so no migration is needed.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: comment only. The `VISION_ASK_COOLDOWN_DAYS` comment now says it also covers I don't know.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. No key changed.
- skipped keys requiring operator action: none.

## Tests run

```text
node --test services/orion-hub/static/js/vision-asks.test.js                      37 pass, 0 fail
pytest services/orion-hub/tests/test_ask_routes.py tests/test_vision_asks_js.py   79 passed (ORION_TEST_PG_URI = throwaway PG)
pytest services/orion-hub/tests -k "ask or vision"                                197 passed, 14 skipped
pytest services/orion-hub/tests/test_orion_is_asking_browser_smoke.py             1 passed (real Chromium)
pytest services/orion-sql-writer/tests/test_vision_confirm_asks.py (+faces)       69 passed (throwaway PG, vision migrations)
pytest services/orion-sql-writer/tests -k "vision or person or ask"               427 passed, 128 skipped (no PG)
pytest tests/test_no_personal_terms.py                                            7 passed
python scripts/check_async_routes_not_blocking.py                                 no blocking calls
Mutation: revert the _NOT_ASKED_SQL change -> test_i_dont_know_... fails.
```

With PG, 3 tests fail that this change does not touch. They fail because of how the throwaway database was set up:

- One test expects the `vision_handoff` migration to be absent, and the throwaway database has it.
- Two osnet eval tests run against `--live` data.

## Evals run

```text
No eval harness covers ask cards. Coverage instead:
- the real-Chromium browser smoke above;
- the production cleanup below, checked by reading the live rows.
```

## Docker/build/smoke checks

```text
No image build: no dependency, compose or env-key change.
UI check (CLAUDE.md section 9):
- Rendered template: templates/index.html loads /static/js/vision-asks.js?v={{HUB_UI_ASSET_VERSION}} (unchanged).
- Linked asset: the browser smoke serves the real panel markup from index.html and the real JS file.
- Interaction: the smoke clicks I don't know at 360 px width. It checks that:
  - all three buttons stay inside the card;
  - the request is v1/dismiss {"reason":"dont_know"}, and the typed "unsure" is not sent;
  - the card leaves the list.
```

## Production cleanup (approved by Juniper)

The rows were snapshotted first, to the operator's `/tmp/clear-vehicle-nonanswer-labels/` (`report.md`, `before_after.csv` and the per-table snapshots).

There is no code path that clears a label. The clear was one guarded UPDATE:

- It set `label`, `labeled_at` and `label_ask_id` to NULL, and `label_copied` to false.
- It changed only rows where kind='vehicle' and the individual id, label and label_ask_id all still matched the snapshot.

It changed 4 vehicle clusters. Their labels were "no", "unsure", "unsure" and "I do not know whose vehicle this is.".

- No other cluster had copied these labels, and these 4 clusters had no `person_link` rows, so there was nothing to revoke.
- One derived row was cleared by its own loop. It was a `vision_person_exemplar` row under the name "unsure" (a "this picture is someone else" marker). The person-model loop revoked it 5 minutes after the clear.
- The `orion_ask` rows were left as they were (still `answered`). Orion does not ask about these 4 vehicles again.

### Rollups

The rollup does not recompute `named` for past periods.

- Each tick rebuilds only the last 3 closed hours.
- Day, week and month rows are built from the stored hour rows, and each stored hour row keeps the label it had when it was built.
- Day rows built after the clear (10-26 onward) do not name these vehicles.
- 24 stored rows still name "no" or "unsure":
  - the days 10-20, 10-21, 10-23, 10-24 and 10-25;
  - 16 hour rows;
  - the week of 10-19;
  - October.
- The month row is rebuilt from day rows. Until the old rows are rebuilt, it keeps saying "no on 5 days, unsure on 2 days".
- A dry run of `scripts/backfill_vision_rollups.py --days 8` produced 0 lines that name "no" or "unsure". I did not apply it, for two reasons:
  - It is outside the approved write.
  - Another agent is changing the rollup code.

## Review findings fixed

- Finding: `(answer_detail->>'dont_know')::boolean` raises an error on a malformed row. That error stops the whole opener query.
  - Fix: use containment instead: `answer_detail @> jsonb_build_object('dont_know', true)`. It has no braces, because `_NOT_ASKED_SQL` goes through `str.format`.
  - Evidence: 69 PG tests pass.
- Finding: an empty Answer on a who-is-this card showed "or press Dismiss". That points at the wrong button.
  - Fix: the card now shows its own hint: "Type a name first, or press I don't know."
  - Evidence: a new JS test.
- Finding: `dont_know_only_on_who_is_this_cards` had no friendly text.
  - Fix: added the text to `errorLine`.
- Finding: I don't know drops any per-picture choices.
  - Fix: this is intended. A comment now says so.
- Not fixed (follow-up): if she always answers I don't know about a vehicle, Orion asks again every 30 days with no limit. The daily ask caps limit how often this happens.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- Severity: low.
- Concern: past rollup rows and the October month line still show "no" and "unsure" until they are rebuilt.
- Concern: the substrate entity nodes written from the original answered events may still carry these labels. UNVERIFIED: the substrate graph store was not reachable from this session.
- Mitigation: after the rollup work in progress lands, run `PYTHONPATH=. python scripts/backfill_vision_rollups.py --days 8 --apply`. It snapshots the rows before it writes.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2910

🤖 Generated with [Claude Code](https://claude.com/claude-code)
