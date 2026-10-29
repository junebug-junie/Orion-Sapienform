## Summary

- A3 of the unified-turn self-prediction arc (#2869). Stance's private bets about its own turn (A1, live since 2026-10-24) are now settled and graded. Before this, 1,075 bets had been placed and none were checked.
- orion-prediction settles each bet's four faculties against the turn's own record (`harness_turn_trace.run_artifact`) every 60 s, into a new table `prediction_turn_self_bet`. The four faculties are: will I ask a question, will I fetch a web page, how many steps, how long the reply.
- `GET /turn-self-bets/live` grades a rolling 7 days, per faculty and per caller:
  - probability bets: Brier score, with skill measured against each caller's own base rate;
  - range bets: realized against stated coverage, median width, and the interval score.
- Two stance prompt fixes, where the prompt and the measurement disagreed: "outside source" counted repo files (and a web search alone), and a "3 to 7 steps" example anchored the guesses low.

## Outcome moved

Orion's self-predictions now get a grade. This is the first time the arc can say whether stance knows how its own turns go.

The first read-only grade, over the 1,075 real bets (884 graded; 87 failed or cut short, 59 never got a turn record):

| faculty | stance's guess | what happens | result |
|---|---|---|---|
| steps | median range 2 wide, about 3–5 | median 14 steps (curiosity 82) | 11% inside, against a stated 80% |
| reply length | median range 600 wide | median 3,217 characters | 9% inside, against a stated 80% |
| fetches a web page | split between 0.02 and 0.98 | 48% of turns | skill 0.01 overall; worse than each caller's own rate |
| asks a question | mostly 0.02 | 16% of turns | skill +0.33 on chat, +0.23 on curiosity, −0.44 on world-pulse reads |

Part of the fetch miss was a definition mismatch, now fixed in the prompt. The motor's repo reads and web searches were never recorded as fetches. The rolling 7-day window lets bets made under the old prompt age out.

## Current architecture

- A1 (#2893): stance writes `thought_decision.self_prediction`. The motor and the judge never see it.
- Outcomes already sit in `harness_turn_trace.run_artifact`: `step_count`, `final_text`, `source_fetches`, `exit_code`.
- The planned settler, the self-monitor organ (#2827), exists only as a design PR. Per Juniper's rule, predictions live in orion-prediction, so A3 is built there, as a model next to action bets.

## Architecture touched

- orion-prediction:
  - a settle loop and a read endpoint;
  - a `/health` block;
  - one new table, which `check_schema` now requires at boot (the same pattern as the other prediction tables).
- Stance prompt: two wording fixes.
- No bus, schema-registry, or producer change.

## Files changed

- `services/orion-prediction/app/turn_self_bets.py`: new. Pure `settle()`/`observe()`/grading, plus the `TurnSelfBets` service object.
- `services/orion-prediction/app/store.py`: `pending_turn_self_bets` (anti-join, `DISTINCT ON`), chunked `insert_turn_self_bets`, windowed `turn_self_bets`, and the `check_schema` entry.
- `services/orion-prediction/app/main.py`: the settle loop, `GET /turn-self-bets/live`, and the `/health` `turn_self_bets` block.
- `services/orion-prediction/app/settings.py`, `.env_example`: `PREDICTION_TURN_SELF_BETS_ENABLED=true`, `PREDICTION_TURN_SELF_BETS_SETTLE_SEC=60`.
- `services/orion-sql-db/manual_migration_prediction_turn_self_bet_v1.sql`: new, additive, `REQUIRED-BY: orion-prediction`.
- `orion/cognition/prompts/stance_react.j2`: the fetch definition matches what is recorded; the step-count example is dropped.
- `services/orion-prediction/tests/test_turn_self_bets.py`: new, 26 tests.

## Schema / bus / API changes

- Added:
  - table `prediction_turn_self_bet` (one row per turn × faculty);
  - `GET /turn-self-bets/live?since=&force=`;
  - a `/health` `turn_self_bets` block.
- Removed: none
- Renamed: none
- Behavior changed: the stance prompt wording for `fetches_outside_p` and `steps_lo/hi`.
- Compatibility notes: none. The bet schema (`turn.self_prediction.v1`) is unchanged.

## Env/config changes

- Added keys: `PREDICTION_TURN_SELF_BETS_ENABLED` (true), `PREDICTION_TURN_SELF_BETS_SETTLE_SEC` (60)
- `.env_example` updated: yes
- local `.env` synced: yes. Both keys were added to `services/orion-prediction/.env`; `check_env_template_parity.py` passes for 95 services.
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-prediction: pytest tests/test_turn_self_bets.py -q                -> 26 passed
services/orion-prediction: pytest tests -q -> 8 failed / 218 passed; the same 8 (backtest/causal/events)
  fail on main in this venv, untouched by this PR
orion/thought/tests/test_stance_self_prediction.py                                 -> 30 passed
orion/thought/tests/test_stance_react_prompt_imperative_discipline.py              -> 6 passed
orion/harness/tests/test_self_prediction_never_reaches_motor.py                    -> 3 passed
python3 scripts/check_env_template_parity.py                                       -> PASS
git diff --check                                                                   -> clean
```

## Evals run

```text
Read-only settle + grade over all 1,075 live bets (table in "Outcome moved"): every bet read from
thought_decision ⋈ harness_turn_trace; nothing written.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-prediction build      -> Built
docker run ... python -c "import app.main, app.turn_self_bets"  -> import ok
pending_turn_self_bets SQL against the live DB inside BEGIN/ROLLBACK (temp stand-in table):
  1075 bets, 971 with a turn record, created_at 2026-10-24 00:33 .. 2026-10-29 19:39 (+00, aware)
Migration applier --dry-run (review agent): NOT APPLIED, would apply (single transaction) -> auto-apply on pull
```

## Review findings fixed

- Finding (medium-high): chat session ids are bare UUIDs. The prefix rule split them into one-letter "callers" (`f9e8...` became `f`, `9c91...` became `none`), so the per-caller base rate was fitted on 1–3 rows and faked the bar.
  - Fix: a UUID is caller `chat`, and a missing id is `unknown`.
  - Evidence: `test_a_chat_session_uuid_is_one_caller_not_split_by_its_hex_letters`. Live, chat is now one caller with 104 settled turns.
- Finding (medium): turns cut short by the harness (`CUT_SHORT_MARKER`, `exit_code` null) were graded as replies. A 230-step stop notice counted as a steps miss.
  - Fix: the marker (imported, not copied) makes the turn void, as `failed_to_execute`.
  - Evidence: `test_a_cut_short_notice_is_void_not_a_graded_reply`.
- Finding (low-medium): a non-object bet or turn record raised an error. It sat at the head of the oldest-first queue, so it stalled every later pass.
  - Fix: void `ungradeable_bad_record`.
  - Evidence: `test_a_malformed_row_settles_void_instead_of_stalling_every_pass`.
- Finding (low): probability faculties reported Brier and skill from 20 rows, while `grade()` treats 50 as enough.
  - Fix: one floor of 50 for both kinds.
  - Evidence: `test_below_the_floor_there_is_a_count_but_no_grade`.
- Finding (low): `thought_decision.correlation_id` is not unique, so the join could pick a row arbitrarily.
  - Fix: `DISTINCT ON` keeps the first stance row.
  - Evidence: the live query in BEGIN/ROLLBACK above.
- Finding (low): the motor can use WebSearch, which `source_fetches` does not record.
  - Fix: the prompt says a web search alone does not count.
- Self-found while running the first live grade:
  - **Fix:** a low-p event that did not happen is not "broken". Probability bets now record `happened`/`not_happened`, and Brier grades the number.
  - **Fix:** the prompt mismatches above (repo files counted as fetches; the "3 to 7 steps" anchor against a live median of 14).
- Not changed: after 6 h with no turn record, a bet is written `expired_unsettled` and is not re-checked. That matches the spec. The live maximum lag is 1 h 51 m.

## Restart required

```text
Post-pull rebuild handles it. The migration is additive with a REQUIRED-BY header, so the post-pull
applier runs it before orion-prediction rebuilds, and the stance prompt is read per call. Manual
equivalent, from the clean main checkout after pull:
scripts/safe_docker_build.sh orion-prediction up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: nothing feeds the grades back into stance yet, so Orion cannot learn from them. That loop is the next patch (the organ's "stance cue"). It changes a cognition loop, so it needs proposal mode.
  - Mitigation: grades are visible at `/turn-self-bets/live` now.
- Severity: low
  - Concern: "asks a question" is the crude `?`-in-the-last-300-characters check. A world-pulse reply that quotes a headline ending in `?` counts as a question.
  - Mitigation: the spec names it as the first thing to replace, never with a word list.
- Severity: low
  - Concern: the per-caller base rate is in-sample, a slightly stricter bar. Callers with a 0% or 100% rate (chat and curiosity never fetch by this measure) get no skill number, only Brier.
  - Mitigation: documented in the module.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2976

🤖 Generated with [Claude Code](https://claude.com/claude-code)
