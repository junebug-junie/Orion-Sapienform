# The sitting record: Orion knows when someone plays (game company, patch 1)

Design: spec PR #3517 (`docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`), ticket #3500.
Live proof: #3626. Nothing in this PR has run live.

## Summary

- **Hub now records when a person plays a game on the Retro service.** A "sitting" is one stretch of play. It opens on the first button press and closes after 5 minutes with no press.
- **Orion is told, in three places.** The chat prompt gets one line while someone plays. The office camera's reach-out prompt gets the same fact. After the sitting, Orion's day holds one play entry: "Someone played A Link to the Past for 2 h 22 min".
- **Nothing guesses who plays.** The game page asks "Who's playing?" once, on the screen that holds the controller. The answers are "Juniper", "Someone else" and "Not now".
- **Orion is credited with nothing.** A person's sitting is never worded as Orion's play, and it never pushes Orion's own play out of a prompt.
- **No model is called and no game picture is read.** The record comes from the controller's input only. Looks, lines and the chat box are patches 2 and 3.
- **A script builds the history for day one** from the game journals, as a dry run. It writes nothing until Juniper says so.

## Outcome moved

Before: 27 of 31 hours of play on Orion's own service left no trace in Orion. On 2026-11-15 a sitting of 249 minutes ran while Orion's day took in 931 events, none about the game, and Orion guessed "computer work" from the office camera.

After (once deployed): each sitting of 2 minutes or more is one row and one play entry in Orion's day, and a chat turn during play knows the game and how long it has run.

## Current architecture

- The Retro engine is sealed: no network, no bus. Hub's gateway process (`hub-retro-gateway`) relays the browser's input to it and proxies its API.
- `retro_play_sessions` held only Orion's own play (58 rows, all The Legend of Zelda). The Temporal Self source `retro_play` turns each row into a play entry of the day.
- A person's play wrote nothing anywhere outside the engine's own journal files.

## Architecture touched

- **Pure rules** (`orion/retro/sitting.py`): what a sitting is. A press or a release counts; the 50 ms repeats of a held button do not. A stick pushed past its dead zone counts (N64).
- **Hub gateway** (`services/orion-hub/scripts/retro_sitting.py`): the relay hands each input message to the record. The hook is synchronous, runs after the message is forwarded, and can never fail or slow the relay.
- **Stores**: Redis key `orion:retro:sitting` (the open sitting, for Hub's other process), table `retro_sitting`, and one `retro_play_sessions` row of the new kind `human_play` at the close.
- **Recovery**: when the gateway's database pool opens, and each time a sitting opens, a row that a dead process left open is closed and a missing play row is written.
- **Temporal Self**: the play source now reads by write time, because a sitting's row is written 5 minutes after its last press. Four readers were changed so a person's sitting is its own fact (stance cue, frame, recent arcs, day block).
- **Prompts**: one `Play:` line in the unified turn's situation text; one sentence in the office reach-out prompt.
- **Page**: the "Who's playing?" box inside the fullscreen stage.

## Files changed

- `orion/retro/sitting.py`: the sitting rules, the input parser, the journal replay.
- `services/orion-hub/scripts/retro_sitting.py`: Redis, Postgres, the loop, recovery, the prompt sentence.
- `services/orion-hub/scripts/retro_transport.py`: the relay's `on_input` hook.
- `services/orion-hub/scripts/retro_routes.py`: two routes, the proxy and socket hooks, a stand-in used when the record cannot load.
- `services/orion-hub/scripts/retro_gateway.py`: recovery at start, close at shutdown.
- `services/orion-hub/scripts/retro_memory_export.py`: `insert_sql` and `row_values`, so both writers of `retro_play_sessions` use one column list.
- `services/orion-hub/scripts/office_reach_out.py`, `orion/hub/turn_orchestrator.py`: the sentence about the open sitting.
- `services/orion-hub/templates/retro.html`, `static/js/retro-panel.js`: the question box.
- `services/orion-hub/app/settings.py`, `.env_example`: three keys.
- `orion/schemas/retro_play.py`: kind `human_play`, basis `human_input`, `player_stated`.
- `orion/temporal_self/sources.py`, `arcs.py`, `stance_cue.py`, `frame.py`, `recent_arcs.py`, `day_block.py`, `now.py`: the label, write-time order, the readers, the salience rule.
- `services/orion-durable-runs/app/temporal_self_sources.py`: the play query.
- `services/orion-durable-runs/evals/temporal_self_fixture_db.py`: a `retro_play` fixture row without `created_at` is written at its `ended_at`.
- `config/integration/organs.yaml`: organ `play` gets `salience_magnitude: {retro_play: sitting_minutes}`.
- `services/orion-sql-db/manual_migration_retro_watch_v1.sql`: table `retro_sitting`, column `retro_play_sessions.player_stated`.
- `scripts/backfill_retro_sittings.py`: the baseline from the journals.
- `docs/runbooks/retro-browser.md`: section "The sitting record".
- `.github/workflows/retro-memory-tests.yml`: the real-database test of the sitting now runs in CI.
- Tests and evals: listed under "Tests run".

## Schema / bus / API changes

- Added:
  - Table `retro_sitting`. Column `retro_play_sessions.player_stated`.
  - `RetroPlaySessionSummaryV1.player_stated` (optional); `SessionKind` value `human_play`; `ControlBasis` value `human_input`.
  - `GET /api/retro/sitting` and `POST /api/retro/sitting/player` on the gateway.
  - Redis key `orion:retro:sitting`.
  - Salience rule `sitting_minutes`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - The Temporal Self source `retro_play` is ordered by the row's write time when that is later than its end (`AVAILABLE_AT_KINDS`). Orion's own play rows are written at their end, so their order does not change.
  - The stance cue can now hold two play clauses: Orion's own latest play, and the latest sitting of a person.
- Compatibility notes:
  - The migration is additive only and safe to apply twice. Its header names `orion-hub` and `orion-durable-runs`, so the deploy gate refuses either without it.
  - No bus channel is added and no event shape changes.

## Env/config changes

- Added keys: `HUB_RETRO_SITTING_ENABLED=true`, `HUB_RETRO_SITTING_GAP_SEC=300`, `HUB_RETRO_SITTING_MIN_SEC=120` (orion-hub; both Hub processes read the same env file).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, `services/orion-hub/.env_example`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes; the three keys are in `services/orion-hub/.env` of the primary checkout.
- skipped keys requiring operator action: none.

## Metric quality gate: sitting minutes

One new number is wired: the salience of a finished sitting is its minutes.

1. **Provenance.** `duration_sec` on the `retro_play_sessions` row, written by `play_row` in `retro_sitting.py`: last button change minus first.
2. **Independence.** It ranks sittings against sittings only. It is in no model with another number.
3. **Theory anchor.** None is claimed. It is a duration, used only to order one organ's own events.
4. **Live data.** From the journals, through the same code as the live counter: 46 sittings in 11 games; 32 of them, in 8 games, lasted 2 minutes or more, from 2 to 142 minutes. Rest is a true zero: no sitting, no row. Per game, one real press or release stands for 3 to 20 journal rows (7.2 for A Link to the Past, 11.4 for Crystalis, 20.4 for Battle Chess), which is why the count uses changes and not rows.
5. **Existing mechanism.** The existing salience rule is reused (`source_salience`).
6. **Reversibility.** One rule name in `organs.yaml`.

## Tests run

```text
tests/retro + orion/temporal_self                 384 passed
services/orion-durable-runs (temporal, chronicle)  72 passed, 21 skipped (need a database)
services/orion-hub -k "retro or office_reach_out or situation or unified_turn"
                                                  477 passed, 5 skipped (need a database)
services/orion-hub tests/test_retro_sitting_pg.py + test_retro_memory_export_pg.py
  against a throwaway Postgres 16                   2 passed
all 52 commands of .github/workflows/orion-static-gates.yml, run locally: 0 failed
git diff --check: clean
```

Not caused by this branch: `services/orion-hub/tests/test_office_reach_out_pg.py` fails 4 of 4 on `main` too when given a bare database (it needs tables that its own setup does not create). CI does not run it that way.

## Evals run

```text
python services/orion-hub/evals/retro_play_panel.py --output /tmp/rsp5      PASS
```

A real headless browser against the real page, script and router. New checks: a key press opens a sitting; the question shows inside the fullscreen stage; "Not now" hides it and a reload does not ask again; a page that only watches is not asked; "Someone else" reaches the server; the game keeps the keyboard.

```text
python3 scripts/backfill_retro_sittings.py        (dry run, read-only)
81 journals, 0 errors, 46 sittings planned, 32 of 2 minutes or more
```

## Docker/build/smoke checks

```text
Not run. This branch is not on main, and a live `up` runs only from the clean primary main checkout.
The real SQL (migration twice, upsert, play row, the chronicle's query, recovery) ran against a throwaway Postgres 16.
```

Every runtime claim in this PR is UNVERIFIED on the live rail until #3626 closes.

## Review findings fixed

A read-only review ran in a subagent (the code-review skill at effort high, then each finding checked by hand). It found 14 items. All are fixed in this PR.

- Finding: a stick push never counted. The parser and the tracker both reduced the stick values, so a full push became 0. Stick-only play opened no sitting.
  - Fix: one place reduces them (the tracker).
  - Evidence: `test_a_stick_push_counts_through_the_live_path` goes parser to tracker, the only live path.
- Finding: about 1 sitting in 16 would never reach Orion's day. The row is written 5 minutes after the last press; the day record read by end time and could pass it by.
  - Fix: the play source is ordered by write time (`AVAILABLE_AT_KINDS`, `created_at` in the query).
  - Evidence: `test_a_play_row_written_after_its_end_is_ordered_by_when_it_was_written`; the Postgres test runs the real query on both windows.
- Finding: a killed gateway left its row open for good; a database that was down at a close dropped the play row.
  - Fix: `recover`, at gateway start and at each sitting's open.
  - Evidence: `test_recover_closes_dead_rows_...`; the Postgres test with four planted rows.
- Finding: the label said "Orion did not play", which is false when a short handoff to Orion sits inside a sitting (1 of 32 long sittings in the journals).
  - Fix: the label names the person as the player and says nothing about Orion.
  - Evidence: `test_a_persons_sitting_is_a_play_arc_that_never_credits_orion`.
- Finding: a slow database could hold the loop, and the Redis copy said "is playing" until the database answered.
  - Fix: one bound covers the wait for a connection and the statement; Redis is written first.
  - Evidence: `test_a_slow_database_never_holds_the_redis_copy_or_the_loop`.
- Finding: a person's sitting pushed Orion's own play out of the stance cue and the frame.
  - Fix: both keep one of each.
  - Evidence: `test_no_reader_words_...`, `test_a_busy_day_keeps_orions_own_play_and_a_persons_sitting_both_in_the_frame`.
- Finding: an answer to "Who's playing?" could raise an error, or land on a later sitting.
  - Fix: the answer names its sitting; the service holds the sitting across its awaits.
  - Evidence: `test_an_answer_lands_on_the_sitting_it_was_asked_about_and_survives_a_close`, `test_the_page_routes`.
- Finding: the backfill would store each live sitting a second time when run after the deploy.
  - Fix: it stops at the first live sitting, or 5 minutes before now.
  - Evidence: `test_a_sitting_the_live_recorder_may_hold_is_never_planned`.
- Finding: a socket that dropped with a button held made a later reconnect count as a press.
  - Fix: a closed input socket's button state is forgotten.
  - Evidence: `test_a_dropped_socket_leaves_no_button_state_behind`.
- Finding: the day block's play line had no cap. On the real closed day 2026-11-15 its four sittings made a 403-character line that pushed the conversation threads line out.
  - Fix: the two longest, placed after the conversation lines. Same day: 2,073 to 2,329 characters of 2,400, no other line lost.
  - Evidence: `test_the_day_block_play_line_has_a_cap`.
- Finding: a setting of 0 was read as unset; the loop was not restarted inside a sitting; a title that failed once was never asked again; the question box could not be dismissed.
  - Fix: each one, as named.
  - Evidence: `test_small_rules_...`; the browser eval for the box.
- Finding: the real SQL never ran in CI, and two column lists for one table could drift.
  - Fix: the Postgres test is in `retro-memory-tests.yml`; one column list.
  - Evidence: the workflow file; `insert_sql(extra=...)`.
- Finding: a fault while the record loads would stop the whole game router.
  - Fix: a stand-in takes its place and the page works as before.
  - Evidence: `test_the_game_page_works_when_the_record_cannot_be_loaded`.

Found by me while fixing: the backfill script's header said "2 N64 journals, 1,446 human rows, all mask 0". That was wrong. Those rows are one SNES session where the page was open and nobody pressed anything. The journals hold one N64 session and it has no input row. The header now says so.

CI found three more (first CI run):

- Finding: the long-sitting test looked at Redis before the loop deleted the key. The tick closes the sitting first, then deletes the key. A slow runner saw the gap.
  - Fix: the test waits for all three outcomes: sitting closed, key gone, play row written. No production change.
  - Evidence: `test_a_long_sitting_ends_as_one_human_play_row_...` passes 5 of 5 locally.
- Finding: the play-line test imports `orion.hub.turn_orchestrator`, which needs `requests`. The native-boundary job runs `tests/retro` with a small dependency set.
  - Fix: the test moved to `services/orion-hub/tests/test_game_sitting_prompt_line.py`. The full Hub job runs it.
  - Evidence: `258 passed` for `tests/retro` plus the moved file.
- Finding: the `retro_play` source reads by write time, but the fixture loader let `created_at` default to the load time. The Postgres test row fell outside its window. It was skipped locally (no database).
  - Fix: the fixture loader gives a `retro_play` row without `created_at` its `ended_at`, the way Orion's own rows are written. The query stays as it is.
  - Evidence: `test_temporal_self_chronicle_postgres.py` 17 passed against a scratch Postgres 16.

## Restart required

The post-pull run on the primary checkout does all of it: it applies the migration first, then rebuilds what changed. By hand, from clean primary `main` after the merge and pull:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

The baseline is a separate step and a production write. It runs only on Juniper's word:

```bash
python3 scripts/backfill_retro_sittings.py --apply
```

## Risks / concerns

- Severity: medium
- Concern: nothing here is proven live. The relay hook is new code on the path of every input message.
- Mitigation: the hook runs after the forward, never awaits, and swallows every fault; `HUB_RETRO_SITTING_ENABLED=false` turns the whole record off. #3626 lists the live checks.

- Severity: low
- Concern: stick play is counted live but is not in the journals, so the baseline would undercount an N64 game. Today the journals hold no N64 input at all.
- Mitigation: stated in the script's report. The live rows replace the baseline as they arrive.

- Severity: low
- Concern: when Hub serves the game API itself, with no gateway, an open sitting is not closed at shutdown.
- Mitigation: recovery closes it and writes its play row when the next sitting opens. The deployed setup uses the gateway, which closes it at shutdown.

- Severity: low
- Concern: the stance cue can grow by one clause on a day with both Orion's own play and a person's sitting.
- Mitigation: the cue's own length cap still applies, and a test checks it.

Issues opened: #3626 (live proof, also holds the baseline's `--apply`).
Issues closed: none. Refs #3500.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3627

🤖 Generated with [Claude Code](https://claude.com/claude-code)
