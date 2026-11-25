## Summary

- A game with no calibrated floor (any floor status but `set`, still calibrating included) now looks on a timer: one look every `HUB_RETRO_WATCH_NO_FLOOR_LOOK_SEC` (60 s), but only while a button changed in the last 60 s.
- A timer look bids through the same want (family `hub.retro_watch`, tier 1, lane chat) and the same GPU hold (class chat, priority system) as a floor look, so chat still wins. Games with a floor are unchanged.
- The reason is visible: the counter `timer_looks` (Redis `orion:retro:watch` counts and `retro_sitting.timer_looks`), and the page/watch statuses `timer`, `timer_waiting` and `timer_input_stale` replace `no_floor`.
- On a timer look, an exact repeat of the previous note is `stored`, not `copied`, so a still screen keeps a fresh note. The commentary line gate skips a stored repeat, so a still screen does not earn a line every 2 minutes.
- Calibration keeps running alongside, so a game can still get a floor later.

## Outcome moved

Live on 2026-11-25, SimEarth sitting `9460d9c7ef644c83a2af8b3bfaba96f6` had 302 ring checks, 0 bids and 0 notes, with watch status `no_floor`. Mario Kart 64, Goddess of Wisdom, Crystalis and Tetrisphere had no floor either. With no stored note, `world:play` attention was never eligible ("not watching: no look note stored in the last 3 min"). With this patch these games get a look a minute while Juniper plays, and the newest note stays under the 3-minute rule.

## Current architecture

`orion/retro/watch.py` `gate()` returned `no_floor` for a game without a floor, and `services/orion-hub/scripts/retro_watch.py` never spawned a look for it. Looks happened only when the screen change beat the floor, a button was pressed in the last 60 s, and 60 s had passed since the last look.

## Architecture touched

- Pure rules: `gate()` gets `timer_sec` and `timer_from`. There are new timer gates, a new counter and new status codes.
- Hub Retro gateway: the timer, the counter (incremented only when the look bids), and `copied` mapped to `stored` on timer looks.
- Hub-app commentary: `NEWEST_NOTE_SQL` skips a stored note that equals the stored note before it.
- DB: additive column `retro_sitting.timer_looks`.

## Files changed

- `orion/retro/watch.py`: timer gates, `NO_FLOOR_LOOK_SEC`, `timer_looks` in `COUNTERS`, timer statuses.
- `services/orion-hub/scripts/retro_watch.py`: timer wiring, `first_check_at`, `_timer_sec`, counting, the repeat rule, `trigger=` in the log line.
- `services/orion-hub/scripts/retro_commentary.py`: the newest-note read skips repeats.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`: `HUB_RETRO_WATCH_NO_FLOOR_LOOK_SEC=60`.
- `services/orion-sql-db/manual_migration_retro_watch_v3.sql`: `ADD COLUMN IF NOT EXISTS timer_looks`, `REQUIRED-BY: orion-hub`.
- Tests: `tests/retro/test_watch_rules.py`, `services/orion-hub/tests/test_retro_watch_service.py`, `test_retro_watch_pg.py`, `test_retro_line_pg.py`.
- Docs: the spec (amendment and the section 3 "Timer looks" bullet) and `docs/runbooks/retro-browser.md`.

## Schema / bus / API changes

- Added: `retro_sitting.timer_looks` (int, default 0); `counts.timer_looks` in the `orion:retro:watch` Redis snapshot; status codes `timer`, `timer_waiting`, `timer_input_stale`.
- Behavior changed: a no-floor game looks on a timer; a stored exact repeat on a timer look; the commentary newest-note read skips a stored repeat.
- Compatibility: no bus channel or schema registry change. The `world:play` rule is unchanged (it still needs a stored note under 180 s old).

## Env/config changes

- Added keys: `HUB_RETRO_WATCH_NO_FLOOR_LOOK_SEC=60` (orion-hub; 0 restores the old rule).
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (primary checkout `services/orion-hub/.env`).
- Skipped keys requiring operator action: none.

## Gate records (timing, not a new metric)

- Freshness: a look starts at most once per 60 s, and its note time is taken at the GPU grant (seat wait at most 30 s). So with one stored note a minute, the newest stays under 180 s. One refused or failed look in between still keeps it under 180 s. Two in a row do not, and then the page and attention honestly say "not watching".
- Other gates that would block a slow screen: `static_skips` is never counted on the timer, because the screen change is not read. The `copied` outcome would block it, so it is handled as described above. The line lock and the watch switch still stop a look before it bids, and such a look is not counted.

## Tests run

```text
tests/retro + orion/dig/tests/test_watch.py + orion/prime/tests/test_prime_regulate.py + tests/test_world_play_attention.py + orion/body_atlas/tests/test_wire_code.py: 150 passed
services/orion-hub tests/retro, test_retro_watch_service, test_retro_commentary_service, test_retro_prime_attention_hub, test_game_sitting_prompt_line: 362 passed
pg (throwaway postgres:16 container): test_retro_watch_pg, test_retro_line_pg, test_retro_sitting_pg, test_retro_prime_attention_pg: 4 passed (v3 applied twice)
scripts/check_integration_seam.py: OK
scripts/check_body_economy_contract.py: OK
scripts/check_env_template_parity.py: PASS
scripts/check_migration_required_by.py: OK
drift.auto_apply_refusals(v3): [] (the post-pull run auto-applies it)
scripts/check_sql_migrations_applied.py --file manual_migration_retro_watch_v3.sql: MISSING (expected until deploy)
```

## Evals run

```text
None new. The look's replay eval (services/orion-hub/evals/retro_look_replay.py) covers the floor statistic, which did not change.
```

## Docker/build/smoke checks

```text
No image build: there is no dependency or Dockerfile change. Live path UNVERIFIED until deploy.
```

## Review findings fixed

- Finding: `timer_looks` was counted before the line-lock, mode-unread and mode-off refusals, so it climbed with the watch off.
  - Fix: count it in `_look_inner`, just before the bid.
  - Evidence: `test_a_timer_game_with_the_watch_off_counts_no_timer_look`.
- Finding: a stored exact repeat reopened the commentary line gate every 120 s on a still screen.
  - Fix: `NEWEST_NOTE_SQL` skips a stored note equal to the stored note before it.
  - Evidence: `test_retro_line_pg.py` (repeat skipped; a new note counts).
- Finding (nit): no explicit tests for an unread calibration or a calibration read late (hub restart).
  - Fix: `test_an_unread_calibration_gives_no_timer_look`, `test_a_calibration_read_late_starts_the_timer_at_once`.
- Not fixed (pre-existing, also on the floor path): a button held for more than 60 s with no new change reads as stale input.

## Restart required

Merge, then pull on the primary checkout. The post-pull run auto-applies `manual_migration_retro_watch_v3.sql` and rebuilds orion-hub (`hub-retro-gateway`, `hub-app`). The deploy gate refuses orion-hub until v3 is applied. For a manual deploy, apply v3 first: `docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_retro_watch_v3.sql`. Until it is applied, the counter UPDATE fails quietly (every counter, not only `timer_looks`), and `dig watch` prints the counts as unread.

## Risks / concerns

- Severity: low. Concern: more chat-lane GPU use, one look a minute during no-floor play. Mitigation: it goes through the same body-economy bid and GPU pool, chat wins, the heat reflex applies, and setting the key to 0 turns it off.
- Severity: low. Concern: a floor game whose stored note repeats a stored note across an unclear or failed look in between is no longer a new note for a line. Mitigation: this is rare and arguably right.
- Issues: live proof belongs on #3726 (verify-live, the look). No new issue opened.
