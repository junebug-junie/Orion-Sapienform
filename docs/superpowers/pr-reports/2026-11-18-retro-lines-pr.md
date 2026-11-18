# Orion speaks while a person plays (game company, patch 3)

Design: spec revision 4 (`docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`).
Juniper's go: 2026-11-18 ("hit it"). Patch 1 (the sitting): PR #3627. Patch 2 (the look): PR #3727.

Nothing in this PR has run live. Every runtime claim below is UNVERIFIED until the live check (#3741) closes.

## Summary

- **Orion can say a line to the person who plays.** When the watch mode is "Watch and talk" and Orion stored a note since it last spoke (at least 2 minutes ago), Orion bids in a new tier 1 family, `hub.retro_commentary`. A funded bid runs a real unified turn (Prime block, stance, motor, finalize) through the outreach door, with the game's 4 newest frames attached. Orion may pass; a pass is recorded and nothing is said.
- **Lines run below Juniper.** The line takes a GPU pool hold of class `chat` at priority `system`, and the whole turn rides that hold, so stance, motor and finalize never run at `interactive`. The outreach door could not carry a hold before; it can now (`compose(gpu_lease=...)`).
- **Juniper goes first.** Any person's message on a chat socket marks a running line `yielded` in its row first, then cancels the turn, its motor and its hold. Delivery claims the row and refuses a yielded one, even when the draft finished after the yield.
- **Orion judges its own looks.** The line turn compares the notes whose frames overlap its own frames with what it sees, and says `holds` or `contradicted` (also when it passes). Only a note that held becomes a Temporal Self event (`retro_look`, "Saw on the game screen: ..."). The play arc now opens at the first settled look and closes with the sitting's row.
- **A chat box inside the game, in fullscreen too.** T or the Chat button opens it; typing never moves the character. "Who's playing?" decides the path: Juniper gets her normal turn; anyone else is stored with source `retro_guest` and is never counted as "Juniper spoke". Both get the game's frames.
- **A line strip with thumbs.** A line shows at the bottom of the screen with thumb up/down. A chat reply within 2 minutes counts as a response. A guest's thumb is stored as a guest's.

## Outcome moved

Before: Orion could look at the game (patch 2) but never speak about it, the notes reached no prompt, and nothing a person typed from the game could be told apart from Juniper.

After (once deployed): each funded line leaves a `retro_line` row with the looks it was shown, the outcome, the seconds by stage and the lease priorities; each shown note gets its bucket; the now is read back by organ `play` (closes the play part of #3457); and the note error rate (contradicted / checkable) becomes measurable from live play.

## Current architecture

- Hub's Retro gateway (`scripts/retro_gateway.py`) records sittings and runs looks; it keeps a ring of frames in memory. hub-app holds the chat sockets and the outreach door.
- Outreach turns (`EndogenousOutreach.compose`) ran on the chat route at `interactive` (551 of 551 motor leases in 7 days, spec section 4), with no way to carry a hold, frames or a source.
- Turn staging copied images only from Hub's chat store and always introduced them as "Juniper attached".
- The websocket path set Juniper as the origin of every message; the temporal speaker rule read any row with a prompt as Juniper.
- `retro_watch_look.verdict` existed but nothing set it; organ `play` was exempt from reading the now.

## Architecture touched

- **Pure rules:** `orion/retro/commentary.py` (gate, near-copy grouping, checkable notes, the game block, the CHECKS block, settlement, error rate).
- **hub-app:** `scripts/retro_commentary.py` (the line loop, Juniper goes first, frames for chat-box turns, the now read-back); `scripts/endogenous_outreach.py` (`compose` extras, `deliver_line`, `juniper_session_id`); `scripts/websocket_handler.py` (one hook: yield + retro chat-box path); `scripts/main.py` (start/stop); `scripts/act_emit.py` (the line act).
- **Gateway:** `scripts/retro_watch.py` (`frames_for_turn`, the line lock), `scripts/retro_routes.py` (`GET /api/retro/sitting/frames`, `POST /api/retro/lines/{id}/thumb`, the three-position switch).
- **Turn:** `orion/hub/turn_orchestrator.py` (ingress carries `source`/`chat_history_source`; `run_unified_turn(utterance_origin=...)`), `orion/harness/attachment_staging.py` + `orion/schemas/harness_finalize.py` (percept kind, `frames_expire_at`, the game-screen sentence).
- **Temporal / Prime:** `orion/schemas/temporal_self.py`, `orion/temporal_self/sources.py`, `orion/temporal_self/arcs.py`, `services/orion-durable-runs/app/temporal_self_sources.py`, `orion/prime/handback.py` (class `play`), `orion/regulation/juniper_turns.py`, `orion/orion_day/gather.py`.
- **Page:** `templates/retro.html`, `static/js/retro-panel.js`.

## Files changed

- `orion/retro/commentary.py`: new; the pure rules.
- `services/orion-hub/scripts/retro_commentary.py`: new; the line service and `prepare_retro_turn`.
- `services/orion-hub/scripts/endogenous_outreach.py`: `compose(attachments, gpu_lease, source, cockpit_sink, session_id, timeout_sec)` (passed on only when set); `deliver_line` (claim under the send lock, no notification, no cap); push and history extras; motor seconds in the debug dict.
- `services/orion-hub/scripts/websocket_handler.py`: the yield hook, the retro chat-box path, the reply into the sitting's recent chat.
- `services/orion-hub/scripts/retro_watch.py`, `retro_routes.py`: frames route (loopback + Retro hub token), thumb route, line lock, `quiet` accepted by the switch.
- `services/orion-hub/scripts/act_emit.py`: `publish_line_said`.
- `services/orion-hub/scripts/main.py`: `CommentaryService` start/stop.
- `services/orion-hub/templates/retro.html`, `static/js/retro-panel.js`: chat box, strip, thumbs, three-position switch, focus guard.
- `orion/hub/turn_orchestrator.py`: ingress keys, origin parameter, staged `source`.
- `orion/harness/attachment_staging.py`, `orion/schemas/harness_finalize.py`: percept staging and the prompt sentence.
- `orion/regulation/juniper_turns.py`: guest rows never "Juniper spoke".
- `orion/orion_day/gather.py`: game lines out of the letter's chat gather.
- `orion/schemas/temporal_self.py`, `orion/temporal_self/sources.py`, `orion/temporal_self/arcs.py`: `retro_look`, the guest speaker, the play arc.
- `orion/prime/handback.py`: `play` class.
- `services/orion-durable-runs/app/{temporal_self_sources,settings,main}.py`, `.env_example`, `docker-compose.yml`, `README.md`, `evals/temporal_self_fixture_db.py`: the source query and its flag.
- `services/orion-sql-writer/app/act_downstream_use.py`: `retro_watch_look` as an arc source table.
- `config/physiology/families.yaml`, `config/physiology/contract.yaml`, `orion/physiology/binding.py`, `config/integration/organs.yaml`: the family, its bid site, its binding producer, organ `play` (kind, family, `reads_now`).
- `services/orion-sql-db/manual_migration_retro_lines_v1.sql`: new. `scripts/sql/2026-11-18_grant_orion_readonly_retro_line.sql`: new (hand-applied).
- `services/orion-hub/.env_example`, `app/settings.py`, `docker-compose.yml` (comment), `Dockerfile` (copies `config/gpu_pool.yaml`), `README.md`; `scripts/sync_local_env_from_example.py` (prefixes).
- `scripts/analysis/check_retro_lines_live.py`: new; the read-only live check.
- `docs/runbooks/retro-browser.md`, the spec (build choices).
- Tests and evals: listed below. `.github/workflows/retro-memory-tests.yml` runs the new Postgres test and the dry eval.

## Schema / bus / API changes

- Added:
  - Table `retro_line`; columns `retro_watch_look.checkable`, `settled_at`, `settled_by_line`.
  - Temporal source kind `retro_look` (organ `play`); Prime hand-back class `play`.
  - Family `hub.retro_commentary` (tier 1, lanes [chat], window 300 s); binding producer `orion-hub.retro_commentary.binding`; act `hub.retro_commentary say retro_sitting:<id>`.
  - Gateway routes `GET /api/retro/sitting/frames` (hub-app only), `POST /api/retro/lines/{id}/thumb`; `POST /api/retro/sitting/watch` accepts `quiet`.
  - `HarnessAttachmentV1.source` (optional).
  - Push payload keys on a line: `source`, `retro_sitting_id`, `line_id`.
- Removed / renamed: none.
- Behavior changed: a chat_history_log row with source `retro_guest` is never read as Juniper (temporal speaker rule, `juniper_turns`, attention, day letter); game lines are not in the day letter's chat gather; turns with game frames introduce them as "frames of the game screen".
- Compatibility notes: no new bus channel. The new `HarnessAttachmentV1.source` is ignored by an older governor (the model does not forbid extra fields). `TemporalNowItemV1` forbids unknown source kinds: the now's readers (orion-thought, hub, mind, prediction) must be rebuilt before durable-runs writes the first `retro_look` event; the post-pull rebuild does both in one run, and durable-runs is refused until the migration is applied (REQUIRED-BY header).

## Env/config changes

- Added keys: `HUB_RETRO_COMMENTARY_ENABLED=true`, `HUB_RETRO_LINE_MIN_GAP_SEC=120`, `HUB_RETRO_LINE_TIMEOUT_SEC=240`, `HUB_RETRO_LINE_SEAT_WAIT_SEC=30` (orion-hub); `TEMPORAL_SELF_RETRO_LOOK_ENABLED=true` (orion-durable-runs). `HUB_RETRO_GATEWAY_PORT` (already in `.env_example`) is now read by hub-app's settings.
- Removed / renamed keys: none.
- `.env_example` updated: yes, both services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (orion-hub: the 4 keys; orion-durable-runs: the flag).
- skipped keys requiring operator action: none. No bus URL changed.

## Metric quality gate: the note error rate (contradicted / checkable)

1. **Provenance.** `orion/retro/commentary.py` `settle` and `error_rate`, from the line turn's CHECKS block (`parse_reply`); stored per line as `retro_line.looks_checkable` / `looks_contradicted` and per note as `retro_watch_look.checkable` / `verdict`. A note is checkable when its frames' window (the 30 s before its grant) overlaps the window of the frames the turn saw (`is_checkable`).
2. **Independence.** It shares the look with every look count, but not the instrument: the look's own model wrote the note; a later unified turn with fresh frames judges it. It is not a transform of `similar_to_previous` or of the screen change.
3. **Theory anchor.** Inter-rater agreement on a checkable subset: a second reading of overlapping evidence, as step 0's per-note verdicts did by hand (9 of 10 place/action right, 4 of 10 with one wrong detail). Spec rev 4, open question 1 and third round point 1.
4. **Live data.** None: no line has run (retro_watch_look held 0 rows on 2026-11-18). The dry eval proves the parsing and the checkable rule on the step 0 notes only. Its rest state is a real 0 (no note contradicted). Expected checkable share is about 8% (spec section 8), so the rate is noisy until many lines have run; it is reported with its count, never alone. UNVERIFIED.
5. **Existing mechanism.** None: `verdict` existed on `retro_watch_look` (patch 2) with nothing setting it.
6. **Reversibility.** Columns on two tables; nothing reads it into a model, a price or a prompt. Cheap to drop.

## Tests run

```text
tests/retro/test_commentary_rules.py (pure rules)                                            28 passed
services/orion-hub tests/test_retro_commentary_service.py (line service, outreach door, guest path,
  review regressions; real EndogenousOutreach.deliver_line)                                  31 passed
services/orion-hub tests/test_retro_line_pg.py + test_retro_watch_pg.py + test_retro_sitting_pg.py
  + test_retro_watch_service.py + tests/retro + test_mind_tier_by_caller + test_orion_unified_turn_tts
  (Postgres tests on a throwaway Postgres 16, port 55989, stopped after)                     331 passed
tests/retro tests/test_harness_attachment_staging.py tests/test_juniper_turns_guest.py orion/physiology/tests
  orion/temporal_self/tests orion/prime/tests orion/orion_day/tests orion/hub                1382 passed
services/orion-durable-runs -k "temporal or retro or chronicle or experience or now or guest"
  (throwaway Postgres for the chronicle and the guest-reader SQL)                             120 passed
services/orion-dream tests/test_idle_counts_only_juniper.py                                  1 passed, 1 skipped
services/orion-sql-writer tests/test_act_downstream_use.py                                   6 passed
retro-preflight CI job simulated in a clean venv with its exact pins:
  tests/retro services/orion-retro/tests 274 passed; services/orion-hub/tests/retro 263 passed, 2 skipped
whole services/orion-hub/tests folder                                                        5198 passed, 6 failed, 4 errors
  (3 failures were mine and are fixed: test_mind_tier_by_caller, test_orion_unified_turn_tts,
   test_boundary route list. The rest fail the same on main: test_check_async_routes_not_blocking and
   test_attention_loops_ui_smoke need the repo root as cwd, test_fcc_model_labels_api reads ~/.fcc/.env,
   test_consult_belief_store_sql needs its own database.)
all steps of .github/workflows/orion-static-gates.yml, run locally: 1 failed (the chat route poacher gate on
  a docstring), fixed, re-run OK; check_integration_seam --require-base OK (31 kinds); env template parity PASS;
  check_settings_defaults orion-hub OK; check_migration_required_by OK; body economy contract OK
post-pull migration classifier (dry run, live DB, read-only): manual_migration_retro_lines_v1.sql "would apply"
rtk proxy git diff --check: clean
```

## Evals run

```text
python services/orion-hub/evals/retro_line_dry_eval.py      PASS (also as evals/test_retro_line_dry_eval.py)
  near-copy cut-off 0.45: 284 different-scene pairs, max 0.4043 (1 word-for-word copy aside);
    5 of 15 same-scene pairs grouped
  synthetic sitting of 28 notes: 12 groups shown, the stuck player's 4 repeats grouped, only N12 checkable,
    no name with nobody stated, "Attached: 4 frames"
  5 reply shapes parsed (plain, markdown, PASS + CHECKS, no block, "CHECKS: none"); a not-checkable note is
    never settled; the CHECKS block never reaches the line
python services/orion-hub/evals/retro_play_panel.py          PASS, 3 of 3 runs after a de-flake (real browser)
  chat box opens with T and with the button in fullscreen; a held button is released by one neutral input
  and the game pauses (no controller); typing sends no non-zero mask; the /ws frame is exactly the contract
  (mode orion, disable_tts true, surface retro, sitting id); reply in panel and strip; reconnect keeps focus;
  a line shows thumbs and a thumb POSTs; a guest never sees Juniper's outreach; after the flip, other
  outreach shows in the panel only; the three-position switch posts quiet.
  Before the de-flake 2 of 5 runs failed: one 50 ms repeat of the held button could land between the
  eval's snapshot and the key press. The eval now checks "a neutral input is sent and nothing held after it".
python3 scripts/analysis/check_retro_lines_live.py           against the live DB: "UNVERIFIED: table
  retro_line does not exist" (exit 0); against a throwaway DB with seeded rows: PASS/FAIL as designed
  (a seeded interactive event made it FAIL).
```

## Docker/build/smoke checks

```text
Not run. Deploys happen only from the clean primary main checkout, after merge.
The new SQL (migration twice, every statement the line service and the thumb route send) ran against a
throwaway Postgres 16. The Dockerfile now copies config/gpu_pool.yaml (the line reads the held route's priority).
```

## Review findings fixed

A read-only review ran in a subagent on `origin/main...HEAD`: 3 must-fix, 6 should-fix, 5 nits. All fixed.

- Finding (must-fix): delivered lines counted on Juniper's daily outreach cap and raised the attention price.
  - Fix: `retro_commentary` joins `CAP_EXEMPT_SOURCES` (`orion/physiology/attention_pacing.py`).
  - Evidence: `orion/physiology/tests/test_attention_pacing.py`.
- Finding (must-fix): a message right after delivery's claim cancelled a line already being said (history row or act lost, text NULL).
  - Fix: the claim marks the line claimed and stores its words at once; a later message no longer yields it.
  - Evidence: `test_a_message_after_the_claim_does_not_cut_a_line_being_said`.
- Finding (must-fix): the yield waited behind the chat-box frames fetch (up to 20 s).
  - Fix: `person_spoke()` sets the yield flag synchronously before anything awaits; chat-box frames wait at most 6 s.
  - Evidence: `test_a_guest_message_on_a_main_chat_socket_cannot_borrow_the_game_sources` (order check), `test_her_message_mid_turn_...`.
- Finding: a message during the bid, or a turn of Juniper's starting before the seat, still let a line run.
  - Fix: the lock and the yield window open before the bid; the turn checks again before it takes its seat.
  - Evidence: `test_a_message_while_the_bid_is_weighed_means_no_line_at_all`, `test_a_turn_of_juniper_running_before_the_seat_yields_the_line`.
- Finding: an outer cancel (Hub stop) could be swallowed and hang shutdown.
  - Fix: the line waits with `asyncio.wait`; an outer cancel cancels the line and re-raises.
  - Evidence: `test_stopping_hub_mid_line_does_not_hang`.
- Finding: a prompt could name frames that staging dropped, and its verdicts could still become events.
  - Fix: the final frame carries `staged_attachment_count`; fewer than named voids the line and its judgments (`frames_not_staged`).
  - Evidence: `test_frames_the_turn_did_not_carry_void_the_line_and_its_judgments`.
- Finding: a timed-out line left its motor running. Fix: `HarnessGovernorClient.cancel` on timeout. Evidence: `test_a_timed_out_line_stops_its_motor`.
- Finding: the game page showed Orion's outreach to Juniper to a guest. Fix: only a line, or Juniper stated, reaches the panel. Evidence: the browser eval.
- Finding: a look could start between the gate and the lock and waste a funded line. Fix: lock before the bid; `look_in_flight` is `no_frames`, not `failed`.
- Nits fixed: the held route read fails closed and is cached; `max(..., default=None)`; an inline `CHECKS:` is cut from the line; a note left `unchecked` by a passed turn can be judged later; no affect capture for a guest's turn; a client cannot send the game sources from a main chat socket.

- Finding (Orion's review of PR #3727, point 5): a funded look that got no GPU seat (heat reflex, pool timeout) counted nowhere.
  - Fix: it now counts as a failed look.
  - Evidence: `test_a_seat_the_pool_refuses_for_heat_...` asserts bids_funded 1, failed_looks 1, model_calls 0.
- Finding (same review, point 2): `retro_watch_change_hist` was left out of the patch 2 read grant.
  - Fix: added to `scripts/sql/2026-11-18_grant_orion_readonly_retro_line.sql` (hand-applied).

## Restart required

After merge, the post-pull rebuild applies the migration and rebuilds `orion-hub` (hub-app and hub-retro-gateway) and `orion-durable-runs` (and every service whose image holds `orion/schemas`). Then, by hand, on Juniper's word:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -f - < scripts/sql/2026-11-18_grant_orion_readonly_retro_line.sql
```

## Risks / concerns

- Severity: medium. Concern: a line cancelled mid-stance lets that one stance call finish on the GPU (orion-thought has no cancel listener, #3658). Juniper's first call can wait for it (11 s median, 19 s p90). Mitigation: the row is yielded first and nothing it returns is said (tested); live check #3741 item 3 measures her wait.
- Severity: medium. Concern: the line holds the chat lane's hold for the whole turn (60 to 90 s estimated). Juniper's interactive calls may use the hold's gaps (pool rule H3), and her message cancels the line. Mitigation: one play call at a time (the gateway bids for no look while a line runs); live check #3741.
- Severity: low. Concern: deploy order. `TemporalNowItemV1` forbids unknown kinds, so the now's readers must run the new schema before durable-runs writes the first `retro_look` event, and durable-runs needs the migration's `settled_at`. Mitigation: one post-pull run rebuilds all; the REQUIRED-BY header refuses durable-runs until the migration is applied; a `retro_look` event needs a delivered or passed line first.
- Severity: low. Concern: the near-copy cut-off (0.45) and the expected checkable share (about 8%) come from step 0, not live play. Mitigation: the dry eval recomputes the cut-off; the live check reports the checkable share.
- Severity: low. Concern: `hub.retro_commentary` keeps the spec's `deferrable_window_s: 300` (families.py accepts it; the brief's "smallest accepted value" note applied to a 0 the spec never set for this family). The producer never holds a want: only `funded` runs.
- Departures from spec, stated: see the spec's "Patch 3 is built" note (any person's message yields, frames route needs the hub token, line act opens and closes at delivery, lines out of the letter's chat gather, the per-turn game block with notes and the stance cancel listener not built).
- Severity: low. Concern: CI's Retro preflight browser eval (`evals/retro_browser.mjs`, a patch 1-era eval this PR does not edit) failed once at a `#stop` click ("Node is either not clickable") after a load-from-save, and passed on rerun; main's last 3 runs passed. Not reproduced. Mitigation: if it recurs, check whether the new chat strip or panel ever covers the "More" controls.
- CI: 60 of 60 checks green after one test fix (the durable-runs cap parity test pinned the old exempt list).
- Issues opened: #3741 (verify-live, needs-juniper), #3742 (chat turns see no look notes), #3743 (play arc staleness close), #3744 (watch switch not in fullscreen). Commented: #3658. Issues closed: none (the play part of #3457 closes when #3741 proves the read-back live).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3745

🤖 Generated with [Claude Code](https://claude.com/claude-code)
