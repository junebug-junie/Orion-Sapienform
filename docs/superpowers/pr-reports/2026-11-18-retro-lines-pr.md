# Orion speaks while a person plays (game company, patch 3)

Design: spec revision 4 (`docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`).
Juniper's go: 2026-11-18 ("hit it"). Patch 1 (the sitting): PR #3627. Patch 2 (the look): PR #3727.

Nothing in this PR has run live. Every runtime claim below is UNVERIFIED until the live check issue closes.

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

REVIEW_AND_RESULTS_PLACEHOLDER
