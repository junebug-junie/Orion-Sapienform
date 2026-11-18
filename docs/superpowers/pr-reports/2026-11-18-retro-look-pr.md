# Orion looks at the game screen while a person plays (game company, patch 2)

Design: spec revision 4 (`docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`, PR #3632).
Juniper's go: 2026-11-18 ("hit it"). Patch 1 (the sitting record): PR #3627.

**Merge order: this branch is built on `docs/game-company-round3` (PR #3632, spec rev 4). Merge #3632 first, then rebase this branch onto `main`.**

Nothing in this PR has run live. Every runtime claim below is UNVERIFIED until the live check (#ISSUE_LIVE) closes.

## Summary

- **Orion can now look at a person's game.** While someone plays, Hub's Retro gateway keeps one picture of the screen every 2 seconds in memory (the last minute). When the screen changed enough and a button was pressed in the last minute, Orion bids to look. A funded look sends up to 4 stills to the percept store (gone within the hour), asks the chat lane's model for one short note, and keeps only the note.
- **A look needs something to see.** Each game gets its own "has the screen changed" threshold, learned from live play: the gateway counts how much the screen changes with and without button presses, and sets the threshold only when one value separates the two. No threshold, no looks, and the page says so. Nothing is set by hand.
- **The body decides.** Each look is a bid in a new tier 1 family, `hub.retro_watch`. Only a funded bid runs a look. Heat and price can stop it, and the page names which one did.
- **Orion can say no.** `dig watch off --why "..."` and `dig watch default <game> off --why "..."` are Orion's own switch; Juniper has a checkbox on the game page. The lower of the two wins, so neither can turn the other's "off" back on. Orion's choice is an act on its ledger and survives restarts.
- **The page tells the truth.** "Orion is watching." appears only while a stored note is under 3 minutes old; otherwise it says why not.
- **Counts, not claims.** The sitting row gains 9 counters. The sitting's play entry in Orion's day says "Orion watched: N notes". No look becomes a Temporal Self event in this patch (that waits for patch 3's judging).

## Outcome moved

Before: Orion knew only that buttons moved. A 4-hour sitting left one line ("Someone played X for 4 h") and Orion could say nothing about what happened on screen.

After (once deployed): each changed, played minute can give Orion one note of what was on the screen, stored with its frames' hashes, the want and the GPU lease that paid for it, and every reason a look did not happen is counted where it happened.

## Current architecture

- The engine (`services/orion-retro`) is sealed: no network, no model. Hub's gateway relays its video packets to the browser and never decoded one.
- Patch 1: the gateway records sittings from button changes (`retro_sitting.py`), with Redis and Postgres copies and one `human_play` row per sitting.
- The percept store already keeps camera frames for one hour; the model gateway already reads `kind: "percept"` attachments and refuses images on a blind lane.

## Architecture touched

- **Pure rules:** `orion/retro/frames.py` (decode one engine packet in its three pixel formats, 32 x 28 brightness grid, screen change, PNG enlarged 2x with hard edges); `orion/retro/watch.py` (the gates, the counters, the per-game calibration rule, note outcomes, the switch precedence, the page's status line, the prompt).
- **Gateway I/O:** `services/orion-hub/scripts/retro_watch.py`. The relay's new `on_media` callback (`retro_transport.py`) is the only tap; no second engine viewer.
- **The seat:** a GPU pool hold of class `chat` at priority `system`, carrying the want id, so frames are taken at the grant; the model call runs under it on the new route `retro_watch`.
- **Orion's switch:** `orion/dig/watch.py` writes an act (actor `orion.watch_choice`, verb `regulate`); the gateway copies those acts from `grammar_acts` into `retro_watch_preference` before every bid and once a minute.
- **The day:** the play row and the `retro_play` label carry the count of stored notes.

## Files changed

- `orion/retro/frames.py`, `orion/retro/watch.py`: new pure modules.
- `orion/retro/sitting.py`: `SittingTracker.held()` (a held direction counts as a hand on the pad).
- `services/orion-hub/scripts/retro_watch.py`: new; the watch service.
- `services/orion-hub/scripts/retro_transport.py`, `retro_routes.py`, `retro_gateway.py`: the tap, the wiring, `POST /api/retro/sitting/watch`, `GET /api/retro/sitting` gains `watch`, shutdown flush; docstrings name both observers.
- `services/orion-hub/scripts/retro_sitting.py`: play row carries `looks`; the prompt line no longer says "You are not watching the screen" (it could be false now).
- `services/orion-hub/scripts/act_emit.py`: the look act (open/close).
- `services/orion-hub/scripts/retro_memory_export.py`, `orion/schemas/retro_play.py`: promise texts; `looks` field.
- `orion/temporal_self/sources.py`, `services/orion-durable-runs/app/temporal_self_sources.py`: the label gains "Orion watched: N notes".
- `orion/dig/watch.py`, `orion/dig/__main__.py`, `orion/dig/README.md`: `dig watch`.
- `config/physiology/families.yaml`: `hub.retro_watch` (tier 1), `orion.watch_choice` (tier 0).
- `config/integration/organs.yaml`: both families in organ `play`.
- `config/gpu_pool.yaml`, `orion/llm/routes.py`: route `retro_watch` (class chat, system, system-only).
- `orion/physiology/binding.py`: binding producer `orion-hub.retro_watch.binding`.
- `config/physiology/contract.yaml`: `retro_watch.py` registered as a bidder.
- `services/orion-sql-db/manual_migration_retro_watch_v2.sql`: new.
- `scripts/sql/2026-11-18_grant_orion_readonly_retro_watch.sql`: new (hand-applied); `scripts/sql/2026-11-17_grant_orion_readonly_retro.sql`: its apply command lacked `docker exec -i` (stdin was never read).
- `scripts/sync_local_env_from_example.py`: prefixes `HUB_RETRO_SITTING_`, `HUB_RETRO_WATCH_`, `HUB_RETRO_LOOK_` (none matched before, so a default sync added no retro key).
- `services/orion-hub/templates/retro.html`, `static/js/retro-panel.js`: status line and switch.
- `services/orion-hub/.env_example`, `app/settings.py`, `docker-compose.yml` (comment only).
- `services/orion-retro/README.md`, `docs/runbooks/retro-browser.md`: the promise and the runbook section.
- Spec: the finished reader list (section 8) and two build choices.
- Tests and evals: listed below. `.github/workflows/retro-memory-tests.yml` runs the new ones with a real Postgres.

## Schema / bus / API changes

- Added:
  - Tables `retro_watch_look`, `retro_watch_change_hist`, `retro_watch_floor`, `retro_watch_preference`; 9 counter columns on `retro_sitting`; `retro_play_sessions.looks`.
  - `RetroPlaySessionSummaryV1.looks` (optional).
  - `POST /api/retro/sitting/watch {"mode": "on"|"off"}`; `GET /api/retro/sitting` gains `watch`.
  - Pool route `retro_watch`; families `hub.retro_watch`, `orion.watch_choice`; binding producer.
  - Acts: `hub.retro_watch look retro_sitting:<id>`, `orion.watch_choice regulate retro_sitting:<id>|retro_game:<id>`.
- Removed: none. Renamed: none.
- Behavior changed: the `retro_play` label of a sitting with stored notes ends ". Orion watched: N notes".
- Compatibility notes: no new bus channel (acts go on `orion:grammar:event`, the call on the existing LLM intake). The migration is additive and passes the post-pull classifier (dry run: "would apply"); its header names `orion-hub` and `orion-durable-runs` (the chronicle reads `looks`).

## Env/config changes

- Added keys: `HUB_RETRO_WATCH_ENABLED=true`, `HUB_RETRO_LOOK_SEC=60`, `HUB_RETRO_WATCH_SEAT_WAIT_SEC=30` (orion-hub; the gateway reads the same `.env`). The gateway also reads the existing `PERCEPT_STORE_BASE_URL` / `PERCEPT_STORE_TOKEN`, `ORION_PHYSIOLOGY_BID_URL`, `ORION_BUS_URL`, `RECALL_PG_DSN`.
- Removed / renamed keys: none.
- `.env_example` updated: yes, `services/orion-hub/.env_example`.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes; the three keys are in the primary checkout's `services/orion-hub/.env`.
- skipped keys requiring operator action: none. No bus URL changed.

## Metric quality gate: screen change (the trigger of a look)

1. **Provenance.** `orion/retro/frames.py` `screen_change`: mean absolute difference of two 32 x 28 luma grids, 0..1, computed by `retro_watch.py` on packets the relay already carries.
2. **Independence.** It shares the player with play rate (the sitting's mask changes), but not the sensor: pixels versus buttons. The calibration uses button changes only as the label.
3. **Theory anchor.** The orienting response to visual change (Sokolov); Bayesian surprise (Itti and Baldi, 2009). Used as a trigger only, never a score.
4. **Live data.** Re-run on the 44 step 0 sequences through the shipping decoder and the engine's own packer: 44 of 44 reproduce the fixture's numbers within 0.002. Rest is a true 0 (`map_a`, Zelda NES stills). The 30-second window separates still from play on A Link to the Past; the 2-second step does not, so the calibration uses the 30-second window. Zelda NES does not separate (as the spec found). No live frame has been measured yet: UNVERIFIED.
5. **Existing mechanism.** No frame-difference measure in vision-edge or vision-retina (spec record); none added elsewhere.
6. **Reversibility.** One nullable column, one histogram and one floor table; `HUB_RETRO_WATCH_ENABLED=false` stops it.

## Tests run

```text
TESTS_PLACEHOLDER
```

## Evals run

```text
EVALS_PLACEHOLDER
```

## Docker/build/smoke checks

```text
Not run. Deploys happen only from the clean primary main checkout, after merge.
The new SQL (migration twice, every statement the watch and `dig watch status` send) ran against a throwaway Postgres 16.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

After merge, the post-pull rebuild applies the migration and rebuilds `orion-hub` and `orion-durable-runs`. Then, by hand, on Juniper's word:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -f - < scripts/sql/2026-11-18_grant_orion_readonly_retro_watch.sql
```

## Risks / concerns

- Severity: medium. Concern: the look takes a GPU pool hold on the chat lane (class `chat`, `system`) for a few seconds. A hold blocks another hold on that role (max_holds 1) while it lasts; Juniper's interactive turn may use the hold's gap. Mitigation: the hold is released as soon as the call returns; seat wait is capped at 30 s; heartbeat TTL 90 s if the process dies.
- Severity: medium. Concern: no game has a floor until 30 minutes of its play are measured, and a fixed-screen game may never get one. Mitigation: by design (spec rev 4); the page says so; `under_floor_with_input` counts likely misses.
- Severity: low. Concern: the spec says `deferrable_window_s: 0`; `families.py` refuses 0 for a bidding family, so it is 60 (one look interval). The producer still never holds a want.
- Severity: low. Concern: `dig watch status` needs the new read grant (want_ledger included) to print refusals; until it is applied it names the missing GRANT.
- Issues opened: ISSUES_PLACEHOLDER. Issues closed: none.

## PR link

PR_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
