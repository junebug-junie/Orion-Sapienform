# Retro: Save/Load states, fullscreen, inert scroll keys, 64-game catalog

## Summary

- A **Save** button now stores the whole emulator state for the current game on the host. A **Load** button restores a chosen save, newest first, with its time, play time and frame. Load works into a running game (restore in place). It also starts a fresh game from the save, for example after a service restart.
- Saves live in a new host directory, `RETRO_SAVES_DIR`, which is bind-mounted at `/saves`. They survive container restarts and rebuilds. Today a restart killed a 3-hour Crystalis game because state lived only in worker memory.
- Only the worker process touches the core. It serializes, restores, writes and reads state through the existing urgent command queue. The API process only lists metadata and pre-checks identity.
- Each save is keyed to its game, ROM SHA-256 and core SHA-256. A mismatched save is refused twice: once in the API and again in the worker, against the core it actually loaded.
- **Fullscreen** now fills the screen with the game display. Esc or an Exit button leaves it. The picture is scaled to fit, keeps its aspect ratio and stays pixelated.
- **The PageUp/PageDown bug:** these keys were not mapped to any game button, so the browser handled them. Its default action scrolled the Hub page, because the Retro frame is taller than the window, and the game slid out of view. While the game has the keyboard, PageUp, PageDown, Home, End, Space and the arrow keys no longer scroll.
- The catalog cap goes from 16 to 64 games. Juniper's staged catalog has 20 games.

## Outcome moved

A service restart, rebuild or crash no longer loses the game. The player saves, and after the restart picks the save and presses Load. PageUp and PageDown no longer move the game off screen.

## Current architecture

- `services/orion-retro`: a FastAPI app on a Unix socket. One spawned worker owns the libretro core (fceumm). `save_enabled=False` was hard-coded. A restart marked the session `failed/service_restart`, and evidence replay cannot rebuild a game.
- `orion/retro/probe.py` already had `snapshot()`/`restore()`, used only by the preflight replay check.
- Hub: `templates/retro.html` and `static/js/retro-panel.js` render in an iframe in `index.html`. `scripts/retro_routes.py` is the same-origin proxy, also served by the `hub-retro-gateway` process. Keydown handling covered only mapped keys on the canvas.

## Architecture touched

- **Retro service:** new `app/saves.py` (`SaveStore`). Worker `save`/`load` ops. Start-from-save. A listing endpoint. Settings, compose mount, `.env_example`.
- **Hub:** proxy allow-list (`save`, `load`, `games/<id>/saves`), retro template and JS, and the iframe `allow="... fullscreen"`.
- **No bus, schema registry or channel changes.** This is a private HTTP API behind the existing operator/hub bearer auth and the Hub origin check, the same path as pause/resume.

## Files changed

- `services/orion-retro/app/saves.py`: new. A store keyed by game, ROM and core. Atomic tmp+rename writes, with metadata written last. Bounds and pruning.
- `services/orion-retro/app/worker.py`: `save`/`load` ops, a load before `ready` for start-from-save, a `played_frames` carry-over, and `save`/`load` evidence events. Failures roll back and do not kill the session.
- `services/orion-retro/app/session.py`: `save_enabled` is now true when `/saves` is writable. Adds `list_saves`, start-from-save with an identity pre-check, `loaded_save` in the session payload, and the catalog cap of 64.
- `services/orion-retro/app/main.py`: `POST /sessions/{id}/save`, `POST /sessions/{id}/load`, `GET /games/{id}/saves`, and an optional `save_id` on `POST /sessions`.
- `services/orion-retro/app/contracts.py`: `Start.save_id` and `LoadSave`.
- `services/orion-retro/settings.py`: `saves`, `save_max_per_game=20`, `save_max_bytes=4 MiB`.
- `services/orion-retro/docker-compose.yml`, `.env_example`: the `RETRO_SAVES_DIR` mount, required in the same `${VAR:?}` style as the other retro directories.
- `services/orion-retro/evals/install_catalog.py`: creates `saves/` and writes `RETRO_SAVES_DIR`.
- `services/orion-retro/README.md`: API and "Save states" section.
- `services/orion-retro/tests/test_saves.py`: new tests.
- `.github/workflows/retro-preflight.yml`: a CI placeholder for the new required compose key.
- `services/orion-hub/scripts/retro_routes.py`: allow-list for the new routes.
- `services/orion-hub/templates/retro.html`: Save/Load/Fullscreen controls and fullscreen CSS. The asset is bumped to `retro-panel.js?v=5`.
- `services/orion-hub/static/js/retro-panel.js`: Save/Load and fullscreen logic, plus a document-level keydown handler that swallows scroll keys.
- `services/orion-hub/templates/index.html`: the iframe may go fullscreen.
- `services/orion-hub/tests/retro/test_boundary.py`, `services/orion-hub/evals/retro_browser.{py,mjs}`: coverage.

## Schema / bus / API changes

- **Added:**
  - Retro HTTP: `POST /sessions/{id}/save`, `POST /sessions/{id}/load {save_id}`, `GET /games/{game_id}/saves`, and `save_id` (optional) on `POST /sessions`.
  - Session payload gains `loaded_save`.
  - Evidence gains the kinds `save` (save_id, frame, played_frames, bytes) and `load` (save_id, frame, save_frame, played_frames).
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:** `save_enabled` in `/games` and in the session payload is now truthful. It is true only when `/saves` is mounted writable.
- **Compatibility notes:** there are no bus or registry changes. The Hub JS hides the save controls when `save_enabled` is false, so a new Hub against an old retro service degrades cleanly.

## Env/config changes

- **Added keys:** `RETRO_SAVES_DIR`, in `services/orion-retro/.env_example`. It is required by compose.
- **Removed / renamed keys:** none.
- **`.env_example` updated:** yes.
- **Local `.env` synced:** yes, with `python3 scripts/sync_local_env_from_example.py orion-retro --all-keys`, run from the worktree, which writes the primary `.env`. The template value is empty, which would fail the compose `:?`. So the local value was set to `/home/athena/.local/share/orion/retro/service/saves`. That directory was created with mode 700, owned by athena (UID 1000 = `RETRO_UID`).
- **Skipped keys requiring operator action:** none.

## Tests run

```text
pytest tests/retro services/orion-retro/tests -q      -> 108 passed
pytest services/orion-hub/tests/retro services/orion-hub/tests/test_hub_panel_reachability.py -q -> 10 passed
python3 scripts/check_env_template_parity.py          -> PASS (95 services)
```

New backend tests in `test_saves.py`:

- **Store:** a round trip, newest-first listing, and refusal on ROM mismatch, core mismatch, a corrupt file, path traversal, the size cap, pruning to the max count, and orphan/temp cleanup.
- **Real native worker loop against the C fixture core:**
  - A save followed by an in-place load replays the same frame hashes frame for frame.
  - A fresh worker started from the save produces the same frames.
  - The `save`/`load` evidence events carry the frame and the save_id.
- **Failure handling:**
  - A mismatched save is refused without a `fatal`.
  - A disk fault returns `save_io_failed` without a `fatal`.
  - A rejected unserialize rolls back and play continues.
- **API:** listing, and a mismatch refused before the worker sees it.
- **Catalog:** 20 games and 64 games are accepted; 65 are refused.

## Evals run

`services/orion-hub/evals/retro_browser.py` was run against a real container. The image was built from this branch with a separate tag, `orion-retro-savetest:local`. The production tag was not touched. The eval uses real Chromium through the real Hub router.

```text
retro_browser.py                    -> PASS
retro_browser.py --ack-delay-ms 350 -> PASS
```

The eval now covers:

- **Save:** the save is listed as compatible and focus returns to the game.
- **Scroll keys:** PageDown, End, Space and PageUp leave `scrollY == 0`. The eval first asserts that the page can scroll, so a 0 means something.
- **Fullscreen:**
  - The stage fills the viewport with `object-fit: contain` and pixelated rendering.
  - PageUp, PageDown, Home and End keep fullscreen.
  - The arrow keys still reach the game, shown by an applied mask of 64.
  - The Exit button works.
- **Load:** an in-place load sets `loaded_save`. After Stop, Load starts a new session from the save.

**Mutation check:** I reverted only the scroll-key `preventDefault`, and the eval failed with "a scroll key moved the page under the game" (scrollY 771). So the eval catches the bug.

The staged 20-game `catalog.next.json` was validated read-only through the real `Sessions` startup checks (ROM hashes and paths). Result: 20 games valid.

## Docker/build/smoke checks

```text
docker build -t orion-retro-savetest:local -f services/orion-retro/Dockerfile .  -> ok
docker compose (primary .env files) -f services/orion-retro/docker-compose.yml config -> /saves bind resolves to the host dir
scripts/rebuild_affected_services.py --plan -> orion-hub build, orion-retro build
```

No `up` was run. Deployment is Juniper's call.

## Review findings fixed

- **Finding:** a disk error (mkdir, unlink, stat or read) during save or load escaped as `OSError`. The worker treated it as fatal and the live game was killed.
  - **Fix:** `SaveStore.write` and `SaveStore.read` now turn `OSError` into `ValueError('save_io_failed')`.
  - **Evidence:** `test_disk_fault_on_load_refuses_without_killing_the_game` and `test_disk_fault_on_save_refuses`.
- **Finding:** a rejected `retro_unserialize` was fatal and could leave the core half-overwritten.
  - **Fix:** the worker snapshots before restoring. On failure it rolls back and returns `save_restore_failed`. Only a failed rollback stays fatal.
  - **Evidence:** `test_rejected_unserialize_rolls_back_and_keeps_playing`.
- **Finding:** an exact metadata field-set match meant a future extra field would hide old saves, and prune would then delete their states as orphans.
  - **Fix:** the field check is now a subset check. Prune deletes a `.state` file only if its `.json` does not exist, and it also sweeps temp files.
  - **Evidence:** `test_metadata_with_extra_fields_still_lists_and_is_not_pruned`.
- **Finding:** `loaded_save` was set before the worker had loaded anything.
  - **Fix:** it is now set when the worker reports ready, and the worker loads before ready.
- **Finding:** the UI reported "Save loaded. Playing." even when no session started, and an in-place load while paused showed the old picture.
  - **Fix:** `newGame` returns success. The paused case now says the picture updates when play resumes.
- **Nit (accepted):** the hub boundary test checks the JS by string. The behavior is covered by the browser eval.

## Restart required

From the clean primary `main` checkout, after merge, in this order:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
mv /home/athena/.local/share/orion/retro/service/catalog/catalog.next.json /home/athena/.local/share/orion/retro/service/catalog/catalog.json
scripts/safe_docker_build.sh orion-retro up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

The catalog `mv` must happen after the new code is pulled and before the retro restart, because the old code refuses more than 16 games. Restarting retro ends any game in progress. That game cannot be saved first, because the old build has no Save.

## Risks / concerns

- **Medium: replay is not exact for every game.** Nova's serialized-state replay is known to be not bit-exact with the pinned core. Load resumes play, but the game may not continue exactly as before. Other games, Crystalis included, are not individually verified. **UNVERIFIED for Crystalis** until Juniper saves and loads once.
  - **Mitigation:** play one save/load round trip early.
- **Low: newest save may be lost on host power loss.** Writes do not fsync. The worker must never block on disk, because a worker silent for 1 s fails the session. A host power loss within the writeback window can lose the newest save. Container restarts and rebuilds are safe.
- **Low: fullscreen inside the Hub iframe is UNVERIFIED.** It depends on the new iframe `allow="fullscreen"`, which only takes effect after the Hub rebuild. The eval verifies fullscreen on `/retro` directly.
  - **Mitigation:** if it is blocked, the UI says "Fullscreen is blocked here. Open /retro directly."
- **Low: the game picker with 20 entries is not exercised by the eval.** The picker is a native `<select>`, so 20 entries render as the browser's own scrolling dropdown and do not change the layout. The longest staged title is 32 characters. The Sessions test covers 20 and 64 games.
- **Low: disk use.** Saves are capped at 20 per game and 4 MiB per state. An NES state is usually under 100 KiB.

- **Note: CI.** The `Static repo gates` check fails on this PR. It fails the same way on `main` at `ca23dbc7a`: the definition drift gate reports `added metric://bus_channel/orion-hub/orion:personality:naming`. That metric comes from the personality-naming merge, not from this PR, so it should be re-locked on main.
  - `service-container` failed once, in the cognitive-goal section of the browser eval ("acknowledged wait never applied"). This PR does not touch that section. A second run of the same commit passed, and a run after merging main passed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2825

🤖 Generated with [Claude Code](https://claude.com/claude-code)
