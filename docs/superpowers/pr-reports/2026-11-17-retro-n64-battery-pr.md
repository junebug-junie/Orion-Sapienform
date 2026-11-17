## Summary

- N64 in-game saves (OoT/MM save slots, Mario Kart records, Controller Pak) now persist per game in `<saves>/<game_id>/battery.srm`. Before this, they were lost whenever a session ended.
- Starting an N64 session from a server save failed every time: the core refuses to save or restore a state before its first frame. The probe now runs one frame first, after the battery is loaded.
- New `evals/import_n64_save.py` imports a player's Mupen64Plus saves (`.eep/.mpk/.sra/.fla`, gzip or raw `.st*` states).
- README corrected. It had wrongly claimed that server saves include cartridge memory.

## Outcome moved

Juniper can bring her existing N64 saves over, and new in-game saves survive between sessions.

## Current architecture

Each session spawns one worker. It creates a fresh core and loads an optional server save state. Nothing ever read or wrote `RETRO_MEMORY_SAVE_RAM`.

## Architecture touched

- `orion/retro/probe.py`: `save_ram()`/`load_save_ram()` (memory functions resolved lazily), `warm_restore` used by both snapshot and restore.
- `services/orion-retro/app/worker.py`: `Battery` loads before the first frame, flushes on change every 5 s and at a clean stop, and records `battery` evidence. Writes stay off when an existing file cannot be loaded.
- `services/orion-retro/app/saves.py`: `read_battery`/`write_battery` (atomic temp file, then rename). Listing and pruning ignore the battery file.
- `services/orion-retro/settings.py`: `battery(system)` is N64-only; `core_host` adds `warm_restore`.
- New: `services/orion-retro/evals/import_n64_save.py`, `services/orion-retro/tests/test_battery.py`.

## Schema / bus / API changes

None. The new file is `battery.srm`, and evidence gains `kind: battery` rows. NES/SNES behaviour is unchanged.

## Env/config changes

None.

## Tests run

```text
pytest tests/retro services/orion-retro/tests -q -> 211 passed
```

## Evals / live checks

```text
Real core, image built from this branch, temporary catalog and saves:
- OoT .sra imported -> Start goes title -> file select -> straight into the save (Link in his house, 19 rupees).
  Control with no battery -> name-entry screen.
- MM .st1 (gzip M64+SAVE v1.7, 16,793,412 B) imported as a server save. The worker's start-from-save order
  (battery load, snapshot, restore) plays 300/300 distinct frames, and the battery is intact afterwards.
- A restore leaves cartridge memory untouched (tested).
- Before the fix, snapshot before the first frame failed: "serialization failed".
```

## Review findings fixed

- Finding: an existing battery file that failed to load, or had the wrong size, could be overwritten with a blank image.
  - Fix: writes are disabled for that session.
  - Evidence: `test_an_unloadable_battery_is_never_overwritten`.
- Finding: the forced flush ran after fatal core errors.
  - Fix: it now runs only on a clean stop.
- Finding: snapshot ran before the warm frame, so starting from a save still failed.
  - Fix: snapshot warms too.
  - Evidence: `test_warm_snapshot_runs_one_frame_first` and the live replay above.
- Rejected: "a state load wipes the battery". Tested on the real core: a restore leaves cartridge memory as it was.
- Accepted: the stop sequence's 0.3 s join can kill a worker before its final flush. At most 5 s of in-game saving is lost; the README says so. No fsync, consistent with the save store's rule that the frame loop must not block on disk.

## Restart required

```bash
# after merge, from clean main:
scripts/safe_docker_build.sh orion-retro up -d --build --force-recreate
```

Then the agent imports Juniper's OoT/MM/Mario Kart saves with the new tool.

## Risks / concerns

- Severity: low. Concern: up to 5 s of in-game saving is lost if a worker is killed rather than stopped. Mitigation: the README documents it.
- Issues opened/closed: none.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
