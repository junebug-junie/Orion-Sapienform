# fix(vision): retina capture task can no longer die silently

## Summary

- The walkway camera service stopped sending pictures for about 3 hours on 2026-10-17 and said "ok" the whole time. This patch fixes the cause and makes that kind of death loud and self-healing.
- Each service now deletes only its own picture files in the shared frames folder. The edge camera service used to delete every file older than 60 s, including the walkway's. This answers Juniper's question "why is vision edge competing here at all?": it should not have been, and now it is not.
- Both pruners now tolerate a file that disappears while they look at it.
- The capture loop can no longer be killed by housekeeping. If a loop that must run forever ends anyway, the process logs CRITICAL and exits, and Docker restarts it.
- Health now says `degraded` with `capture_stalled` when nothing was published for `RETINA_PUBLISH_STALL_SECONDS` (default 120). A watchdog exits for restart when the capture loop is hung, or when the camera still streams but nothing gets published.

## Outcome moved

Failure mode closed: "eye dark, health green, no log line". Before: one lost race on a shared folder ended capture forever, silently. After: the race cannot end capture; any forever-task death is a CRITICAL log line plus an automatic restart; a publisher that stops while the camera streams is restarted within about 2 minutes; health tells the truth in between.

## Current architecture

- `services/orion-vision-retina/app/main.py` `capture_loop` ran `cleanup_old_frames` outside its try. `app/frame_store.py` did `isfile()` then an unguarded `getmtime()` on every file in the shared dir.
- `services/orion-vision-edge/app/capture_worker.py` pruned every file older than 60 s in the same dir, whoever wrote it.
- The three long-lived tasks were plain `asyncio.create_task` with a held reference: an exception ended the task with no log.
- Health status was `ok` whenever the last capture attempt had no error, so a dead capture task (no attempts at all) read as healthy.

Root cause, reproduced against the pre-fix code: a file deleted between `isfile` and `getmtime` raises `FileNotFoundError` (`OLD CODE RAISES: FileNotFoundError`, scratch repro; pinned by `test_the_old_isfile_then_getmtime_pattern_raised_on_this_race` and `test_cleanup_survives_files_vanishing_between_listing_and_stat`).

## Architecture touched

orion-vision-retina (capture loop, task supervision, health, settings, compose), orion-vision-edge (pruner only). No bus/schema contract change: `system.health.v1` `details` is a free-form dict; two keys were added.

## Files changed

- `services/orion-vision-retina/app/frame_store.py`: `owned_frame_pattern`; `cleanup_old_frames` prunes only this camera/stream's files, one stat per file, never raises.
- `services/orion-vision-retina/app/main.py`: cleanup inside the protective try; `_spawn_forever` + done callback -> `_die` (CRITICAL, `os._exit(1)`, once); `publish_stall` / `publisher_dead`; health loop reports and enforces them.
- `services/orion-vision-retina/app/health.py`: `capture_stalled` drives `status=degraded`; `capture_stalled`, `capture_stalled_reason` details.
- `services/orion-vision-retina/app/settings.py`: `RETINA_PUBLISH_STALL_SECONDS` + validator.
- `services/orion-vision-retina/.env_example`, `docker-compose.yml` (base, walkway, doorbell): new key; base service gets `restart: unless-stopped` (it already exits on purpose for wedged RTSP readers, and had no restart policy).
- `services/orion-vision-retina/README.md`: new section.
- `services/orion-vision-retina/tests/test_capture_task_supervision.py`: repro + regression tests.
- `services/orion-vision-edge/app/frame_prune.py` (new), `app/capture_worker.py`: ownership-scoped, race-tolerant pruner, split out so it can be tested without the camera/detector imports.
- `services/orion-vision-edge/tests/test_frame_prune.py`, `README.md`.

## Schema / bus / API changes

- Added: `system.health.v1` details `capture_stalled` (bool), `capture_stalled_reason` (str|null) from orion-vision-retina.
- Removed / Renamed: none.
- Behavior changed: retina health `status` is `degraded` while capture is stalled.
- Compatibility notes: details is an open dict; no consumer reads these keys yet.

## Env/config changes

- Added keys: `RETINA_PUBLISH_STALL_SECONDS=120.0` (orion-vision-retina; base, walkway, doorbell). ON by default; `0` disables.
- Removed / Renamed: none.
- `.env_example` updated: yes.
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys orion-vision-retina`: yes (`+RETINA_PUBLISH_STALL_SECONDS='120.0'`).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. .venv/bin/python -m pytest services/orion-vision-retina/tests -q
  125 passed, 1 failed (test_connect_without_token_reports_login_needed: ring_doorbell
  not installed in the local venv; pre-existing, CI installs requirements.txt)
PYTHONPATH=. .venv/bin/python -m pytest services/orion-vision-edge/tests/test_frame_prune.py -q
  3 passed
(services/orion-vision-edge/tests as a whole fails collection on main too:
 test_camera_reconnect.py needs env Settings; pre-existing, edge has no CI workflow)
python3 scripts/check_env_template_parity.py  -> PASS
Pre-fix repro: old cleanup_old_frames + a delete between isfile and getmtime
  -> FileNotFoundError
```

## Evals run

```text
No eval change: this is a liveness fix. services/orion-vision-retina/evals (frame_smear)
is unaffected. The watchdog/health behavior is covered by gate tests.
```

## Docker/build/smoke checks

```text
docker build -f services/orion-vision-retina/Dockerfile .   -> OK
docker run ... python -c "import app.main"                    -> RETINA_PUBLISH_STALL_SECONDS 120.0
docker compose --profile walkway --profile doorbell config    -> new key in all 3 services,
                                                                 restart: unless-stopped in all 3
Live: UNVERIFIED until deploy (no deploy from this branch).
```

## The 181 threads

Not a leak. The walkway process restarted 7 minutes before the check and already had exactly 181 threads, the same count as the dead process after 3 h. 177 of them carry the inherited `uvicorn` name (native threads), 4 are `libuv-worker`; py-spy saw only 3 Python threads. The base retina (folder source, no FFmpeg) has 69. The host has 96 CPUs and `cv2.getNumThreads()` is 96 in the container. So the threads are native pools sized to the CPU count: OpenCV's parallel pool, the two bundled OpenBLAS copies, and FFmpeg's H.264 decoder threads. Fixed-size, idle (futex waits). Optional follow-up: cap them (`cv2.setNumThreads`, `OPENBLAS_NUM_THREADS`) to save memory; not done here.

## Follow-up (not in this PR)

- sql-writer's story-strip capture (`services/orion-sql-writer/app/vision_story_strip.py`) was tuned to a ~60 s walkway frame lifetime. Walkway frames now live `FRAME_RETENTION_SECONDS` (300 s), so that capture timing can relax later. Its docstring still says ~60 s. Its frame reads already tolerate vanishing files (`OSError` on scandir and `Image.open` are caught).
- Nobody prunes files in the shared dir that no live writer owns (none today: the live dir holds only `frame_*` and `walkway_walkway_*`).

## Review findings fixed

Code review ran in a subagent (no must-fix). Fixed:

- Finding: the "streaming but not publishing" exit would restart-loop on failures a restart cannot fix (disk full, bus down, percept upload refused) -- about 17 camera-session drops an hour.
  - Fix: capture errors after a frame was read stamp `_last_capture_error_mono`; the exit stands down while they recur. Health still reads degraded with `last_error`.
  - Evidence: `test_watchdog_does_not_exit_for_failures_a_restart_cannot_fix`, `test_capture_loop_records_a_capture_error`.
- Finding: `os._exit` skipped `source.stop()`, so no RTSP TEARDOWN; the wedged-reader exit used its own `os._exit`.
  - Fix: one async `_die()`: CRITICAL log, bounded `source.stop()`, exit once. Done callbacks schedule it; the wedged-reader path uses it too.
  - Evidence: `test_a_forever_task_ending_exits_the_process` (asserts stop), `test_two_tasks_dying_exit_once`, `test_wedged_rtsp_readers_exit_through_the_same_path`.
- Finding: loop-tick check used wall clock; an NTP step or carbon laptop resume would exit.
  - Fix: `time.monotonic()` for the loop tick and capture-error clocks.
  - Evidence: `test_wall_clock_jump_alone_is_not_a_hang`.
- Finding: validator floor had no margin for clip `pause_device` (stop + clip + start).
  - Fix: floor uses `RETINA_CLIP_TIMEOUT_SEC + SOURCE_RECONNECT_SECONDS + 10` when clips are on.
  - Evidence: settings test rejects 31 s with clips on.
- Finding: boot grace clocks set at module import, not `start()`.
  - Fix: reset in `start()`.
- Finding: one test was tautological, one mislabeled; heal-backoff false positive unpinned.
  - Fix: removed the tautological test, relabeled the other, added `test_watchdog_quiet_during_a_long_heal_backoff`.
- Finding: orphan files and the Ring false negative undocumented.
  - Fix: README "Known gaps" line.

## Restart required

Pull rebuild covers it (retina + edge changed). The walkway waits its 90 s RTSP quiet period on restart, by design.

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-vision-retina --profile walkway up -d --build orion-vision-retina-walkway
```

## Risks / concerns

- Severity: low. Concern: the base `orion-vision-retina` service gains `restart: unless-stopped`; a misconfigured base instance now restart-loops instead of staying dead. Mitigation: intended -- every exit is a CRITICAL log line; `docker compose stop` still stops it.
- Severity: low. Concern: a Ring camera whose publishes all fail does not read `capture_stalled`. Mitigation: `last_error`/`source_ok` already report it; documented.
- Severity: low. Concern: a watchdog exit that hits a truly wedged RTSP reader still loses TEARDOWN (bounded stop). Mitigation: the persisted quiet gate makes the next boot wait 90 s.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
