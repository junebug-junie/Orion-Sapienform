## Summary

The walkway camera's picture broke for hours whenever the retina reconnected to it too soon after dropping a connection. This PR makes the retina wait before every reconnect, close every connection cleanly, and recover by itself if the stream turns corrupt anyway.

- **Quiet period.** The retina does not open a new camera connection until 90 s (`RETINA_RTSP_RECONNECT_QUIET_SECONDS`) have passed since the last one. This applies on every path: a reconnect after failed reads, a watchdog replacement, `pause_device`, a heal, and process start.
- **Survives a restart.** The time of the last connection is saved to `/var/lib/orion-vision-retina/rtsp_session_<stream_id>.json`. The host directory `/mnt/telemetry/vision/retina_state` is mounted there. A container restart or deploy therefore waits too. If the time is unknown at boot, the retina waits one full period.
- **Clean teardown.** Every connection is closed with `release()`. In this image, `release()` sends the RTSP TEARDOWN message (checked against a fake server). `stop()` now waits long enough for a blocked open or read to finish, so the reader closes its own connection. It also waits for readers that were retired earlier. Compose `stop_grace_period` is 30 s on both services.
- **Self-heal.** The retina scores every 10th served frame with the same math as `evals/frame_smear.py`. That math now lives in `app/frame_corruption.py`, so there is one copy. If at least 80% of the checks over 120 s are corrupt, the retina logs a WARNING, drops the connection, waits the quiet period, and reconnects. Each heal in a row doubles the wait. After 3 heals in a row it stops dropping the connection and only reports. It never reboots the camera.
- **Health.** System health now reports `source_quiet_wait_remaining_sec`, `source_last_session_end_ts`, `source_corrupt_frame_rate`, `source_corruption_checks`, `source_corrupt_frames`, `source_corruption_heals` and `source_corruption_heal_suppressed`.
- **CI.** A new workflow, `orion-vision-retina-tests.yml`, runs this service's tests. The service had no CI before.

## Outcome moved

The failure mode was a fast reconnect to the walkway Reolink. The camera then mixed the old connection into the new one and every frame came out smeared, for hours. Now:

- No connect path can reconnect inside the quiet window, deploys included.
- If corruption gets through anyway, the retina recovers by itself in about 3.5 min (120 s of checks plus a 90 s wait). Before, it needed a person and a camera reboot.

### Live evidence (root cause, 2026-10-15/16)

- **Corrupt stream:** about 6,000 `RTP: PT=60/61: bad cseq X expected=Y` lines per minute, in two alternating sequence series. `frame_smear.py` showed every frame smeared.
- **2026-10-15:** started after a stall at 17:14 UTC and a restart. Only a camera reboot cleared it.
- **2026-10-16:** started with the 07:41 UTC deploy restart and lasted about 10 hours.
- **Live test, 2026-10-16 17:53 UTC:** `docker stop` the walkway retina, wait 180 s, `docker start`. Result: 0 bad cseq, 0 decode errors, 20/20 clean frames, no camera reboot.
- **Outside report:** this matches the Reolink forum thread "rtsp ffmpeg rtp pt xx bad cseq if stopped restarted quickly" (https://community.reolink.com/topic/563/). In that thread, a 1-2 s restart broke the stream and a 5 s pause worked for one user. A 30 s pause did not work for another user. One user suggests some cameras clean up a dropped stream after about 30 s.

### Why 90 s is the default

- **Lower bound:** 90 s is above the 60 s default session timeout in RFC 2326 (section 12.37), which FFmpeg also assumes. It is also above the 30 s that failed for one forum user.
- **Upper bound:** it is half the 180 s that was proven clean live.
- **No vendor figure:** Reolink does not document its RTSP session timeout. I searched and found nothing.
- **Status:** 90 s is **UNVERIFIED** live. Only 180 s has been tested.
- **What happens if 90 s is too short:**
  - A deploy can still corrupt the stream.
  - The self-heal recovers it within minutes.
  - `source_corruption_heals` would rise after deploys. That is the signal to raise the default to 180. Raising it is one env value.
- **Cost:** about 90 s with no frames after every deploy or restart. That gap is intended.

## Current architecture

`services/orion-vision-retina/app/sources.py` `RtspFrameSource` works like this:

- A reader thread keeps only the newest frame.
- After 3 failed reads, the reader releases the capture, waits 5 s (`SOURCE_RECONNECT_SECONDS`) and reopens.
- The stall watchdog from PR #2692 abandons a wedged reader and starts a new one immediately.
- `stop()` waits 5 s for the reader thread.
- The walkway service had `stop_grace_period: 15s`. The base service had none (the Docker default is 10 s).
- Nothing enforced a minimum gap between connections, and nothing detected corrupt frames live.

## Architecture touched

- Service `orion-vision-retina` only.
- New classes in `sources.py`: `RtspSessionGate` and `CorruptionTracker`.
- `frame_corruption.py` is new.
- Settings, health, main, compose (both services), `.env_example` and README are updated.
- The eval now imports the shared math from `app/frame_corruption.py`.
- No bus, schema or channel changes. The new health fields are additive keys in the free-form `details` dict of `SystemHealthV1`.

## Files changed

- `services/orion-vision-retina/app/sources.py`: `RtspSessionGate` (quiet period, persisted on the wall clock, waits on the monotonic clock, writes serialized). `CorruptionTracker`. Quiet wait in the reader before every open. A stop check after an open. `_release()` restarts the quiet clock. The watchdog skips the quiet wait. Heal with backoff and a cap. `stop()` joins retired readers. The default join is the open timeout + 2 s.
- `services/orion-vision-retina/app/frame_corruption.py`: the smear and decoder-gray math, moved here from the eval so the image ships it.
- `services/orion-vision-retina/evals/frame_smear.py`: imports that math. The CLI is unchanged.
- `services/orion-vision-retina/app/settings.py`: six new keys and validators, including the minimum of 5 checks per window. Adds `rtsp_session_state_path()`.
- `services/orion-vision-retina/app/main.py`: passes the new settings to the source. A quiet wait is reported as a quiet wait, not counted as a failed read. Copies the source metrics into health.
- `services/orion-vision-retina/app/health.py`: the new health fields.
- `services/orion-vision-retina/docker-compose.yml`: new env keys and the state volume on both services. `stop_grace_period: 30s` on both.
- `services/orion-vision-retina/.env_example`, `README.md`: document the new keys and behavior.
- `services/orion-vision-retina/tests/test_rtsp_quiet_period.py`: 38 new tests.
- `.github/workflows/orion-vision-retina-tests.yml`: the new CI workflow.

## Schema / bus / API changes

- Added: health `details` keys (listed in the Summary).
- Removed: none.
- Renamed: none.
- Behavior changed: an RTSP source now waits before connecting at boot and on every reconnect.
- Compatibility notes: webcam, folder and mock sources are unaffected.

## Env/config changes

- Added keys:
  - `RETINA_RTSP_RECONNECT_QUIET_SECONDS=90.0`
  - `RETINA_RTSP_SESSION_STATE_DIR=/var/lib/orion-vision-retina`
  - `RETINA_RTSP_HEAL_ENABLED=true`
  - `RETINA_RTSP_HEAL_CHECK_EVERY=10`
  - `RETINA_RTSP_HEAL_WINDOW_SECONDS=120.0`
  - `RETINA_RTSP_HEAL_MIN_SHARE=0.8`
- Removed or renamed keys: none.
- `.env_example` updated: yes. All flags are ON.
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys orion-vision-retina`: yes. It added all 6 keys to the primary checkout's `services/orion-vision-retina/.env`. There were 0 diverged lines; a dry-run afterwards reports "No changes needed".
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-vision-retina: pytest tests -q (host venv, py3.12)        -> 80 passed (3 consecutive runs)
same suite inside the retina image (py3.10 = CI interpreter)              -> 80 passed
mutation: removing the stop-after-open check                              -> regression test fails
all 34 run-steps of .github/workflows/orion-static-gates.yml locally      -> 0 failures
scripts/check_env_template_parity.py                                      -> PASS
git diff --check                                                          -> clean
```

## Evals run

```text
evals/frame_smear.py (the heal's scorer) on real walkway frames:
  2026-10-15 clean archive        frames=131 corrupt=0   (false positives: 0)
  2026-10-15 corrupt archive      frames=75  corrupt=75  (smear up to 0.83: detected)
  live frames dir 2026-10-16 ~18:20 UTC (after the 17:53 clean restart)
                                  frames=130 corrupt=0
```

These are all daytime frames. There are no archived night, IR, fog or rain walkway frames, so the false-positive rate in those scenes is UNVERIFIED. The 3-in-a-row heal cap limits the damage if the scorer misreads one of them.

## Docker/build/smoke checks

These ran in the existing retina image (OpenCV 4.10.0, avformat 59.27.100 = FFmpeg 5.1). The container had `--network none`. The worktree's `app/` was mounted, and a local fake RTSP server logged each RTSP method. No real camera was contacted.

```text
release() sends TEARDOWN:          METHOD TEARDOWN / CLOSED by client right after cap.release()
boot, no state file:               "rtsp quiet period: waiting 4s before connecting" -> PLAY 4.0 s later
stop():                            TEARDOWN; state file written {"last_session_activity_ts": ...}
new process, persisted state:      boot_state=persisted remaining=4.0 -> PLAY 4.0 s later
stream goes quiet -> 3 failed reads: TEARDOWN, "waiting 3s" (1 s reconnect already elapsed) -> PLAY 4.0 s after TEARDOWN
```

## Review findings fixed

A code review subagent reported 0 must-fix, 4 should-fix and 7 nit findings.

- Finding: a stop during a blocking open was followed by a blocking read. That is up to 15 s, longer than the 12 s join, so the process could exit with no TEARDOWN.
  - Fix: the reader checks the stop flag right after the open, then releases the capture.
  - Evidence: `test_stop_during_a_blocking_open_releases_without_reading`. That test fails when the check is removed.
- Finding: a false positive (fog or IR night) could trigger heal after heal, and the doubling wait would keep the camera blind most of the time.
  - Fix: heals stop after `HEAL_MAX_IN_A_ROW=3` with no clean window between them. Then `heal_suppressed=true` is set, a WARNING is logged, and streaming continues. A clean window resets it.
  - Evidence: `test_heal_stops_after_max_in_a_row_and_resumes_after_a_clean_window`, plus the false-positive eval above.
- Finding: the 90 s default is below the only gap proven clean (180 s).
  - Fix: kept 90 as the brief asked. Labeled UNVERIFIED with the trade-off and the signal to watch (see "Why 90 s is the default").
  - Evidence: n/a.
- Finding: two threads could write the state file at once, through the same tmp file.
  - Fix: the write is serialized with a lock, and the newest timestamp always wins.
  - Evidence: `test_concurrent_persists_keep_the_newest_timestamp` (200 threads).
- Finding: the quiet-wait flag was shared by all readers.
  - Fix: only the live reader sets or clears it.
- Finding: at a low frame rate the self-heal can never fire.
  - Fix: settings refuse an RTSP config with heal ON and fewer than 5 checks per window.
  - Evidence: `test_heal_needs_enough_checks_per_window_at_low_fps`.
- Finding: the gate ran on the wall clock, so an NTP step could shorten a wait.
  - Fix: waits now use the monotonic clock. The wall clock is used only for the persisted value.
  - Evidence: `test_a_wall_clock_step_does_not_shorten_the_wait`.
- Finding: readers retired by a heal were not joined at shutdown.
  - Fix: `stop()` joins them within the same budget.
  - Evidence: `test_stop_joins_retired_readers_so_they_tear_down`.
- Finding: two timing-sensitive tests could flake.
  - Fix: they now use 1.0 s quiet waits and wait on observed state.
- Not changed, accepted:
  - A small state-file write (throttled) can run on the event loop during a heal, watchdog or stop. The file is on a local disk.
  - A full-frame scoring check allocates about 59 MB, every 5 s, in a worker thread. It takes about 170 ms per check on a 2560x1920 frame.

## Restart required

Deploy from the clean primary `main` checkout after merge. **The deploy restart itself waits the quiet period. That is the point. Expect about 90 s with no walkway frames.**

```bash
scripts/safe_docker_build.sh orion-vision-retina --profile walkway up -d --build orion-vision-retina-walkway
```

The base `orion-vision-retina` instance only needs a rebuild if it runs on an RTSP source. For webcam and folder sources the changes have no effect.

Post-deploy check, about 3 min after the start:

```bash
docker logs --since 3m orion-vision-retina-walkway 2>&1 | grep -c 'bad cseq'     # expect 0
docker logs --since 3m orion-vision-retina-walkway 2>&1 | grep 'rtsp quiet period' # wait logged, then "over; connecting"
.venv/bin/python services/orion-vision-retina/evals/frame_smear.py /mnt/telemetry/vision/frames --glob 'walkway_*.jpg' --last 20   # corrupt=0
ls -l /mnt/telemetry/vision/retina_state/rtsp_session_walkway.json                 # state file exists
```

## Risks / concerns

- Severity: medium
  - Concern: the 90 s default is UNVERIFIED live. Only 180 s was tested.
  - Mitigation: the self-heal catches a failure. Watch `source_corruption_heals` after deploys. Raising the default is one env value.
- Severity: medium
  - Concern: the scorer's false-positive rate at night, in IR mode, fog or rain is unmeasured.
  - Mitigation: after 3 heals in a row, the self-heal only reports. Health shows `source_corruption_heal_suppressed`.
- Severity: low
  - Concern: a SIGKILL or crash cannot send TEARDOWN.
  - Mitigation: the state file is refreshed every 5 s while streaming, so the next boot still waits about the full period.
- Severity: low
  - Concern: if the camera is down, retries now come every 90 s, not every 5 s.
  - Mitigation: this is intended. A failed open may already have created a session on the camera side.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
