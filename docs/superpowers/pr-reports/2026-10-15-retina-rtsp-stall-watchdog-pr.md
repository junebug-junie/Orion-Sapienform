## Summary

- The walkway camera stopped sending pictures for about 34 minutes on 2026-10-15. The container stayed up and nothing was logged. This PR makes the RTSP reader recover on its own and say so in the logs.
- RTSP cameras now use TCP instead of UDP. Over UDP the walkway lost packets: 50 of the last 50 walkway frames had the bottom smeared into vertical streaks (worst: 71% of the picture).
- A silent stream can no longer block a frame read forever. FFmpeg gets a 5 s socket timeout, and OpenCV gets open and read timeouts.
- A watchdog notices when no frame has arrived for 30 s. It logs a WARNING. If the reader thread is stuck inside FFmpeg, the watchdog replaces it with a fresh one.
- System health now reports how old the newest frame is (`last_frame_age_sec`) and how many times the reader was replaced (`source_stall_restarts`).
- A new eval, `evals/frame_smear.py`, measures the smear in saved frames, so a post-deploy check can confirm the fix.

## Outcome moved

- **Silent outage.** Before: a stalled stream gave no log line, and only a manual `docker restart` recovered it. After: the stall is named in the log and in health, and the reader reconnects on its own. If readers stay stuck, the process exits and the walkway's `restart: unless-stopped` policy restarts it.
- **Smeared frames.** Before: 50/50 recent walkway frames were smeared (`smear_fraction` up to 0.71), and the log showed many `RTP: PT=60: bad cseq` lines. After: UNVERIFIED until deploy, because I could not open a second session to the live camera. The check is in "Restart required" below.

## Current architecture

- `services/orion-vision-retina/app/sources.py` `RtspFrameSource`: a background thread calls `cv2.VideoCapture.read()` and keeps only the newest frame. It had no transport setting, so FFmpeg's default was used: UDP RTP. It had no FFmpeg socket timeout; only OpenCV's built-in 30 s interrupt applied. Nothing checked how old the newest frame was.
- `app/main.py` `capture_once`: when there was no frame, it added 1 to `frames_failed` and logged nothing.
- Bus `system.health.v1` `details`: no frame age and no reconnect count.

## Architecture touched

- One service, `orion-vision-retina`. Both compose instances (`orion-vision-retina` and the `walkway` profile) pass the new keys through.
- No bus, schema, or channel changes. The two new health `details` keys are additive. `SystemHealthV1.details` is a free-form dict, and a test validates the new keys against it.

## Files changed

- `services/orion-vision-retina/app/sources.py`: adds `rtsp_capture_options()`, a default opener that passes `CAP_PROP_OPEN_TIMEOUT_MSEC`/`CAP_PROP_READ_TIMEOUT_MSEC`, a progress-aware stall watchdog, an abandoned-reader cap with an exit request, and a stop-event check under the lock.
- `services/orion-vision-retina/app/main.py`: passes the RTSP settings to the source, counts stall restarts, reports the stall reason in `last_error`, and exits if the source asks for a process restart.
- `services/orion-vision-retina/app/health.py`: adds `last_frame_age_sec` and `source_stall_restarts` to health details.
- `services/orion-vision-retina/app/settings.py`: adds 3 new keys, a transport validator, and a stall-window validator.
- `services/orion-vision-retina/.env_example`, `docker-compose.yml`, `README.md`: the new keys and documentation.
- `services/orion-vision-retina/evals/frame_smear.py`: new smear eval.
- `services/orion-vision-retina/tests/*`: watchdog, options, compose parity, smear eval, and `conftest.py`.
- `tests/test_vision_retina_capture_once.py`, `tests/test_vision_retina_health.py`: new cases. The two old health tests were already failing on main (they did not pass `heartbeat_interval_sec`); they are fixed here.

## Schema / bus / API changes

- Added: health `details.last_frame_age_sec`, `details.source_stall_restarts`.
- Removed / Renamed: none.
- Behavior changed: RTSP sources set the process env `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|timeout;5000000`. Webcam, folder, and mock sources do not touch it (tested).
- Compatibility notes: these keys are additive in a free-form dict.

## Env/config changes

- Added keys: `RETINA_RTSP_TRANSPORT=tcp`, `RETINA_RTSP_IO_TIMEOUT_SECONDS=5.0`, `RETINA_RTSP_STALL_SECONDS=30.0`. All ship ON.
- `.env_example` updated: yes. Both compose services pass the keys through with the same defaults.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-vision-retina`: yes.
  - The stall key first synced as 15.0. After review the default became 30.0, and the new validator refuses 15 when the IO timeout is 5. The sync script does not overwrite a diverged key without `--force`, so I changed that one line by hand in the primary checkout's `services/orion-vision-retina/.env`. Verified: `Settings` loads it, and `compose --profile walkway config` renders `tcp / 5.0 / 30.0` for both instances.
- Skipped keys requiring operator action: none.

### Which FFmpeg option name works (measured in the built image)

OpenCV 4.10.0 bundles its own FFmpeg: avformat 59.27.100 / avcodec 59.37.100, which is FFmpeg 5.1. The system `ffmpeg` binary in the image is 7.1, but OpenCV does not use it.

I tested against a local fake RTSP server inside the image (`--network none`). The server streams H.264 for 4 s over TCP, then goes quiet without closing the connection. These are the times the read blocked after the stream went quiet:

| `OPENCV_FFMPEG_CAPTURE_OPTIONS` | blocked read returned after |
|---|---|
| none (production before this PR) | 30.0 s (OpenCV's default interrupt) |
| `rtsp_transport;tcp\|timeout;5000000` | **5.1 s** |
| `rtsp_transport;tcp\|stimeout;5000000` | 30.0 s (ignored) |
| `rtsp_transport;tcp\|rw_timeout;5000000` | 30.0 s (ignored) |
| `CAP_PROP_READ_TIMEOUT_MSEC=4000` param | 4.0 s |

Opening against a server that accepts the connection and never answers returned in 5.1 s.

So OpenCV's own 30 s timeout already existed. The 34-minute outage was therefore probably not one read blocked forever. More likely, the reader kept cycling and never got a good frame: the UDP stream was corrupted, and the `bad cseq` lines alternate between two sequence spaces, which suggests a stale second session sending to the same UDP port. That is my inference; it is unproven because nothing recorded the reader's state at the time. TCP removes that failure class. The watchdog and the logs cover the rest.

## Tests run

```text
cd services/orion-vision-retina && pytest tests -q          -> 26 passed (x6 runs, plus 4 runs under 4 CPU burners: all green)
pytest services/orion-vision-retina/tests tests/test_vision_retina_*.py -q   -> 83 passed
python scripts/check_env_template_parity.py                 -> PASS (93 services)
```

Key tests:
- A wedged reader (`read()` blocks on an Event after N frames) is replaced after the stall window. Frames resume from the new capture. When the wedged read is released, its frame never reaches `_latest`.
- A reader that keeps cycling (camera down) gets one stall message and no restart. The message clears when frames come back.
- Abandoned readers are capped at 2. After 4 windows stuck at the cap, the source asks for a process restart.
- The capture options are set for RTSP and not for webcam, folder, or mock. The opener passes open and read timeouts.
- The settings validator refuses a stall window that is not longer than 3 x IO timeout + reconnect.

## Evals run

```text
python evals/frame_smear.py /mnt/telemetry/vision/frames --glob 'walkway_*.jpg' --last 50
  frames=50 smeared=50 worst_smear_fraction=0.71    (live walkway, UDP, before this PR)
python evals/frame_smear.py /mnt/telemetry/vision/frames --glob '[!w]*.jpg' --last 200
  frames=200 smeared=0 worst_smear_fraction=0.00    (indoor cam0)
```

How the metric works: for each 16-row strip, it divides the change going down by the change going across. Real scenes measured 0.56 or more. The smeared walkway strips went down to 0.17. Strips with no horizontal texture are skipped, such as flat sky or a dark night band (tested). Limit: no clean walkway frame existed yet to calibrate against, so the post-deploy run is that check.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-vision-retina build orion-vision-retina      -> Built (twice, final after review fixes)
scripts/safe_docker_build.sh orion-vision-retina --profile walkway config       -> both instances render RETINA_RTSP_* = tcp/5.0/30.0
In-image e2e (final code, real RtspFrameSource, fake server goes quiet after 4s, io=2s, stall=8s):
  WARNING rtsp read failed 3 times; reconnecting
  WARNING rtsp stalled: no frame for 8s; reader alive and reconnecting
  INFO    rtsp frames resumed after 10s (stall_restarts=0)
```

The wedged-in-FFmpeg replacement path is proven by unit tests only. With the timeouts now working, real FFmpeg could not be made to wedge in the harness.

## Review findings fixed

- Finding: `stall_error` could stay set after frames resumed, through the cap branch.
  - Fix: it now clears whenever a frame is served.
  - Evidence: `test_stall_error_clears_after_a_capped_window_and_resume`.
- Finding: the watchdog could not tell a camera that is down from a wedged reader. It would restart and WARN every window for the whole outage.
  - Fix: a reader progress clock. The watchdog restarts a reader only if it made no progress for a whole window. Otherwise it logs one WARNING.
  - Evidence: `test_a_reader_that_keeps_cycling_is_named_but_not_replaced`, plus the in-image e2e above.
- Finding: a 15 s stall window was the same length as the reader's own recovery cycle (3 x 5 s), so the two raced and opened a second camera session.
  - Fix: the default is now 30 s, and a validator requires a window longer than 3 x IO + reconnect.
  - Evidence: the settings test.
- Finding: the smear eval would flag a flat sky or a night scene.
  - Fix: it now uses the direction ratio and skips untextured strips.
  - Evidence: `test_flat_sky_or_dark_band_is_not_smear`, plus live measurements.
- Finding: when stuck at the abandoned-reader cap, the camera stayed dark until someone restarted it by hand.
  - Fix: after 4 capped windows the process exits, and the restart policy brings it back.
  - Evidence: `test_readers_stuck_at_the_cap_ask_for_a_process_restart`.
- Finding: abandoned readers could overwrite `last_error`.
  - Fix: those writes are guarded by the reader's own stop event.
- Finding: the process-global env var leaked between tests, and service tests failed to collect when run from the repo root.
  - Fix: `tests/conftest.py`.
  - Evidence: 83 tests pass from the repo root.
- Not fixed (nit): `start()` after a timed-out `pause_device` stop does not apply the abandoned cap. Only RTSP with clips enabled reaches this, and the walkway has clips off.

## Restart required

The app directory is bind-mounted from the primary checkout. A rebuild still picks up the env keys. Deploy from the clean primary `main` after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-vision-retina up -d --build orion-vision-retina && scripts/safe_docker_build.sh orion-vision-retina --profile walkway up -d --build orion-vision-retina-walkway
```

Post-deploy checks (after ~2 min, so the old UDP frames have aged out of the frames dir):

```bash
docker logs --since 2m orion-vision-retina-walkway 2>&1 | grep -c 'bad cseq'    # expect 0 (was continuous)
/mnt/scripts/Orion-Sapienform/.venv/bin/python /mnt/scripts/Orion-Sapienform/services/orion-vision-retina/evals/frame_smear.py /mnt/telemetry/vision/frames --glob 'walkway_*.jpg' --last 20    # expect smeared=0 (was 50/50)
```

## Risks / concerns

- Severity: medium.
  - Concern: some cameras refuse RTSP over TCP. The Reolink is expected to support it (it is standard), but this is unverified against this unit.
  - Mitigation: set `RETINA_RTSP_TRANSPORT=udp` in the local `.env` to revert without a code change. If TCP is refused, the log shows `rtsp open failed` / reconnect WARNINGs instead of silence.
- Severity: low.
  - Concern: the process exits after 4 windows stuck at the cap. The base `orion-vision-retina` service has no restart policy, but it is a folder source on this host, so the path cannot trigger there. The walkway has `restart: unless-stopped`.
- Severity: low.
  - Concern: the cause of the 34-minute outage is inferred, not proven (see the measurement section above).

## PR link

(filled in after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
