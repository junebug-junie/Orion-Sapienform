## Summary

- The room camera's capture service (orion-vision-edge) now **reconnects** when the camera's video connection dies: after 5 s without a frame, retrying at 5, 10, 20 s and so on, up to once a minute.
- It **never hands out a stale frame as live**. Any frame older than 5 s reads as "no frame", which every consumer already handles.
- Its bus heartbeat says **"degraded" while it is blind**, and `/health` shows the newest frame's age and the reconnect count.
- Connections time out after 5 s instead of FFmpeg's ~30 s, and a decoder error counts as a failed read instead of killing the reader thread.

## Outcome moved

On 10-11 the room camera rebooted at 02:00 (its scheduled reboot) and dropped the stream. The old reader never reopened it. It failed about 200,000 reads in a row and kept serving the last good picture, a dark empty room stamped "02:00:00 am SUN", as live for about 16 hours:
- the frame router ran ~7,000 scene checks on that one frozen frame and never saw a person;
- Orion's presence record said "absent";
- the heartbeat said "ok".

Juniper was in front of the camera during that window, and Orion never saw her. No face check ran and no face frame was saved, so the full-resolution face checks from #2611 never got a chance to run.

## Current architecture

`CameraSource` opened the RTSP substream once at boot. A thread looped on `cap.read()` and, on failure, only slept. `last_frame` was never cleared, and the capture loop republished it every 1/FPS s. If the very first open failed, the thread slept forever.

## Architecture touched

orion-vision-edge only: `app/capture.py`, the heartbeat in `app/main.py`, `/health` in `app/routes.py`, and the walkway instance's compose block. No bus, schema or env changes.

## Files changed

- `services/orion-vision-edge/app/capture.py`: `step()` (read, or reconnect with backoff), stale-frame gate in `get_frame()`, `frame_age_sec()`, 5 s open/read timeouts, the reader thread owns the release.
- `services/orion-vision-edge/app/main.py`: heartbeat `status` is `degraded` when there is no live frame.
- `services/orion-vision-edge/app/routes.py`: `camera_frame_age_sec`, `camera_reopens` in `/health`.
- `services/orion-vision-edge/docker-compose.yml`: walkway instance heartbeats as `vision-edge-walkway`.
- `services/orion-vision-edge/tests/test_camera_reconnect.py`: new.

## Schema / bus / API changes

- Behavior changed: the `system.health.v1` heartbeat from vision-edge can now be `degraded`. orion-equilibrium computes distress from `last_seen_ts` only, so this adds ok/degraded transition rows but no distress.
- Added: two fields in `GET /health`.

## Env/config changes

- None. `.env_example` is unchanged, so there is nothing to sync.

## Tests run

```text
orion-vision-edge  pytest tests -q (in its image)   16 passed, 1 failed
  the failure is test_activity::test_publish_activity_if_allowed_publishes_person_detection, which also fails on main
```

## Evals run

```text
None for this service (it has no eval harness). Live evidence below stands in.
```

## Docker/build/smoke checks

```text
Throwaway container, new CameraSource against the real cam0 substream: live frame (480x640), age 0.12 s, 0 reopens; reader thread exits and releases on stop.
Unreachable camera host: get_frame() None, reopen #1 at 5 s, #2 at 13 s.
Wrong RTSP port: get_frame() None, reopens at 5 s and 10 s.
Before fix, live: newest published frame stamped 10/11 02:00:00 am SUN at 10-12 00:00Z; edge log "Failed to read frame (attempt 203061)".
```

## Review findings fixed

- Finding (material, latent): cam0 and walkway heartbeat under the same `service@node`, so one camera down would flip a shared record between ok and degraded.
  - Fix: walkway sets `SERVICE_NAME=vision-edge-walkway`; cam0's identity is unchanged.
  - Evidence: `test_walkway_has_its_own_heartbeat_identity`.
- Finding: an exception in the reader thread killed it for good.
  - Fix: read and reopen exceptions count as failed reads; `_run` has a backstop.
  - Evidence: `test_a_read_that_raises_counts_as_a_failure_and_still_reopens`.
- Finding: FFmpeg's ~30 s default timeouts let a slow open use up the backoff.
  - Fix: 5 s open/read timeouts for stream URLs; the next reopen is timed from after the open.
  - Evidence: unreachable-host smoke above.
- Finding: `stop()` could release the capture while a read was blocked inside it.
  - Fix: the reader thread releases on exit; `_open` refuses after stop.
- Nits: the unlocked frame/timestamp ordering is now commented as load-bearing; `/health` reads the age once.

## Restart required

From the primary checkout on main after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-vision-edge/.env -f services/orion-vision-edge/docker-compose.yml up -d --build vision-edge
```

Proof after deploy: `curl -s localhost:7100/health | jq '.camera_frame_age_sec, .camera_reopens'` shows an age under 1 s. The next camera reboot logs `Reopening camera ...` followed by `Camera stream recovered after ...s`.

## Risks / concerns

- Low: during a long camera outage the capture loop logs "No frames from camera" every ~5 s, which is intended.
- Low: a stream that really delivers under 0.2 frames/s would read as stale between frames. cam0 runs at 15 fps.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
