## Summary

- The Ring doorbell's live view decodes video again. It had been blind since about 2026-10-21 03:41 UTC: every live view ended with "no frame within 15s", while Ring motion and ring notifications kept arriving.
- Cause: Ring now sends 1920x1920 H.264 with keyframes of about 250 RTP packets. aiortc's video buffer holds 128 packets. When a frame overflows, aiortc drops the frame's head and gives the decoder only the tail, so nothing ever decoded.
- Fix: the doorbell's video receivers now buffer 512 packets (`ensure_video_jitter_capacity`, `ring_source.py`).
- Health now reports when the doorbell is blind: `ring_consecutive_failed_bursts`, and `ring_blind` with status `degraded` after 3 failed live views in a row. Before this, health said `source_ok: true` for 2.5 days while the camera was blind.

## Outcome moved

Live, 2026-10-23, one instrumented live view inside the doorbell container:
- 128-packet buffer: about 7,000 good video packets received (8 SPS, 8 PPS, 8 IDR starts), 0 frames decoded.
- 512-packet buffer: first frame after 2.7 s, 261 frames decoded in 12 s, 1920x1920.

## Current architecture

orion-vision-retina-doorbell: a Ring push event opens a 15 s aiortc WebRTC live view. Frames go out on `orion:vision:frames`. aiortc 1.15.0 `RTCRtpReceiver` hard-codes `JitterBuffer(capacity=128, is_video=True)`.

## Architecture touched

orion-vision-retina only. No bus, schema or env change.

## Files changed

- `services/orion-vision-retina/app/ring_source.py`: video buffer capacity, failed-in-a-row counter, `blind` property, setup failures counted.
- `services/orion-vision-retina/app/health.py`: `ring_consecutive_failed_bursts`, `ring_blind`; status `degraded` when blind.
- `services/orion-vision-retina/app/main.py`: copies the two new fields into metrics.
- `services/orion-vision-retina/tests/test_ring_source.py`: regression tests.

## Schema / bus / API changes

- Added: two health detail fields.
- Removed: none
- Renamed: none
- Behavior changed: health status is `degraded` when the doorbell is blind.
- Compatibility notes: mesh-guardian acts on `degraded` only when heartbeats are also late, so a blind doorbell does not trigger a restart loop.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
retina image, PYTHONPATH=.: pytest services/orion-vision-retina/tests -> 152 passed
New tests:
- a real 250-packet FU-A keyframe never comes out complete from aiortc's default buffer, and does after the fix
- capacity, idempotency, audio untouched
- failed-in-a-row count through _burst: 3 no-answer views -> blind; a frame resets it to 0
- a setup failure counts toward blind
- health payload is degraded with ring_blind=true
Mutation: removing the counter reset -> test_burst_delivers_frames fails.
```

## Evals run

```text
Live diagnostic (not a repeatable eval): /tmp/claude-1000/.../scratchpad/ringdiag{,4,5,6}.py inside the doorbell container. Results are above.
```

## Docker/build/smoke checks

```text
Tests ran inside the deployed retina image (aiortc 1.15.0, av 17.1.0).
```

## Review findings fixed

- Finding: the blind test set the counter directly, so a dropped reset would pass.
  - Fix: the count is now checked through `_burst` (fail x3 -> blind; a frame -> 0).
  - Evidence: mutation test above.
- Finding: the shipped capacity (1024) was not the capacity verified live (512).
  - Fix: ship 512. A lost packet stalls decoding for about 1 s instead of about 2 s.
  - Evidence: live 512 run, 261 frames.
- Finding: a failure building the peer connection never counted toward blind.
  - Fix: it is now caught, counted and logged.
  - Evidence: `test_a_setup_failure_counts_toward_blind`.

## Restart required

```bash
git pull --ff-only && scripts/safe_docker_build.sh orion-vision-retina up -d --force-recreate orion-vision-retina-doorbell
```

## Risks / concerns

- Severity: low
  - Concern: this patches aiortc's internal class lookup. A future aiortc may change it.
  - Mitigation: the test fails if the swap stops taking effect, and the module import is pinned (`aiortc==1.15.0`).
- Severity: low
  - Concern: Ring could grow keyframes past 512 packets.
  - Mitigation: `ring_blind` now degrades health within 3 live views, and the cause shows in the log.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
