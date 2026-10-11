## Summary

- Face checks on the room camera (cam0) now run on a **full-resolution still (2560x1920)** instead of the 640x480 frame the camera streams to Orion all day.
- The camera-capture service (orion-vision-edge) gets `POST /still`: it opens the camera's main stream, grabs one frame, writes it to the shared frame folder and returns the path. The camera password stays in that service.
- The frame router asks for that still right before a face check, without holding up anything else, and falls back to the small frame if the still fails. Each face check records which image it used.
- vision-host keeps each detected face cropped at full resolution next to the saved frame (`<stem>_face<N>.jpg`), so Juniper can be re-enrolled from this camera.

## Outcome moved

Juniper's own cam0 frames from 10-11 04:44Z (5 frames, all her, all "unsure") scored **0.09-0.18 against each other**, and a back-of-the-head frame scored 0.34-0.38 against the desk frames. At 640x480 her face at the desk is about 25 px wide, while the face model works from 160 px, so the model was matching "a blur in that chair", not her face. Re-enrolling from those frames would have taught it exactly that. A full-resolution still puts the same face at about 100 px.

## Current architecture

orion-vision-edge reads the camera substream (`Preview_01_sub`) continuously and publishes frame pointers. The frame router dispatches `identity_face` on the same frame when a person was seen recently (5 s spacing on cam0). vision-host ran MTCNN + facenet on that frame. The camera's HTTP snapshot API returns 500 on every request, so stills come from the RTSP main stream.

## Architecture touched

orion-vision-edge (new route), orion-vision-frame-router (identity dispatch path), orion-vision-host (identity face check + face frame store), `config/vision_frame_router.yaml`. No bus or schema changes.

## Files changed

- `services/orion-vision-edge/app/still.py`: one-frame grab from the main stream, atomic write.
- `services/orion-vision-edge/app/routes.py`: `POST /still` (one at a time), `hires_still` in `/health`.
- `services/orion-vision-edge/app/settings.py`, `.env_example`, `docker-compose.yml`: `HIRES_SOURCE` from `REOLINK_HIRES_URL` (cam0 service only; walkway stays off).
- `services/orion-vision-frame-router/app/dispatcher.py`: identity publish factored into `_publish_identity`; `_identity_with_still` fetches the still in the background; `request_still`.
- `services/orion-vision-frame-router/app/policy.py`, `state.py`: no second face check for a camera while its still is being fetched.
- `services/orion-vision-frame-router/app/metrics.py`: `identity_still_total`, `identity_still_fallback_total` in health.
- `config/vision_frame_router.yaml`: cam0 `still_url`, `still_timeout_sec`.
- `services/orion-vision-host/app/runner.py`: `mtcnn.detect` + `mtcnn.extract` (what `mtcnn(img)` runs with `keep_all=True`) so the face boxes are kept; sidecar records boxes, source size and image source.
- `services/orion-vision-host/app/face_frames.py`: full-resolution face crops.
- Tests in all three services.

## Schema / bus / API changes

- Added: HTTP `POST /still` on orion-vision-edge.
- Behavior changed: `identity_face` requests from cam0 carry `image_source` and point at the still. Request is a free dict; no schema change.
- Compatibility: if the still is unavailable, the face check runs on the substream frame as before.

## Env/config changes

- Added keys: `REOLINK_HIRES_URL` (orion-vision-edge; passed in as `HIRES_SOURCE`).
- `.env_example` updated: yes (placeholder only).
- Local `.env`: `REOLINK_HIRES_URL` written by hand into the primary checkout's `services/orion-vision-edge/.env`, built from the existing `REOLINK_URL` with `Preview_01_main`. The sync script would have copied the placeholder, which carries no password.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion-vision-frame-router  pytest tests -q                         85 passed
orion-vision-edge          pytest tests -q (in its image)           11 passed, 1 failed (test_activity, also fails on main)
orion-vision-host          pytest tests -q (in its image)          242 passed, 1 skipped
check_env_template_parity.py                                        PASS
check_service_env_compose_parity.py orion-vision-edge               N/A (env_file)
```

## Evals run

```text
Read-only similarity probe on the 5 saved cam0 frames (in a throwaway vision-host container):
  pairwise 0.09-0.68, close frontal shot vs others 0.09-0.18, back-of-head vs desk 0.34-0.38
  vs current enrollment 0.14-0.31
Face detection cost on the P4: 640x480 0.04 s, 2560x1920 0.25 s.
```

The before/after eval (same probe on full-resolution face crops) runs once stills are live and Juniper is in front of the camera.

## Docker/build/smoke checks

```text
Live grab with app/still.py against the real camera main stream:
  2560x1920 in 2.6 s, then 1.8 s; wrong port -> None
```

## Review findings fixed

- Finding (material): a slow camera stream could build an unbounded queue of grabs on the capture service, each running after the router had stopped waiting.
  - Fix: `POST /still` answers 409 "busy" instead of queueing; `grab_still` stops at an 8 s deadline and caps open/read timeouts at it.
  - Evidence: `test_gives_up_at_the_deadline`.
- Finding: an error publishing the delayed face check vanished (background task).
  - Fix: recorded in the router's `last_error`; the pending flag still clears.
- Finding: `frame_ts` describes the triggering frame, not the still.
  - Fix: `meta.still_frame_ts` added; `frame_ts` kept as the trigger's time.
- Finding: a face box outside the image would fail the whole frame save.
  - Fix: that crop is skipped. Evidence: extended `test_faces_are_cropped_at_source_resolution_before_the_frame_shrinks`.
- Finding: a future `HIRES_SOURCE` in the shared .env would turn stills on for the walkway instance.
  - Fix: walkway sets `HIRES_SOURCE=` explicitly.
- Noted, not changed: the delayed publish can exceed `max_inflight_total` by one per camera (commented); `/still` has no auth on the LAN port (the busy answer bounds abuse); sidecar boxes are in source pixels (commented).
- Checked by review: `detect` + `extract` returns exactly what `mtcnn(img)` did with `keep_all=True`; no password reaches logs, health or responses; nothing downstream reads the identity request's image path.

## Restart required

From the primary checkout on main after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-vision-edge orion-vision-host orion-vision-frame-router; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build; done
```

The frame router's config is a single-file bind mount, so its container must be recreated (the loop above does that), not just pulled.

Proof after deploy: `curl -s localhost:7100/health | jq .hires_still` is `true`; router logs show `identity_dispatch ... image_source=hires_still`; new `face_frames/cam0/<date>/*_face0.jpg` files appear when someone is in the room.

## Risks / concerns

- Low: each face check now opens the camera's main stream (2-5 s, one at a time, at most every 5 s while someone is in the room). The camera may limit concurrent main-stream viewers; a refusal falls back to the substream frame and is counted.
- Low: the still is taken 2-5 s after the frame that triggered the check, so a person who just left gives a still with no face. That check reports no face, as it should.
- Low: face crops add a small file per face to the 14-day face frame store.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
