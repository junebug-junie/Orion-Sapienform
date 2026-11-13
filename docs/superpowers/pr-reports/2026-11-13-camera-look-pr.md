## Summary

- Orion can look at a camera on request: `python -P -m orion.dig look office` (or `walkway`, `doorbell`). The command copies that camera's newest saved frame into the sandbox and prints its path. Orion opens it with the Read tool and sees it.
- `dig look` with no camera lists each camera's newest-frame age.
- The harness-governor (where turns run) gets a read-only mount of the frames folder.
- The one-line dig brief in each turn's prompt now mentions `look`.
- Gateway: when the GPU pool puts a call carrying an image on a lane that cannot see, the image becomes a short text note. Before, the call failed, and a long curiosity run could die on one picture.
- Skills stay off. A skill is instructions, not a capability. Design: `docs/superpowers/specs/2026-11-13-camera-look-design.md`.

## Outcome moved

Orion can choose to perceive. Before this, camera frames reached Orion only as captions other services wrote. Now a chat turn or curiosity run can open the raw frame itself. Prime reaches this through its existing `curiosity_focus` move.

## Current architecture

- The office (`cam0`) and walkway retinas save a frame every 0.5 s to `/mnt/telemetry/vision/frames` and keep 5 minutes of them.
- The chat lane (8011) and the agent lane (8015) report `vision: true` on `/props`. fast/metacog (8013) and agent-gpu2 (8016) report false.
- The Claude CLI's Read tool already hands image files to the model (verified 2026-08-15).
- The gateway checked vision only on its bus path, not on the passthrough path that harness turns use.

## Architecture touched

- `orion/dig` (Orion's tool kit)
- `orion-harness-governor` compose: one read-only bind mount
- `orion-llm-gateway` passthrough dispatch

## Files changed

- `orion/dig/look.py`: new. Finds the newest complete frame, copies it, logs it, caps looks, sweeps old copies.
- `orion/dig/__main__.py`: adds the `look` command and its help line.
- `orion/dig/brief.py`: adds a camera pointer to the brief line. Cap raised from 420 to 520 characters.
- `orion/dig/README.md`: documents `look`.
- `orion/dig/tests/test_look.py`: new tests.
- `services/orion-harness-governor/docker-compose.yml`: frames folder mounted read-only.
- `services/orion-llm-gateway/app/vision.py`: adds `count_images` and `replace_images_with_note`.
- `services/orion-llm-gateway/app/passthrough_proxy.py`: checks the granted lane's vision after placement, for llamacpp and Strata backends only.
- `services/orion-llm-gateway/tests/test_blind_lane_images.py`: new tests.
- `docs/superpowers/specs/2026-11-13-camera-look-design.md`: the design doc.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: on a blind lane, gateway passthrough calls carry a text note in place of each image.
- Compatibility notes: text-only calls are never probed and pass through unchanged. vLLM grants are never probed.

## Env/config changes

- Added keys: none (`ORION_DIG_FRAMES_DIR` is a test override with a fixed default, like `ORION_DIG_REPO`).
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/dig/tests                      -> 174 passed
pytest services/orion-llm-gateway/tests     -> 502 passed
pytest services/orion-harness-governor/tests -> 105 passed, 1 failed
  (test_fcc_warm_pool_wiring: PermissionError on ~/.fcc/.env; it fails the same way on main and is unrelated)
static gates: diff --check, async-routes, compose relative mounts, compose claude.json, hostname refs,
  env-key single source, definition drift, metric lineage, no personal terms -> all PASS
```

## Evals run

```text
Live model check (direct to the chat-lane worker, 8011): sent a frame from `dig look walkway`.
The model answered: "a nighttime view from a porch, featuring a white Adirondack chair in the
foreground and a residential street with houses across the lawn ... timestamp reading 12:38:43 am".
The frame was taken at 00:38:43 MST. About 1,000 prompt tokens per image; 4.7 s.

Not run through the gateway route: both sighted lanes were held by running work and the pool
refused a test lease (deadline). Gateway behaviour is covered by tests. UNVERIFIED live: a real
chat turn running `look` and then Read.
```

## Docker/build/smoke checks

```text
docker compose ... orion-harness-governor config -> renders /mnt/telemetry/vision/frames:ro
dig look (host, live frames) -> office 0.2 s old (live), walkway 0.5 s old (live); copy + looks.jsonl written
live frames end with FF D9 (the complete-JPEG check matches real files)
```

## Review findings fixed

- Finding: the newest frame can be read while the retina is still writing it (cv2.imwrite writes in place).
  - Fix: a frame counts only if it starts with FF D8 and ends with FF D9. Otherwise `look` uses one of the next two newest frames.
  - Evidence: `test_half_written_newest_frame_falls_back_to_the_one_before`, `test_all_newest_frames_cut_off_is_an_error`.
- Finding: when `/props` did not answer, the note wrongly said the lane cannot see.
  - Fix: the note now says "could not confirm this lane can see images".
  - Evidence: `test_unreachable_props_note_does_not_claim_the_lane_is_blind`.
- Finding: no test covered the vLLM gate, streaming, or the retry after an overflow.
  - Fix: added all three tests.
  - Evidence: `test_vllm_grant_is_never_probed`, `test_streaming_call_on_a_blind_lane_is_stripped_too`, `test_retry_after_overflow_on_a_sighted_lane_gets_the_real_image`.
- Finding: a log row with no time zone would crash the hourly cap.
  - Fix: such rows are skipped.
  - Evidence: `test_malformed_log_rows_do_not_crash_the_cap`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
```

Run both from the clean primary `main` checkout after merge. The post-pull rebuild does the same, and then resets the FCC sandbox, which is what makes `dig look` live for Orion.

## Risks / concerns

- Severity: low
  - Concern: whether the turn record (`cockpit_turn_sighting`) stores the image bytes of a Read.
  - Mitigation: check the first real look's row after deploy (noted in the design doc).
- Severity: low
  - Concern: the vision check is cached for 5 minutes. A lane that swaps from a sighted model to a blind one still gets images for up to 5 minutes. This risk existed before this PR.
  - Mitigation: none in this PR.
- Issues opened: #3320 (camera looks are not on the act ledger).
- Issues closed: none.

## PR link

(added after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
