# PR report: ask_camera skill — on-demand VQA, fresh capture, real chat wiring

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1793
Branch: `feat/ask-camera-skill` (built on top of #1791, `feat/vision-host-vqa`, which merged to `main` while this branch was in progress)

## Summary

- Completes what `LookAtCameraVerb`'s own docstring named as separate, larger work: "triggering an on-demand capture (bypassing the window/council chain entirely, e.g. via a direct vision-host RPC)".
- New skill `skills.perception.ask_camera.v1`: posts a real question to vision-host's `/v1/vision/task` (`task_type=vqa`) against the most recently captured frame, bypassing `orion-vision-window`/`orion-vision-council` entirely. This is the actual consumer PR #1791's `_run_vlm_vqa` was missing — it had zero real callers in the cognition loop until this.
- New "fresh-capture resolution" in `orion-vision-host`: `request.use_latest_frame: true` opt-in resolves to whatever `orion-vision-edge` captured most recently (it already captures continuously at ~5s cadence regardless of any downstream consumer) — the practical equivalent of "look now" for a slow-moving room, without inventing a second capture path.
- Two real bugs found and fixed via code review + live verification, not theoretical:
  1. `capability_bridge.py`'s `_SEMANTIC_VERB_TO_SKILL` map is documented as "avoids family ordering bugs" but `look_at_camera` was never actually in it — it only resolved correctly by accident as the sole perception-family skill. Adding `ask_camera` as a second one broke it immediately (confirmed live). Fixed by pinning both explicitly.
  2. `executor.py::_plan_request_from_step_ctx`'s user-text injection (the same pattern already fixed twice for `docker_prune_stopped_containers` and `notify_chat_message`, both after a live miss) was missing a third case for `ask_camera` — meaning a real chat-driven "ask the camera" turn would have silently seen an empty question and returned `missing_question`, even though every direct-call unit test passed. Caught by code review before shipping, fixed with the same pattern, live-verified end-to-end.
- Also recalibrated `VISION_VRAM_RESERVE_MB`/`VISION_VRAM_HARD_FLOOR_MB` (flagged out-of-scope in #1791; fixed now because it was blocking full live verification of both patches).

## Outcome moved

Orion can now actually ask a real question about the current camera scene — via the real chat/planner dispatch path, not just direct API calls. Verified live, inside the real running containers, composing the real functions together (not mocks):

```
_plan_request_from_step_ctx(ctx={"raw_user_text": "Is the door open?"})
  -> skill_args = {"question": "Is the door open?"}
  -> AskCameraVerb.execute()
  -> real HTTP POST to orion-athena-vision-host
  -> real GPU inference (Salesforce/blip-image-captioning-base)
  -> {"ok": true, "status": "no_answer"/"ok", "question": "...", "answer": "...", ...}
```

Both `task_type=vqa` and the pre-existing `task_type=caption_frame` now succeed via the real scheduler-gated HTTP endpoint after the VRAM recalibration.

## Current architecture

`LookAtCameraVerb` (PR #1679) reads `orion-vision-window`'s passive projection — no fresh capture, no real question, read-only summary. `_run_vlm_vqa` (PR #1791) implemented real VQA execution in `orion-vision-host` but had no caller anywhere in the cognition loop. This PR connects them.

## Architecture touched

- `services/orion-vision-host/app/runner.py`: `_resolve_latest_frame_path()`, `use_latest_frame` opt-in on `_load_image_from_request()`.
- `services/orion-vision-host/app/settings.py`: `VISION_FRAMES_DIR`.
- `services/orion-cortex-exec/app/verb_adapters.py`: `AskCameraVerb`, `_http_json_post()`.
- `services/orion-cortex-exec/app/capability_bridge.py`: `_SEMANTIC_VERB_TO_SKILL` gains both `look_at_camera` and `ask_camera`.
- `services/orion-cortex-exec/app/executor.py`: `ask_camera` case in `_plan_request_from_step_ctx`.
- `orion/cognition/verbs/skills.perception.ask_camera.v1.yaml`, `orion/cognition/verbs/ask_camera.yaml`: new manifests.
- `services/orion-vision-host/.env_example`: `VISION_FRAMES_DIR`, VRAM recalibration.
- `services/orion-cortex-exec/.env_example` + `docker-compose.yml`: `VISION_HOST_SERVICE_URL`, `VISION_HOST_HTTP_TIMEOUT_SEC`.

## Files changed

See PR #1793's own "Files changed" list — matches the "Architecture touched" section above plus:
- `services/orion-vision-host/tests/test_fresh_capture_resolution.py` (new)
- `services/orion-cortex-exec/tests/test_skill_verbs.py` (AskCameraVerb tests added)
- `services/orion-cortex-exec/tests/test_executor_ask_camera_skill_args.py` (new)

## Schema / bus / API changes

- Added: none published to the bus. `task_type=vqa` (already valid per #1791) now has a real caller.
- Removed / renamed: none.
- Behavior changed: `_load_image_from_request` accepts a new opt-in `use_latest_frame` field — additive, every existing caller keeps identical behavior when it's absent.
- Compatibility notes: none needed.

## Env/config changes

- Added keys: `VISION_FRAMES_DIR` (vision-host), `VISION_HOST_SERVICE_URL`/`VISION_HOST_HTTP_TIMEOUT_SEC` (cortex-exec).
- Changed values: `VISION_VRAM_RESERVE_MB` (3500→1200), `VISION_VRAM_HARD_FLOOR_MB` (1400→800), `VISION_VRAM_SOFT_FLOOR_MB` (2200→1000, confirmed dead/unread config, changed only for documentation consistency).
- `.env_example` updated: yes, both services.
- local `.env` synced: yes, both services, hand-verified `git check-ignore`'d.
- skipped keys requiring operator action: none.

## Tests run

```text
Venv:
  services/orion-cortex-exec/tests/test_skill_verbs.py -> 42 passed, 1
    pre-existing unrelated failure (confirmed via git stash on base branch)
  services/orion-cortex-exec/tests/test_executor_ask_camera_skill_args.py -> 5 passed
  services/orion-cortex-exec/tests/test_executor_docker_prune_skill_args.py,
    test_executor_direct_dispatch_skill_args.py,
    test_executor_capability_bridge_skill_args.py -> all passing
  (Run as separate invocations -- pre-existing cross-file @verb-registration
  collision when the whole tests/ dir runs together, confirmed identical
  on base branch via git stash, same documented precedent as PR #1679.)

  orion/vision/tests/test_caption_echo.py + services/orion-vision-host/tests/
    (excluding torch-dependent files) -> 66 passed

Real environment (torch, inside orion-athena-vision-host):
  test_fresh_capture_resolution.py -> 7 passed
```

## Evals run

None — no eval harness exists for either service.

## Docker/build/smoke checks

```text
bash scripts/safe_docker_build.sh orion-vision-host up -d --build
bash scripts/safe_docker_build.sh orion-cortex-exec up -d --build
  -> both rebuilt/redeployed multiple times, each time md5-verified against
     the worktree source before trusting it.
  -> Hit a live recurrence of the cross-worktree Docker deploy collision
     (PR #1776's documented mechanism) mid-session: a concurrent build from
     elsewhere reverted both orion-vision-host's env and image between
     deploys. Caught via the same md5 discipline, redeployed with --build,
     re-confirmed correct before trusting the final live verification.

Final live verification:
  1. Full chain: _plan_request_from_step_ctx -> AskCameraVerb.execute() ->
     real HTTP POST -> orion-athena-vision-host -> real GPU inference ->
     ok=True response with the real threaded question.
  2. VRAM after both models resident: 4033/7680MB -- comfortable headroom.
     Both task_type=vqa and task_type=caption_frame succeed via the real
     scheduler-gated HTTP endpoint (both were blocked before this patch).
```

## Review findings fixed

- Finding (HIGH): `skill_args["question"]` never reached `AskCameraVerb` via the real chat/planner dispatch path -- `_plan_request_from_step_ctx` had no `ask_camera` case in its user-text-injection logic (the same gap already hit and fixed twice for other verbs).
  - Fix: added a third case mirroring the existing two exactly.
  - Evidence: new `test_executor_ask_camera_skill_args.py` (5 tests, including an integration test composing both real functions) plus live verification inside the real running containers.
- Finding (MEDIUM): `vision_host_service_url`/`vision_host_http_timeout_sec` were missing from `.env_example`, local `.env`, and `docker-compose.yml`'s `environment:` block.
  - Fix: added to all three, matching the existing `VISION_WINDOW_*` sibling pattern.
- All other review findings: none — reviewer confirmed backward compatibility, response-shape correctness, the `no_answer`-is-`ok=True` design choice, the `_SEMANTIC_VERB_TO_SKILL` fix's sufficiency, and the VRAM scheduler's concurrency behavior all check out.

## Restart required

```bash
bash scripts/safe_docker_build.sh orion-vision-host up -d --build
bash scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

Already deployed and live-verified during this session.

## Risks / concerns

- Severity: should-know, not blocking — answer quality is honestly weak (BLIP-base is a captioner, not an instruction-tuned VQA model) -- same documented limitation as #1791.
- Severity: note, not blocking — `_SEMANTIC_VERB_TO_SKILL` requires a manual pin per new ambiguous verb; no automatic generalization for a third+ perception skill later.
- Severity: note, not blocking — the scheduler's VRAM admission check samples NVML free VRAM once at admission, before acquiring the per-GPU semaphore -- pre-existing design, not introduced here, but the tighter margins leave less slack. Mitigated by `VISION_MAX_INFLIGHT_PER_GPU=1` on this single-GPU host. Worth a follow-up if a second GPU is ever added.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1793
