# PR report: route the foveal probe through orion-llm-gateway instead of a dedicated vision-host

## Summary

- The foveal probe (`POST /debug/foveal-probe`) originally RPC'd a dedicated, isolated `orion-vision-host` instance. Juniper deployed real config for that path this session (`CHANNEL_FOVEAL_HOST_REQUEST`, `FOVEAL_PERCEPT_STORE_URL`, etc.) and asked how it looked to proceed.
- Live-tested end-to-end against the real, newly-deployed config: the pipeline worked mechanically (frame resolve → percept-store upload → RPC → decode all succeeded, `ok: true` throughout) but returned an **empty** caption (`"text": "", "caption_rejected:too_short"`). Root cause: `config/vision_profiles.yaml`'s `vlm_caption`/`vlm_vqa` profiles still carry the unfilled placeholder `model_id: "REPLACE_ME/qwen2-vl_or_llava_next"`, which falls back to `orion-vision-host`'s own `VISION_VLM_MODEL_ID` default — itself another BLIP-family model (`blip2-opt-2.7b`), not a real VLM. A live instance of CLAUDE.md's own "empty-shell cognition" anti-pattern.
- Per the existing-mechanism check (CLAUDE.md's metric/mechanism gate, item 5): a real, already-warm, already-proven vision-capable model already exists — circe's `chat` lane (Qwen3.6-35B-A3B, `modalities.vision=true`, confirmed live via `/props`) — and `orion-llm-gateway/app/vision.py` already implements a `kind="percept"` attachment path built specifically for camera frames. `orion-vision-council` was already a registered producer on `orion:exec:request:LLMGatewayService` (used for its own metacog calls), and the gateway's live config already allowlisted `orion-athena-percept-store` as an attachment host.
- Juniper chose this direction explicitly ("b" — the gateway route, over filling in the `REPLACE_ME` placeholder with a dedicated local VLM).
- Rewrote `app/foveal_probe.py`'s hop 3 to build a `ChatRequestPayload` with a `kind="percept"` attachment instead of a `VisionTaskRequestPayload`, RPC'd to the gateway's existing `CHANNEL_LLM_REQUEST` on the `chat` route. Retired `CHANNEL_FOVEAL_HOST_REQUEST`/`CHANNEL_FOVEAL_HOST_REPLY_PREFIX` and the now-producer-less `orion:exec:request:VisionHostService:*` channels.yaml registration outright.
- Live-verified the new path end-to-end after deploy: a real, rich, multi-sentence caption of the actual room scene came back from `circe-worker-1` (`Qwen3.6-35B-A3B-UD-Q5_K_M.gguf`), not an empty rejection.

## Outcome moved

The Foveal tier now actually delivers what it was built for — a real, richer-than-BLIP interpretation of the current frame — instead of mechanically round-tripping to the same weak captioner the always-on peripheral pipeline already uses. Before: `caption.text = ""`, `caption_rejected:too_short`. After: a real multi-sentence, concrete description of the actual captured room (desk, monitor, printer, chairs, closet, curtain, door, hallway), served by circe's 35B chat-lane model.

## Current architecture

`orion-vision-council`'s `/debug/foveal-probe` endpoint (`app/main.py::debug_foveal_probe`) → `run_foveal_probe` (`app/foveal_probe.py`): resolve newest local frame → upload to `orion-percept-store` → (previously) RPC a dedicated vision-host instance on an isolated channel → (now) RPC `orion-llm-gateway` on `CHANNEL_LLM_REQUEST` with a `kind="percept"` attachment on the `chat` route → decode `ChatResultPayload` → return the real caption/answer.

## Architecture touched

`services/orion-vision-council` (settings, foveal_probe.py, main.py, new llm_reply.py, tests, README, .env_example, docker-compose.yml), `orion/bus/channels.yaml` (retired one channel registration), `config/metrics/metric_definitions.lock.json` (re-locked against that retirement). No changes needed in `orion-llm-gateway` itself — its `vision.py`/`kind="percept"` attachment path and `LLM_GATEWAY_PERCEPT_BASE_URL`/`LLM_GATEWAY_ATTACHMENT_ALLOWED_HOSTS` config were already live and correct.

## Files changed

- `services/orion-vision-council/app/foveal_probe.py`: hop 3 rewritten from a `VisionTaskRequestPayload` RPC to a `ChatRequestPayload` RPC with a `kind="percept"` `AttachmentRefV1`; new `FovealNotConfiguredError` (percept-store + route preflight), `FovealTaskFailedError` now carries `error_code`; 0-byte frame reads now rejected instead of clamped.
- `services/orion-vision-council/app/llm_reply.py` (new): shared `extract_chat_result_text`/`GATEWAY_ERROR_PREFIX`, so this module and `main.py`'s own metacog call use identical reply-parsing logic instead of two independently-drifting copies.
- `services/orion-vision-council/app/main.py`: imports the shared `llm_reply` helper instead of a local copy; debug endpoint now shares the council's existing LLM-gateway semaphore and surfaces `FovealTaskFailedError.error_code`.
- `services/orion-vision-council/app/settings.py`: retired `CHANNEL_FOVEAL_HOST_REQUEST`/`CHANNEL_FOVEAL_HOST_REPLY_PREFIX`; added `FOVEAL_LLM_ROUTE` (default `"chat"`).
- `services/orion-vision-council/.env_example`, `docker-compose.yml`: matching key changes.
- `services/orion-vision-council/README.md`: documented the incident and the architecture change.
- `services/orion-vision-council/tests/test_foveal_probe.py`: rewritten for the new envelope shape; added coverage for the whitespace-question bug, 0-byte frame rejection, route preflight, and `error_code` values.
- `orion/bus/channels.yaml`: removed the `orion:exec:request:VisionHostService:*` registration (sole producer was `orion-vision-council`, which no longer uses it).
- `config/metrics/metric_definitions.lock.json`: re-locked (`scripts/check_definition_drift.py --update`) to reflect the retired channel.

## Schema / bus / API changes

- Removed: `orion:exec:request:VisionHostService:*` channel registration (channels.yaml) — its one producer (`orion-vision-council`) no longer uses it. Not touched: the base `orion:exec:request:VisionHostService` channel and `orion:vision:reply:*` wildcard, which the frame-router's continuous pipeline and other consumers still use.
- No new channel: the foveal probe now reuses the already-registered `orion:exec:request:LLMGatewayService` (`orion-vision-council` was already a listed producer for its own metacog calls).
- Compatibility notes: the `/debug/foveal-probe` HTTP response shape gained a `caption` field; `reply` is now a `ChatResultPayload`-shaped dict instead of a `VisionTaskResultPayload`-shaped one. This is a debug-only, manually-triggered endpoint with no automatic consumer, so no migration is needed.

## Env/config changes

- Added keys: `FOVEAL_LLM_ROUTE` (default `chat`).
- Removed keys: `CHANNEL_FOVEAL_HOST_REQUEST`, `CHANNEL_FOVEAL_HOST_REPLY_PREFIX`.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced: hand-edited directly in both the worktree (used to build) and the primary checkout (`/mnt/scripts/Orion-Sapienform/services/orion-vision-council/.env`) — `sync_local_env_from_example.py` can't see worktree-only `.env_example` changes pre-merge, same limitation as this session's earlier vision-council PR.
- Skipped keys requiring operator action: none.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-foveal-probe-via-gateway
/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-vision-council/tests -q
  89 passed, 18 warnings (pre-existing pydantic protected-namespace noise, unrelated)

/mnt/scripts/Orion-Sapienform/.venv/bin/python3 scripts/check_definition_drift.py --gate
  615 metric definitions (0 changed, 0 high severity) -- PASS

/mnt/scripts/Orion-Sapienform/.venv/bin/python3 scripts/check_service_env_compose_parity.py orion-vision-council
  N/A (env_file: already delivers every .env_example key regardless of environment: list)
```

## Evals run

No dedicated eval harness exists for this service. Covered instead by the live end-to-end smoke test below, which is the actual quality bar this patch was built to clear (the retired path passed every unit test it had too, and still returned an empty caption in production).

## Docker/build/smoke checks

```text
cd /mnt/scripts/Orion-Sapienform-foveal-probe-via-gateway
bash scripts/safe_docker_build.sh orion-vision-council build   # built clean
bash scripts/safe_docker_build.sh orion-vision-council up -d   # container recreated

curl -s http://localhost:8025/health
  {"ok":true}

curl -s -X POST "http://localhost:8025/debug/foveal-probe" --max-time 60
  ok: true
  caption: "A white desk holding a computer monitor, printer, and various items
  sits against a wall next to two rolling office chairs on a carpeted floor.
  An open closet filled with clothes is partially covered by a blue curtain,
  located next to a closed white door and a doorway leading to a hallway."
  reply.meta.served_by: "circe-worker-1"
  reply.model_used: "/models/gguf/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
  reply.raw.vision: {"attachment_count": 1, "vision": true, "status": "attached"}
```

Real, rich, multi-sentence caption of the actual room, served by the real 35B vision-capable model — not the empty BLIP rejection the retired path produced.

## Review findings fixed

- Finding: `run_foveal_probe` bypassed the council's existing `_llm_semaphore`, risking two concurrent outbound `CHANNEL_LLM_REQUEST` calls from this process if a probe fires while the always-on metacog loop has one in flight.
  - Fix: the debug endpoint now wraps the call in `async with service._llm_semaphore`.
  - Evidence: `services/orion-vision-council/app/main.py::debug_foveal_probe`.
- Finding: `config/metrics/metric_definitions.lock.json` still carried the retired channel entry, which `scripts/check_definition_drift.py --gate` (run in CI) would flag as a removed definition.
  - Fix: ran `scripts/check_definition_drift.py --update` and committed the re-locked file.
  - Evidence: gate re-run shows `615 metric definitions (0 changed, 0 high severity)`.
- Finding: a whitespace-only `?question=` (e.g. `%20`) passed the truthy check before `.strip()`, sending the model an empty instruction string instead of falling back to the default caption prompt.
  - Fix: strip before the truthiness check.
  - Evidence: `test_build_foveal_chat_envelope_whitespace_only_question_falls_back_to_caption_prompt`.
- Finding: `foveal_probe.py` extracted reply text via `ChatResultPayload.text` (top-level `content`/`text` only), while `main.py`'s `_call_llm_raw` used a richer extractor with `choices[]`/`raw.choices[]` fallbacks for the identical reply contract — the two could disagree on the same payload.
  - Fix: extracted the shared logic into `app/llm_reply.py`; both call sites now use it.
  - Evidence: `services/orion-vision-council/app/llm_reply.py`, updated imports in both `foveal_probe.py` and `main.py`.
- Finding: the debug endpoint collapsed every `FovealTaskFailedError` into the same constant `error_code="foveal_task_failed"`, losing the diagnosable distinction between "gateway reported a failure" and "genuinely empty answer."
  - Fix: `FovealTaskFailedError` now carries `error_code` (`empty_response`/`gateway_error`), surfaced by the endpoint.
  - Evidence: `test_run_foveal_probe_raises_when_response_is_empty`, `test_run_foveal_probe_raises_on_embedded_error_content`.
- Finding: `FOVEAL_LLM_ROUTE` was not preflight-checked, so a blank/typo'd route would still spend a real percept-store upload before the gateway eventually replied with an embedded routing error.
  - Fix: added to the same preflight check as `FOVEAL_PERCEPT_STORE_URL`.
  - Evidence: `test_run_foveal_probe_refuses_when_llm_route_unconfigured`.
- Finding: a genuinely empty (0-byte) frame read was clamped to a fabricated `bytes=1` instead of surfaced as an error — `sha256(b"")` is well-defined and would pass the upload hash check, letting a truncated capture through as a "successful" empty attachment.
  - Fix: reject a 0-byte frame read explicitly before upload.
  - Evidence: `test_run_foveal_probe_raises_when_frame_is_empty`.
- Considered, not changed: retiring the `orion:exec:request:VisionHostService:*` channel entirely (rather than keeping the isolated-per-host-channel mechanism dormant for a possible future dedicated-VLM deployment). The mechanism itself (`check_single_consumer_channels.py`'s glob resolution) is generic and untouched — only this one specific, now-producer-less registration is gone. Re-adding a registry entry if a dedicated per-host deployment is wanted again later is a few lines, not a rebuild.
- Not independently added: a dedicated test exercising the new `async with service._llm_semaphore` wrap at the FastAPI layer. The fix mirrors `_call_llm_raw`'s existing, already-tested-by-inspection pattern one-for-one; standing up `TestClient`-based endpoint tests (no existing pattern in this test suite) for one line felt disproportionate. Flagged here rather than silently skipped.

## Restart required

```text
No restart required beyond what this PR already performed: the vision-council
container was rebuilt and recreated live during this session
(scripts/safe_docker_build.sh orion-vision-council build && ... up -d), and
the smoke test above ran against that already-running container.
```

## Risks / concerns

- Severity: low
  Concern: the `/debug/foveal-probe` endpoint is still manually-triggered only (surprise-driven foveation is P2, blocked on `want_embeddings` per the design doc) — this PR makes the tier actually useful when triggered, it does not wire it to any automatic cadence.
  Mitigation: none needed yet; matches the design doc's own pragmatic ladder.
- Severity: low
  Concern: `config/vision_profiles.yaml`'s `vlm_caption`/`vlm_vqa` profiles still carry the unfilled `REPLACE_ME` placeholder and still serve the weak BLIP fallback for every OTHER consumer of those profiles (e.g. `orion-vision-host`'s own always-on peripheral pipeline) — this PR routes only the foveal probe around the problem, it does not fix the underlying placeholder.
  Mitigation: out of scope for this patch (the peripheral pipeline's captioning quality is a separate, larger question); flagged as a known follow-up if that path's quality is ever revisited.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1872

🤖 Generated with [Claude Code](https://claude.com/claude-code)
