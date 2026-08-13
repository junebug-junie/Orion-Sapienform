# Muse Glimmer 30B model card — Circe 1x V100 32GB agent lane

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1627
Branch: `feat/muse-glimmer-30b-profile`

## Summary

- Adds `muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision` to `config/llm_profiles.yaml`: `unsloth/Muse-Glimmer-30B-GGUF` UD-Q4_K_XL text weights + Q8_0 mmproj, tools + vision enabled.
- Sampling matches the model card exactly: `temperature=1.0`, `top_p=0.95`, `top_k=64`.
- Sized for 1x V100 32GB, wired for `ATLAS_AGENT_PROFILE_NAME` (Circe's Agent compute GPU lane, per `services/orion-llamacpp-host/README.md`'s worker table).
- Speculative decoding (DFlash) documented but **not wired**.
- Model's "thinking" control documented but **not wired** via `chat_template_kwargs`.
- Adds a regression test (`test_muse_glimmer_30b_agent_vision_profile_forwards_flags_and_leaves_draft_unset`) mirroring the existing gemma4 multimodal forwarding test.

## Outcome moved

New GPU lane (Circe, agent/tool-use + vision) has a validated, schema-checked model card ready to deploy today on the currently pinned llama-server build — no code or infra changes required to go live.

## Current architecture

`orion-llamacpp-host` is a profile-driven llama.cpp wrapper: `config/llm_profiles.yaml` is the source of truth for model path/download spec/runtime knobs, `LLM_PROFILE_NAME` selects the active profile per container, and `docker-compose.atlas-workers.yml` defines fixed worker slots (`chat`/`metacog`/`fast`/`agent`) each bound to an env-supplied profile name. The `agent` slot (`ATLAS_AGENT_PROFILE_NAME`, port 8014) is the "Agent compute GPU lane" — already the physical target for tool-use/agentic workers (see `qwen3-coder-next-*-agent-*` profiles).

## Architecture touched

Config only: `config/llm_profiles.yaml` (new profile block) + a matching test in `services/orion-llamacpp-host/tests/test_profile_forwarding.py`. No service code, schema, bus, or env-key changes.

## Files changed

- `config/llm_profiles.yaml`: why — new `muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision` profile.
- `services/orion-llamacpp-host/tests/test_profile_forwarding.py`: why — new forwarding-flag regression test for the profile.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: n/a — pure additive config entry.

## Env/config changes

- Added keys: none (profile *values*, not new env keys — assign via existing `ATLAS_AGENT_PROFILE_NAME` on the Circe worker's `.env`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable (no new key).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable, no `.env_example` change.
- skipped keys requiring operator action: none.

## Research findings baked into the profile

- **Build tag question (explicitly asked)**: current pin `LLAMACPP_IMAGE_TAG=server-cuda-b8740` (~April 2026) is fine for everything in this profile *except* DFlash speculative decoding. Chat/tools/vision (mmproj/mtmd) all already work on this pin — the existing `gemma4-*-multimodal` profiles prove that pattern live.
- **Speculative decoding (DFlash) — intentionally not wired.** Muse Glimmer ships a DFlash drafter (`dflash-kquant.gguf`), architecturally different from a plain small-model draft: it needs `--spec-type draft-dflash`/`--dflash` + `--spec-draft-n-max`, neither of which `services/orion-llamacpp-host/app/main.py`'s CLI builder emits today (it only knows the generic `--model-draft`/`--draft-min`/`--draft-max`/`--n-gpu-layers-draft` path used by `qwen36-35b-a3b-udq5km-2xv100-32gb-deep-cognition`'s commented block). DFlash merged upstream 2026-06-28 ([ggml-org/llama.cpp#22105](https://github.com/ggml-org/llama.cpp/pull/22105)), well after this stack's b8740 pin. `LLAMACPP_IMAGE_TAG` is also one shared env var across every atlas-workers service (chat/metacog/fast/agent), so bumping it is a fleet-wide change, not scoped to this profile. Left a fully commented block in the profile documenting the exact 3-step follow-up: (1) add `spec_type`/`spec_draft_n_max` fields to `LlamaCppConfig` + CLI emission in `app/main.py`, (2) bump `LLAMACPP_IMAGE_TAG` past the DFlash merge and smoke-test the other lanes on that tag, (3) then point `draft_filename`/`draft_repo_id` at `dflash-kquant.gguf` in the same HF repo.
- **Thinking control is different for this model.** No `chat_template_kwargs.enable_thinking`/`--reasoning-budget` equivalent — the card controls reasoning depth via literal `"Reasoning strength: <low|medium|high|xhigh>"` text in the system prompt. Left `llamacpp.reasoning*`/`chat_template_kwargs` unset rather than wiring a flag that would silently no-op for this model; documented the system-prompt convention in the profile's `notes`.
- **VRAM budget**: UD-Q4_K_XL weights (15.9GB) + Q8_0 mmproj (~1.9GB) ≈ 17.8GB, leaving ~14GB of the 90%-of-32GB budget for 32k KV + batch/ubatch overhead — same order as `qwen36-35b-a3b-udq4km-v100-32gb-balanced`, already proven to fit this card class.

## Tests run

```text
$ /mnt/scripts/Orion-Sapienform/venv/bin/python3 -m pytest services/orion-llamacpp-host/tests -q
22 passed, 1 failed (pre-existing, unrelated), 19 warnings

FAILED tests/test_profile_forwarding.py::test_qwen3_8b_atlas_metacog_profile_q5km_single_lane_16k
  -> pydantic ValidationError: llm_profile_name Field required (Settings env-parse
     ordering issue). Reproduces identically against unmodified origin/main -- confirmed
     pre-existing, not caused by this change.
```

Also manually validated the new profile parses cleanly against the live `LLMProfile`/`LlamaCppConfig` pydantic schema in `app/profiles.py` (separately from the full-registry parse, which fails on an unrelated pre-existing `llama3-1-cola` backend-literal mismatch).

## Evals run

No eval harness exists for `orion-llamacpp-host`; this is a config-card addition, not a behavior change to an existing model. No live-inference eval is possible pre-deploy (model not yet downloaded to Circe).

## Docker/build/smoke checks

Not run — no Dockerfile/compose/runtime change in this PR. Deploying the profile requires no rebuild: point `ATLAS_AGENT_PROFILE_NAME=muse-glimmer-30b-udq4kxl-v100-32gb-agent-vision` at the Circe worker's `.env` and restart that one container.

## Review findings fixed

- `/code-review medium` (background subagent) against `feat/muse-glimmer-30b-profile`: **0 findings**. Independently re-verified every `LlamaCppConfig`/`GPUConfig` field against the pydantic schema, the CLI-flag-emission path the new test exercises, absence of profile-name/port/model_id collisions, and cross-checked the profile also validates against the separate `services/orion-llm-gateway/app/profiles.py` schema copy (pre-existing drift risk between the two copies, not introduced here — flagged as FYI, not a finding).

## Restart required

```bash
# On the Circe worker host, after setting ATLAS_AGENT_PROFILE_NAME in its .env:
scripts/safe_docker_build.sh orion-llamacpp-host up -d --build atlas-agent
curl -fsS http://localhost:8014/health
```

## Risks / concerns

- Severity: low
  - Concern: first boot on Circe will pull ~17.8GB of GGUF weights (Q4_K_XL text + Q8_0 mmproj) from HuggingFace; no smoke test has run against real Circe VRAM yet.
  - Mitigation: `notes` field documents an explicit OOM fallback ladder (ctx_size 32768 → 24576 → 16384, then image_max_tokens 1024 → 560, then drop to UD-Q3_K_XL) before declaring the lane broken.
- Severity: low
  - Concern: DFlash speculative decoding is documented but not usable yet — needs its own scoped follow-up (schema/CLI fields + build-tag bump + bench), separate from this PR.
  - Mitigation: exact follow-up sequence is written directly in the profile's commented block so it isn't lost.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1627
