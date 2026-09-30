## Summary

- New service `orion-llamacpp-bonsai-host`: a separate llama.cpp worker for **Ternary-Bonsai-2-27B**. It needs PrismML's llama.cpp fork, because stock llama.cpp rejects PQ2_0.
- The image compiles the fork (`prism` @ `88c4bc6`) for Volta (sm_70) on CUDA 12.8, then copies the binaries onto the existing `orion-llamacpp-host` wrapper. The same profile launcher starts it, and the chat/agent/fast/metacog images never see the fork.
- New profile `ternary-bonsai2-27b-pq2-v100-32gb-circe-np4`: 4 concurrent runs, 65,536 tokens each, on gpu2, port 8017, `reasoning_effort: medium`.
- **Shared wrapper fix:** the launcher misread the fork's version line as build 0. It would then have treated the fork as an ancient binary and silently dropped `--flash-attn off` and `--reasoning`.

## Outcome moved

Circe can now run the four-runs-on-one-V100 bake-off. Before this, nothing could load the model. The fork is **proven to compile for Volta**: every CUDA kernel built for sm_70, including the ternary ones. It is **not yet proven to run correctly on a V100**, since athena has no Volta card.

## Current architecture

One `orion-llamacpp-host` image, built from upstream `ghcr.io/ggml-org/llama.cpp`, serves every lane from `config/llm_profiles.yaml`. The DeepSeek soak already layers a forked `llama-server` onto that wrapper (`Dockerfile.dsv41-porte`). This PR follows the same pattern as its own service.

## Architecture touched

- New service directory, not managed by the GPU pool (`restart: "no"`, announces the non-pool role `bonsai-bakeoff`).
- `services/orion-llamacpp-host/app/main.py`: build-number parsing only.
- CI: the bonsai contract tests run inside `orion-gpu-pool-tests.yml`.

## Files changed

- `services/orion-llamacpp-bonsai-host/Dockerfile`: fork build (blobless clone, stub `libcuda.so.1`, asserts build > 5332 and the required flags), layered onto the wrapper image.
- `services/orion-llamacpp-bonsai-host/docker-compose.yml`, `.env_example`: worker on gpu2 / 8017.
- `services/orion-llamacpp-bonsai-host/scripts/build-bonsai-volta.sh`: build plus a post-build check of the final image.
- `services/orion-llamacpp-bonsai-host/README.md`: why the service exists, how to run it, the gpu2 out-of-memory risk, the profile rationale, and what to measure.
- `services/orion-llamacpp-bonsai-host/tests/`: pins, isolation from the shared host, pool-role check, launch argv.
- `config/llm_profiles.yaml`: the new profile.
- `services/orion-llamacpp-host/app/main.py`, `tests/test_profile_forwarding.py`: `_parse_llama_build` plus a regression test.
- `.github/workflows/orion-gpu-pool-tests.yml`: runs the bonsai tests.

## Schema / bus / API changes

- Added: none. The worker announces through the existing `LlmWorkerAnnounceV1` with role `bonsai-bakeoff`. That role is not in `gpu_pool.yaml`, so the pool shows it as `unclaimed`. No traffic is routed to it, because routing uses configured role ports.
- Behavior changed: the wrapper's build detection now prefers `(build N)` over `version: N`. Upstream `version: 8740 (hash)` still parses to 8740.

## Env/config changes

- Added keys (new service only): `BONSAI_LLAMACPP_IMAGE`, `BONSAI_PROFILE_NAME`, `BONSAI_CUDA_VISIBLE_DEVICES`, `BONSAI_HOST_PORT`, plus the standard `SERVICE_*`, `ORION_BUS_*`, `LLM_*` and `HF_*` keys.
- `.env_example` updated: yes (new file).
- Local `.env`: created from the template in the worktree. `sync_local_env_from_example.py` cannot bootstrap a new service. The deploying host needs a one-time `cp .env_example .env` (see README).
- Skipped keys: none.

## Tests run

```text
pytest services/orion-llamacpp-bonsai-host/tests -q           -> 4 passed
pytest services/orion-llamacpp-host/tests -q (LLM_PROFILE_NAME=ci)
                                                              -> 40 passed, 1 failed
  the failure (test_qwen3_8b_atlas_metacog_profile_q5km_single_lane_16k) also fails on clean main
check_env_template_parity.py orion-llamacpp-bonsai-host       -> PASS
check_gpu_pool_config.py                                      -> ok
check_compose_no_relative_mounts / _no_host_claude_json_mount / check_service_hostname_refs -> pass
docker compose ... config -q                                  -> ok
```

## Evals run

```text
None. This is infrastructure for a bake-off; the bake-off itself is the eval (README "Bake-off measurements").
```

## Docker/build/smoke checks

```text
athena, image llamacpp-bonsai-prism:buildcheck:
  fork compiled for sm_70 on CUDA 12.8                      -> ok
  llama-server --version -> "0.2.0-dev (build 10750, commit 88c4bc60)"
  --help lists --ctx-checkpoints, --jinja, --reasoning, --reasoning-format,
    --chat-template-kwargs, --flash-attn, --no-context-shift -> ok
  final image (docker run --gpus all) runs the fork binary  -> ok
Not run: a model load or generation on a V100. UNVERIFIED: whether Prism's kernels run correctly on Volta.
```

## Review findings fixed

- Finding: a shallow clone makes the build number `rev-list --count` = 1, so the wrapper drops `--flash-attn off`.
  - Fix: blobless clone, plus a build assertion. The first rebuild then showed a second cause of the same bug: the fork's semver'd version line parsed as 0. Fixed at the root in `_parse_llama_build`.
  - Evidence: the build log shows `build 10750`; a regression test covers both formats.
- Finding: `LLM_ROLE=experiment` collides with the DeepSeek soak's pool role, and announcements are keyed by role.
  - Fix: `bonsai-bakeoff`, plus a test that the role is not in `gpu_pool.yaml`.
- Finding: CUDA stubs lack `libcuda.so.1`.
  - Fix: symlink in the build stage. Evidence: the first athena build failed at link time with exactly this error.
- Finding: the tests were not run by CI.
  - Fix: added to `orion-gpu-pool-tests.yml`, and path triggers added.
- Finding: the argv test was fully mocked.
  - Fix: the real flag and build probe now runs in the Docker build itself.
- Finding: dead `ENSURE_MODEL_DOWNLOAD` / `WAIT_FOR_MODEL_SECONDS` in compose.
  - Fix: removed.
- Not fixed: the pool can still load onto gpu2 while Bonsai holds it (see Risks).

## Restart required

```bash
# On circe, from a worktree at this branch or at main after merge:
cp services/orion-llamacpp-bonsai-host/.env_example services/orion-llamacpp-bonsai-host/.env
services/orion-llamacpp-bonsai-host/scripts/build-bonsai-volta.sh
nvidia-smi -i 2   # must be empty
scripts/safe_docker_build.sh orion-llamacpp-bonsai-host up -d
curl -fsS http://localhost:8017/health
```

The shared-wrapper parse change only takes effect on the next `orion-llamacpp-host` rebuild. Upstream builds parse the same as before, so there is no need to restart existing lanes.

## Risks / concerns

- Severity: high (bake-off only). The pool does not know Bonsai is on gpu2. An agent backlog past the swap wait, or a diffusion request, will load onto the same card and run it out of memory. Mitigation: run in a quiet window, or stop the lane controller for the duration (README). A real card reservation for an outside worker would be a pool change and is not in this PR.
- Severity: medium. Volta runtime is UNVERIFIED. Prism publishes no numbers for cards older than Ampere.
- Severity: medium. VRAM is estimated (~24 GiB before compute buffers) from Prism's ~64 KB/token figure, not measured.
- Severity: low. UNVERIFIED that the Bonsai chat template honours `preserve_thinking: false`.
- Pre-existing, not touched: `Dockerfile.dsv41-porte` has the same `--depth 1` and `ARG`-before-`FROM` bugs.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2420

🤖 Generated with [Claude Code](https://claude.com/claude-code)
