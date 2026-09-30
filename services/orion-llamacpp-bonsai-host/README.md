# orion-llamacpp-bonsai-host

A separate llama.cpp worker for **Ternary-Bonsai-2-27B**, Prism's ternary
build of Qwen3.8-27B. Every weight is -1, 0 or +1. The PQ2_0 file is 7.21 GB,
against 17.6 GB for the Q4 build the agent lane runs today.

It exists to answer one question on Circe: **can one 32 GB V100 run four
curiosity runs at once, with answers that still make sense?**

## Why a separate service

Stock llama.cpp cannot run this model. It rejects PQ2_0 as an unknown type,
or it loads Q2_0 and emits garbage, because it has no Hadamard activation
runtime. The model needs PrismML's fork. Keeping the fork in its own image and
compose file means chat, agent, fast and metacog on `orion-llamacpp-host`
never pick up its binaries.

This service has no Python of its own. The image is built `FROM
orion-llamacpp-host:0.1.0` with the fork's `llama-server` copied over `/app`,
so the same profile wrapper (`services/orion-llamacpp-host/app/main.py`)
launches it from `config/llm_profiles.yaml`. This is the same pattern as the
DeepSeek soak (`Dockerfile.dsv41-porte`).

| Piece | Value |
|---|---|
| Fork | `PrismML-Eng/llama.cpp`, branch `prism` @ `88c4bc6` (includes the 2026-09-21/23 fixes in Prism's KNOWN_ISSUES) |
| Build | CUDA 12.8, `CMAKE_CUDA_ARCHITECTURES=70`, `GGML_CUDA_FA_ALL_QUANTS=ON` |
| Image | `llamacpp-bonsai-prism:server-local-volta` |
| Profile | `ternary-bonsai2-27b-pq2-v100-32gb-circe-np4` |
| Card / port | gpu2 (`BONSAI_CUDA_VISIBLE_DEVICES=2`), host port `8017` |

## Build and run (on Circe, from a worktree)

```bash
# 1. The base wrapper image must exist locally.
docker image inspect orion-llamacpp-host:0.1.0 >/dev/null

# 2. Compile the fork for Volta (a long build; the llama.cpp CUDA compile dominates).
services/orion-llamacpp-bonsai-host/scripts/build-bonsai-volta.sh

# 3. Only while gpu2 is free (see below).
scripts/safe_docker_build.sh orion-llamacpp-bonsai-host up -d
curl -fsS http://localhost:8017/health
```

The first boot downloads the 7.21 GB GGUF into `${LLM_CACHE_DIR}/gguf`.

## gpu2 is borrowed: out-of-memory risk for the whole run

`orion-gpu-lane-controller` owns gpu2 (`config/gpu_pool.yaml`: `agent-gpu2`,
which diffusion reclaims). This worker is **not** in the pool, and the pool
has no verb that reserves a whole card for an outside worker. So the
controller does not know Bonsai is there. If an agent backlog passes its swap
wait, or a diffusion request lands, it will launch its own worker (17.6 GB+)
or diffusion (about 24 GB) onto a card already holding Bonsai's roughly 24 GB,
and one side runs out of memory. Checking `nvidia-smi -i 2` before `up` only
covers the start.

For a bake-off, pick one:

- Run it in a window with no image generation and no agent backlog, and
  watch `nvidia-smi -i 2`.
- Stop the controller for the duration. This also stops every other pool
  swap, including gpu1's affect/agent flips.

The worker announces `LLM_ROLE=bonsai-bakeoff`, which is not a pool role, so
the pool's discovery view lists it under `unclaimed`. It does not use
`experiment`: that role belongs to the DeepSeek soak, and announcements are
keyed by role, so the two would overwrite each other. No traffic is routed
here. The pool and the gateway build URLs from configured role ports, never
from announcements.

`restart: "no"` keeps the worker from coming back on its own after a reboot.
Stop it when done:

```bash
scripts/safe_docker_build.sh orion-llamacpp-bonsai-host down
```

## New-service `.env`

`scripts/sync_local_env_from_example.py` cannot bootstrap a new service.
On the deploying host, create the `.env` once from the template:

```bash
cp services/orion-llamacpp-bonsai-host/.env_example services/orion-llamacpp-bonsai-host/.env
```

## Profile choices

- `ctx_size: 262144`, `n_parallel: 4`. llama-server divides the context
  across slots, so each run gets 65,536 tokens.
- VRAM, **UNVERIFIED on Volta**: 7.21 GB of weights, plus about 16 GiB of f16
  KV cache at Prism's stated ~64 KB per token, comes to about 24 GiB before
  compute buffers.
- `reasoning: auto`. Prism: `--reasoning on` overrides a client's
  `reasoning_effort: "none"`.
- `reasoning_effort: medium`. The template accepts only `low`, `medium` and
  `xhigh`. `high` returns HTTP 500, and `low` behaves like `xhigh`.
- `preserve_thinking: false`. Prism: re-rendering earlier reasoning into later
  turns makes the prompt cache miss in tool loops (Bonsai-demo #183). That is
  exactly the cost four slots would multiply. **UNVERIFIED** that the Bonsai
  template honours this kwarg the way Qwen3.8's does.
- `n_predict: 16384`. Smaller output caps end generation mid-thought.
- `flash_attn: off`, matching the live Qwen3.8-27B agent lane on the same
  hardware. The wrapper only emits `--flash-attn off` when the binary reports
  a build number above b5332. That number is `git rev-list --count`, so the
  Dockerfile uses a blobless clone, not `--depth 1`, and asserts the number.

## Bake-off measurements

Replay four real curiosity runs at once and record, per run:

1. Whether the answers make sense, compared with the same runs on the Q4 agent lane.
2. Generation tokens per second (`timings.predicted_per_second`).
3. Prompt tokens re-processed per step (`timings.prompt_n` against `tokens_cached`). This shows whether prompt caching survives four slots.
4. Peak VRAM (`nvidia-smi -i 2`).

## Tests

```bash
pytest services/orion-llamacpp-bonsai-host/tests -q
```

These check that the fork pin, the Volta arch and the CUDA 12.8 base stay in
place, that the shared `orion-llamacpp-host` files never mention the fork, and
the exact `llama-server` argv the wrapper builds from the profile.
