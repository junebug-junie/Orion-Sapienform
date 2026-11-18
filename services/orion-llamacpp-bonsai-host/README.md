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
| Card / port | gpu1 or gpu2 only, never gpu0 (`BONSAI_CUDA_VISIBLE_DEVICES`, required, template `2`), host port `8017` |

## Build and run (on Circe, from a worktree)

```bash
# 1. The base wrapper image must exist locally.
docker image inspect orion-llamacpp-host:0.1.0 >/dev/null

# 2. Compile the fork for Volta (a long build; the llama.cpp CUDA compile dominates).
services/orion-llamacpp-bonsai-host/scripts/build-bonsai-volta.sh

# 3. Only while gpu2's pool seat is idle (see below).
scripts/safe_docker_build.sh orion-llamacpp-bonsai-host up -d
curl -fsS http://localhost:8017/health
```

The first boot downloads the 7.21 GB GGUF into `${LLM_CACHE_DIR}/gguf`.

## Cards: gpu1 or gpu2 only, never gpu0

gpu0 is chat's card. Bonsai does not take it and does not evict chat
(Juniper, 2026-09-30). The target cards are the agent cards, gpu1 and gpu2
(`docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md`).
`BONSAI_CUDA_VISIBLE_DEVICES` has no compose default, so a missing value fails
every compose command for this service (including `down`) instead of picking a
card, and
`tests/test_bonsai_contract.py::test_no_bonsai_config_targets_chats_card` fails
if any Bonsai config points at chat's card.

That test only covers the repo. A host `.env` written before this change can
still say `0`, and the sync script does not overwrite existing values. Check
it before `up`:

```bash
grep BONSAI_CUDA_VISIBLE_DEVICES services/orion-llamacpp-bonsai-host/.env
```

gpu2 is not free either. The pool lends it to `agent-gpu2` and diffusion, and
this worker is not a pool role, so the lane controller does not know Bonsai is
there. If it launches its Q4 worker (17.6 GB) or diffusion onto a card already
holding Bonsai's ~24 GB, one side runs out of memory. For a bake-off on gpu2:
wait until the `agent-gpu2` seat has unloaded and diffusion is not resident
(`nvidia-smi -i 2` shows no agent or diffusion process; the small world-model
lane may stay), then pause pool actuation so nothing is launched onto it, and
resume when Bonsai is down. Pausing does not unload a worker that is already
running.

```bash
ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python scripts/gpu_pool_pause.py pause
# ... bake-off ...
ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python scripts/gpu_pool_pause.py resume
```

### This service is a bake-off tool

It is retired in stage 7.6 of the stage 7 spec, once the Bonsai image lives in
`orion-llamacpp-host` and the pool launches it on the agent cards itself.
Until then it exists only for manual bake-offs.

The worker announces `LLM_ROLE=bonsai-bakeoff`, which is not a pool role, so
the pool's discovery view lists it under `unclaimed`. It does not use
`experiment`: that role belongs to the DeepSeek soak, and announcements are
keyed by role, so the two would overwrite each other. No traffic is routed
here. The pool and the gateway build URLs from configured role ports, never
from announcements. `restart: "no"` keeps it from coming back after a reboot
and holding a card the pool thinks is free.

Manual only (Juniper, 2026-09-30). It is in no host's auto-rebuild list:
absent from `mesh-utilities/common/include_services_circe.txt`, and listed in
`exclude_services.txt`. A merge never starts it on its own. Run
`up -d --build` by hand; the `build:` section rebuilds with the current
wrapper and profiles, reusing the cached fork compile.

## The image carries its own wrapper and profiles

The image is `FROM orion-llamacpp-host:0.1.0` for the Python deps only. It
copies `services/orion-llamacpp-host/app`, `config/` and `orion/` fresh, because
the base image's baked copies are only as new as its last rebuild. On circe
(2026-09-30) they lacked this profile and the fork build-number fix.

## New-service `.env`

`scripts/sync_local_env_from_example.py` cannot bootstrap a new service.
On the deploying host, create the `.env` once from the template:

```bash
cp services/orion-llamacpp-bonsai-host/.env_example services/orion-llamacpp-bonsai-host/.env
```

## Profile choices

- `ctx_size: 262144`, `n_parallel: 4`. llama-server divides the context
  across slots, so each run gets 65,536 tokens.
- VRAM, measured on circe gpu0 during the 2026-09-30 bake-off (before the
  gpu1/gpu2-only rule): 24.1-24.3 GB with flash attention on (two hand
  readings), 27.2-28.6 GB with it off (sampled every second). That covers all four 65K slots, whether idle or
  full.
- `reasoning: auto`. To turn thinking off per request, send
  `chat_template_kwargs: {"enable_thinking": false}`. This template rejects
  `reasoning_effort: "none"` with HTTP 500, despite Prism's docs, and
  `reasoning_budget: 0` does not stop thinking.
- `reasoning_effort: medium`. The template accepts only `low`, `medium` and
  `xhigh`. `high` returns HTTP 500, and `low` behaves like `xhigh`.
- `preserve_thinking: false`. Prism: re-rendering earlier reasoning into later
  turns makes the prompt cache miss in tool loops (Bonsai-demo #183). That is
  exactly the cost four slots would multiply. **UNVERIFIED** that the Bonsai
  template honours this kwarg the way Qwen3.8's does.
- `n_predict: 16384`. Smaller output caps end generation mid-thought.
- `flash_attn: on`. Measured on circe: at 61K tokens of context, decode is
  32.3 tok/s with it on against 17.6 off, and it uses about 4 GB less memory.
  The one cost is that exactly two concurrent runs get 23 tok/s each, against
  41 with it off. See `docs/2026-09-30-ternary-bonsai2-27b-1xv100-circe.md`.
  The wrapper only emits `--flash-attn` for binaries it reads as newer than
  b5332. That number is `git rev-list --count`, so the Dockerfile uses a
  blobless clone, not `--depth 1`, and asserts the number.

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
