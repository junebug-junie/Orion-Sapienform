# Hecate GLM 5.3 Flash: Athena preparation and later activation

## What is ready

Athena can now discover and route to a vLLM role through the existing GPU pool.
The Hecate profile, immutable source/model revisions, candidate config renderer,
worker announcement and inference smoke are staged. No active route is changed.
No model weights have been downloaded, no Hecate service started, and no Circe
service changed. GPU build, memory fit and model quality are **UNVERIFIED**.

## Current architecture

- Athena: `orion-gpu-pool` grants GPU leases; `orion-llm-gateway` serves HTTP
  and bus clients. Their settings live in `app/settings.py`; there is no
  service-root `settings.py` for either. Both have tests and evals.
- Circe: existing llama.cpp roles are launched by the lane controller. Keep
  their roles, models and ports as they are during this preparation.
- Hecate: current inventory records one 32 GB V100 for `agent-deep`, port 8021,
  Tailscale `100.87.202.68`. The requested checkpoint requires a four-card plan;
  actual installed hardware must be checked when `ssh hecate@hecate` is available.
- vLLM host: existing wrapper, `app/settings.py`, compose, requirements and
  tests. A new `evals/smoke_glm.py` covers opt-in real inference acceptance.
  The legacy stock-vLLM Dockerfile does **not** build the required fork.
- Contracts: existing `orion:llm:worker:announce` and GPU pool lease/state
  channels; no new bus kinds or registry entries.

The missing seam was llama.cpp-only discovery and dispatch. This patch makes
that seam explicit without changing Orion's memory, identity or cognition loops.

## Pinned upstream build

Sources: [checkpoint card](https://huggingface.co/philbert440/GLM-5.3-Flash-EXL3),
[pinned API server](https://github.com/1CatAI/1Cat-vLLM/blob/ec60883ad7187dc8aa29b3c30cae5a69e1cfa5cd/vllm/entrypoints/openai/api_server.py),
[development-only router registration](https://github.com/1CatAI/1Cat-vLLM/blob/ec60883ad7187dc8aa29b3c30cae5a69e1cfa5cd/vllm/entrypoints/serve/instrumentator/__init__.py).

The source and weights are locked in
`services/orion-vllm-host/scripts/glm53.lock.json`. This EXL3 packing needs the
SM70 path in 1Cat-vLLM. Stock ExLlama and the legacy Orion image are unsuitable.
The upstream recipe uses four 32 GB V100s, CUDA 12.8 and GCC/G++ 14.
Its throughput/context claims are upstream reports, not Orion measurements.

Prepare source on a writable disk (no weights, CUDA build or deployment):

```bash
python3 services/orion-vllm-host/scripts/prepare_source.py /mnt/scripts/hecate-glm53-source
```

The helper fetches exact commits and merges them in the locked order. It stops
on conflicts. Inspect each conflict in `CMakeLists.txt` and `csrc/ops.h`; retain
both independent additions only after checking them. Commit the resolution and
resume with `--resume`. Never blindly remove conflict markers. Preserve the
final source commit and build output as deployment artifacts. The helper checks
its final source-tree hash against the reviewed lock. Athena's assembled copy is
`/mnt/scripts/hecate-glm53-source-20261011`, commit
`58531acc450ca0226f71fd1756e3357c7897c479`, tree
`c5838487a6c58c079cc9e86f3fa391ba5ff1cca0`; compilation is still UNVERIFIED.

Once Hecate is online, first inspect GPUs, free memory, topology, free disk,
driver/toolkit and compilers:

```bash
ssh hecate@hecate 'hostname; nvidia-smi; nvidia-smi topo -m; df -h; nvcc --version; gcc-14 --version'
```

Do not start the four-GPU build/serve plan on a one-GPU inventory. If cards need
to move from Circe, decide that separately; it changes existing service capacity.
Transfer/assemble the pinned source on Hecate. In that source checkout:

```bash
uv venv --python 3.12 --seed
uv pip install -r requirements/build/cuda.txt --torch-backend=cu128
TORCH_CUDA_ARCH_LIST=7.0 CMAKE_CUDA_ARCHITECTURES=70 MAX_JOBS=10 \
  CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14 \
  uv pip install -e . --torch-backend=cu128 --no-build-isolation
uv run pytest tests/kernels/quantization/test_exl3_moe.py tests/models/glm5next/test_sm70_sparse.py
hf download philbert440/GLM-5.3-Flash-EXL3 \
  --revision 4ba4b9afb2191578378454cdd28aa35e9484960c --local-dir /models/GLM-5.3-Flash-EXL3
```

Check that `/models` is writable and has sufficient free space first; choose a
mount at that path rather than changing the profile/engine identity independently.
Install Orion host requirements into this same environment. Do not use the
stock image. In the Hecate Orion worktree, set the following **local-only**
service `.env` values, with the absolute profile path for that worktree:

```dotenv
NODE_NAME=hecate
SERVICE_NAME=orion-vllm-host
ORION_BUS_URL=redis://100.92.216.81:6379/0
VLLM_PROFILE_NAME=glm-5.3-flash-exl3-hecate
VLLM_MODEL_ID=
LLM_PROFILES_CONFIG_PATH=/absolute/orion-worktree/config/llm_profiles.yaml
VLLM_HOST=100.87.202.68
VLLM_PORT=8021
LLM_ROLE=agent-deep
LLM_ANNOUNCE_HOST=hecate
LLM_ANNOUNCE_PORT=8021
```

Run the wrapper with the fork's Python and Orion on `PYTHONPATH` from
`services/orion-vllm-host` (its settings read `.env`). Replace these two absolute
paths with the verified checkouts on Hecate:

```bash
PYTHONPATH=/absolute/orion-worktree:/absolute/orion-worktree/services/orion-vllm-host \
  /absolute/1Cat-vLLM/.venv/bin/python -m app.main
```

The launch profile supplies the constrained sequence count, precision, cache,
parsers, speculation and engine environment. It also registers only the
read-only `/orion/server-info` middleware. Do **not** enable
`VLLM_SERVER_DEV_MODE`; the pool does not need the fork's other developer APIs.
A foreground run is the first acceptance step; a permanent service/container
and its exact rollback should be reviewed after the hardware smoke succeeds.

## Candidate routing and safe rollout

Render a full candidate to a scratch file; the renderer does not edit defaults:

```bash
python services/orion-vllm-host/scripts/stage_pool.py > /tmp/hecate-pool.candidate.yaml
```

It replaces the existing Hecate `agent-deep` allocation with all four cards and
`backend: vllm`, retaining port 8021. All four lend flags must be enabled before
other classes can borrow it. Do not run the previous Hecate llama.cpp worker
on those cards/port at the same time. Do not activate the candidate while an
old `agent-deep` lease is active; drain and verify no outstanding holds first.

The added backend field is omitted for unchanged llama.cpp roles. Once a vLLM
role is activated, older strict `PoolConfig` readers cannot decode it. Before
activation, update the gateway and every reader of shared pool configuration
(including Hub, durable-runs and Circe's lane controller) from the same branch.
The lane controller's Circe role definitions and launch digests remain unchanged.
Deploying those reader updates and switching the candidate require a concrete
operator-approved maintenance step; neither is done by this preparation patch.

After approval, from the selected Orion worktree with synchronized local env:

```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

These are future restart commands, not commands run during Athena preparation.
The active `config/gpu_pool.yaml` must contain the reviewed candidate before
building the activated images. Reader upgrades on Circe must precede activation.

## Acceptance and rollback

1. From Athena: `/health`, `/v1/models`, `/orion/server-info` on Hecate answer.
   Model path/alias, four CUDA indices and live capacity agree. Inspect startup
   logs for the EXL3 and SM70 kernels. A heartbeat alone proves none of this.
2. Run direct synthetic inference:
   `python services/orion-vllm-host/evals/smoke_glm.py --url http://100.87.202.68:8021`.
   Save the nonempty outputs, model ID and usage. The set checks arithmetic,
   a planted key and JSON; it is not a comprehensive quality benchmark.
3. After approved activation, verify a fresh announcement, confirmed pool role,
   lease grant/release and a bus response with Hecate `served_by` and the exact
   model. Run the same smoke through Athena using
   `--url http://127.0.0.1:8210 --route agent-deep`.
4. Check streamed text and tool calls separately. Anthropic Messages and images
   remain disabled/unvalidated for this integration; the fork having an endpoint
   or vision tower is not proof that Orion's paths work.
5. Test Hecate unavailable: no fabricated answer, no false healthy capacity,
   no leaked lease, existing Circe routes still serve.

For rollback, drain Hecate work, restore the previous pool config and rebuild
pool/gateway from the chosen worktree. Stop the new worker before restoring
the old one. No schema/database migration or persistent model-state rewrite is
introduced. Leave the source and graph artifacts intact.
