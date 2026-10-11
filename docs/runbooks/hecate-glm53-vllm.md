# Hecate GLM 5.3 Flash: Athena preparation and later activation

## What is ready

Athena can now discover and route to a vLLM role through the existing GPU pool.
The Hecate profile, immutable source/model revisions, candidate config renderer,
worker announcement and inference smoke are staged. No active route is changed.
The follow-up on 2026-10-11 provisions Hecate's NVMe and builds a pinned SM70
image. The four-GPU GLM plan does not fit the live one-GPU inventory; the operator
authorized a single-GPU Qwen validation model. No Circe service changed.
GPU build, memory fit and model quality remain **UNVERIFIED** until smoke results
are recorded below.

## Current architecture

- Athena: `orion-gpu-pool` grants GPU leases; `orion-llm-gateway` serves HTTP
  and bus clients. Their settings live in `app/settings.py`; there is no
  service-root `settings.py` for either. Both have tests and evals.
- Circe: existing llama.cpp roles are launched by the lane controller. Keep
  their roles, models and ports as they are during this preparation.
- Hecate: current inventory records one 32 GB V100 for `agent-deep`, port 8021,
  Tailscale `100.87.202.68`. The requested checkpoint requires a four-card plan;
  live SSH and PCI inventory on 2026-10-11 confirmed exactly one V100-SXM2-32GB.
  The current four-GPU checkpoint plan is **BLOCKED** on hardware/model choice.
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

The helper cherry-picks six exact patches onto the tested base, then merges
only the EXL3 loader branch. Merging all upstream branches imports
`d4ce513991b79a9cac2a4d72fbe8bcac5f566d5e`, a known MTP startup regression.
The lock now rejects that ancestor and verifies the tree after every step.

It stops on conflicts. Inspect each conflict in `CMakeLists.txt` and `csrc/ops.h`;
retain both independent additions after checking them. Commit the resolution and
resume with `--resume`. Resume accepts only clean, reviewed intermediate trees.
Never blindly remove conflict markers. Preserve the final source commit and
build output as deployment artifacts.

**Retired source, do not build:** `/mnt/scripts/hecate-glm53-source-20261011`,
commit `58531acc450ca0226f71fd1756e3357c7897c479`, tree
`c5838487a6c58c079cc9e86f3fa391ba5ff1cca0`. It contains the regression.
The corrected Athena copy is `/mnt/scripts/hecate-glm53-source-corrected-20261011`,
commit `def8a636157d5e3766c6c03df2b816a521549ada`, tree
`b321701ed88fc9882ae7c71cdb5968a6635505ab`. Compilation remains **UNVERIFIED**.

## Hecate live inventory (2026-10-11)

- SSH and PCI enumeration: one Tesla V100-SXM2-32GB, driver 580.178.04.
- Existing `orion-hecate-hecate-llamacpp-agent-deep` uses about 24,678 MiB
  and publishes port 8021; it was not stopped or changed.
- 183 GiB host RAM; host RAM does not satisfy the documented TP4 GPU allocation.
- System Python 3.14.4 and GCC 15.2; `nvcc`, `gcc-14` and `uv` absent from PATH.
- `/mnt/scripts` is a separate 1.6 TiB filesystem. Recheck free capacity before
  downloading; the root filesystem has only about 58 GiB free.
- Operator explicitly authorized erasing Toshiba NVMe serial `49JA22UKK03N`.
  It is now ext4 at `/mnt/models`, UUID `19d66b9a-2234-4a60-9180-1862285cd74c`,
  with a persistent `/etc/fstab` entry. Original partition table and fstab are
  recorded in `/mnt/scripts/hecate-nvme-layout-before-20261011` (metadata only,
  not a backup of erased Windows data).
- Both pinned snapshots downloaded successfully: Qwen has 3 weight shards
  (8,044,982,000 bytes), GLM has 13 (104,011,208,249 bytes). All index-referenced
  shards exist and download metadata matches the pinned revisions. The CUDA
  image build is in progress. Existing llama.cpp worker remains running
  until the image is ready. No pool route activated.

Recheck before any eventual activation:

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
uv pip install pytest tblib
.venv/bin/python -m pytest tests/kernels/quantization/test_exl3_moe.py tests/models/glm5next/test_sm70_sparse.py
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

## Explicit Hecate container and single-GPU validation

Athena's post-pull classifier and bring-up sweeps exclude `orion-vllm-host`.
The base Compose file also requires the explicit `vllm-worker` profile; a generic
`up` cannot start it. Never add the worker to Athena's automatic rebuild list.

`Dockerfile.sm70` uses CUDA 12.8.1, GCC 14, Python 3.12 and the verified source tree.
`docker-compose.hecate.yml` supplies the external source build context and binds
the real NVMe `/mnt/models` into `/models`. The wrapper rejects other hosts and
an absent/wrong NVMe mount. This is separate from the legacy stock-vLLM image.

Hecate worktree: `/mnt/scripts/Orion-Sapienform-hecate-vllm`.
Set its local service env (also synchronize the primary checkout's ignored env):

```dotenv
NODE_NAME=hecate
SERVICE_NAME=orion-vllm-host
ORION_BUS_URL=redis://100.92.216.81:6379/0
VLLM_SOURCE_DIR=/mnt/scripts/hecate-glm53-source-verified-20261011
VLLM_PROFILE_NAME=qwen3-4b-vllm-hecate
VLLM_HOST_PORT=8021
VLLM_ENFORCE_EAGER=true
LLM_ROLE=
LLM_ANNOUNCE_PORT=
```

Qwen checkpoint: `Qwen/Qwen3-4B-Instruct-2507` at
`cdbee75f17c01a7cc42f958dc650907174af0554`, downloaded to
`/mnt/models/Qwen3-4B-Instruct-2507`. The same downloader then fetches the pinned
GLM checkpoint into `/mnt/models/GLM-5.3-Flash-EXL3` for later hardware use.
Host RAM does not supply the missing three GPUs in the documented GLM TP4 recipe.

```bash
services/orion-vllm-host/scripts/hecate.sh build
# After draining the existing role and stopping its worker:
services/orion-vllm-host/scripts/hecate.sh up -d
python3 services/orion-vllm-host/evals/smoke_glm.py \
  --url http://100.87.202.68:8021 \
  --route qwen3-4b-instruct --expected-model qwen3-4b-instruct
```

Build log: `/mnt/scripts/hecate-vllm-build-headers.log` on Hecate.
Download progress: `docker logs --tail 20 hecate-model-download`.
Role announcement stays disabled during direct acceptance: the production pool
still describes the previous llama.cpp worker. Do not advertise Qwen as GLM.
After acceptance, render the one-card candidate with `stage_pool.py --gpus 1`.
It requires the Qwen profile and exact served alias; the default four-card GLM
candidate below must not be activated on this inventory.

Rollback: stop `orion-hecate-vllm-host` before starting the preserved
`orion-hecate-hecate-llamacpp-agent-deep` container. The old image and model remain.

## Candidate routing and safe rollout

Render a full candidate to a scratch file; the renderer does not edit defaults:

```bash
python services/orion-vllm-host/scripts/stage_pool.py > /tmp/hecate-pool.candidate.yaml
# Live single-card inventory, for the Qwen fallback:
python services/orion-vllm-host/scripts/stage_pool.py --gpus 1 > /tmp/hecate-qwen-pool.candidate.yaml
```

The default four-card candidate replaces the Hecate `agent-deep` allocation with all four cards and
`backend: vllm`, retaining port 8021. All four lend flags must be enabled before
other classes can borrow it. Do not run the previous Hecate llama.cpp worker
on those cards/port at the same time. Do not activate the candidate while an
old `agent-deep` lease is active; drain and verify no outstanding holds first.

The added backend field is omitted for unchanged llama.cpp roles. Once a vLLM
role is activated, older strict `PoolConfig` readers cannot decode it. Before
activation, update the gateway and every reader of shared pool configuration
(including Hub, mesh-guardian, durable-runs and Circe's lane controller) from the same branch.
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
