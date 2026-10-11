# Hecate vLLM: live Qwen and staged GLM

## Live state (2026-10-11)

Hecate serves `qwen3-4b-instruct` through Athena's `agent-deep` gateway route.
The runtime is the pinned 1Cat-vLLM SM70/EXL3 build, not the legacy stock image.
Live discovery confirms one slot, 8,192 tokens, TP1 and GPU0. HTTP requests,
streamed tool calls and bus RPC reached Hecate with the exact model identity.
The maintenance hold was released; ordinary requests work and no test lease remains.

The requested GLM checkpoint is fully downloaded to NVMe, but Hecate has exactly
one Tesla V100-SXM2-32GB. The documented GLM recipe requires four. Juniper
explicitly authorized the smaller model to validate vLLM. Host RAM does not
supply the missing GPU capacity. Do not enable the four-card profile on this host.

**Quality limitation:** the strict arithmetic eval returns `17 × 23 = 391` rather
than only `391`, including in streaming. That gate still fails. Recall, JSON,
structured arithmetic and automatic/streamed tool calls pass. This does not
establish 4B quality equivalent to the previous 27B/128K worker. GLM inference,
TP4, Anthropic Messages and vision remain **UNVERIFIED**.

## Architecture and deployment

Deployment used the reviewed worktree. Merge PR #2615 before rebuilding the pool
or gateway from shared main: main still has the previous llama.cpp role config
until that merge. Athena's existing worker exclusion is already present on main.

- Athena GPU pool owns leases and live worker discovery; the gateway dispatches
  HTTP and bus calls to the granted role. `config/gpu_pool.yaml` selects
  `agent-deep.backend=vllm`; other role definitions and launch digests are unchanged.
- Hecate: `100.87.202.68:8021`, container `orion-hecate-vllm-host`, profile
  `qwen3-4b-vllm-hecate`. The old `orion-hecate-hecate-llamacpp-agent-deep`
  container is stopped and preserved with its model on telemetry storage.
- Existing `orion:llm:worker:announce` and GPU pool contracts are used. Announcement
  alone is insufficient: the pool checks health, model ID/path, capacity and GPUs.
- With explicit operator approval, deployed compatibility readers on Athena
  durable-runs/mesh-guardian and Circe's lane controller before activating the
  pool/gateway images. Hub already had the compatible reader. Circe's model workers
  were not restarted; its `quick` route still returns a real answer.
- Services use `app/settings.py`; service-root `settings.py` files are absent.
  The vLLM host, pool and gateway have tests; vLLM's opt-in model eval is
  `services/orion-vllm-host/evals/smoke_glm.py`.

Active image IDs (prefixes):

| Service | Image |
|---|---|
| Hecate vLLM (image config) | `8d4eac1c38a7` |
| Athena GPU pool | `562b90014859` |
| Athena gateway | `c86645e65422` |
| Athena durable-runs | `870f7945992b` |
| Athena mesh-guardian | `f23a5ef16030` |
| Circe lane-controller | `126f18023f68` |

## Model and image storage

Juniper explicitly approved erasing Toshiba NVMe serial `49JA22UKK03N`.
It is ext4 at `/mnt/models`, UUID `19d66b9a-2234-4a60-9180-1862285cd74c`, with a
persistent fstab entry. Saved partition/fstab metadata under
`/mnt/scripts/hecate-nvme-layout-before-20261011` is not an erased-data backup.

| Model | NVMe directory | Locked revision | Verified weight bytes |
|---|---|---|---:|
| Qwen/Qwen3-4B-Instruct-2507 | `/mnt/models/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | 8,044,982,000 (3 shards) |
| philbert440/GLM-5.3-Flash-EXL3 | `/mnt/models/GLM-5.3-Flash-EXL3` | `4ba4b9afb2191578378454cdd28aa35e9484960c` | 104,011,208,249 (13 shards) |

All index-referenced shards exist and download metadata matches these revisions.
The container binds this NVMe to `/models`; missing host paths are not auto-created.

Docker data-root was already `/mnt/docker` on `/dev/sdb1`, but containerd's separate
store still used `/var/lib/containerd`. Image unpacking left root with 3.5 GiB free.
Docker's data-root does not relocate that store; see
[Docker daemon documentation](https://docs.docker.com/engine/daemon/).

The operator-approved correction is complete:

- Active containerd root: `/mnt/docker/containerd-store`, with a systemd
  `RequiresMountsFor` dependency. Live snapshot mounts and the nonroot guard pass.
- Original store retained intact at `/mnt/docker/containerd-root-backup-20261011`.
- Root: 79 GiB free, 16% used; original 100 GiB LV unchanged. Docker SSD: 686 GiB
  free; model NVMe: 341 GiB free at verification. No image pruning/root expansion.
- First attempt exercised recovery when biometrics' startup connection reset.
  Original config and both workers recovered. Second attempt added bounded
  retries/curl-managed output and succeeded. Biometrics HTTP health and preserved
  image identity were verified before moving the old store off root.
- Evidence/log on Hecate: `/mnt/scripts/hecate-containerd-move-20261011-attempt2`
  and the adjacent `.log`; attempt-one evidence is preserved separately.
  **Do not rerun** `/mnt/scripts/hecate-containerd-relocate.sh`.

The launch wrapper checks the configured filesystem and actual snapshot mounts
from a running container (normally biometrics). It reads the mount table without
stat-ing root-only Docker directories. Editing config without moving the live
store cannot satisfy the guard. Stop/down and inspection bypass storage checks.

## Pinned source and build

The [checkpoint card](https://huggingface.co/philbert440/GLM-5.3-Flash-EXL3) requires
the SM70/EXL3 path in 1Cat-vLLM. The lock is
`services/orion-vllm-host/scripts/glm53.lock.json`: base, six cherry-picks, EXL3
merge and per-step tree checks. It rejects the known MTP-regression ancestor
`d4ce513991b79a9cac2a4d72fbe8bcac5f566d5e`.

Verified source on Athena and Hecate:
`/mnt/scripts/hecate-glm53-source-verified-20261011`, commit
`08b29dcd5211cc85502f7ae6a3ef66ff303f20f6`, tree
`b321701ed88fc9882ae7c71cdb5968a6635505ab`.

**Retired source, do not build:** `/mnt/scripts/hecate-glm53-source-20261011`,
tree `c5838487a6c58c079cc9e86f3fa391ba5ff1cca0`, which includes the regression.
Preserve source and graph artifacts.

For a fresh source checkout:

```bash
python3 services/orion-vllm-host/scripts/prepare_source.py /mnt/scripts/hecate-glm53-source-new
```

If the helper stops on conflicts, inspect both additions in `CMakeLists.txt` and
`csrc/ops.h`, retain the necessary independent declarations, commit, then use
`--resume`. Never blindly strip conflict markers. The helper verifies ancestry
and resulting trees before accepting a checkpoint.

`Dockerfile.sm70` builds CUDA12.8/GCC14/Python3.12/torch2.10 for SM70 and reconstructs
the pinned source from Git objects to avoid upstream Docker-ignore omissions.
Native compilation, installation, export and actual GPU execution passed.
Image: `orion-vllm-host:sm70-glm53-b321701e`.
Build log on Hecate: `/mnt/scripts/hecate-vllm-build-headers.log`.

## Explicit launch and configuration

Athena's generic post-pull classifier and bring-up sweeps exclude `orion-vllm-host`.
The base Compose service additionally requires `--profile vllm-worker`. A generic
`up` cannot launch it. Use the Hecate wrapper from a worktree, never shared main.

Hecate worktree: `/mnt/scripts/Orion-Sapienform-hecate-vllm`. Its ignored local env,
also synchronized to the primary checkout, contains:

```dotenv
NODE_NAME=hecate
SERVICE_NAME=orion-vllm-host
ORION_BUS_URL=redis://100.92.216.81:6379/0
VLLM_SOURCE_DIR=/mnt/scripts/hecate-glm53-source-verified-20261011
VLLM_PROFILE_NAME=qwen3-4b-vllm-hecate
LLM_PROFILES_CONFIG_PATH=/app/config/llm_profiles.yaml
VLLM_HOST_PORT=8021
VLLM_ENFORCE_EAGER=true
LLM_ROLE=agent-deep
LLM_ANNOUNCE_HOST=hecate
LLM_ANNOUNCE_PORT=8021
```

The profile supplies FP16, TP1, GPU0, context8192, one sequence, 0.85 memory
fraction and Hermes automatic tool parsing. Do not enable `VLLM_SERVER_DEV_MODE`;
the read-only `/orion/server-info` middleware supplies live discovery facts.

```bash
# Hecate only; explicit future rebuild/start commands:
services/orion-vllm-host/scripts/hecate.sh build
services/orion-vllm-host/scripts/hecate.sh up -d --no-build
```

Blank both `LLM_ROLE` and `LLM_ANNOUNCE_PORT` for a deliberately undiscovered
worker during future isolated acceptance. Current live deployment has them enabled.

## Verification and evidence

The actual V100 passed all 80 EXL3/SM70 kernel tests:

```bash
services/orion-vllm-host/scripts/hecate.sh run --rm --no-deps \
  --entrypoint python --workdir /opt/vllm vllm-host -m pytest \
  tests/kernels/quantization/test_exl3_moe.py tests/models/glm5next/test_sm70_sparse.py -q
```

Run this only with the GPU drained and model worker stopped; it uses the GPU.
Runtime reports torch `2.10.0+cu128`, CUDA12.8, vLLM `0.1.dev20452+g08b29dcd5`,
SM70 and EXL3 available. `/health`, `/v1/models` and `/orion/server-info` agree.

The strict model smoke remains useful and currently exits nonzero for formatting:

```bash
python3 services/orion-vllm-host/evals/smoke_glm.py \
  --url http://100.87.202.68:8021 \
  --route qwen3-4b-instruct --expected-model qwen3-4b-instruct
# Gateway equivalent (may wait for a normal pool lease):
python3 services/orion-vllm-host/evals/smoke_glm.py \
  --url http://127.0.0.1:8210 --route agent-deep --expected-model qwen3-4b-instruct
```

Committed synthetic evidence lives at
`docs/superpowers/evidence/2026-10-11-hecate-vllm/`. It records failing exact-text
arithmetic, successful recall/JSON, completed streaming, automatic and streamed
tool calls, structured arithmetic, context rejection, fresh announcement,
HTTP/bus dispatch and an ordinary request after maintenance release.

Bus correlation `1b29fb78-10e7-4701-ab92-ec4ed953ff18` proves
`served_by=hecate-worker-agent-deep` and exact model identity. Normal HTTP correlation
`a6eabc40-bbf8-40cc-8b80-9eaac5d94da9` returned `{"answer":391}` after release.
Circe `quick` correlation `0fb5c43c-36cb-421c-81b0-75dbe3ef8ed0` reached
`circe-worker-fast` and returned `READY.`.

During warm-up, the pool reported down/zero slots. A queued client's timeout
cancelled cleanly with no remaining child lease. This verifies cancellation while
unavailable, not a completed server-side timeout/error response. The test hold
`367216bbc21340ec82679b1f6e9c6165` was explicitly released and no test lease remains.

## Rollback and future four-card activation

Drain Hecate and hold the role before changing its worker. Stop vLLM before
starting the preserved llama.cpp worker; they share GPU0 and port8021. Restore
pool/gateway config and images to the previous llama.cpp backend. Athena's
pre-activation images are tagged:

```text
orion-gpu-pool-gpu-pool:hecate-rollback-20261011
orion-llm-gateway-llm-gateway:hecate-rollback-20261011
```

Activation/rollback Compose overrides are under `/tmp/hecate-{activate,rollback}-<service>.yml`
on Athena; image IDs are also in the PR report. If those temporary files are gone,
create an override selecting the rollback image for `gpu-pool` or `llm-gateway`,
then use the worktree's `scripts/safe_docker_build.sh` with that override and
`up -d --no-build`. The additive compatible-reader upgrades can stay in place.
Do not overwrite shared main or remove storage/model rollback artifacts.

For later hardware availability, render a candidate without mutating live defaults:

```bash
python3 services/orion-vllm-host/scripts/stage_pool.py > /tmp/hecate-tp4.candidate.yaml
python3 services/orion-vllm-host/scripts/stage_pool.py --gpus 1 > /tmp/hecate-tp1.candidate.yaml
```

TP4 must be independently approved and verified against four physical GPUs,
model memory fit and live GLM inference. Do not move Circe cards or advertise
Qwen as GLM. All readers of the shared pool config must support the selected
backend before activation. Confirm exact model/alias/context in discovery and
repeat HTTP, bus, streaming, tool and lease tests before releasing maintenance.
