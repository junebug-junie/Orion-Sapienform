# Orion World Model

GPU-backed world-model service: a small MLP fusion encoder + a Transformer
dynamics model, intended to run on host `circe`, sharing circe's 4th V100
32GB card via MPS with a separately-designed, out-of-scope Infinity 2B
diffusion service. This document is the follow-up to
`docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md` §3/§8,
which explicitly deferred "whatever 'world model' turns out to mean" as "a
new, 4th physical V100 32GB on Circe... separate, later, undefined" — this
patch is that later, undefined thing: service scaffolding, schema/channel
contract, and GPU scheduling. It is not the reverie visual chain itself and
does not touch `orion-vision-host` or `orion-diffusion-host` internals.

## Status: untrained inference-capable scaffolding

**This service does a real forward pass through real, randomly-initialized
PyTorch weights. It does not fake output, and it is not a stub that skips
the model. But there is no training pipeline yet, so a prediction from this
service carries no learned signal.** `WorldModelPredictionPayload
.model_untrained` is always `true` in this patch, `GET /health` reports
`model_untrained: true`, and every prediction round-trips that flag over the
bus. Do not read a prediction from this service as a working next-state
estimate — read it only as proof that the encoder/dynamics/GPU-
scheduling/bus seam works end to end with real shapes.

## Assumptions (flagged explicitly, not silently decided)

Two things were not nailed down before context was lost to compaction on the
originating task. Both are stated here as assumptions, not specs:

- **Dynamics model size**: target was "~100-350M params." Concrete default
  chosen: a 12-layer, `d_model=1024` Transformer encoder, which measures at
  **154,661,376 parameters (~154.7M)** with the encoder included (see
  `tests/test_model_shapes.py::test_world_model_param_count_within_assumed_range`,
  which asserts the real, computed count stays within the stated 100-350M
  range using the service's actual default hyperparameters -- not a
  hardcoded/hand-waved number).
- **Feature group input dims**: `WM_DIM_BIOMETRICS`, `WM_DIM_AFFECT`,
  `WM_DIM_EXECUTION_CONTEXT`, `WM_DIM_MEMORY_POINTERS`, `WM_DIM_TEMPORAL`,
  `WM_DIM_VISION_EMBEDDING` in `.env_example` are placeholder defaults, not a
  spec of what any real upstream producer emits. An operator wiring a real
  feature producer must set these to match that producer's actual vector
  lengths. A trajectory step whose vector length disagrees with its declared
  `dim`, or whose `dim` disagrees with the configured group dim, is rejected
  at forward-pass time with `error_code=bad_trajectory` (`app/main.py::
  trajectory_steps_to_tensors`) -- not silently truncated or padded.

## What this is for

- **Encoder** (`app/model.py::WorldModelEncoder`): fuses six already-computed
  feature groups per trajectory step -- biometrics, affect,
  execution-context signals, memory pointers, temporal features, and a
  vision embedding -- via small per-group `Linear -> LayerNorm -> GELU`
  branches plus a fusion layer. It is deliberately small (an MLP fusion
  layer, not a new backbone). It does **not** run any vision backbone
  itself: `vision_embedding` is expected to already be
  `orion-vision-host`'s `VisionArtifactPayload.outputs.embedding.vector`
  (`orion/schemas/vision.py::VisionEmbedding`), computed elsewhere and
  passed in as a plain float vector.
- **Dynamics model** (`app/model.py::WorldModelDynamics`): a causal
  (autoregressive-masked) Transformer encoder over a trajectory window of
  fused encoder states, predicting a next-state Gaussian (`mean`, `log_var`)
  from the final window step. Real `nn.TransformerEncoder` forward pass,
  real causal mask (`nn.Transformer.generate_square_subsequent_mask`) so a
  step only attends to history, not future steps within the same window.
- **GPU scheduling** (`app/gpu.py::GpuInspector`): VRAM-aware pick with a
  reserve + hard-floor pattern adapted from `services/orion-vision-host/app/
  gpu.py` (not cross-imported -- CLAUDE.md §5 service-boundary rule). Sized
  via `WM_VRAM_*` env keys so this service caps itself well under a 32GB
  card: a ~150M-param fp32 model is itself only ~600MB of weights, and the
  default reserve (`WM_VRAM_RESERVE_MB=4000`) is deliberately generous
  headroom for the separately-scheduled Infinity 2B diffusion process
  sharing the same physical card via MPS -- not because this model needs
  that much.

## Port: why `HOST_PORT=6613`, not 8014

Port 8014 on circe is already bound to `circe-worker-agent-1` (llama.cpp
Muse Glimmer 30B -- `services/orion-llm-gateway/.env_example`'s
`ATLAS_AGENT_HOST_PORT=8014` and `services/orion-llamacpp-host/README.md`).
This service follows `orion-vision-host`'s pattern instead: a fixed internal
container port (`6701`, see `Dockerfile`), with the **host** port assigned
via `HOST_PORT` so it's operator-set per node, not hardcoded. `6613` was
picked by grepping every `services/*/.env_example` and `docker-compose.yml`
in this repo for numeric port collisions (Tailscale makes every service port
reachable network-wide, so a same-node-only check isn't sufficient) --
it continues the existing `orion-mind` (`6611`) / `orion-diffusion-host`
(`6612`) sequence and was confirmed unclaimed. Note the internal container
port `6701` differs from `orion-diffusion-host`'s own internal `6700` even
though internal ports don't collide across separate bridge-network
containers -- picked to avoid any confusion if the two are ever run under
host networking.

## Node: why `NODE_NAME=athena` by default, not `circe`

The design doc (§3/§8) explicitly deferred "the 4th Circe GPU's physical
provisioning [and] `node_catalog.yaml` entry" as separate, later, undefined
work -- `circe` does not have an entry in `config/biometrics/node_catalog.yaml`
as of this patch. `.env_example` defaults `NODE_NAME=athena` so a fresh
operator `.env` still boots locally; whoever provisions circe's 4th GPU sets
`NODE_NAME=circe` at that point.

## Bus contract

| Channel (env) | Default | Kind | Direction |
| :--- | :--- | :--- | :--- |
| `CHANNEL_WORLDMODEL_INTAKE` | `orion:exec:request:WorldModelService` | `world_model.task.request` | In (request) |
| `CHANNEL_WORLDMODEL_REPLY_PREFIX` | `orion:worldmodel:reply` | `world_model.prediction` | Out (reply) |
| `CHANNEL_WORLDMODEL_PUB` | `orion:worldmodel:predictions` | `world_model.prediction` | Out (broadcast) |

Schemas: `orion/schemas/world_model.py` --
`WorldModelTaskRequestPayload` / `WorldModelPredictionPayload` (registered in
both `_REGISTRY` and `SCHEMA_REGISTRY`, `orion/schemas/registry.py`), plus
nested `WorldModelFeatureGroupV1` / `WorldModelTrajectoryStepV1` (registered
in `_REGISTRY` only, same split as `VisionObject`/`VisionArtifactOutputs` in
`orion/schemas/vision.py`). Channels registered in `orion/bus/channels.yaml`.

### No real downstream consumer yet

**Flagged honestly, not hidden** (known repo anti-pattern per CLAUDE.md).
This patch wires a real producer (this service publishing
`WorldModelPredictionPayload` after a genuine forward pass) and a real
consumer for its own intake channel (the service consumes its own
`orion:exec:request:WorldModelService`). But nothing in the rest of the repo
issues world-model requests yet, and nothing consumes
`orion:worldmodel:predictions` except `scripts/tap_predictions.py`, a
documented tap/smoke script (mirrors `orion-vision-host`'s
`scripts/tap_artifacts.py`). The only current end-to-end producer of a real
request is `scripts/publish_test_task.py`. Wiring a real upstream caller
(e.g. a reverie/attention consumer) is out of scope for this patch.

## Probes

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness: process up, device picked, `model_untrained: true`. |
| `GET /ready` | Readiness: HTTP **503** until model loaded + bus connected (if `ORION_BUS_ENABLED`) + at least one CUDA device clears the VRAM hard floor (skipped when no `cuda:*` device is configured, so a CPU-only dev box can still go ready). |
| `POST /v1/world-model/predict` | Optional HTTP entrypoint mirroring vision-host's `/v1/vision/task` -- local smoke without a bus. Real integration is expected to go over the bus intake channel. |

## Operator checklist

1. **GPU visibility**: `CUDA_VISIBLE_DEVICES` and `WM_DEVICES` must list
   indices the container can see. NVML (`nvidia-ml-py`) must work for
   VRAM-aware scheduling; without it (or without any `cuda:*` device
   configured), the service falls back to CPU -- untrained-scaffolding
   forward passes still work there, just slower.
   **Index parity is not automatic**: `app/gpu.py::GpuInspector` picks a
   candidate index via NVML (always PCI-bus-id order, same as `nvidia-smi`),
   then `app/main.py::_select_device` hands that same integer straight to
   torch as `cuda:{idx}`. On a heterogeneous multi-GPU host, CUDA's default
   enumeration is *not* PCI-bus-id order (a "fastest first" heuristic), so
   the two index spaces can silently disagree -- NVML's `cuda:2` and torch's
   `cuda:2` can be two different physical cards. Confirmed live on circe
   (2026-08-25): `WM_DEFAULT_DEVICE=cuda:2` intended for the empty
   PG500-216 (host index 2) instead loaded weights onto host index 3, a
   busy V100-32GB, while `/health`/`/ready` reported `device=cuda:2` and
   looked correct. The Dockerfile now sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` to
   force torch's enumeration to match NVML's -- the same fix
   `orion-llamacpp-host`/`orion-vllm-host` already apply for their spawned
   subprocesses. **`orion-vision-host` (this service's `GpuInspector`
   source pattern) has the same latent gap and has not been fixed** -- flagged,
   not fixed here, out of scope for this patch.
   To verify index parity after a deploy, don't trust the log line alone:
   `docker exec <container> python3 -c "import torch; print(torch.cuda.get_device_name(N), torch.cuda.mem_get_info(N))"`
   and cross-check the name/free-bytes against `nvidia-smi --query-gpu=index,name,memory.free --format=csv` for
   the same physical index.
2. **Bus**: when `ORION_BUS_ENABLED=true`, Redis must be reachable before
   `/ready` goes green.
3. **VRAM floors**: tune `WM_VRAM_RESERVE_MB` and `WM_VRAM_HARD_FLOOR_MB` to
   match whatever else is co-hosted on the same card (Infinity 2B via MPS on
   circe) -- these two are the ones `app/main.py::_select_device` actually
   reads. `WM_VRAM_SOFT_FLOOR_MB` is declared for parity with
   `orion-vision-host`'s floor pattern but is **not read** by this scaffold:
   there is no queue (unlike vision-host's async `VisionScheduler`) for a
   soft floor to gate a queue-vs-reject decision on yet.
4. **Feature dims**: `WM_DIM_*` must match whatever real upstream producer
   you wire in (see "Assumptions" above) -- there is no auto-detection.
5. **Device strategy / dtype / timeout**: `WM_DEVICE_STRATEGY`
   (`best_free_vram` default, or `fixed` to pin `WM_DEFAULT_DEVICE` and skip
   scanning other GPUs), `WM_DTYPE` (`auto` resolves to fp32, not a
   device-conditional guess -- set `fp16`/`bf16`/`fp32` explicitly to opt
   into lower precision), and `WM_TIMEOUT_S` (bounds how long a caller waits
   for a reply; cannot forcibly kill an already-dispatched worker thread,
   same limitation as vision-host's identical pattern) are all real, read
   knobs -- `app/main.py::_select_device` / `resolve_dtype` /
   `run_prediction_task`.
6. **`MODEL_CACHE_DIR`**: declared and volume-mounted for path-convention
   parity, but **not read** by any code in this patch -- weights are always
   randomly initialized in-process (see "Status" above). Reserved for a
   future patch that adds real checkpoint loading.

## Smoke scripts

- `scripts/publish_test_task.py` -- publish a real trajectory request over
  the bus and print the reply.
- `scripts/tap_predictions.py` -- subscribe to the prediction broadcast
  channel.

## Tests

```bash
cd services/orion-world-model
PYTHONPATH=.:../.. python3 -m pytest tests/ -q
```

Also see `orion/schemas/tests/test_world_model_registry.py` for the bus
contract (schema registration + round trip):

```bash
python3 -m pytest orion/schemas/tests/test_world_model_registry.py -q
```

No CUDA/GPU is required for either test suite -- the model tests run real
forward passes on CPU with small hyperparameters, and `app/gpu.py`'s tests
mock `pynvml`.
