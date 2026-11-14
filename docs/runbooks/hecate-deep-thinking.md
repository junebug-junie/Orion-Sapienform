# Hecate deep thinker

Status: **UNVERIFIED** until the live acceptance evidence below exists. Do not infer
132K or 256K capacity from the profile or successful startup.

## Hardware and compatibility (2026-11-14)

Read-only evidence: `docs/bench/2026-11-14-hecate-deep/hardware.json`.
Two V100-SXM2-32GB cards, driver 580.178.04, NV2 between the cards, two active
25.781 GB/s links per card. NVIDIA read/write P2P reports OK; the existing
CUDA12.8/torch2.10 container reports `can_device_access_peer(0,1)=True`.
This is connectivity proof, not a measured transfer bandwidth. Both cards sit
on NUMA node 1 (CPUs 20-39,60-79); two Xeon Gold 6148 sockets, 183 GiB RAM,
172 GiB available at reconnaissance. CPU↔GPU uses PCIe, not CPU NVLink.
Initial free space: model NVMe 254 GiB, Docker SSD 683 GiB, root 79 GiB.
The existing 4B worker used GPU0; biometrics also runs here and must remain up.
The host has no `nvcc`; the existing container has CUDA12.8 and GCC14.
The new image carries its own CUDA12.8 toolkit/compiler environment.

## Recommendation and source verification

The correctness baseline is the pinned **QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4**
checkpoint on **1Cat-vLLM 1.5.3**, FP16 compute, FP8 E4M3 KV, FLASH_ATTN_V100,
TP2, one sequence, chunked prefill 4096, prefix caching, 262144 context.
The exact revision, five shard hashes and wheel digest are in
`services/orion-vllm-host/scripts/deep-thinker.lock.json`.
The 20.56 GB checkpoint was downloaded to `/mnt/models/Qwen3.8-27B-QUASAR-NVFP4`.

This uses software SM70 kernels; Volta has neither native FP8 nor NVFP4 tensor
cores. Source at pinned commit `cb97f379b67eb9f5703709db961d9dabcdecbcff`:
`compressed_tensors_w4a4_nvfp4.py` selects minimum capability 70 and the SM70
QPN kernel; `vllm/v1/attention/backends/flash_attn_v100.py` supplies attention.
[Release 1.5.3](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3)
was published November 14. Its four-card results do not establish TP2 capacity.
CUDA Graphs, native MTP4 and DFlash2 are supported upstream; speculation starts
disabled so its extra weights/KV and correctness can be measured separately.
No CPU weight/KV offload is enabled. Do not add offload to hide an OOM.

The [dual PCIe benchmark #1202](https://github.com/1CatAI/1Cat-vLLM/issues/1202)
used v1.5.1, TP2, QUASAR revision d9af17e, DFlash2 q7, E4M3 target KV, FP16
draft KV, memory .85, prefill 8192, four sequence slots and prefix caching.
Its machines lacked CUDA P2P/NVLink and disabled custom all-reduce/NCCL P2P.
It also disabled an experimental prefill kernel after a crash. Its largest
input was 261888 plus 256 generated tokens; retrieval passed twice, shared-cache
256K was not tested. This is useful external evidence, not Hecate validation.
Do not copy its transport workarounds onto Hecate without a failing measurement.

## Memory engineering

`scripts/deep_memory.py` reads the pinned model geometry and tensor byte count.
The model has 16 full-attention layers, 4 KV heads and head dimension 256:
FP8 attention KV is `2 × 16 × 4 × 256 × tokens` bytes across both GPUs.
FP16 KV doubles that component. Linear/GDN layers need state plus retained
prefix snapshots, rather than ordinary per-token KV.

| Tokens | Attention KV, both cards | Total planning budget, both cards |
| --- | ---: | ---: |
| 132000 | 4.03 GiB | 41.71 GiB |
| 196608 (192 Ki) | 6.00 GiB | 44.66 GiB |
| 262144 (256 Ki) | 8.00 GiB | 47.79 GiB |

The total includes 19.15 GiB checkpoint tensors/quantization metadata, estimated
GDN snapshots, CUDA/graphs 4 GiB, attention workspace 2 GiB, temporary prefill 4 GiB,
safety/fragmentation 6 GiB, and no draft. These are **estimates**, not allocator
measurements. Replicated tensors, packed kernel copies, per-rank imbalance and
cache page rounding can invalidate a simple equal split. Runtime profiling and
the full-output endurance test decide feasibility. Exact component output is
committed in `memory-estimates.json`.

## Orion integration and budgets

Gateway OpenAI `model: deep.thinker.v1` maps to existing class/role `agent-deep`.
The role owns both `hecate-gpu0` and `hecate-gpu1`; Circe launch digests and
default routes do not change. The existing pool supplies priority/age ordering,
queue state, heartbeats, cancellation and one active inference seat. Hecate is
a fixed warm resident, with no actuator that unloads it between requests.
Keep both cards un-lent for dedicated use; borrowing requires both lent.

At context 262144, the default prompt budget is 229376 and output budget 32768,
including reasoning. Explicit output 16384 permits prompt 245760. Even a tiny
requested answer reserves 16384. The worker counts the actual rendered prompt
with its own tokenizer, supports OpenAI Chat and Anthropic Messages, and rejects
truncation/overflow rather than allowing the engine to silently shrink output.
Other generation protocols fail closed on this worker. Its policy is text only.
The pool's route view exposes live context, model, endpoint, status, hardware and
budget information. Cost remains null/UNMEASURED until measured.

The explicit deep route has an 1800s bus budget (callers may shorten it); Hecate
HTTP upstream reads have an 1800s inactivity timeout. Pool queue waits retain
their existing policy. A caller's own RPC deadline must cover its queue plus
inference budget. HTTP streaming is bounded by output tokens and caller
cancellation; inactivity timeout is not a wall-clock SLA.

## Build, start, stop and rollback

Builds may run in a worktree. Live `up` must run from Hecate's **clean primary
main, synchronized with origin/main**. The wrapper enforces this and the model
NVMe/containerd storage guards. Preserve the old image/model and cached Strata.

```bash
# Hecate, after checking resources; needs huggingface_hub in the Python environment
python services/orion-vllm-host/scripts/prepare_deep_model.py /mnt/models --download
services/orion-vllm-host/scripts/hecate-deep.sh build
services/orion-vllm-host/scripts/hecate-deep.sh config --quiet
# Drain agent-deep using the existing GPU-pool operator hold; confirm no active child.
# Deploy pool/Gateway with this config before accepting deep.thinker.v1 traffic.
services/orion-vllm-host/scripts/hecate-deep.sh up -d --no-build
curl -f http://localhost:8021/health
curl -f http://localhost:8021/orion/server-info
services/orion-vllm-host/scripts/hecate-deep.sh logs --tail=100
# Stop only this worker:
services/orion-vllm-host/scripts/hecate-deep.sh stop
```

Local env templates are unchanged; the explicit Compose override selects the
profile, port and resident role. Bus remains `redis://100.92.216.81:6379/0`.
Do not run a generic Hecate rebuild sweep or restart biometrics. Do not pull
Hecate's old main without first inspecting its post-merge rebuild hook.

Rollback: drain the deep role, stop the new worker, restore the single-card role
config and old Gateway route set through a reviewed revert, then use the existing
`services/orion-vllm-host/scripts/hecate.sh up -d --no-build` with its preserved
4B image/env. Verify health, model identity, one GPU/slot and a leased answer.
The old worker is an 8K operational rollback and does **not** satisfy this mission.
Never delete model volumes, containerd stores or graphify artifacts for rollback.

## Acceptance and benchmarking

Use `evals/deep_context.py` through `operator_lane_sync`; no direct inference
URLs. Operator work waits behind Orion's work and exits on recall. During the
planned isolated benchmark window, lend both Hecate cards and take the operator
hold before inference. Restore original lending flags afterward.

In one Hecate host terminal, start sampling **before** inference:

```bash
python services/orion-vllm-host/evals/sample_host.py --seconds 7200 > /tmp/hecate-host.jsonl
```

In a second terminal, using the pinned image's Python dependencies and repository
on PYTHONPATH, run the benchmark; stop the sampler with Ctrl-C afterward:

```bash
export ORION_BUS_URL=redis://100.92.216.81:6379/0
PYTHONPATH=. python services/orion-vllm-host/evals/deep_context.py \
  --tokenizer /mnt/models/Qwen3.8-27B-QUASAR-NVFP4 --model deep.thinker.v1 \
  --gateway http://100.92.216.81:8210 --out /tmp/hecate-deep-results.jsonl \
  --repeats 3 --output-budget 32768 --endurance
```

The grid uses actual prompts 8192/32768/65536/132000/196608/229376 with a 32768
output reserve. With reserve 16384 the last prompt is 245760. Fresh-prefix and
exact-replay modes record actual cached-token counts; replay is not assumed to
hit. TTFT includes queue/transport; prompt-per-TTFT is a proxy, not pure ingestion
speed. Repeat output includes model, usage, seeds, prompt hash, reasoning, final
answer and finish reason. Empty final, wrong answer/model, missing usage,
truncation and length-stop fail quality checks. Three distant random identifiers
and distributed inventory arithmetic must all be correct.

`--endurance` separately forces the full output reserve, verifies actual usage
sums to 262144, and tests recovery. It makes no reasoning-quality claim. Add
representative architectural/code/planning tasks from the existing Hecate hard
reasoning suite; the synthetic ledger alone cannot choose the best reasoner.
`evals/deep_quality.py` runs four fixed code, resource-planning, memory-provenance
and transaction-recovery problems with exact JSON answers, three repeats by
default, through the same leased Gateway path. Pass `--tokenizer`, `--model`,
`--gateway` and an exclusive `--out` JSONL file as above. This is a narrow
deterministic suite, not evidence of general reasoning superiority. Preserve
reasoning/final text for review and compare failures as well as latency/energy.
The resource-plan gold is checked by enumerating all topological schedules;
its final lexicographic tie-break avoids penalizing equally optimal schedules.
Capture NVML memory/utilization/power and `/proc` CPU/RAM samples. NVLink byte
traffic, memory bandwidth and KV residency remain null unless independently
measured. Query actual engine metrics/logs for cache capacity, running/waiting
requests and preemptions. Test cancel/timeout recovery and queued short work
behind a full-context request. Do not induce production OOM just to test recovery.

Measurement gate: samples originate in NVML, monotonic clocks, provider usage
and `/proc`; throughput, time, power and energy are causally dependent and not
independent cognition signals. These files are offline benchmark evidence,
never new detector inputs. Live rest points and unavailable counters must be
recorded; all instrumentation is removable without changing cognition defaults.

## Strata and Legion comparison

Cached Strata `0.2.7-cuda12` digest
`04b696b7890b258adfcdbd813cd7eed9033764092c19f86c567452a20502206f`
and Flash-Next IQ3_S already exist on Hecate. The October 30 single-V100 run
passed 20/20 hard checks and 22/22 retrieval checks to 75K, with about 94 output
tokens/s on that hard set. These are historical results, not a dual-card 256K
comparison. See `docs/bench/2026-10-30-strata-hecate/README.md`.

[Strata's multi-GPU design](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md)
splits contiguous layers and their expert caches across cards; it is not TP.
Host-resident experts and mapped host handoffs remain relevant. The cached
container explicitly accepts `GPUS=0,1`, `CONTEXT=256k`, `PARALLEL=1`, `KV=int8`.
Verify engine startup logs, both cards, host RSS/NUMA residency and actual
context with the same tests; use a separate preset without rewriting Circe's
config. More parameters do not prove better task quality. Full dual-card
Strata 256K quality and offload penalties remain **UNVERIFIED**.

[Legion](https://github.com/eelgaev/Legion) targets unified-memory platforms,
especially POWER9 AC922 with CPU–GPU NVLink/ATS and DGX Spark. Hecate's Xeon
PCIe host path lacks that CPU link. Transferable ideas are expert locality,
NUMA placement and bounded hot caches; published AC922 bandwidth/speed numbers
do not transfer. No new inference framework is proposed.
