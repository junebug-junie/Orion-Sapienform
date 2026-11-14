# Hecate deep thinker

Status: **262144 actual user tokens VERIFIED** through Gateway on the pinned
dense baseline: 245760 input + 16384 generated, followed by a correct recovery
request. The hard distributed-reasoning grid passed **36/36** fresh/cached requests
across six depths, including 132000 and the maximum prompt. See `1cat-hard-grid.jsonl`
and its resource summary in the dated benchmark directory. All five operational checks passed, including serialized queueing, budget refusal,
stream cancellation and client-timeout recovery. The four task-quality cases passed all three
repeats (12/12). DFlash2 and Strata comparisons remain in progress. Measured
latency/energy tables are in the dated benchmark directory’s `RESULTS.md`.

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

Optional DFlash2 experiment: the pinned draft is 3848817896 bytes (3.58 GiB),
SHA256 `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
Its five attention layers use a 2048-token sliding window. Runtime profiling must
still measure weight sharding, target-feature buffers, draft KV and graph overhead.
The optional override keeps the baseline context and sequence limits, uses seven
speculative tokens, TP2 for the draft, and `kv_cache_dtype=auto` (FP16 with this
target's FP16 compute). Preparation does not activate speculation:

```bash
python services/orion-vllm-host/scripts/prepare_deep_model.py /mnt/models --draft --download
# On clean synchronized Hecate primary main, with the lane private and drained:
mkdir -p .orion
python services/orion-vllm-host/scripts/stage_dflash.py > .orion/hecate-dflash-profiles.yaml
services/orion-vllm-host/scripts/hecate-deep.sh \
  -f services/orion-vllm-host/docker-compose.dflash-eval.yml up -d --no-build
```

Use the same quality/context harness and record the speculative engine configuration.
Restore baseline with `hecate-deep.sh up -d --no-build` (no DFlash override). Keep
the candidate private until its own context and quality results pass.

The first cold 132K ledger request took 98.08s total, 86.00s to first token,
1574 uncached prompt tokens/s from provider prefill timing, and 42.79 generated
tokens/s. Exact replay reached the first token in 2.57s with 131968 cached tokens.
Both cards reached 100% utilization; sampled peak memory was 28699/28693 MiB.
Sampled GPU board energy was 52.08 kJ cold and 5.98 kJ replay. These are one
simple task each, not a broad reasoning score or a wall-plug energy measurement.

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
URLs. Operator work keeps its background priority, heartbeat, heat gate and
recall behavior. During commissioning keep both Hecate cards **un-lent** and
make their owner the existing `operator_agent_deep` class using a temporary
candidate config. This prevents production spill onto an unvalidated model.
From clean synchronized Athena main, after reserving/draining `agent-deep`:

```bash
mkdir -p .orion
docker run --rm --network none --entrypoint python -e PYTHONPATH=/repo -v "$PWD:/repo:ro" \
  orion-gpu-pool-gpu-pool /repo/services/orion-vllm-host/scripts/stage_pool.py \
  --gpus 2 --commissioning > .orion/hecate-eval-pool.yaml
scripts/safe_docker_build.sh orion-gpu-pool -f services/orion-gpu-pool/docker-compose.hecate-eval.yml up -d --no-build
```

Only the pool needs this ownership override for the vLLM baseline: Gateway
routes/backend remain identical. Release the maintenance hold once discovery
is confirmed; the benchmark takes its own background hold. After validation,
recreate the pool from its base Compose file to restore owner `agent-deep`.
Verify lending flags are still false and the production deep route works.

In one Hecate host terminal, start sampling **before** inference:

```bash
python services/orion-vllm-host/evals/sample_host.py --seconds 7200 > /tmp/hecate-host.jsonl
# Optional read-only engine gauges; vLLM names are not assumed for Strata.
python services/orion-vllm-host/evals/sample_engine.py --worker http://127.0.0.1:8021 \
  --seconds 7200 > /tmp/hecate-engine.jsonl
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

`evals/deep_operations.py` requires an otherwise idle, privately owned Hecate lane.
It observes a second hold queued behind a 132K request, then tests budget/n refusal,
stream disconnect, and a five-second client timeout during forced long generation.
Both cancellation cases require an idle engine within 30 seconds and a correct
subsequent response. Use `--worker` only for read-only engine metrics and `--pool`
for the existing read-only pool snapshot; all inference still uses Gateway.

`evals/summarize_deep.py --host HOST.jsonl --results RESULTS.jsonl --out SUMMARY.jsonl`
joins raw measurements to request timestamps. Power comes from `nvidia-smi` board
power; trapezoidal integration gives sampled GPU energy (not independent of power,
and not wall-plug energy). CPU busy time comes from `/proc/stat` counter differences,
excluding duplicated guest counters; it includes unrelated host services. Gaps over
six seconds or absent boundary coverage yield null energy. GPU utilization, memory,
power and engine KV occupancy are related resource diagnostics, not independent
quality signals. These offline artifacts reuse existing measurements and introduce
no cognition metric, schema default or production telemetry channel.

The pinned Strata server has `CTX_SLACK=8` and defaults to silently shrinking
`max_tokens` near its limit. The challenger now runs setup without serving, then
merges `hecate-deep-defaults.json` with `fit_max_tokens=false` before launch.
For Strata use explicit `--context-slack 8`: its maximum tested prompt with a
16384 output allowance is 245752, or 262136 user tokens total. This is an
eight-token capacity difference from vLLM, not a 262144-token passing result.
Strata lacks generation `min_tokens`/`ignore_eos`; its pinned server declares EOS
IDs 248044 and 248046. `--endurance-eos-ban 248044 248046` uses its supported
request-scoped `logit_bias` bans to force a capacity-only run, which must still
report the full requested output count and a length stop to pass.

The grid uses actual prompts 8192/32768/65536/132000/196608/229376 with a 32768
output reserve. With reserve 16384 the last prompt is 245760. Fresh-prefix and
exact-replay modes record actual cached-token counts; replay is not assumed to
hit. TTFT includes queue/transport; prompt-per-TTFT is a proxy, not pure ingestion
speed. Repeat output includes model, usage, seeds, prompt hash, reasoning, final
answer and finish reason. Empty final, wrong answer/model, missing usage,
truncation and length-stop fail quality checks. Three distant random identifiers
and distributed inventory arithmetic must all be correct.

`--suite hard` replaces the ledger with six distant records: a keeper replacement,
an unaffected decoy, pounds/kg arithmetic, two date updates and original-keeper
recall. All five answers must match. This reproduces the reasoning families of
the historical Hecate hard suite at exact token depths. Endurance and recovery
remain separate ledger cases, even when the scored grid uses the hard suite.

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

The explicit challenger stack is
`services/orion-llamacpp-host/docker-compose.hecate-strata-eval.yml`. It uses the
existing announce-only wrapper, a separate writable state volume, the cached
model files read-only, both GPUs, CPU node 1, 262144 context, int8 KV, MTP and one
slot. This is a layer split, not tensor parallelism. CPU pinning provides local
first-touch placement; it does not enforce NUMA memory binding. Check actual
`numastat`/`numa_maps` before interpreting offload performance.

Build the announcer on Hecate and **both pool and Gateway images on Athena**
from the same reviewed worktree revision first. The pool image must contain the
new challenger entry in `llm_profiles.yaml`; mounting a candidate pool config
alone cannot make that profile discoverable. Merge the source and synchronize
clean primary main on Athena/Hecate. Reserve/drain `agent-deep`. On Athena render
the candidate from that exact main (only Hecate's backend and commissioning
owner change; both cards remain un-lent):

```bash
mkdir -p .orion
docker run --rm --network none --entrypoint python -e PYTHONPATH=/repo -v "$PWD:/repo:ro" \
  orion-gpu-pool-gpu-pool /repo/services/orion-vllm-host/scripts/stage_pool.py \
  --gpus 2 --backend strata --commissioning > .orion/hecate-eval-pool.yaml
scripts/safe_docker_build.sh orion-gpu-pool -f services/orion-gpu-pool/docker-compose.hecate-eval.yml up -d --no-build
# Wait for a quiet Gateway request boundary before restarting the shared Gateway.
scripts/safe_docker_build.sh orion-llm-gateway -f services/orion-llm-gateway/docker-compose.hecate-eval.yml up -d --no-build
```

On Hecate stop only vLLM, then start the alternate stack. The two wrappers refuse
`up` while the competing engine is running; neither stops it automatically.

```bash
services/orion-vllm-host/scripts/hecate-deep.sh stop
services/orion-llamacpp-host/scripts/hecate-strata-eval.sh up -d --no-build
```

Confirm pool discovery reports the challenger profile, both cards, one slot,
actual model file and context before releasing the maintenance hold. Benchmark
through `operator_agent_deep`; `deep.thinker.v1` intentionally refuses this
unvalidated challenger. Do not substitute its model identifier in the dense
profile. Use the Flash-Next tokenizer and verify its rendered count against
actual provider usage before scoring context capacity. Record engine versions,
cache hit/miss evidence, CPU RAM, and both cards' allocations; a populated card
alone does not prove useful expert execution.

To return to the dense baseline: reserve/drain, stop the Strata stack, start
`hecate-deep.sh up -d --no-build`, then recreate pool and Gateway from their base
Compose files without either Strata override. The generated local candidate and
Strata state volume may remain for reproducibility; neither is a default input.
Verify the original deep profile is confirmed before releasing the hold.

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
