## Summary

Hecate now has verified 262144-token capacity through Orion Gateway: 245760 prompt
plus 16384 generated tokens, followed by a correct recovery request. The dense
baseline passed all 36 distributed-context checks, all five operational checks,
and all 12 runs of four task-quality problems.

DFlash2 and Strata comparisons remain in progress; production promotion is pending.

## Architecture and changes

The existing Gateway, GPU pool and resident worker remain the control path. This
patch records benchmark evidence, pairs reasoning facts across Strata's eight-token
internal context reserve, and extends the offline sampler to capture existing
Strata cache/offload facts and vLLM draft counters. No new control-plane service,
cognitive metric, bus event or routing default is introduced.

## Tests and runtime evidence

- Context harness: 12 focused tests passed, including paired boundary seeds.
- Engine sampler: two focused tests passed; independent review found no issues.
- All CI checks passed at commit 75caa7cc8; later commits require their own CI pass.
- Full baseline results, exact outputs, provider usage and sampled resource costs:
  `docs/bench/2026-11-14-hecate-deep/RESULTS.md` and neighboring JSONL artifacts.
- Gateway cancellation fix from PR #3360 is live. Streaming cancellation returned
  to idle within the first 0.11-second sample; timed-out active nonstreaming work
  returned to idle after 1.36 seconds. Both subsequent answers were correct.
- Baseline model remained warm from 21:10 UTC through all baseline tests; no
  preemptions or CUDA OOMs were observed. Circe workers and biometrics were unchanged.

## Measurement provenance and limits

These are offline performance observations, not cognitive detector inputs.

1. GPU samples come from `sample_host.py:sample` invoking NVIDIA's existing NVML
   query through `nvidia-smi`; CPU comes from `/proc/stat`. Provider usage and
   stream timing come from `deep_context.py:stream_chat`. Strata's pinned
   `serve/server.py:Service.metrics` exposes engine/live state and numeric history;
   its history producer derives hit rate from engine hits/lookups and PCIe share
   from offloaded/routed counts. vLLM draft counters originate in its pinned
   `vllm/v1/spec_decode/metrics.py`.
2. Energy is causally derived from power; throughput from tokens and time; cache
   rates from the same engine's counts. None is an independent quality signal.
3. Energy uses the physical definition integral(P dt), with trapezoidal sampled
   integration. CPU busy fraction uses counter differences. Cache/draft ratios
   are empirical event frequencies, not theory of cognition or reasoning scores.
4. Live baseline observations vary with actual work and return to zero active
   requests/utilization at idle. Idle board power remains about 119 W combined;
   prefix-cache allocation can remain nonzero without active requests. Missing
   bandwidth/counter data stays null. Strata/DFlash live sanity checks are pending.
5. Existing engine and host instruments are reused; no production signal is added.
6. Samplers and benchmark artifacts are removable independently of cognition,
   schema defaults, routing and training. They feed no detector.

## Config, deployment and rollback

No env template changes. Ignored worker env files exist locally with the Tailscale
bus URL and were synchronized. The optional DFlash profile/Compose override and
isolated Strata stack were prepared in earlier merged PRs. Live activation must
use clean synchronized primary main and the existing operator maintenance hold.
The dense baseline image/model are preserved for rollback. See the runbook.

## Risks and remaining work

- Candidate comparisons and production promotion are still pending.
- Four task-quality problems and synthetic distributed reasoning are limited
  evidence; a perfect score does not establish broad reasoning superiority.
- Energy is GPU board energy, not wall-plug energy. Whole-host CPU includes other
  processes. NVLink byte counters and host memory-bandwidth counters are unavailable.
- Deliberate CUDA exhaustion and multi-request full-context batching were not run;
  admission is initially one request and tested invalid allocations fail closed.
- Graphify refresh is unavailable on this host; existing graphs were not modified.
- Issues opened/closed: none; remaining mission work is continuing here.

## PR

https://github.com/junebug-junie/Orion-Sapienform/pull/3361
