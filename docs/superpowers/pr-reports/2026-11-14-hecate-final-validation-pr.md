## Summary

Hecate now has verified 262144-token capacity through Orion Gateway: 245760 prompt
plus 16384 generated tokens, followed by a correct recovery request. The dense
baseline passed all 36 distributed-context checks, all five operational checks,
and all 12 runs of four task-quality problems.

DFlash2 passed the same 36/36 context grid, the endurance run with recovery and
12/12 task-quality runs, with 2.2-4x decode (104-190 vs 38-48 tokens/s) and
unchanged prefill. This PR promotes it: `deep.thinker.v1` now carries the DFlash2
`speculative_config`, and the commissioning-only overlay (`stage_dflash.py`,
`docker-compose.dflash-eval.yml`) is retired. The Strata comparison stays open; it
should run against a harder, discriminating task set, since both current suites
are saturated.

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

No env template changes. After merge, deploy from clean synchronized primary main:

1. Hecate: `hecate-deep.sh build`, then `hecate-deep.sh up -d --no-build`. The
   image bakes in `config/llm_profiles.yaml`, so the build is what picks up
   DFlash2. Confirm with `docker top orion-hecate-vllm-host -o args`, which must show
   `--speculative-config`. Then delete `.orion/hecate-dflash-profiles.yaml`.
2. Athena: recreate `orion-gpu-pool` from its base Compose file only, dropping
   `docker-compose.hecate-eval.yml`, so `agent-deep` returns to owner `agent-deep`
   and Orion's agent/metacog/fast classes can use the lane.

Rollback to the dense baseline: delete the `speculative_config` block and redeploy
step 1 (build, then up). The model files are unchanged.

## Review findings fixed

- Finding: deploy/rollback used `up --no-build`, but the image bakes in the profile,
  so DFlash2 would not start (and a rollback would not stop it).
  - Fix: build before up in the runbook and this report, plus a process-args check.
- Finding: source-text filler containing special-token strings (`<|im_end|>`) could
  inject chat-role markers into the prompt.
  - Fix: decode filler with `skip_special_tokens=True`; test covers it.
- Finding: retired overlay file left on Hecate. Fix: runbook says to delete it.
- Not changed: re-tokenization drift at filler slice boundaries. The builder already
  re-counts every attempt and raises instead of returning a wrong count.

## Risks and remaining work

- Strata comparison is still pending.
- DFlash2 decode speed depends on draft acceptance and varies with the text.
  Reasoning length differed between engines on the same correct answers.
- DFlash2 completed-task energy and the source-text distractor run were not measured.
- Four task-quality problems and synthetic distributed reasoning are limited
  evidence; a perfect score does not establish broad reasoning superiority.
- Energy is GPU board energy, not wall-plug energy. Whole-host CPU includes other
  processes. NVLink byte counters and host memory-bandwidth counters are unavailable.
- Deliberate CUDA exhaustion and multi-request full-context batching were not run;
  admission is initially one request and tested invalid allocations fail closed.
- Graphify refresh is unavailable on this host; existing graphs were not modified.
  Follow-up: https://github.com/junebug-junie/Orion-Sapienform/issues/3363.
- Issues opened: #3363 (Graphify storage/tooling). Closed: none. Project-board
  placement is unavailable through the connected GitHub tools; `gh` is absent
  locally and on Athena. Codex session ran out; DFlash2 evidence recovered from Hecate and finished by Claude.

## PR

https://github.com/junebug-junie/Orion-Sapienform/pull/3361
