## Summary

- Add pinned 1Cat 1.5.3 / QUASAR Qwen3.8 27B deployment for Hecate's two V100s.
- Keep one full-context request resident; reuse the existing Gateway and GPU-pool queue.
- Enforce an actual rendered-token budget and reject silent truncation across supported protocols.
- Add reproducible hardware capture, memory estimates, leased retrieval/reasoning and full-output endurance/recovery tests.

## Outcome moved

Hecate's two-card deployment and deep route are concrete and testable. Production inference,
132K acceptance, full-window residency and the dual-card Strata comparison remain UNVERIFIED
until live results are attached. No benchmark speed or quality is inferred from configuration.

## Current architecture

Gateway leases a fixed `agent-deep` vLLM worker on Hecate. Before this patch it advertised
one V100, Qwen3 4B and 8192 context. Circe runs existing Strata and other workers. The
GPU pool already owns priorities, waiting, cancellation, operator eval holds and discovery.

## Architecture touched

Existing vLLM host, Gateway dispatch, GPU-pool config and shared route view. No new service,
queue authority, schema or cognition loop. Deep inference uses both cards and one slot.

## Files changed

- `config/llm_profiles.yaml`: `deep.thinker.v1` TP2 runtime profile.
- `config/gpu_pool.yaml`: second Hecate card and explicit deep route.
- `services/orion-vllm-host/`: pinned wheel image, deployment wrapper, budget gate, staged download and benchmarks.
- `services/orion-llm-gateway/app/`: deep timeout, worker identity fence, verified deep Anthropic support.
- `orion/llm/routes.py`, `orion/gpu_pool/route_view.py`: route vocabulary and inspectable budget.
- Focused service/pool tests: two-card lending, protocol dispatch, no false capacity success.
- `docs/runbooks/hecate-deep-thinking.md`: compatibility, memory, operation and rollback.
- `docs/bench/2026-11-14-hecate-deep/`: captured hardware and explicitly labeled estimates.

## Schema / bus / API changes

- Added: logical Gateway model/route `deep.thinker.v1`.
- Removed: none.
- Renamed: none.
- Behavior changed: deep worker rejects truncation and insufficient reserved output; unsupported generation APIs fail closed.
- Compatibility notes: existing routes/launch digests remain; Hecate role now needs both cards. Legacy vLLM Anthropic remains refused.

## Env/config changes

- Added keys: none in env templates.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not required; no template change.
- skipped keys requiring operator action: none.
- Bus remains the existing Tailscale address `redis://100.92.216.81:6379/0`.

## Tests run

- vLLM host: 44 passed.
- GPU-pool plus route vocabulary: 588 passed.
- Gateway: 542 passed initially, one new refusal classification failure fixed by reusing existing `gpu_pool_unavailable`; affected grammar/protocol suite then 51 passed.
- Focused pool and Compose logging checks: 64 passed.
- GPU pool static config gate and `git diff --check`: passed.
- Independent review: no remaining findings after two passes. CI pending.

## Evals run

Hardware probe ran on Hecate. Analytical memory calculator ran on the pinned checkpoint config.
Real-runtime protocol boundary eval and live inference benchmark remain pending image build.
Historical Strata measurements are labeled October30 single-card results, not new measurements.

## Docker/build/smoke checks

Hecate wheel image build in progress in an isolated worktree. Existing worker and biometrics
remain healthy. Model checkpoint downloaded after available disk/RAM and size were established.

## Review findings fixed

- Finding: Anthropic could bypass the worker budget gate.
  - Fix: cover conversion to the actual chat request; unsupported APIs fail closed.
  - Evidence: real-runtime protocol boundary eval plus Gateway protocol tests.
- Finding: tokenization could omit generation-specific template settings.
  - Fix: delegate template parameters to the actual ChatCompletionRequest.
  - Evidence: protocol eval checks reasoning_effort, tools and Anthropic history.
- Finding: a tiny answer did not prove full context residency.
  - Fix: separate full-output endurance and subsequent recovery; no quality claim for forced output.
  - Evidence: harness requires actual prompt+completion usage equal 262144.
- Finding: frozen tests assumed one Hecate GPU and no new route.
  - Fix: update fixtures and add a half-lent TP2 refusal regression.
  - Evidence: focused pool tests.

## Restart required

After review/CI and merge, from clean synchronized primary main only:

```bash
# Athena; preserve existing workers on Circe
scripts/safe_docker_build.sh orion-gpu-pool up -d --no-build
scripts/safe_docker_build.sh orion-llm-gateway up -d --no-build
# Hecate; drain agent-deep first
services/orion-vllm-host/scripts/hecate-deep.sh up -d --no-build
```

Build the corresponding images before these commands. Hecate's old main must be inspected
before synchronization; no broad post-pull rebuild is part of this deployment.

## Risks / concerns

- High: runtime capacity/quality remains UNVERIFIED until actual long requests and full-output endurance pass.
- Medium: 1Cat 1.5.3 is a newly published stable release; preserve the old image/checkpoint for rollback.
- Medium: historical Strata data cannot choose the dual-card winner; run the same quality workload on both.
- Medium: CUDA Graphs, FP8 accuracy, speculation, cache fairness and failure recovery require live evidence.
- Metrics are offline measurements only; no new cognition signal or detector is wired in.
- Issues opened/closed: none at this implementation checkpoint; remaining mission work is active, not deferred.

## PR link

Pending creation.
