## Summary

- Add exact-depth reasoning and operational failure benchmarks for Hecate.
- Record passing 8K/132K ledger requests and sampled resource use through Gateway.
- Cancel upstream HTTP work when a granted caller disconnects before a reply.
- Disable Strata output clamping and record its eight-token internal reserve explicitly.

## Outcome moved

132000 input tokens are now verified with correct retrieval/arithmetic, exact provider
usage, and a 16384-token output allowance. A real-socket regression exposed a Gateway
disconnect gap after admission; the fix closes upstream before releasing its lease.
The larger context/quality/comparison runs remain active work, not completed claims.

## Current architecture

Existing Gateway and GPU pool control the fixed dual-V100 Hecate worker. The prior
patches deployed pinned 1Cat and prepared a cached Strata challenger. Commissioning
ownership keeps Hecate private to background operator leases while Circe stays live.

## Architecture touched

Gateway HTTP passthrough cleanup and worker offline evaluation scripts only. No new
control-plane service, bus event, default routing change or production metric.

## Files changed

- `deep_context.py`: six distributed records with updates and objectively scored reasoning.
- `deep_operations.py`: queue, unsafe-request refusal, disconnect and recovery checks.
- `sample_engine.py`, `summarize_deep.py`: raw gauges and sampled energy/resource reports.
- Gateway passthrough, lifecycle tests and README: disconnect cleanup after grant.
- Dated benchmark artifacts and Hecate runbook: live evidence and reproduction steps.

## Schema / bus / API changes

No schema or bus changes. HTTP client disconnect after grant now stops upstream and
releases as `cancelled/http_client_disconnected`; Gateway records 499. Successful
response semantics and stream ownership are unchanged.

## Env/config changes

No env keys or templates changed; local env synchronization is not required. The
existing Tailscale bus URL remains `redis://100.92.216.81:6379/0`.

## Tests run

- Gateway: 545 passed, including two real-socket regressions that failed before the fix.
- Worker: 52 tests passed; reviewer checked hard-suite gold answers over 1000 seeds.
- Resource-summary mathematics: four tests passed.
- Strata defaults: eight tests passed; pinned setup CLI options verified from its help.
- Independent reviews: no remaining findings; diff and syntax checks passed.

## Evals run

Fresh/cached 8192 and 132000 input tokens passed through Gateway. Cold 132K took
98.08s total, TTFT 86.00s, decode 42.79 tokens/s; replay TTFT 2.57s. Both GPUs
reached 100% utilization; sampled peak memory 28699/28693 MiB. Sampled GPU board
energy was 52.08 kJ cold and 5.98 kJ replay, not wall-plug energy. Full hard grid,
262144-token endurance, operational live tests and Strata comparison are in progress.

## Docker/build/smoke checks

Gateway image rebuilt successfully on Athena. Deployment
must use clean synchronized primary main after merge. Current live baseline remains
healthy and its active benchmark must finish before the approved maintenance restart.

## Review findings fixed

The timeout benchmark initially allowed natural completion to look like cancellation.
It now requires observed worker activity, forces 16384 output tokens, and demands idle
within 30 seconds plus a correct recovery request. Energy gaps produce null, not zero.
Pinned Strata source revealed default output clamping and eight internal context tokens;
setup now merges the explicit no-clamp policy before serving, and benchmark rows record
the reserve. Its capacity-only run uses supported EOS logit bans rather than unsupported
generation min_tokens/ignore_eos options.

## Restart required

From clean synchronized Athena primary main, after the Hecate benchmark exits:

```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --no-build --timeout 30
```

Juniper explicitly approved a brief shared Gateway maintenance restart. Circe model
workers are not restarted. Hecate stays warm throughout this Gateway update.

## Risks / concerns

- Larger context, Strata and live cancellation checks remain active work.
- The synthetic suites are limited evidence, not a comprehensive ranking of reasoning.
- Resource counters are sampled; CPU includes unrelated host work. Host memory bandwidth
  and NVLink byte counters are unavailable with current counter access.
- Issues opened/closed: none; remaining mission work is continuing in this session.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3360
