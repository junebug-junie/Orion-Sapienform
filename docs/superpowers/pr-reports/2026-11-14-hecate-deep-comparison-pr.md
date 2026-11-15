## Summary

Prepare an isolated Strata challenger using cached Flash-Next IQ3_S and the existing announce-only wrapper. A temporary Hecate ownership configuration lets background operator benchmarks run while both cards remain un-lent, preventing production spill onto an unvalidated model.

## Outcome moved

The dense baseline from PR #3358 is merged and healthy on both Hecate V100s. Startup and actual runtime protocol checks are recorded. No inference-capacity or quality result is claimed yet; 132K/256K requests and comparison remain UNVERIFIED.

## Current architecture

Existing GPU pool and Gateway own admission and routing. Hecate is a fixed resident at port 8021. The alternate Strata stack replaces that worker during commissioning and announces the same role; explicit matching pool/Gateway overrides select its protocol. No new control-plane service.

## Architecture touched

Existing worker Compose configuration, candidate-config generator, model profile and benchmark evidence. Circe classes, routes and workers remain unchanged. The temporary owner is the existing operator_agent_deep class, with normal background/heartbeat/heat policy.

## Files changed

- Strata standalone Compose, existing announce-only wrapper image and Hecate startup guard.
- Matching optional pool/Gateway candidate-config mounts.
- stage_pool.py backend/commissioning switches and scheduler regression.
- Hecate candidate model profile, runbook and startup evidence.

## Schema / bus / API changes

No new event or payload. No default route change. The deep.thinker.v1 identity gate continues refusing the unvalidated Strata challenger; comparisons use the existing operator_agent_deep route.

## Env/config changes

No .env_example changes or new env-template keys. Local env sync is not required for this patch. Candidate YAML is generated into ignored .orion state, never applied implicitly. Bus remains redis://100.92.216.81:6379/0. Base Compose restores normal ownership/backend.

## Tests run

- Worker tests and candidate scheduler contract: 47 passed.
- Existing wrapper announcement/Strata defaults: 14 passed.
- Compose journald checks: 2 passed.
- Pool config static validation and git diff --check passed.
- Independent review: no remaining findings after read-only-root and rebuild-prerequisite fixes.

## Evals run

Pinned 1Cat real protocol boundary check and actual CLI parser passed. Live tokenizer alignment matches 8192, 132000, 229376 and 245760 tokens after fixing the installed Transformers dictionary return default. Its live startup reports TP2, one slot and 262144 configured context. Allocated KV capacity is a runtime sizing report, not request acceptance. Actual long-context, task-quality and Strata inference tests remain pending.

## Docker/build/smoke checks

1Cat and shared pool deployed from clean primary main after PR #3358. No quiet boundary appeared in 15 minutes; Juniper approved a brief shared Gateway maintenance restart. The separate Strata announcer image built and its real profile import passed; alternate Compose validates. Strata engine startup is UNVERIFIED.

## Review findings fixed

- CI requires explicit journald settings even in override files and absolute host mounts; both were corrected.
- Transformers returns BatchEncoding by default: benchmark requests token IDs explicitly, with a regression and live alignment evidence.
- Read-only data root prevented Strata startup: keep the data root writable and bind cached models/packs/mtp children read-only. Cached pack prerequisites and matching draft vocabulary were inspected in the pinned image.
- Candidate profile must exist in the pool image: require rebuilding both pool and Gateway before applying backend overrides.
- Commissioning isolation: scheduler regression proves operator-only use of un-lent Hecate and preserved heat shedding.

## Restart required

Build pool/Gateway and announcer from this revision, merge, then use clean synchronized primary main and the runbook's explicit commissioning commands. Dense ownership isolation only needs a pool override. The Strata protocol switch requires matching pool and Gateway overrides. Reserve/drain Hecate before each engine swap; startup wrappers refuse simultaneous engines.

## Risks / concerns

- Actual 132K/256K inference and quality comparison remain UNVERIFIED; mission work continues.
- Gateway is shared and continuously busy; a brief maintenance restart is explicitly approved.
- CPU pinning is not strict NUMA memory binding; measure residency and offload impact.
- Existing cached Strata artifacts must be reused successfully in live preparation.
- Issues opened/closed: none; remaining work is active, not deferred.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3359
