# Route finalizers to agent and preserve reading JSON

## Status

DONE

## Summary

- Route `harness_finalize_reflect` and `orion_voice_finalize` explicitly to the physical `agent` route and logical agent lane, with chat fallback disabled.
- Enable gateway lane routing by default while preserving unlabeled callers through the legacy body-route behavior.
- For trusted `reading_only` Stage 1/2 turns, validate and canonicalize the motor JSON instead of sending it through prose-oriented voice finalization.
- Keep ordinary turn behavior backward compatible: `preserve_structured_output` defaults false and the existing 5c voice pass still runs.

## Runtime evidence that motivated the patch

- The 24-hour gateway trace contained 31 completed chat-route calls: 14 `orion_voice_finalize`, 13 `harness_finalize_reflect`, and 4 `stance_react`.
- Reading request `f3283057-e329-59e6-8ce9-72287aa58510` reached Stage 2 but failed after valid structured output was rewritten as prose by voice finalization.
- Production behavior after deployment is UNVERIFIED; this PR only builds images and does not restart services.

## Contract and compatibility

- No bus or schema shape changes.
- `HarnessRunRequestV1.reading_only` is the existing trusted switch; the governor maps it to the new optional finalize-chain argument.
- Invalid structured output fails through the existing finalize failure-artifact path rather than returning empty-shell success.
- `LLM_LANE_ROUTING_ENABLED=false` remains the rollback to body-route-only behavior.
- `LLM_LANE_DEFAULT=chat` matches the settings default and preserves an explicit body route when lane metadata is absent.

## Tests and evals

- Harness focused tests: 8 passed.
- Harness governor tests: 14 passed.
- Cortex route/lane tests: 44 passed.
- Gateway lane tests: 11 passed.
- Reading Stage 1/2 tests and eval: 51 passed.
- Full harness suite: 323 passed, 3 pre-existing failures reproduced on current `main` (two `mind_coloring` fixture omissions and one FCC error-code assertion).
- Env template parity and compose env reachability: passed.
- Python compile and `git diff --check`: passed.
- Docker builds: `orion-llm-gateway`, `orion-cortex-exec`, and `orion-harness-governor` passed.
- Graphify safe update: 77,966 -> 77,975 nodes; accepted within the 10% safety threshold.

## Review findings fixed

- Finding: `LLM_LANE_DEFAULT=quick` was inconsistent with the gateway's logical lane vocabulary and code default.
  - Fix: changed the env contract and main's ignored local env to `chat`.
  - Evidence: added a gateway regression test proving an unlabeled request keeps its explicit body route.

Independent review reported no remaining material correctness or security findings.

## Deployment

After merge, rebuild/restart these services from the primary checkout using the safe wrapper:

```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

Then verify `/routes`, one ordinary unified turn, and one reading Stage 1/2 run. The reading trace must contain `harness_structured_output_preserved`, return schema-valid JSON, and contain no `orion_voice_finalize` RPC for that correlation ID.
