## Summary

- Turns on the LLM-gateway inference lane from #2327 in the operator env templates: the gateway reports its own call outcomes, the substrate reducer folds them, and the field digester puts the failure share on `capability:llm_inference`.
- `LLM_GATEWAY_GRAMMAR_ENABLED=true`, `ENABLE_LLM_INFERENCE_REDUCER=true`, `ENABLE_LLM_INFERENCE_FIELD_DIGESTION=true` in `.env_example`; local `.env` files flipped to match.
- Code defaults in `settings.py` and compose fallbacks stay `false`, so a fresh host without these env files keeps the lane off.
- Gateway README updated to say the template turns it on.

## Outcome moved

`capability:llm_inference` `reliability_pressure` gets a real input (gateway-reported backend failure share per serving node) instead of having no producer.

## Current architecture

#2327 merged with all three flags off. The prerequisite migration (`services/orion-sql-db/manual_migration_llm_inference_substrate_loop.sql`) was applied to production on 2026-09-25: table `substrate_llm_inference_projection` exists, cursor `llm_inference_grammar_reducer` seeded with a null position.

## Architecture touched

Env templates only: orion-llm-gateway, orion-substrate-runtime, orion-field-digester.

## Files changed

- `services/orion-llm-gateway/.env_example`: grammar flag on.
- `services/orion-substrate-runtime/.env_example`: reducer flag on, comment notes migration applied.
- `services/orion-field-digester/.env_example`: field digestion flag on.
- `services/orion-llm-gateway/README.md`: default wording.
- `docs/superpowers/pr-reports/2026-09-25-llm-inference-lane-flags-on-pr.md`: this report.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: the gateway starts publishing `llm_gateway.inference:` traces on `orion:grammar:event` once rebuilt with the new env.
- Compatibility notes: producer already cataloged in `orion/bus/channels.yaml` by #2327.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- Changed values: the three flags above, `false` -> `true`.
- `.env_example` updated: yes.
- local `.env` synced: yes, the three keys flipped in the primary checkout's service `.env` files; `python scripts/sync_local_env_from_example.py` exit 0.
- skipped keys requiring operator action: none.

## Tests run

```text
python scripts/check_env_template_parity.py  -> env template parity: PASS (90 service(s) compared)
git diff --check                              -> clean
```

## Evals run

```text
None: config-only change. Lane evals shipped and ran in #2327.
```

## Docker/build/smoke checks

```text
See Restart required; proof queries per stage below.
```

## Review findings fixed

- None: 4-line config flip plus doc wording; no code.

## Restart required

Run in order, each from an up-to-date main worktree, checking each proof before the next.

```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
# proof: grammar traces arriving
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*), max(emitted_at) from grammar_events where source_service='orion-llm-gateway' and emitted_at > now()-interval '5 minutes'"

scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
# proof: reducer receipts + projection row
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*), max(created_at) from substrate_reduction_receipts where reducer_name like 'llm_inference%' and created_at > now()-interval '5 minutes'"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select generated_at from substrate_llm_inference_projection"

scripts/safe_docker_build.sh orion-field-digester up -d --build
# proof: node:circe carries inference_failure_pressure (expect 0.0 at rest)
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select field_json->'node_vectors'->'node:circe'->>'inference_failure_pressure', field_json->'capability_vectors'->'capability:llm_inference'->>'reliability_pressure' from substrate_field_state order by generated_at desc limit 1"
```

## Risks / concerns

- Severity: medium. Concern: a non-zero failure share has only been proven by tests, never observed live. Mitigation: proof queries above; `make substrate-ladder-check` covers freshness.
- Severity: low. Concern: this lands the gpu-pool plan's stage-6 work before stages 3-5 (noted in #2327). Mitigation: the lane reads only `served_by`; flip any flag back to `false` and rebuild that one service to roll back.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2329
