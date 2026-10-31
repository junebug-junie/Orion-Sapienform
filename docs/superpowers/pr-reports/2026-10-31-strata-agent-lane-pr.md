## Summary

- circe's `agent` role (lane 1, gpu1, port 8015; also `memory_distill`) moves from llama.cpp + Qwen3.8-27B Q4 to Strata + Qwen3.8-Flash-Next IQ3_S, the engine and model the chat lane has run since PR #3016.
- Context is 131,072 tokens, the same as `agent-gpu2` (Bonsai, 2 x 131072), so a recalled curiosity run can resume on either lane. One request at a time, as before.
- Thinking stays on at high effort, with the 27B lane's sampling (`strata/agent-defaults.json`).
- Compose: shared `x-strata-engine` / `x-strata-announce` blocks drive both Strata lanes. `atlas-agent` is the engine (compose profile `agent-split`, as before), `atlas-agent-announce` is its announcer, and `atlas-agent-llamacpp` is the 27B rollback.
- The lane runs on the other socket's CPUs (18-35,54-71) and shares the chat lane's model files.

**Merge after #3017** (this branch is built on it).

## Outcome moved

Lane 1, and memory distill with it, gets the stronger model. In the hecate bench (PR #3015) Flash-Next scored 20/20 on the hard set at 94 tok/s. The same-harness single-V100 run for the 27B Q4 scored 20/20 at 26 tok/s.

## Current architecture

`atlas-agent` ran orion-llamacpp-host with the 27B Q4, 131072 x 1 slot, thinking on (xhigh). Since #3016 the chat lane runs Strata; the pool confirms both kinds through `/props`.

## Architecture touched

`services/orion-llamacpp-host` (compose, env, defaults, README), `config/gpu_pool.yaml`, `config/llm_profiles.yaml`, tests, `scripts/check_circe_worker_refs.py`.

## Files changed

- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`:
  - anchors for the shared engine and announcer
  - `atlas-agent` / `atlas-agent-announce` / `atlas-agent-llamacpp`
  - the wrapper reads `STRATA_PRESET` / `STRATA_CONTEXT`
- `services/orion-llamacpp-host/strata/agent-defaults.json`: thinking high, 27B sampling.
- `services/orion-llamacpp-host/.env_example`: `ATLAS_AGENT_STRATA_PROFILE_NAME`, `ATLAS_AGENT_STRATA_DATA_DIR`, `ATLAS_AGENT_STRATA_CPUS`.
- `config/gpu_pool.yaml`: `agent` gets `backend: strata`; the memory_distill comment is updated.
- `config/llm_profiles.yaml`: adds `qwen38-flash-next-iq3s-strata-1xv100-circe-agent` (131072).
- Tests:
  - agent role discovery
  - backend serialization
  - route-view golden
  - stock-image list
  - agent defaults merge
- `scripts/check_circe_worker_refs.py`: new container names in the fallback list.
- `services/orion-llamacpp-host/README.md`: agent-lane runbook and rollback.

## Schema / bus / API changes

- None. `backend: strata` already exists (#3016).

## Env/config changes

- Added keys: `ATLAS_AGENT_STRATA_PROFILE_NAME`, `ATLAS_AGENT_STRATA_DATA_DIR`, `ATLAS_AGENT_STRATA_CPUS`.
- `ATLAS_AGENT_PROFILE_NAME` keeps its meaning: the llama.cpp 27B, used by the rollback and as the gpu2 burst worker's fallback.
- Sync on circe: `python3 scripts/sync_local_env_from_example.py --all-keys orion-llamacpp-host`.

## Tests run

```text
orion/gpu_pool/tests                  521 passed
services/orion-gpu-pool/tests         199 passed, 15 skipped
services/orion-llm-gateway/tests      470 passed
services/orion-llamacpp-host/tests    158 passed, 1 failed
  (the failure, test_qwen3_8b_atlas_metacog_profile_q5km_single_lane_16k, also fails on main)
services/orion-llamacpp-bonsai-host   7 passed
services/orion-durable-runs/tests     727 passed, 1 skipped (throwaway Postgres;
  test_admission_review_regressions.py not collected locally because of missing deps, CI runs it)
gates: check_gpu_pool_config, check_circe_worker_refs, check_env_template_parity,
       check_compose_no_relative_mounts, check_env_key_single_source: pass
Wrapper with a fake strata binary: STRATA_PRESET/STRATA_CONTEXT reach the launch (--context 131072 --save-preset agent);
  SIGTERM -> launch --stop --preset agent, exit 0
```

## Evals run

```text
Model eval: hecate bench (PR #3015). Live on circe's chat lane (same model): 241K-token cold prompt, 3/3 needles.
The agent lane at 131K is UNVERIFIED until cutover.
```

## Docker/build/smoke checks

```text
docker compose config (--profile agent-split --profile agent-llamacpp):
  atlas-chat   cpuset 0-17,36-53, /mnt/telemetry/strata:/data, 8011, 262144 (unchanged)
  atlas-agent  cpuset 18-35,54-71, /mnt/telemetry/strata-agent:/data + /mnt/telemetry/strata/Strata-data, 8015, 131072
  announcers: their profiles, healthcheck to $STRATA_ENGINE
  atlas-agent-llamacpp: old container name, 27B profile
```

## Review findings fixed

- Review in progress; findings will be added here.

## Restart required

Cutover on circe, on Juniper's go: README "Agent lane on Strata". Merge order #3017, then this. On athena, the pool and gateway pick up `backend: strata` for agent on post-pull rebuild.

## Risks / concerns

- Severity: medium. Concern: RAM. A second Strata adds ~57 GB of its own memory, minus ~10 GB freed by the 27B; circe goes from ~118 GB available to ~60-70 GB. Mitigation: shared model files; watch `free -g` at cutover.
- Severity: low. Concern: curiosity runs that spill to chat (262K) and grow past 131K can't resume on lanes 1 or 2. The turn is refused, not truncated. Mitigation: September agent-lane prompts had a 24K median and 82K maximum.
- Severity: low. Concern: memory distill (shadow tables only) changes model. Mitigation: Juniper's call ("move distil over too").

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3020

🤖 Generated with [Claude Code](https://claude.com/claude-code)
