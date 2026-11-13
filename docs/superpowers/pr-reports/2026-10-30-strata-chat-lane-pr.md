## Summary

- circe's `chat` role (gpu0, port 8011) moves from llama.cpp + Qwen3.6-35B-A3B to Strata + Qwen3.8-Flash-Next IQ3_S on the same single V100.
- Context goes from 65,536 to 262,144 tokens, still one request at a time.
- GPU pool: a new role backend, `strata`. Discovery confirms it through Strata's llama.cpp-shaped `/props`.
- Gateway: `strata` gets OpenAI chat with template kwargs, logprobs and response_format. Schema formats are rewritten to OpenAI's named `json_schema`. It never gets llama.cpp's `/apply-template` + `/completion`.
- Compose: `atlas-chat` is now the Strata engine, under a stay-up wrapper. `atlas-chat-announce` announces it. `atlas-chat-llamacpp` is the rollback, behind profile `chat-llamacpp`.

## Outcome moved

Orion's main chat model gets about 4x the context window on the same card. In the hecate bench (PR #3015, 128K) this model scored 20/20 on the hard set at 94 tok/s average output. The 35B on circe scored 19/20 at 89 tok/s.

## Current architecture

`atlas-chat` ran `orion-llamacpp-host`, which launched llama-server and announced role `chat` on `orion:llm:worker:announce`. The pool confirmed the role by comparing the announced profile's file to `/props`.

## Architecture touched

- `orion/gpu_pool` (config, discovery)
- `services/orion-gpu-pool` (probe)
- `services/orion-llm-gateway` (backend extras, structured output)
- `services/orion-llamacpp-host` (announce-only mode, profile filter, compose, env)
- `config/gpu_pool.yaml`, `config/llm_profiles.yaml`
- harness-governor env

## Files changed

- `orion/gpu_pool/config.py`: `backend: strata`.
- `orion/gpu_pool/discovery.py`: a profile's `strata.model_file` names the file to match.
- `services/orion-gpu-pool/app/main.py`: the probe keeps the role's backend.
- `services/orion-llm-gateway/app/llm_backend.py`, `profiles.py`: `OPENAI_COMPAT_BACKENDS` adds strata. Native completion stays llamacpp-only.
- `services/orion-llm-gateway/app/structured_output.py`: `response_format_for_backend`.
- `services/orion-llamacpp-host/app/{main,settings,profiles}.py`:
  - Strata profiles load.
  - A strata profile announces only.
  - `LLM_REQUIRE_BACKEND` refuses any other backend.
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`: the engine, the announcer and the rollback service.
- `services/orion-llamacpp-host/strata/chat-defaults.json`, `scripts/strata_apply_defaults.py`: Strata starts with thinking off and the old sampling.
- `config/gpu_pool.yaml`: `chat` gets `backend: strata`.
- `config/llm_profiles.yaml`: adds `qwen38-flash-next-iq3s-strata-1xv100-circe-chat`.
- `scripts/check_fcc_context_parity.py`: reads `ATLAS_CHAT_STRATA_PROFILE_NAME` first.
- `scripts/check_circe_worker_refs.py`: new container names in the fallback list.
- Tests: pool discovery and probe, gateway payload and schema, announcer (real loader), defaults script.
- `services/orion-llamacpp-host/README.md`: runbook and rollback.

## Schema / bus / API changes

- Added: role backend value `strata` (`RoleSpec.backend`) and profile backend `strata`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Chat grants report `backend: strata`.
  - Native-completion logprob probes that land on chat use OpenAI chat logprobs instead.
- Compatibility notes:
  - The bus payload is unchanged.
  - The `RoleSpec` serializer still omits `backend` for llamacpp roles, so the pool digest of other roles is unchanged.

## Env/config changes

- Added keys (orion-llamacpp-host): `ATLAS_CHAT_STRATA_PROFILE_NAME`, `ATLAS_CHAT_STRATA_IMAGE`, `ATLAS_CHAT_STRATA_DATA_DIR`, `ATLAS_CHAT_STRATA_CPUS`.
- Changed default: `HARNESS_FCC_MAX_CONTEXT_TOKENS` goes from 65536 to 262144 (orion-harness-governor).
- `.env_example` updated: yes.
- Local `.env` synced:
  - The new keys arrive through `scripts/sync_local_env_from_example.py` on circe after merge.
  - `HARNESS_FCC_MAX_CONTEXT_TOKENS` is a value change. The sync keeps local values, so set it by hand on athena (see Restart).
- Skipped keys requiring operator action: `HARNESS_FCC_MAX_CONTEXT_TOKENS` (value), athena `services/orion-harness-governor/.env`.

## Tests run

```text
orion/gpu_pool/tests                         520 passed
services/orion-gpu-pool/tests                199 passed, 15 skipped
services/orion-llm-gateway/tests             470 passed
services/orion-llamacpp-host/tests           157 passed, 1 failed
  (the 1 failure, test_qwen3_8b_atlas_metacog_profile_q5km_single_lane_16k, also fails on main)
check_gpu_pool_config / check_circe_worker_refs / check_env_template_parity /
check_compose_no_relative_mounts / check_env_key_single_source / check_fcc_context_parity   all pass
Real-loader announcer tests fail with the old profile filter: 2 failed (verified).
Compose stay-up wrapper with a fake strata binary: server gone -> exit 1; SIGTERM -> --stop, exit 0; launch fails -> exit 1.
```

## Evals run

```text
The model eval is the hecate bench (PR #3015) at 128K:
  reasoning at depth 12/12, hard set 20/20, recall 22/22 to 75K.
256K on circe is UNVERIFIED until cutover (smoke plan below).
```

## Docker/build/smoke checks

```text
docker compose config (atlas-workers, circe-shaped env):
  atlas-chat: Strata image, port 8011->8080, gpu device 0, cpuset 0-17,36-53 (gpu0's NUMA node 0, nvidia-smi topo)
strata launch --plan on circe gpu0, 256K, IQ3_S: accepted, nothing to download
Model data on circe /mnt/telemetry/strata: byte-identical to hecate's; run config seeded with chat defaults
```

## Review findings fixed

- Finding: the announcer crashed at boot, because the host's profile filter skipped `strata`.
  - Fix: the filter accepts strata.
  - Evidence: real-loader tests fail without the fix (2 failed) and pass with it.
- Finding: circe's `ATLAS_CHAT_PROFILE_NAME` names the llama.cpp profile, so the announcer would launch llama-server or announce a mismatch.
  - Fix: the announcer uses `ATLAS_CHAT_STRATA_PROFILE_NAME`, and `LLM_REQUIRE_BACKEND=strata` refuses anything else.
  - Evidence: `test_announcer_refuses_a_llamacpp_profile`.
- Finding: `strata launch --no-tui` returns once serving, so the container would exit and restart forever.
  - Fix: a stay-up wrapper copied from the image's entrypoint loop.
  - Evidence: three-case fake-binary run.
- Finding: Strata ignored json_object schemas and returned 400 on bare json_schema.
  - Fix: `response_format_for_backend` sends OpenAI's named json_schema.
  - Evidence: `test_schema_formats_become_the_openai_wrapper_strata_accepts`.
- Finding: the docs cited a missing bench; the runbook used raw compose.
  - Fix: cite PR #3015; use `safe_docker_build.sh --compose-file`.
  - Evidence: README.
- Finding: one gateway test passed without the change.
  - Fix: removed it.

## Restart required

Order: merge -> athena pool + gateway rebuild (post-pull) -> circe cutover. Juniper approves the cutover.

```bash
# athena (primary main): set the harness fallback window
sed -i 's/^HARNESS_FCC_MAX_CONTEXT_TOKENS=.*/HARNESS_FCC_MAX_CONTEXT_TOKENS=262144/' services/orion-harness-governor/.env
# circe (primary main, after git pull): new env keys, then the cutover in services/orion-llamacpp-host/README.md "Chat lane on Strata"
python3 scripts/sync_local_env_from_example.py
```

## Risks / concerns

- Severity: medium. Concern: chat is a spill role for the agent, metacog and fast classes, and Strata serves one request at a time (as the 35B did). Mitigation: unchanged from today.
- Severity: medium. Concern: IQ3_S at 262K is not measured by upstream or by us. Mitigation: KV streaming keeps 32K on the GPU and about 3.6 GB in RAM; circe has about 190 GB available. Smoke-test one long conversation at cutover.
- Severity: low. Concern: a cold 256K prompt takes about 4-4.5 minutes to first token. Mitigation: Strata caches the conversation, so follow-up turns read only the new part.
- Severity: low. Concern: `response_format` with `tools` returns 400 on Strata for passthrough clients. The gateway's own payload never combines them.
- Severity: low. Concern: logprob-uncertainty readings from chat change source (native n_probs to OpenAI logprobs), so readings before and after are not directly comparable.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3016

🤖 Generated with [Claude Code](https://claude.com/claude-code)
