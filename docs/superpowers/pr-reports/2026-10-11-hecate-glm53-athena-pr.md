## Summary

- Prepare Athena to discover and route to a Hecate vLLM worker through the existing GPU pool.
- Forward the server's exact model ID on bus/OpenAI calls, including spills and streaming.
- Add a staged GLM profile, read-only engine discovery, worker announcements and a candidate pool renderer.
- Pin and assemble the custom upstream source stack; keep active Hecate/Circe routing unchanged.
- Add CPU regression tests, CI coverage, a real-inference smoke and a deployment runbook.

## Outcome moved

Athena's discovery/dispatch path can now represent a vLLM worker without
pretending it is llama.cpp. Tested negative cases refuse capacity instead of
claiming readiness. Hecate GPU compilation and inference remain **UNVERIFIED**.
This is the requested Athena preparation phase, not a live model cutover.

## Current architecture

Athena's GPU pool placed every LLM request; the gateway dispatched the grant.
Both assumed llama.cpp. Existing Hecate inventory had one V100 and a llama.cpp
`agent-deep` role. The old vLLM wrapper used a stock image and had no pool
announcement. Settings for these services live in `app/settings.py`.

## Architecture touched

Shared pool config/discovery/route view, pool HTTP probing, gateway dispatch,
and the existing vLLM wrapper. No new service, cognition loop or metric.
The staged source uses the fork's supported middleware hook rather than
turning on its developer APIs.

## Files changed

- `orion/gpu_pool/{config,discovery,route_view}.py`: explicit backend and verified engine facts.
- `orion/schemas/gpu_pool.py`, `orion/bus/channels.yaml`: contract descriptions and producer registration.
- `services/orion-gpu-pool/app/main.py`: vLLM HTTP discovery.
- `services/orion-llm-gateway/app/{main,llm_backend,pool_placement,passthrough_proxy,anthropic_passthrough}.py`: dispatch, exact alias, retries and catalog.
- `services/orion-vllm-host/app/{main,settings,discovery}.py`: launch, announcements and read-only endpoint.
- `config/llm_profiles.yaml`: unselected GLM launch profile.
- `services/orion-vllm-host/{.env_example,docker-compose.yml,requirements.txt}`: announcement and middleware requirements.
- `services/orion-vllm-host/scripts/`: revision lock, source assembly and candidate rendering.
- `services/orion-vllm-host/evals/smoke_glm.py`: opt-in synthetic inference acceptance.
- Tests under shared pool and all three services: discovery, dispatch, streaming, startup, announcement and preparation.
- `.github/workflows/orion-gpu-pool-tests.yml`: host tests on CPU and path triggers.
- Service READMEs and `docs/runbooks/hecate-glm53-vllm.md`: rollout and rollback.

## Schema / bus / API changes

- Added: optional role `backend: vllm`; read-only `GET /orion/server-info` middleware.
- Removed: none.
- Renamed: none.
- Behavior changed: vLLM `model_file` is its exact served ID; `model_path` is the loaded path.
- Compatibility notes: no bus wire fields/kinds added; registry unchanged. Default llama.cpp backend is omitted during serialization. Older strict config readers must be updated before activating a vLLM role. Existing active role configuration is unchanged.
- Announcement producer list now includes `orion-vllm-host` using the existing registered kind.
- Anthropic requests receiving a vLLM grant return a clear refusal and release the lease. The fork has an Anthropic endpoint; this integration is unvalidated. Vision is also unvalidated.

## Env/config changes

- Added keys: `LLM_ROLE`, `LLM_ANNOUNCE_HOST`, `LLM_ANNOUNCE_PORT` on vLLM host.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes; role and port disabled by default.
- local `.env` synced: ran `python3 scripts/sync_local_env_from_example.py`, then the service-specific `--all-keys` sync against the primary checkout.
- Skipped keys: host-specific `ORION_BUS_URL` was missing and manually synced to `redis://100.92.216.81:6379/0`. No changed key remains unsynced. Unrelated pre-existing env warnings were not folded into this patch.
- `.env` remains ignored and uncommitted.

## Tests run

```text
Gateway tests + existing eval tests: 434 passed.
Pool core + service tests: 609 passed, 15 Postgres-dependent skips.
vLLM host tests: 13 passed.
Focused real loopback HTTP: 5 passed, including bus response identity and SSE lease cleanup.
Shared schema registry: 2 passed. Pool config gate, vLLM env parity and compose parity: PASS.
```

The pool suite's database tests require `GPU_POOL_TEST_POSTGRES_URI`; local
skips are reported explicitly, with Postgres coverage also running in CI.

## Evals run

```text
services/orion-gpu-pool/evals/run_pool_day_eval.py: VERDICT PASS.
services/orion-gpu-pool/evals/run_controller_stale_eval.py: PASS (incident alert and retryable-refusal cases).
```

The new GLM inference eval was not run against a model: Hecate is offline.
Its refusal of empty/reasoning-only/wrong-model output is unit tested. CPU
fixtures and policy replay are not claims about GLM quality.

## Docker/build/smoke checks

```text
Athena gateway Docker image: built successfully.
Athena GPU-pool Docker image: built successfully.
vLLM compose config: validated; legacy stock image is not the GLM image.
Source stack: all seven locked commits verified and assembled on Athena.
Source commit: 58531acc450ca0226f71fd1756e3357c7897c479
Source tree: c5838487a6c58c079cc9e86f3fa391ba5ff1cca0
Local source: /mnt/scripts/hecate-glm53-source-20261011
Pinned prepare_source.py --resume: passed with source-tree verification.
Live gateway /health read: status ok; no live services restarted.
Hecate GPU build, memory fit, direct inference and live pool traffic: UNVERIFIED.
```

## Review findings fixed

- Finding: upstream `/server_info` needs global developer mode.
  - Fix: selectively expose minimal initialized engine facts through middleware.
  - Evidence: endpoint tests cover not-initialized 503, live values, GET-only and absent developer endpoints.
- Finding: bad announcement settings could disappear into a detached task.
  - Fix: validate the full tuple before starting anything.
  - Evidence: incomplete settings fail before heartbeat/server startup.
- Finding: retry payload could retain a previous vLLM alias; catalog claimed every route was llama.cpp.
  - Fix: rebuild each dispatch body and report backend from role configuration.
  - Evidence: dispatch suite and independent re-review.
- Independent requesting-code-review skill verdict: **ready to merge as staged preparation**, no remaining material findings. GPU execution explicitly excluded from the review verdict.

## Restart required

No restart required for this staged preparation. No active role is changed.
Future activation, after Hecate hardware/smokes and reader upgrades are verified:

```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

Use the runbook for the separate Hecate foreground launch and fleet upgrade
order. These commands have not been run against live services.

## Risks / concerns

- Severity: deployment blocker.
- Concern: repository inventory records one Hecate V100; the candidate assumes four 32 GB V100s. Hecate is offline, so hardware, CUDA kernels and model output are unverified.
- Mitigation: bring Hecate online, inventory it, then compile and smoke before activating candidate routing. Any Circe GPU moves need their own capacity decision.
- Anthropic and images remain deferred; the current candidate must not be presented as validated support for those paths.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2610
