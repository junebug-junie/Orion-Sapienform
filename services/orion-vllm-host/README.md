# orion-vllm-host

A thin wrapper around `vllm.entrypoints.openai.api_server`, resolving model + GPU config from
`llm_profiles.yaml` (`VLLM_PROFILE_NAME`) or a direct `VLLM_MODEL_ID` override before launching
the real vLLM OpenAI-compatible server subprocess.

## Usage

```bash
docker compose -f services/orion-vllm-host/docker-compose.yml up -d
```

Health: `GET http://localhost:${VLLM_HOST_PORT:-7000}/health`

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), on its own independent bus connection, separate from
the vLLM server subprocess this service launches. See
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.

## Hecate GLM 5.3 Flash (staged)

`glm-5.3-flash-exl3-hecate` is a launch profile for a **custom, pinned 1Cat-vLLM
build**, not this service's legacy `vllm/vllm-openai:v0.6.0` Docker image.
See [the build and activation runbook](../../docs/runbooks/hecate-glm53-vllm.md).
The active GPU pool configuration is unchanged. GPU inference is **UNVERIFIED**.

The profile's `vllm` block supplies serving arguments and a small allow-list of
engine environment settings. `pool_discovery: true` loads
`app.discovery.pool_server_info` through vLLM's `--middleware` option. Its
read-only `GET /orion/server-info` exposes only initialized engine model path,
context limit, scheduler sequence limit and parallel layout. It does not enable
developer mode; the fork's built-in `/server_info` is otherwise hidden behind it.

`LLM_ROLE`, `LLM_ANNOUNCE_HOST`, and `LLM_ANNOUNCE_PORT` identify a pool role and
its externally reachable host/port. Role/port are disabled by default. Setting
either requires both and an explicit `VLLM_PROFILE_NAME`, checked before startup.
The worker publishes the existing `llm.worker.announce.v1` every 30 seconds;
the pool requires live HTTP facts to agree before making grants.

CPU checks (from repository root):

```bash
PYTHONPATH=.:services/orion-vllm-host python -m pytest services/orion-vllm-host/tests -q
python services/orion-vllm-host/scripts/stage_pool.py > /tmp/hecate-pool.candidate.yaml
```

`evals/smoke_glm.py` is an opt-in inference acceptance check against a real
endpoint; a passing CPU test is not evidence of model quality or GPU readiness.
