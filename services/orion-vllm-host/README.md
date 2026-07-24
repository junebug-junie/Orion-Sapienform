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
