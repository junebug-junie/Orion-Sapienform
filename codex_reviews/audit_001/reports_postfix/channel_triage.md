# Channel triage (heuristic)

| channel | classification | evidence |
|---|---|---|
| `${CORTEX_GATEWAY_REQUEST_CHANNEL:-orion-cortex-gateway:request}` | `unknown` | services/orion-cortex-gateway/docker-compose.yml:24 |
| `${CORTEX_LOG_CHANNEL}` | `unknown` | services/orion-cortex-exec/docker-compose.yml:31 |
| `${CORTEX_REQUEST_CHANNEL:-orion-cortex:request}` | `unknown` | services/orion-spark-introspector/docker-compose.yml:31 |
| `${ORCH_REQUEST_CHANNEL:-orion-cortex:request}` | `unknown` | services/orion-cortex-gateway/docker-compose.yml:21, services/orion-cortex-orch/docker-compose.yml:35 |
| `${STATE_REQUEST_CHANNEL:-orion-state:request}` | `unknown` | services/orion-cortex-orch/docker-compose.yml:25 |
| `${VOIP_BUS_COMMAND_CHANNEL:-orion:voip:command}` | `unknown` | services/orion-voip-endpoint/docker-compose.yml:44 |
| `${VOIP_BUS_STATUS_CHANNEL:-orion:voip:status}` | `unknown` | services/orion-voip-endpoint/docker-compose.yml:45 |
| `orion:*` | `pattern` | orion/cognition/hub_gateway/bus_harness.py:91 |
