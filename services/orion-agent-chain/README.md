# Orion Agent Chain

The **Agent Chain** service executes linear sequences of tool-use actions. It is invoked by Cortex Exec when a plan involves known toolsets.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-exec:request:AgentChainService` | `AGENT_CHAIN_REQUEST_CHANNEL` | `agent.chain.request` | Execution requests. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-exec:request:PlannerReactService` | `PLANNER_REQUEST_CHANNEL` | `agent.planner.request` | Sub-planning requests (optional). |
| (Caller-defined) | (via `reply_to`) | `agent.chain.result` | Execution result. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `AGENT_CHAIN_REQUEST_CHANNEL` | `orion-exec:request:AgentChainService` | Intake channel. |
| `PLANNER_REQUEST_CHANNEL` | `orion-exec:request:PlannerReactService` | Downstream planner channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-agent-chain
```
