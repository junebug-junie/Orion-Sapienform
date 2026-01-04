# Orion Cortex Exec

**Cortex Exec** is the execution engine for the Cognitive Runtime. It receives a `PlanExecutionRequest` from the Orchestrator, decomposes it into steps (using Planner/Agents), and coordinates workers (LLM, Recall, Tools) to fulfill the request.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-cortex-exec:request` | `CHANNEL_EXEC_REQUEST` | `cortex.exec.request` | Request from Orchestrator. |

### Published Channels (Worker RPC)
Exec sends requests to these channels and listens for replies on ephemeral `reply_to` channels.

| Target Worker | Channel (Env Var) | Default Value | Kind |
| :--- | :--- | :--- | :--- |
| **LLM Gateway** | `CHANNEL_LLM_INTAKE` | `orion-exec:request:LLMGatewayService` | `llm.chat.request` |
| **Recall** | `CHANNEL_RECALL_INTAKE` | `orion-exec:request:RecallService` | `recall.query.request` |
| **Planner** | `CHANNEL_PLANNER_INTAKE` | `orion-exec:request:PlannerReactService` | `agent.planner.request` |
| **Agent Chain** | `CHANNEL_AGENT_CHAIN_INTAKE` | `orion-exec:request:AgentChainService` | `agent.chain.request` |
| **Council** | `CHANNEL_COUNCIL_INTAKE` | `orion:agent-council:intake` | `council.request` |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_EXEC_REQUEST` | `orion-cortex-exec:request` | Main input. |
| `CHANNEL_LLM_INTAKE` | ... | LLM worker channel. |
| `CHANNEL_RECALL_INTAKE` | ... | Recall worker channel. |
| `CHANNEL_PLANNER_INTAKE` | ... | Planner worker channel. |
| `CHANNEL_AGENT_CHAIN_INTAKE` | ... | Agent Chain worker channel. |
| `CHANNEL_COUNCIL_INTAKE` | ... | Council worker channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-cortex-exec
```

### Smoke Test
Exec is tested via the Orchestrator flow.
```bash
python scripts/bus_harness.py brain "plan a party"
```
