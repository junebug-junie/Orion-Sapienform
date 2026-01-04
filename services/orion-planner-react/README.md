# Orion Planner (ReAct)

The **Planner React** service implements a ReAct (Reasoning + Acting) loop to break down high-level goals from Cortex Exec into actionable steps. It utilizes the LLM Gateway for reasoning.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-exec:request:PlannerReactService` | `PLANNER_REQUEST_CHANNEL` | `agent.planner.request` | Planning requests from Exec. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-exec:request:LLMGatewayService` | `CHANNEL_LLM_INTAKE` (implicit) | `llm.chat.request` | Reasoning calls to LLM. |
| (Caller-defined) | (via `reply_to`) | `agent.planner.result` | Execution plan returned to Exec. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `PLANNER_REQUEST_CHANNEL` | `orion-exec:request:PlannerReactService` | Intake channel. |
| `CORTEX_REQUEST_CHANNEL` | `orion-cortex:request` | Optional upstream channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-planner-react
```

### Smoke Test
```bash
python scripts/bus_harness.py agent "plan a trip to mars"
```
