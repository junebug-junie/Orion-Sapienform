# Orion Context Exec

Bounded depth-2 investigation organ replacing planner-react/agent-chain for grounded probes.

## HTTP

- `GET /health`
- `POST /context-exec/run` — native `ContextExecRequestV1` → `ContextExecRunV1`
- `POST /agent/chain/run` — `AgentChainRequest` compatibility shim

## Bus

- Intake: `orion:exec:request:ContextExecService` (`context.exec.request.v1`)
- Optional compat alias: `agent.chain.request` on `CONTEXT_EXEC_AGENT_CHAIN_INTAKE_ALIAS` when enabled
- Events: `orion:context_exec:event`

## Safety defaults

Read-only; no network/shell/repo writes. RLM runs via `FakeRLMEngine` unless `CONTEXT_EXEC_RLM_ENGINE` is changed.

## Local run

```bash
docker compose --env-file .env -f services/orion-context-exec/docker-compose.yml up -d --build
curl -s http://127.0.0.1:8096/health | python -m json.tool
```
