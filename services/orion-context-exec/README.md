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

**Bus alias safety:** `CONTEXT_EXEC_COMPAT_AGENT_CHAIN_ENABLED` defaults to **false**. Keep it false while `orion-agent-chain` is running — use native `ContextExecService` intake only. HTTP `/agent/chain/run` remains for local harness tests.

**Fake evidence:** `CONTEXT_EXEC_FAKE_ORGANS_ENABLED=false` (default) means FakeRLM runs but memory/trace stubs return empty until real organs are wired (#662).

## Local run

```bash
docker compose --env-file .env -f services/orion-context-exec/docker-compose.yml up -d --build
curl -s http://127.0.0.1:8096/health | python -m json.tool
```
