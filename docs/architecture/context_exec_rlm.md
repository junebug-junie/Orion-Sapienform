# Context Exec RLM integration

Bounded depth-2 investigation organ supervised by Cortex. Replaces planner-react → agent-chain loops for grounded probes when `agent_runtime_engine=context_exec`.

## Flow

```
cortex-orch (DecisionRouter)
  → depth 2 + context_exec_mode
cortex-exec (Supervisor)
  → ContextExecClient RPC
orion-context-exec
  → FakeRLMEngine (default) / future AlexZhangRLMEngine
  → AgentChainResult-compatible payload
```

## Feature flags

- `CONTEXT_EXEC_ENABLED` on cortex-exec (default false)
- Legacy fallback via `CONTEXT_EXEC_LEGACY_FALLBACK=true`

## Safety

Read-only by default; sandbox mode `docker`; max depth 1.
