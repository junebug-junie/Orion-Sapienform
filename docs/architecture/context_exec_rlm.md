# Context Exec RLM integration

Bounded depth-2 investigation organ supervised by Cortex. Replaces planner-react → agent-chain loops for grounded probes when `agent_runtime_engine=context_exec`.

## Flow

Note (2026-09-05): `DecisionRouter` below is confirmed unreachable from any
current Hub UI mode -- this diagram's first step does not fire in practice
today. See this change's PR description.

```
cortex-orch (DecisionRouter)
  → depth 2 + context_exec_mode
cortex-exec (Supervisor)
  → ContextExecClient RPC
orion-context-exec
  → FakeRLMEngine (default) / AlexZhangRLMEngine (opt-in via CONTEXT_EXEC_RLM_ENGINE=alexzhang)
  → AgentChainResult-compatible payload
```

## Feature flags

- `CONTEXT_EXEC_ENABLED` on cortex-exec (default false)
- Legacy fallback via `CONTEXT_EXEC_LEGACY_FALLBACK=true`
- Beta runbook: [context-exec-beta-runbook.md](../context-exec-beta-runbook.md)

## Safety

Read-only by default; sandbox mode `docker`; max depth 1.
