# Context-exec adoption manifest (post-#660 / #661)

## Safe deploy defaults

| Service | Key | Live value |
|---------|-----|------------|
| orion-context-exec | `CONTEXT_EXEC_COMPAT_AGENT_CHAIN_ENABLED` | **false** — do not subscribe to agent-chain bus lane while agent-chain runs |
| orion-context-exec | `CONTEXT_EXEC_FAKE_ORGANS_ENABLED` | **false** — FakeRLM may run, but memory/trace stubs return empty until real organs wired |
| orion-context-exec | `CONTEXT_EXEC_RLM_ENGINE` | `fake` |
| orion-cortex-exec | `CONTEXT_EXEC_ENABLED` | `true` only after context-exec health passes |
| orion-cortex-exec | `CONTEXT_EXEC_LEGACY_FALLBACK` | `true` until golden-path parity |

## Intake channels (no collision)

- Native: `orion:exec:request:ContextExecService` ← cortex-exec `ContextExecClient`
- Legacy HTTP only: `POST /agent/chain/run` on context-exec (local/harness)
- Legacy bus alias: **off by default** — enable only when agent-chain is stopped

## Live smoke

```bash
python3 scripts/sync_local_env_from_example.py orion-context-exec orion-cortex-exec
bash scripts/context_exec_live_smoke.sh
```

## Golden Hub probes (manual, after cortex-exec enabled)

1. Belief provenance — Denver claim origin → `context_exec_mode=belief_provenance`
2. Trace autopsy — `corr N fail open` → `trace_autopsy`
3. Repo impact — replace agent-chain → `repo_impact_analysis`

Verify in cortex-exec logs: `ContextExecService` RPC, not `AgentChainService`.

## Next PRs

- **#662** — read-only real organs (recall, traces, repo)
- **#663** — golden-path evaluation vs agent-chain
- **#664** — AlexZhangRLMEngine (not default)
