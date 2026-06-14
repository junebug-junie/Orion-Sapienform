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

Read-only; no network/shell/repo writes. RLM runs via `FakeRLMEngine` by default (`CONTEXT_EXEC_RLM_ENGINE=fake`). Opt in to `AlexZhangRLMEngine` with `CONTEXT_EXEC_RLM_ENGINE=alexzhang` (falls back to fake when `CONTEXT_EXEC_RLM_FALLBACK_ENABLED=true`).

**Bus alias safety:** `CONTEXT_EXEC_COMPAT_AGENT_CHAIN_ENABLED` defaults to **false**. Keep it false while `orion-agent-chain` is running — use native `ContextExecService` intake only. HTTP `/agent/chain/run` remains for local harness tests.

**Fake evidence:** `CONTEXT_EXEC_FAKE_ORGANS_ENABLED=false` (default) means FakeRLM runs but memory/trace stubs return empty until real organs are wired (#662).

## Beta runtime

Context-exec is in **beta** for three investigative modes. Full runbook: [docs/context-exec-beta-runbook.md](../../docs/context-exec-beta-runbook.md).

**Supported beta modes:** `belief_provenance`, `trace_autopsy`, `repo_impact_analysis` → artifacts `BeliefProvenanceReportV1`, `TraceAutopsyReportV1`, `RepoImpactAnalysisReportV1`.

**Proposal-mode scaffold (not beta-certified):** `patch_proposal` → `ProposalEnvelopeV1` wrapping `PatchProposalV1`. It may emit structured patch blueprints inside a review envelope; it may not execute changes, write files, or mutate runtime state. Proposal artifacts are not actions. Proposal envelopes are review objects. Executors are separate. Cortex/human approval is required before mutation. Context-exec may draft proposals, but it may not approve or execute them.

**Proposal ledger (system of record):** `ProposalLedgerRecordV1` stores proposals after context-exec drafts them. Triage via `ProposalTriageDecisionV1` prevents approval-queue spam — only `pending_review` proposals require Hub attention. Review via `ProposalReviewDecisionV1` creates execution eligibility (`ProposalExecutionEligibilityV1`) but does not execute. Hub is an attention router, not the system of record or an infinite approval queue.

**Lifecycle:** context-exec proposal → `ProposalEnvelopeV1` → `ProposalLedgerRecordV1(status=stored)` → `ProposalTriageDecisionV1` → `pending_review` (only if attention-worthy) → `ProposalReviewDecisionV1` → `ProposalExecutionEligibilityV1` → future executor → future receipt.

**Engine selection:** `fake` (default) | `alexzhang` (opt-in). Fallback must be explicit (`CONTEXT_EXEC_RLM_FALLBACK_ENABLED=true`) and visible in `runtime_debug`.

**Safety defaults:** read-only, `CONTEXT_EXEC_MAX_DEPTH=1`, write/network off, compat AgentChain bus alias off.

**Verification:**

```bash
bash scripts/context_exec_beta_gate.sh
bash scripts/context_exec_golden_probes.sh   # live Hub stack required
```

**AgentChain replacement goal:** Context-exec is the bounded recursive workbench intended to replace the legacy ReAct planner / AgentChain runtime. Cortex remains sovereign; artifacts are the contract. AgentChain is legacy compatibility only during beta (`CONTEXT_EXEC_LEGACY_FALLBACK` on cortex-exec).

## Local run

```bash
docker compose --env-file .env -f services/orion-context-exec/docker-compose.yml up -d --build
curl -s http://127.0.0.1:8096/health | python -m json.tool
```
