# Orion Context Exec

Bounded depth-2 investigation organ replacing planner-react/agent-chain for grounded probes.

## HTTP

- `GET /health`
- `POST /context-exec/run` — native `ContextExecRequestV1` → `ContextExecRunV1`
- `POST /agent/chain/run` — `AgentChainRequest` compatibility shim

## Proposal review API (Hub-facing seam)

Control-plane HTTP surface over the proposal ledger. **Disabled by default** (`PROPOSAL_REVIEW_API_ENABLED=false`). Enable only with an explicit `PROPOSAL_LEDGER_STORE_PATH` (JSON file path; no repo default).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service health + `proposal_review_api` store status block |
| GET | `/proposals` | List ledger records (`?status=` optional) |
| GET | `/proposals/{proposal_id}` | Proposal detail + review history + eligibility |
| POST | `/proposals/{proposal_id}/triage` | Apply `ProposalTriageDecisionV1` |
| POST | `/proposals/{proposal_id}/review` | Apply `ProposalReviewDecisionV1` |
| GET | `/proposals/{proposal_id}/eligibility` | Inspect `ProposalExecutionEligibilityV1` |

Routes mount only when `PROPOSAL_REVIEW_API_ENABLED=true`. `/health` always reports `proposal_review_api.enabled`, `store_configured`, `store_path_present`, `ok`, and `error`.

Approval creates execution eligibility only — no executor, no receipts, no auto-approval. Context-exec cannot approve (`reviewer_id=context-exec` → 403). Hub must call this API, not read JSON ledger files.

```bash
export PROPOSAL_REVIEW_API_ENABLED=true
export PROPOSAL_LEDGER_STORE_PATH=/tmp/orion-proposals.json
bash scripts/proposal_review_api_smoke.sh
```

See [docs/proposal-review-api.md](../../docs/proposal-review-api.md).

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

**Proposal-mode scaffold (not beta-certified):** `patch_proposal` → `ProposalEnvelopeV1` wrapping `PatchProposalV1`; `memory_correction_proposal` → `ProposalEnvelopeV1` wrapping `MemoryCorrectionProposalV1`. These modes may emit structured proposals inside a review envelope; they may not execute changes, write files, update memory backends, or mutate runtime state. Memory correction proposals are not memory writes. Context-exec may draft a correction proposal, but it may not update cards, RDF, Graphiti, Chroma, SQL timeline, or any other memory backend. Cortex/human approval is required before a separate executor can apply a correction. Proposal artifacts are not actions. Proposal envelopes are review objects. Executors are separate. Cortex/human approval is required before mutation. Context-exec may draft proposals, but it may not approve or execute them.

**Proposal ledger (system of record):** `ProposalLedgerRecordV1` stores proposals after context-exec drafts them. Triage via `ProposalTriageDecisionV1` prevents approval-queue spam — only `pending_review` proposals require Hub attention. Review via `ProposalReviewDecisionV1` creates execution eligibility (`ProposalExecutionEligibilityV1`) but does not execute. Hub is an attention router, not the system of record or an infinite approval queue.

**Lifecycle:** context-exec proposal → `ProposalEnvelopeV1` → `ProposalLedgerRecordV1(status=stored)` → `ProposalTriageDecisionV1` → `pending_review` (only if attention-worthy) → `ProposalReviewDecisionV1` → `ProposalExecutionEligibilityV1` → future executor → future receipt.

**Operator CLI:** `scripts/orion_proposal_cli.py` is the first operator review surface. It lists/shows ledger records, applies triage/review via lifecycle validators, prints execution eligibility, and supports dry-run execution receipts (`dry-run-execute`). Dry-run is not real execution — it produces a simulated `ProposalExecutionReceiptV1` with `mutation_performed=false` and does not mutate memory/repo/runtime. It does not execute proposals or change proposal status to `executed`. Use an explicit `--store /path/to/proposals.json` path (never repo files). Only `pending_review` records require active human attention by default; stored/blocked/discarded/expired/superseded proposals stay out of the default review queue.

**Proposal ledger intake (opt-in):** Context-exec can persist `ProposalEnvelopeV1` outputs as `ProposalLedgerRecordV1` when explicitly enabled. Disabled by default (`CONTEXT_EXEC_PROPOSAL_LEDGER_ENABLED=false`). When enabled, `CONTEXT_EXEC_PROPOSAL_LEDGER_STORE_PATH` must be set to an explicit JSON path (e.g. `/tmp/orion-proposals.json`) — no repo-local defaults. Ledger persistence is not execution. Context-exec still cannot approve or execute. Optional `CONTEXT_EXEC_PROPOSAL_LEDGER_AUTO_TRIAGE` applies conservative deterministic triage (no LLM): it may leave proposals quietly stored, block for evidence, or promote to `pending_review` — it never approves, requests execution, or mutates anything. Runtime debug includes `proposal_id`, `ledger_status`, and `triage_action` when persisted.

```bash
export CONTEXT_EXEC_PROPOSAL_LEDGER_ENABLED=true
export CONTEXT_EXEC_PROPOSAL_LEDGER_STORE_PATH=/tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py list --status stored --store /tmp/orion-proposals.json
```

**Engine selection:** `fake` (default) | `alexzhang` (opt-in). Fallback must be explicit (`CONTEXT_EXEC_RLM_FALLBACK_ENABLED=true`) and visible in `runtime_debug` (`engine_requested`, `engine_selected`, `fallback_used`, `fallback_reason`).

**LLM profile / route binding:** Hub Agent mode passes `llm_profile` (`chat`, `quick`, `agent`, `metacog`) on `ContextExecRequestV1`. Context-exec resolves the profile to a trusted gateway route id, records `llm_profile_requested`, `llm_profile_selected`, and `route_used` in `runtime_debug`, and binds RLM `llm.subcall` / bus RPC to that route. Invalid profile ids are rejected at schema validation. Route unavailability fails closed unless `CONTEXT_EXEC_LLM_PROFILE_FALLBACK_ENABLED=true` (records `fallback_used` + `fallback_reason`). Profile selection does not change permissions (read-only, proposal-gated).

**Safety defaults:** read-only, `CONTEXT_EXEC_MAX_DEPTH=1`, write/network off, compat AgentChain bus alias off.

**Verification:**

Run from the **repository root** (the smoke ladder requires a `.git` directory; git worktrees with a `.git` file pointer must use the main checkout or run the individual steps below).

```bash
# Context-exec beta gate only (pytest + RLM eval fake/alexzhang)
ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh

# Full proposal-control spine + beta gate (schema → CLI → ledger → review API → Hub → RLM evals → Denver vertical → CLI approve/eligibility/dry-run → beta gate)
ORION_PY=orion_dev/bin/python \
STORE=/tmp/orion-proposals.json \
./scripts/repl/orion_fresh_main_smoke.sh

# Live Hub golden probes (stack required)
bash scripts/context_exec_golden_probes.sh
```

**AgentChain replacement goal:** Context-exec is the bounded recursive workbench intended to replace the legacy ReAct planner / AgentChain runtime. Cortex remains sovereign; artifacts are the contract. AgentChain is legacy compatibility only during beta (`CONTEXT_EXEC_LEGACY_FALLBACK` on cortex-exec).

## Local run

```bash
docker compose --env-file .env -f services/orion-context-exec/docker-compose.yml up -d --build
curl -s http://127.0.0.1:8096/health | python -m json.tool
```
