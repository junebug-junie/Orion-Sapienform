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
| GET | `/health` | Service health + `proposal_review_api` store status + bus dependency readiness (`recall`, `llm_gateway`) |
| GET | `/proposals` | List ledger records (`?status=` optional) |
| GET | `/proposals/{proposal_id}` | Proposal detail + review history + eligibility |
| POST | `/proposals/{proposal_id}/triage` | Apply `ProposalTriageDecisionV1` |
| POST | `/proposals/{proposal_id}/review` | Apply `ProposalReviewDecisionV1` |
| GET | `/proposals/{proposal_id}/eligibility` | Inspect `ProposalExecutionEligibilityV1` |

Routes mount only when `PROPOSAL_REVIEW_API_ENABLED=true`. `/health` always reports `proposal_review_api.enabled`, `store_configured`, `store_path`, `store_path_present`, `store_parent_present`, `store_parent_writable`, `reject_store_path`, `ok`, and `error`, plus a `storage` block for mounted arena paths.

Approval creates execution eligibility only — no executor, no receipts, no auto-approval. Context-exec cannot approve (`reviewer_id=context-exec` → 403). Hub must call this API, not read JSON ledger files.

```bash
export PROPOSAL_REVIEW_API_ENABLED=true
export PROPOSAL_LEDGER_STORE_PATH=/var/lib/orion/context-exec/ledger/orion-proposals.json
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

**Proposal ledger intake (opt-in):** Context-exec can persist `ProposalEnvelopeV1` outputs as `ProposalLedgerRecordV1` when explicitly enabled. Disabled by default (`CONTEXT_EXEC_PROPOSAL_LEDGER_ENABLED=false`). When enabled, `CONTEXT_EXEC_PROPOSAL_LEDGER_STORE_PATH` must point at a JSON file under the mounted storage arena (default `/var/lib/orion/context-exec/ledger/orion-proposals.json` inside the container). Ledger persistence is not execution. Context-exec still cannot approve or execute. Optional `CONTEXT_EXEC_PROPOSAL_LEDGER_AUTO_TRIAGE` applies conservative deterministic triage (no LLM): it may leave proposals quietly stored, block for evidence, or promote to `pending_review` — it never approves, requests execution, or mutates anything. Runtime debug includes `proposal_id`, `ledger_status`, and `triage_action` when persisted.

```bash
export CONTEXT_EXEC_PROPOSAL_LEDGER_ENABLED=true
export CONTEXT_EXEC_PROPOSAL_LEDGER_STORE_PATH=/var/lib/orion/context-exec/ledger/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py list --status stored --store /var/lib/orion/context-exec/ledger/orion-proposals.json
```

**Engine selection:** `fake` (default) | `alexzhang` (opt-in). Fallback must be explicit (`CONTEXT_EXEC_RLM_FALLBACK_ENABLED=true`) and visible in `runtime_debug` (`engine_requested`, `engine_selected`, `fallback_used`, `fallback_reason`).

**LLM profile / route binding:** Hub Agent mode passes `llm_profile` (`chat`, `quick`, `agent`, `metacog`) on `ContextExecRequestV1`. Context-exec resolves the profile to a trusted gateway route id, records `llm_profile_requested`, `llm_profile_selected`, and `route_used` in `runtime_debug`, and binds RLM `llm.subcall` / bus RPC to that route. Invalid profile ids are rejected at schema validation. Route unavailability fails closed unless `CONTEXT_EXEC_LLM_PROFILE_FALLBACK_ENABLED=true` (records `fallback_used` + `fallback_reason`). Profile selection does not change permissions (read-only, proposal-gated).

**Agent synthesis pass (route-bound):** After deterministic organ collection, supported Agent modes (`belief_provenance`, `trace_autopsy`, `repo_impact_analysis`, `patch_proposal`, `memory_correction_proposal`) may run a read-only synthesis pass via `CONTEXT_EXEC_AGENT_SYNTHESIS_ENABLED` (default `true`). Synthesis consumes the deterministic artifact only — no shell, writes, memory mutation, or proposal execution. It uses the selected `route_used` for LLM RPC. When synthesis is unavailable or ungrounded, deterministic output is preserved and `model_synthesis_used`, `fallback_used`, and `fallback_reason` are recorded explicitly. `operator_summary` on `ContextExecRunV1` exposes mode, route, synthesis status, proposal id/status, and safety posture for Hub.

**Investigation v2:** When `CONTEXT_EXEC_INVESTIGATION_V2_ENABLED=true`, requests use neutral `mode=investigation_v2` with profile-derived permissions (Hub compute lane / agent compat `response_profile`). The evidence sweep fans out read-only probes (repo, traces, recall, memory, runtime, health) with per-source status isolation; reducers build `InvestigationReportV2` sections and a deterministic summary before optional route-bound LLM synthesis. Synthesis failure preserves deterministic evidence and records a limitation (`synthesis_unavailable`). Before recall RPC, context-exec runs a fast bus NUMSUB preflight on `CHANNEL_RECALL_INTAKE`; dead consumers mark recall unavailable without waiting for recall timeout. Before LLM synthesis, context-exec runs the same preflight on `CHANNEL_LLM_INTAKE`; dead gateway consumers skip the LLM RPC and return a deterministic report with `LLM synthesis unavailable: LLM gateway bus consumer not ready`. `GET /health` exposes aggregate `bus_consumer_ready` plus per-dependency subscriber counts when the bus is enabled. Health probe metadata includes recall/LLM gateway readiness snapshots when the bus is connected. Configure per-probe timeout via `CONTEXT_EXEC_INVESTIGATION_V2_PROBE_TIMEOUT_SEC` and readiness timeout via `CONTEXT_EXEC_BUS_READINESS_TIMEOUT_SEC`. Default flag is `false` (legacy keyword mode inference preserved).

**Safety defaults:** read-only, `CONTEXT_EXEC_MAX_DEPTH=1`, write/network off, compat AgentChain bus alias off.

## Storage arena

Durable state lives on a host-mounted NVMe path mapped into the container at `/var/lib/orion/context-exec`. The app never references host device paths — only container paths (`CONTEXT_EXEC_STORAGE_ROOT`, sub-roots, and proposal ledger JSON). Docker compose binds `${CONTEXT_EXEC_HOST_STORAGE_ROOT:-/mnt/rlm-nvme/context-exec}` → `/var/lib/orion/context-exec`. `GET /health` includes a `storage` block (`configured`, `ok`, `dirs`, `run_ledger_enabled`). Startup calls `ensure_storage_dirs()` to create the sub-directory skeleton when the mount is writable.

**Run ledger (forensic bundles):** When `CONTEXT_EXEC_RUN_LEDGER_ENABLED=true` (default), every completed run writes an immutable evidence bundle to `{CONTEXT_EXEC_RUN_ROOT}/{run_id}/` (e.g. `/var/lib/orion/context-exec/runs/ctxrun_abc123/`): `manifest.json` (`schema=orion.context_exec.run_ledger.v1`), `run.json` (full serialized `ContextExecRunV1`), `final.md`, and — when present — `artifact.json`, `runtime_debug.json`, `verb_trace.json`, and the redacted `request.json`. Writes are atomic (`*.tmp` → `fsync` → `os.replace`) and persistence is **fail-open**: a ledger write failure logs a warning and never fails the run. Secret-named keys (`token`, `secret`, `password`, `authorization`, `api_key`, `apikey`, `key`) are redacted to `[REDACTED]` across all persisted payloads (best-effort, key-name based — free text is not scrubbed). This PR persists evidence only — no workspaces, no repo writes, no agent-behavior changes.

**Verification:**

Run from the **repository root** (the smoke ladder requires a `.git` directory; git worktrees with a `.git` file pointer must use the main checkout or run the individual steps below).

```bash
# Context-exec beta gate only (pytest + RLM eval fake/alexzhang)
ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh

# Full proposal-control spine + beta gate (schema → CLI → ledger → review API → Hub → RLM evals → Denver vertical → CLI approve/eligibility/dry-run → beta gate)
ORION_PY=orion_dev/bin/python \
STORE=/var/lib/orion/context-exec/ledger/orion-proposals.json \
./scripts/repl/orion_fresh_main_smoke.sh

# Live Hub golden probes (stack required)
bash scripts/context_exec_golden_probes.sh

# Hub Agent mode route probe (stack required)
HUB_BASE_URL=http://127.0.0.1:8080 AGENT_ROUTE=chat \
  AGENT_TEXT="Where did the Denver belief come from?" \
  bash scripts/context_exec_agent_route_probe.sh
```

**AgentChain replacement goal:** Context-exec is the bounded recursive workbench intended to replace the legacy ReAct planner / AgentChain runtime. Cortex remains sovereign; artifacts are the contract. AgentChain is legacy compatibility only during beta (`CONTEXT_EXEC_LEGACY_FALLBACK` on cortex-exec).

## Local run

```bash
docker compose --env-file .env -f services/orion-context-exec/docker-compose.yml up -d --build
curl -s http://127.0.0.1:8096/health | python -m json.tool
```
