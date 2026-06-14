# Context-exec beta runtime runbook

## 1. Purpose

Context-exec is Orion's bounded recursive workbench for read-only investigation. During beta, it replaces the legacy ReAct planner / AgentChain runtime for supported modes while Cortex remains sovereign over routing, policy, identity, synthesis, and mutation gates.

This runbook answers:

- Is context-exec ready to be treated as beta-safe?
- Which modes are supported?
- Which env posture is required?
- Which tests/probes must pass?
- What is explicitly forbidden?
- How should new verbs be added?
- How does this replace AgentChain without becoming goblin soup?
- How should mutation be introduced later without violating Cortex sovereignty?

**Core stance:** Cortex remains sovereign. Context-exec becomes Orion's bounded recursive workbench. RLM is the execution pattern. Artifacts are the contract. AgentChain becomes legacy compatibility. Human/Cortex gates control mutation.

> Fallback is temporary. Artifact contracts are permanent. Context-exec is the replacement substrate. Cortex is still sovereign.

## 2. Current supported modes

Beta-safe modes (schema-valid artifacts, eval fixtures, golden probes):

| Mode | Artifact schema | Purpose |
|------|-----------------|---------|
| `belief_provenance` | `BeliefProvenanceReportV1` | Trace where a belief/claim entered Orion's runtime |
| `trace_autopsy` | `TraceAutopsyReportV1` | Explain why a correlation id failed or fell back |
| `repo_impact_analysis` | `RepoImpactAnalysisReportV1` | Grounded repo impact for a proposed change |

Other modes exist in schema (`runtime_debug`, `grammar_collision_audit`, etc.) but are **not** beta-certified in this release.

### patch_proposal (proposal-mode scaffold)

- **Purpose:** Emit a read-only patch blueprint wrapped in `ProposalEnvelopeV1`. Inner payload is `PatchProposalV1`. Context-exec may **propose** mutation; it may **not perform** mutation.
- **Input shape:** Natural-language request to diagnose a weakness and propose a patch (e.g. weak trace-autopsy synthesis).
- **Artifact type:** `ProposalEnvelopeV1` — review envelope with `proposal_type`, `title`, `summary`, `evidence`, `risk`, `requires_human_approval` (always `true`), `mutation_allowed` (always `false`), `review_status` (`draft` or `pending_review` only from context-exec), inner `artifact_type=PatchProposalV1`, and nested `artifact` payload.
- **Inner payload:** `PatchProposalV1` — `problem`, `evidence`, `files_to_change`, `proposed_change_summary`, `risk`, `tests_to_run`, `rollback_plan`, `open_questions`, `mutation_allowed` (always `false`).
- **Review lifecycle:** `draft` → `pending_review` → (`approved` | `rejected` | `superseded`) → `executed` (by Cortex/human + separate executor only; context-exec never emits `approved` or `executed`).
- **Expected behavior:** Read-only repo grounding via existing organ hooks; schema-valid envelope; `mutation_allowed=false` and `requires_human_approval=true` in artifact and `runtime_debug`; `proposal_enveloped=true` in `runtime_debug`.
- **Boundary:** Proposal artifacts are not actions. Proposal envelopes are review objects. Executors are separate. Cortex/human approval is required before mutation. Context-exec may draft proposals, but it may not approve or execute them.
- **Not beta-certified:** Eval fixtures exist; Hub/Cortex auto-routing is not wired in this release.

### memory_correction_proposal (proposal-mode scaffold)

- **Purpose:** Emit a read-only memory correction blueprint wrapped in `ProposalEnvelopeV1`. Inner payload is `MemoryCorrectionProposalV1`. Context-exec may **propose** a memory correction; it may **not perform** mutation or write to any memory backend.
- **Input shape:** Natural-language request to draft a correction for an unsupported or contradicted belief (e.g. unsupported Denver location claim).
- **Artifact type:** `ProposalEnvelopeV1` — review envelope with `proposal_type=memory_correction_proposal`, `requires_human_approval` (always `true`), `mutation_allowed` (always `false`), `review_status` (`draft` or `pending_review` only from context-exec), inner `artifact_type=MemoryCorrectionProposalV1`, and nested `artifact` payload.
- **Inner payload:** `MemoryCorrectionProposalV1` — `current_belief`, `proposed_belief`, `correction_type`, `rationale`, `supporting_evidence`, `contradicting_evidence`, `missing_evidence`, `target_memory_domains`, `risk`, `tests_to_run`, `rollback_plan`, `open_questions`, `mutation_allowed` (always `false`).
- **Review lifecycle:** Same as `patch_proposal` — context-exec never emits `approved` or `executed`; a separate executor applies corrections after Cortex/human approval.
- **Expected behavior:** Read-only recall/memory/trace grounding; schema-valid envelope; no cards/RDF/Graphiti/Chroma/SQL writes.
- **Boundary:** Memory correction proposals are not memory writes. Context-exec may draft a correction proposal, but it may not update cards, RDF, Graphiti, Chroma, SQL timeline, or any other memory backend. Cortex/human approval is required before a separate executor can apply a correction.
- **Not beta-certified:** Eval fixtures exist; Hub/Cortex auto-routing is not wired in this release.

### belief_provenance

- **Purpose:** Investigate origin and support status of a user/runtime claim.
- **Input shape:** Natural-language question about a claim; optional `answer_contract` with `requires_runtime_grounding`.
- **Artifact type:** `BeliefProvenanceReportV1` — fields include `claim`, `status`, `likely_origin`, `source_chain`, `findings`, `missing_evidence`.
- **Expected behavior:** RLM reads recall/memory/trace organs (read-only); returns schema-valid report; does not echo the question as the claim.
- **Known limitations:** Depends on real recall/trace when `CONTEXT_EXEC_FAKE_ORGANS_ENABLED=false`; sparse evidence yields `unknown`/`unsupported`.

### trace_autopsy

- **Purpose:** Autopsy a specific correlation id's failure chain.
- **Input shape:** Question referencing `corr <uuid>`; Hub probes use `CONTEXT_EXEC_PROBE_CORR_ID`.
- **Artifact type:** `TraceAutopsyReportV1` — fields include `target`, `status`, `failure_chain`, `root_cause`, `evidence`.
- **Expected behavior:** Reads trace organ; root cause must not echo the user question; missing corr → `unknown` with insufficient evidence.
- **Known limitations:** Requires real trace data for known corr; fake engine uses seeded fixtures in eval only.

### repo_impact_analysis

- **Purpose:** Repo-grounded impact analysis for a proposed architectural change.
- **Input shape:** Question about replacing/changing components; requires `read_repo` permission.
- **Artifact type:** `RepoImpactAnalysisReportV1` — fields include `proposed_change`, `affected_paths`, `breaking_surfaces`, `migration_steps`, `risk`.
- **Expected behavior:** Greps/reads repo files under `CONTEXT_EXEC_REPO_ROOT`; no invented paths.
- **Known limitations:** Fake engine greps fixed patterns; AlexZhang engine required for engine-file grounding quality bar.

## 3. Runtime architecture

### Legacy ReAct / AgentChain pattern

```
task
  → freeform thought/action loop
  → tool-ish calls
  → observe
  → next thought
  → maybe final answer
```

### Replacement context-exec pattern

```
task
  → Cortex selects bounded mode
  → context-exec runs recursive episode
  → verbs call read-only organs
  → result is schema-valid artifact
  → Cortex/Hub renders or gates next action
```

The point is to replace the planner primitive, not Cortex.

### Stack flow

```
Hub → Cortex-Orch (DecisionRouter, mode selection)
    → Cortex-Exec (Supervisor, depth-2 routing)
    → ContextExecService (RLM engine + read-only organs)
    → schema-valid artifact → Hub render
```

See also: [context_exec_rlm.md](./architecture/context_exec_rlm.md).

## 4. AgentChain replacement goal

Context-exec is intended to replace the legacy ReAct planner / AgentChain runtime.

During beta, AgentChain may remain as an explicit fallback, but it should not be treated as the long-term planning substrate.

### Migration rules

- Cortex remains sovereign over routing, policy, identity, and mutation gates.
- Context-exec becomes the bounded recursive execution workbench.
- ReAct/AgentChain becomes legacy compatibility only.
- All successful context-exec work returns schema-valid artifacts, not freeform agent transcripts.
- Mutation must be proposed through artifacts and approved by Cortex/human gates.
- AgentChain fallback must be visible in `runtime_debug` and eventually disabled by default.

### Deprecation target

1. Make context-exec primary for supported modes.
2. Add eval/golden coverage for each replacement mode.
3. Disable legacy fallback by default.
4. Convert remaining AgentChain call sites to context-exec modes.
5. Retain `AgentChainService` as a compatibility shim only if needed.
6. Remove or archive AgentChain runtime once parity is proven.

## 5. Required env posture

### orion-context-exec (beta-safe defaults)

```bash
CONTEXT_EXEC_ENABLED=true
CONTEXT_EXEC_SANDBOX_MODE=docker
CONTEXT_EXEC_MAX_DEPTH=1
CONTEXT_EXEC_WRITE_ENABLED=false
CONTEXT_EXEC_NETWORK_ENABLED=false
CONTEXT_EXEC_COMPAT_AGENT_CHAIN_ENABLED=false
CONTEXT_EXEC_FAKE_ORGANS_ENABLED=false
CONTEXT_EXEC_REAL_TRACE_ENABLED=true
CONTEXT_EXEC_REAL_RECALL_ENABLED=true
CONTEXT_EXEC_REAL_REPO_ENABLED=true
```

### RLM engine selection

| Value | Role |
|-------|------|
| `CONTEXT_EXEC_RLM_ENGINE=fake` | Default / compatibility engine |
| `CONTEXT_EXEC_RLM_ENGINE=alexzhang` | Opt-in beta engine |
| `CONTEXT_EXEC_RLM_FALLBACK_ENABLED=true` | AlexZhang failures fall back to fake; must be visible in `runtime_debug` |

**Rules:** `fake` is default. `alexzhang` is opt-in only. Do not make AlexZhang default unless an existing env explicitly opts in. Do not remove `FakeRLMEngine`.

### Legacy fallback (cortex-exec)

```bash
# services/orion-cortex-exec/.env
CONTEXT_EXEC_LEGACY_FALLBACK=true   # current beta: allowed, must be visible
# Future target:
# CONTEXT_EXEC_LEGACY_FALLBACK=false
```

Current beta may allow legacy fallback when context-exec returns non-ok status, but it must be visible in supervisor logs / `runtime_debug` and treated as temporary. **Do not flip the default in this PR** unless tests and repo policy already support it.

### orion-cortex-exec

```bash
CONTEXT_EXEC_ENABLED=true
CONTEXT_EXEC_LEGACY_FALLBACK=true   # temporary; see above
```

## 6. Verification checklist

Before treating a deployment as beta-safe:

- [ ] `orion-context-exec` env matches section 5 (read-only, real organs, compat alias off).
- [ ] `orion-cortex-exec` has `CONTEXT_EXEC_ENABLED=true`.
- [ ] `CONTEXT_EXEC_WRITE_ENABLED=false` and `CONTEXT_EXEC_NETWORK_ENABLED=false` on health + `runtime_debug`.
- [ ] `bash scripts/context_exec_beta_gate.sh` passes (pytest + RLM evals).
- [ ] Live golden probes pass when stack is up (`--live`).
- [ ] No routing changes deployed without separate review.

## 7. Golden probe commands

Requires: context-exec `:8096`, cortex-exec + cortex-orch with `CONTEXT_EXEC_ENABLED=true`, Hub `:8080`.

```bash
bash scripts/context_exec_golden_probes.sh
```

With trace autopsy corr id:

```bash
CONTEXT_EXEC_PROBE_CORR_ID=5506f854-2606-42b5-a2ee-89775b3a8ed5 \
HUB_BASE_URL=http://127.0.0.1:8080 \
bash scripts/context_exec_golden_probes.sh
```

Probes assert Hub routing through context-exec (not chat_general stall) via `scripts/context_exec_probe_lib.py`.

## 8. Eval commands

```bash
PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/ -q

PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine fake
PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine alexzhang
```

Combined beta gate (non-live):

```bash
bash scripts/context_exec_beta_gate.sh
```

Live gate (includes golden probes):

```bash
CONTEXT_EXEC_PROBE_CORR_ID=5506f854-2606-42b5-a2ee-89775b3a8ed5 \
HUB_BASE_URL=http://127.0.0.1:8080 \
bash scripts/context_exec_beta_gate.sh --live
```

## 9. Known remaining limitations

- Only three modes are beta-certified; other schema modes are not eval/golden gated.
- Legacy AgentChain fallback still active on cortex-exec when `CONTEXT_EXEC_LEGACY_FALLBACK=true`.
- Proposal envelope scaffold (`ProposalEnvelopeV1`) is implemented for `patch_proposal` (wrapping `PatchProposalV1`) and `memory_correction_proposal` (wrapping `MemoryCorrectionProposalV1`); other proposal types remain documented only.
- AlexZhang engine requires opt-in; fake remains default.
- Repo impact quality differs between fake (pattern grep) and alexzhang (engine-file grounding).
- No write/network/shell; mutation is out of scope for beta.

## 10. Safety boundaries

### Explicitly forbidden (beta)

- Write access (`CONTEXT_EXEC_WRITE_ENABLED=true`)
- Network access (`CONTEXT_EXEC_NETWORK_ENABLED=true`)
- Shell execution
- Direct memory/repo/runtime mutation from context-exec
- Bypassing Cortex policy or Hub routing
- Treating freeform agent transcripts as success contracts
- Keeping AgentChain as permanent escape hatch without visibility

### runtime_debug must expose

- `write_enabled: false`
- `network_enabled: false`
- `engine` / `engine_selected`
- `sandbox_mode`
- `rlm_depth` ≤ configured max depth
- Legacy fallback usage when cortex-exec falls back

## 11. How to add a new verb

Checklist:

1. Define/confirm artifact schema in `orion/schemas/context_exec.py` and register in `orion/schemas/registry.py`.
2. Add mode routing only if Cortex should select it (cortex-orch / cortex-exec — separate PR).
3. Add RLM engine handling in `FakeRLMEngine` and/or `AlexZhangRLMEngine`.
4. Add read-only organ usage only (recall, trace, repo grep).
5. Add eval fixture in `app/rlm_eval_harness.py` + pytest in `tests/test_rlm_eval_fixtures.py`.
6. Add golden probe if user-visible through Hub.
7. Ensure no mutation in the verb path.
8. Add proposal artifact type if mutation is needed (proposal only — separate executor).

**Do not:**

- Add a new autonomous agent.
- Let freeform text be the success contract.
- Bypass Cortex policy.
- Mutate memory/repo/runtime directly from context-exec.
- Keep AgentChain alive as a permanent escape hatch.

## 12. How not to add a new goblin

- One mode → one artifact schema → one eval case minimum.
- No parallel planner loops inside context-exec.
- No hidden compat channels (`CONTEXT_EXEC_COMPAT_AGENT_CHAIN_ENABLED` stays false in production beta).
- No env flags that silently enable write/network.
- Every new mode needs golden or eval proof before beta certification.
- If Cortex doesn't select the mode, don't wire it to Hub auto-routing.

## 13. Future proposal-artifact path

Proposal-mode scaffold ( **partially implemented** ):

| Proposal artifact | Status | Purpose |
|-------------------|--------|---------|
| `ProposalEnvelopeV1` | Scaffold | Shared review wrapper for all proposal artifacts |
| `PatchProposalV1` | Scaffold (inner payload) | Propose code/doc patch — read-only blueprint only |
| `MemoryCorrectionProposalV1` | Scaffold (inner payload) | Propose memory correction — read-only blueprint only |
| `ProposalLedgerRecordV1` | Contract | Durable ledger row — system of record for proposals |
| `ProposalTriageDecisionV1` | Contract | Policy triage — prevents approval-queue spam |
| `ProposalReviewDecisionV1` | Contract | Human/policy review — does not execute |
| `ProposalExecutionEligibilityV1` | Contract | Future executor gate — eligibility only |
| `RuntimeConfigProposalV1` | Not implemented | Propose runtime config change |
| `TestPlanProposalV1` | Not implemented | Propose test additions |

`patch_proposal` emits `ProposalEnvelopeV1` wrapping `PatchProposalV1`. `memory_correction_proposal` emits `ProposalEnvelopeV1` wrapping `MemoryCorrectionProposalV1`. Neither mode may execute changes or write to memory backends.

### Proposal ledger and Hub attention model

**Hub is an attention router, not the system of record.**

The proposal ledger is the system of record.

Hub should display:

1. Inline proposals caused by the current conversation
2. A compact pending decisions drawer
3. Batch review views for maintenance windows

Hub should not display every stored draft. Hub should only actively surface `pending_review` proposals. Weak, duplicate, stale, unsupported, or blocked proposals should not become Juniper chores.

**Status visibility for Hub:**

| Status | Hub visibility |
|--------|----------------|
| `pending_review` | Show as decision-worthy |
| `blocked` | Optionally show in diagnostics, not active queue |
| `stored` | Hidden by default |
| `discarded` / `expired` / `superseded` | Hidden by default |
| `approved` / `rejected` / `executed` | History only |

**Lifecycle:**

```text
context-exec proposal
  → ProposalEnvelopeV1
  → ProposalLedgerRecordV1(status=stored)
  → ProposalTriageDecisionV1
  → pending_review only if attention-worthy
  → ProposalReviewDecisionV1
  → ProposalExecutionEligibilityV1
  → future executor
  → future receipt
```

Triage actions:

- `store_only`: keep proposal, do not bother human
- `promote_to_review`: make `pending_review` and show in Hub surfaces
- `block_for_evidence`: proposal needs more grounding before review
- `discard`: weak/duplicate/unsafe/noisy proposal should not be reviewed
- `supersede`: replace with newer/better proposal
- `expire`: proposal is stale

**Rule:**

- Proposal artifacts are not actions.
- Proposal envelopes are review objects.
- Executors are separate.
- Cortex/human approval is required before mutation.
- Context-exec may draft proposals, but it may not approve or execute them.
- Context-exec may **propose** mutation.
- Context-exec may **not perform** mutation.
- Cortex/human gate approves.
- A separate executor performs the action.
- Result is traced.

### Operator proposal review CLI

The proposal ledger is the system of record. Hub is an attention router. The operator CLI is the first inspectable review surface for proposal ledger records.

Not every proposal becomes a human chore. Only `pending_review` records require active attention. Stored, blocked, discarded, expired, or superseded proposals are hidden by default when filtering for operator review queues.

The CLI can list, show, triage, and review proposals. It cannot execute proposals, write memory, mutate repo files, or call Hub/Cortex/context-exec to generate new proposals (except `seed-demo`, which writes fixture records only to an explicit `--store` path).

Approval creates future execution eligibility only; it does not execute anything.

### Proposal review API (Hub-facing seam)

HTTP control-plane over the proposal ledger on orion-context-exec. **Disabled by default** (`PROPOSAL_REVIEW_API_ENABLED=false`). Hub must call this API — not read JSON ledger files directly.

| Setting | Default | Purpose |
|---------|---------|---------|
| `PROPOSAL_REVIEW_API_ENABLED` | `false` | Mount proposal review routes when `true` |
| `PROPOSAL_LEDGER_STORE_PATH` | unset | Explicit JSON store path (required when enabled) |

`/health` reports `proposal_review_api` with `enabled`, `store_configured`, `store_path_present`, `ok`, and `error`. Missing store path must not look healthy.

Approval creates execution eligibility only — no executor, no receipts, no memory/repo/runtime mutation. Context-exec cannot approve (`reviewer_id=context-exec` → 403).

```bash
export PROPOSAL_REVIEW_API_ENABLED=true
export PROPOSAL_LEDGER_STORE_PATH=/tmp/orion-proposals.json
bash scripts/proposal_review_api_smoke.sh
```

See [docs/proposal-review-api.md](proposal-review-api.md).

### Hub read-only proposal review surface

Hub exposes a read-only **Pending Decisions** panel on the main Hub tab. It calls the proposal review API only — never JSON ledger files.

| Variable | Default | Description |
|----------|---------|-------------|
| `HUB_PROPOSAL_REVIEW_ENABLED` | `false` | When true, Hub fetches pending-review proposals from context-exec. |
| `HUB_PROPOSAL_REVIEW_API_URL` | `http://orion-context-exec:8096` | Base URL for proposal review API (host dev: `http://127.0.0.1:8096`). |

Hub routes (read-only):

```text
GET /api/proposal-review/health
GET /api/proposal-review/pending?status=pending_review
GET /api/proposal-review/proposals/{proposal_id}
GET /api/proposal-review/proposals/{proposal_id}/eligibility
```

Default filter is `pending_review` only. Optional secondary filters (blocked, stored, approved/rejected history) are hidden in the UI until the operator reveals the filter control.

When disabled or upstream unavailable, Hub shows a quiet empty/unavailable state — no error spam. Hub does not POST triage/review and does not execute proposals.

Manual smoke with Hub:

```bash
rm -f /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py seed-demo --store /tmp/orion-proposals.json
PROPOSAL_REVIEW_API_ENABLED=true PROPOSAL_LEDGER_STORE_PATH=/tmp/orion-proposals.json \
  # start context-exec on :8096
HUB_PROPOSAL_REVIEW_ENABLED=true HUB_PROPOSAL_REVIEW_API_URL=http://127.0.0.1:8096 \
  # start Hub on :8080 — Pending Decisions shows pending_review demo proposal
```

### Proposal ledger intake (opt-in)

Context-exec can persist real `ProposalEnvelopeV1` outputs into the proposal ledger when explicitly configured. Ledger intake is **disabled by default**. Persistence is not execution — context-exec still cannot approve, execute, write memory, or mutate repo files.

| Setting | Default | Purpose |
|---------|---------|---------|
| `CONTEXT_EXEC_PROPOSAL_LEDGER_ENABLED` | `false` | Opt in to ledger persistence |
| `CONTEXT_EXEC_PROPOSAL_LEDGER_STORE_PATH` | unset | Explicit JSON store path (required when enabled) |
| `CONTEXT_EXEC_PROPOSAL_LEDGER_AUTO_TRIAGE` | `false` | Conservative deterministic triage after store (no LLM; never approves or executes) |

`AUTO_TRIAGE=true` is deterministic policy only. It never approves, never executes, and only selects `stored`, `blocked`, or `pending_review` after the initial quiet store. Hub attention should only turn on for `pending_review` records (`attention_required=true`).

When ledger intake is enabled without a store path, persistence is skipped and `runtime_debug` reports `proposal_ledger_error=store_path_required`. When disabled, `proposal_ledger_persisted=false` — not a failure.

Example intake + CLI inspection:

```bash
export CONTEXT_EXEC_PROPOSAL_LEDGER_ENABLED=true
export CONTEXT_EXEC_PROPOSAL_LEDGER_STORE_PATH=/tmp/orion-proposals.json

curl -sS http://127.0.0.1:8096/context-exec/run \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "memory_correction_proposal",
    "text": "Propose a memory correction for the unsupported claim that I am from Denver.",
    "max_depth": 1,
    "permissions": {
      "write": false,
      "network": false,
      "shell": false
    }
  }' | jq

PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py list \
  --status stored \
  --store /tmp/orion-proposals.json
```

Example commands:

```bash
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py seed-demo --store /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py list --status pending_review --store /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py show <proposal_id> --store /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py triage <proposal_id> --action promote_to_review --reason "identity memory correction" --store /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py review <proposal_id> --decision reject --reason "unsupported evidence" --reviewer human:june --store /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py eligibility <proposal_id> --store /tmp/orion-proposals.json
```

## 14. Deprecation path for legacy planner / AgentChain

### AgentChain retirement milestones

**Milestone 1:** context-exec primary for investigative verbs

- `belief_provenance`
- `trace_autopsy`
- `repo_impact_analysis`

**Milestone 2:** context-exec primary for proposal verbs

- `patch_proposal`
- `memory_correction_proposal`
- `test_plan_proposal`
- `runtime_config_proposal`

**Milestone 3:** legacy fallback disabled by default

- `CONTEXT_EXEC_LEGACY_FALLBACK=false`

**Milestone 4:** AgentChainService compatibility shim

- Old AgentChain requests translate into context-exec modes where possible

**Milestone 5:** legacy planner removal

- Remove/archive ReAct planner and AgentChain runtime after parity/evals/golden probes exist
