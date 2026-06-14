# PR: Harden Denver vertical slice after merge

**Branch:** `feat/denver-vertical-harden`  
**Title:** Harden Denver vertical slice after merge  
**Create PR:** https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/denver-vertical-harden?expand=1

## Summary

Hardens the merged Denver memory-correction vertical slice.

This verifies the end-to-end read-only path from context-exec proposal generation through proposal ledger intake, deterministic auto-triage, proposal review API exposure, and Hub Pending Decisions display.

## Changes

- Strengthen `assert_denver_vertical_slice_safety` with inner-artifact checks (belief, correction type, rationale, risk, confidence).
- Expand smoke script assertions: envelope/inner types, safety flags, eligibility `eligible=false`, richer stdout.
- Add `_GetOnlyClientSession` GET-only HTTP guard in Hub proposal review client (blocks POST/PUT/PATCH/DELETE).
- Add Hub tests: GET path allowlist, non-GET method rejection, stronger Denver card + read-only UI assertions.
- Strengthen UI wiring test for evidence/risk/confidence fields and absence of action buttons.
- Document expected smoke output in runbook, proposal-review-api, and Hub README.

## Flow

```text
memory_correction_proposal (Denver)
  → ProposalEnvelopeV1
  → ProposalLedgerRecordV1
  → auto-triage pending_review
  → GET /proposals (proposal review API)
  → Hub Pending Decisions (read-only)
```

## Safety

- Read-only Hub surface
- No approve/reject/triage actions
- No executor
- No execution receipts
- No memory/repo/runtime mutation
- Hub performs GET only
- `mutation_allowed=false`, `requires_human_approval=true`, eligibility `eligible=false` before approval

## Verification

| Command | Exit | Result |
|---------|------|--------|
| `ORION_PY=orion_dev/bin/python bash scripts/denver_memory_correction_vertical_smoke.sh` | 0 | PASS |
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_proposal_ledger_intake.py tests/services/test_proposal_review_api.py services/orion-hub/tests/test_proposal_review_hub.py services/orion-hub/tests/test_proposal_review_ui.py -q` | 0 | 56 passed |
| `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine fake` | 0 | all cases ok |
| `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine alexzhang` | 0 | all cases ok |

## Test plan

- [ ] Run smoke script from clean main — one Denver `pending_review` proposal
- [ ] Confirm API GET checks: `eligible=false`, safety flags set
- [ ] Hub Pending Decisions shows Denver card with belief/correction/rationale/evidence/risk — no action buttons
- [ ] Confirm Hub client rejects POST to `/triage` and `/review`
