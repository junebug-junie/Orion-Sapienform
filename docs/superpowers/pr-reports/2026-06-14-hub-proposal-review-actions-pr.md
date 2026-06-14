# PR: Add Hub proposal review actions without execution

**Branch:** `feat/hub-proposal-review-actions`  
**Title:** Add Hub proposal review actions without execution

## Summary

Adds Hub proposal review actions without execution.

Hub can now approve, reject, or request changes for pending proposals through the proposal review API. Approval creates future execution eligibility only; it does not execute, mutate memory, mutate repo files, or create execution receipts.

## Changes

- Extend Hub proposal review client with POST allowlist for `/proposals/{proposal_id}/review` only (no triage, execute, or arbitrary POST).
- Add Hub route `POST /api/proposal-review/proposals/{proposal_id}/review` with server-side rationale validation.
- Add Pending Decisions detail actions: **Approve**, **Reject**, **Request changes** (required rationale; optional constraints on approve).
- After action: refresh detail, refresh pending list, show status update.
- Add six safety-focused Hub tests plus test isolation fix for combined pytest runs.
- Update docs: Hub records review decisions; Hub cannot execute; approval creates eligibility only.

## Architecture

```text
Hub UI (Pending Decisions detail card)
  → POST /api/proposal-review/proposals/{id}/review
  → proposal_review_client (POST allowlist: /proposals/{id}/review only)
  → context-exec POST /proposals/{id}/review
  → JsonFileProposalLedgerRepository (context-exec only — Hub never reads/writes ledger files directly)
```

## Safety

- No executor
- No execution receipts
- No memory/repo/runtime mutation
- Hub only records review decisions via review API
- Hub does not POST triage
- Approval only creates future execution eligibility (`eligible=true`, `execution_requested=false`)

## Verification

| Command | Exit | Result |
|---------|------|--------|
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-hub/tests/test_proposal_review_hub.py services/orion-hub/tests/test_proposal_review_ui.py tests/services/test_proposal_review_api.py -q` | 0 | 34 passed |
| `ORION_PY=orion_dev/bin/python bash scripts/denver_memory_correction_vertical_smoke.sh` | 0 | PASS |
| `orion_dev/bin/python -m compileall services/orion-hub/scripts/proposal_review_client.py services/orion-hub/scripts/proposal_review_routes.py -q` | 0 | OK |

## Manual smoke (operator)

1. Seed Denver proposal (`denver_memory_correction_vertical_smoke.sh` or CLI seed-demo).
2. Start context-exec with proposal review API enabled.
3. Start Hub with `HUB_PROPOSAL_REVIEW_ENABLED=true`.
4. Open Hub Pending Decisions → Denver detail.
5. Reject with rationale → card leaves pending list or status updates to `rejected`.
6. Repeat with approve → `eligible=true`, no execution receipt, no memory mutation.

## Test plan

- [ ] Hub review buttons visible on pending_review detail card
- [ ] Empty rationale rejected (client + server 422)
- [ ] Approve sets `status=approved`, `eligible=true`, `execution_requested=false`
- [ ] Reject sets `status=rejected`, `eligible=false`
- [ ] Request changes sets `status=request_changes`, `eligible=false`
- [ ] Network tab shows POST to `/api/proposal-review/proposals/{id}/review` only (no `/triage`, no execute)
- [ ] Denver smoke still passes
