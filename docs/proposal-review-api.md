# Proposal Review API

Future Hub-facing control-plane seam over the proposal ledger. Exposes safe review operations without execution, memory writes, or repo mutation.

## Flow

```text
context-exec emits ProposalEnvelopeV1
  → proposal ledger stores ProposalLedgerRecordV1
  → deterministic triage marks stored / blocked / pending_review
  → review API exposes pending_review and detail surfaces
  → operator or future Hub records review decisions
  → future executor may consume approved proposals only later
```

## Service

Mounted on **orion-context-exec** (same port as context-exec HTTP, default `8096`).

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `PROPOSAL_REVIEW_API_ENABLED` | No (default `true`) | Feature flag for the review API block in `/health` |
| `PROPOSAL_LEDGER_STORE_PATH` | **Yes** for proposal routes | Explicit JSON ledger file path. No repo default. |

Proposal routes return **503** when `PROPOSAL_LEDGER_STORE_PATH` is unset or the store file is malformed.

## Endpoints

```text
GET  /health
GET  /proposals
GET  /proposals/{proposal_id}
POST /proposals/{proposal_id}/triage
POST /proposals/{proposal_id}/review
GET  /proposals/{proposal_id}/eligibility
```

### List proposals

```bash
curl -s "http://127.0.0.1:8096/proposals?status=pending_review"
```

### Triage

```bash
curl -s -X POST "http://127.0.0.1:8096/proposals/{id}/triage" \
  -H 'Content-Type: application/json' \
  -d '{"action":"promote_to_review","rationale":"identity memory correction"}'
```

### Review

```bash
curl -s -X POST "http://127.0.0.1:8096/proposals/{id}/review" \
  -H 'Content-Type: application/json' \
  -d '{"decision":"approve","rationale":"bounded and reversible","reviewer_type":"human","reviewer_id":"june"}'
```

### Eligibility

```bash
curl -s "http://127.0.0.1:8096/proposals/{id}/eligibility"
```

## Safety

- No Hub UI (API only)
- No executor
- No execution receipts
- No approval automation or auto-approval
- No memory writes
- No repo writes
- No runtime mutation
- Approval creates future execution eligibility only
- Context-exec cannot approve or execute proposals

## Operator CLI

`scripts/orion_proposal_cli.py` remains the first operator surface. The API mirrors its list/show/triage/review/eligibility behavior for future Hub integration.

## Verification

```bash
PYTHONPATH=. orion_dev/bin/python -m pytest tests/services/test_proposal_review_api.py -q
PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/ -q
```
