# Proposal Review API

Future Hub-facing control-plane seam over the proposal ledger. Exposes safe review operations without execution, memory writes, or repo mutation.

Hub now includes a proposal review surface (`Pending Decisions`) that calls this API. Hub can record review decisions (`approve`, `reject`, `request_changes`) through `POST /proposals/{proposal_id}/review`. Hub does not read JSON ledger files, does not own lifecycle logic, and does not triage or execute.

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

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PROPOSAL_REVIEW_API_ENABLED` | No | **`false`** | When false, proposal routes are not mounted and `/health` reports `enabled=false`. |
| `PROPOSAL_LEDGER_STORE_PATH` | **Yes** when enabled | empty | Explicit JSON ledger file path. No repo default. |

**Default-safe posture:** The API is disabled unless `PROPOSAL_REVIEW_API_ENABLED=true` **and** `PROPOSAL_LEDGER_STORE_PATH` points at an explicit JSON file. Missing store path must not look healthy.

Proposal routes return **503** when enabled but `PROPOSAL_LEDGER_STORE_PATH` is unset or the store file is malformed/invalid. When the API is disabled, routes are not mounted and return **404**.

## Health semantics

`/health` includes a `proposal_review_api` block:

| Field | Meaning |
|-------|---------|
| `enabled` | `PROPOSAL_REVIEW_API_ENABLED` |
| `store_configured` | non-empty `PROPOSAL_LEDGER_STORE_PATH` |
| `store_path_present` | configured path exists on disk |
| `ok` | enabled, configured, and store opens cleanly |
| `error` | human-readable reason when not `ok` |

Hub must call this API — not read JSON ledger files directly.

## Endpoints

Mounted only when `PROPOSAL_REVIEW_API_ENABLED=true`:

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

- Hub records review decisions only via `POST /proposals/{proposal_id}/review`
- Hub cannot triage (`POST /triage` is not exposed from Hub)
- Hub cannot execute proposals
- No executor
- No execution receipts
- No approval automation or auto-approval
- No memory writes
- No repo writes
- No runtime mutation
- Approval creates future execution eligibility only — does **not** execute
- Context-exec cannot approve proposals (`reviewer_id=context-exec` → 403)

## Operator CLI

`scripts/orion_proposal_cli.py` remains the first operator surface. The API mirrors its list/show/triage/review/eligibility behavior for future Hub integration.

## Smoke

```bash
bash scripts/proposal_review_api_smoke.sh
ORION_PY=orion_dev/bin/python bash scripts/denver_memory_correction_vertical_smoke.sh
```

### Denver memory correction vertical slice

Proves `memory_correction_proposal` → ledger intake → auto-triage `pending_review` → proposal review API → Hub Pending Decisions.

Denver smoke verifies read-only listing/detail. Hub review actions are tested separately and still do not execute or mutate memory during smoke.

```bash
ORION_PY=orion_dev/bin/python bash scripts/denver_memory_correction_vertical_smoke.sh
```

Or manually:

```bash
rm -f /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py seed-demo --store /tmp/orion-proposals.json
PROPOSAL_REVIEW_API_ENABLED=true PROPOSAL_LEDGER_STORE_PATH=/tmp/orion-proposals.json \
  PYTHONPATH=. orion_dev/bin/python -m pytest tests/services/test_proposal_review_api.py -q
```

## Verification

```bash
PYTHONPATH=. orion_dev/bin/python -m pytest tests/services/test_proposal_review_api.py -q
PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_health.py -q
PYTHONPATH=. orion_dev/bin/python -m pytest tests/scripts/test_orion_proposal_cli.py -q
```
