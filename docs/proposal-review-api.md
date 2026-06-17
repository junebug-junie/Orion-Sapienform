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
  → dry-run executor produces ProposalExecutionReceiptV1 (simulated; no mutation)
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
| `store_path` | configured ledger JSON path |
| `store_path_present` | configured path exists on disk |
| `store_parent_present` | parent directory of ledger file exists |
| `store_parent_writable` | parent directory is writable (readiness signal before first write) |
| `reject_store_path` | sibling reject ledger path (`orion-proposals.reject.json`) |
| `ok` | enabled, configured, parent writable, and store opens cleanly |
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
- No real executor (dry-run receipt scaffold only)
- Dry-run execution is not real execution — it proves handoff shape only
- Dry-run receipts use `dry_run=true`, `status=simulated`, `mutation_performed=false`
- No approval automation or auto-approval
- No memory writes
- No repo writes
- No runtime mutation
- Approval creates future execution eligibility only — does **not** execute
- Dry-run does not transition proposals to `executed` or `execution_requested`
- Context-exec cannot approve proposals (`reviewer_id=context-exec` → 403)

## Dry-run execution (CLI only)

Dry-run execution produces a `ProposalExecutionReceiptV1` without mutating memory, repo, or runtime. Real executor integration remains future work.

Requirements:

- `status=approved`
- `eligibility.eligible=true`
- `execution_requested=false`

```bash
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py dry-run-execute <proposal_id> \
  --store /var/lib/orion/context-exec/ledger/orion-proposals.json \
  --executor dry-run
```

Receipt fields: `dry_run=true`, `status=simulated`, `mutation_performed=false`. Proposal status stays `approved`.

## Operator CLI

`scripts/orion_proposal_cli.py` remains the first operator surface. The API mirrors its list/show/triage/review/eligibility behavior for future Hub integration. Dry-run execution is CLI-only for now.

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

Expected: `denver_memory_correction_vertical_smoke PASS` with one `pending_review` Denver proposal, API GET checks showing `eligible=false`, `mutation_allowed=false`, `requires_human_approval=true`. Hub live GET is optional (`HUB_SMOKE=true HUB_BASE_URL=...`).

Or manually (host-run smoke paths):

```bash
rm -f /mnt/rlm-nvme/context-exec/ledger/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py seed-demo \
  --store /mnt/rlm-nvme/context-exec/ledger/orion-proposals.json
PROPOSAL_REVIEW_API_ENABLED=true \
PROPOSAL_LEDGER_STORE_PATH=/mnt/rlm-nvme/context-exec/ledger/orion-proposals.json \
  PYTHONPATH=. orion_dev/bin/python -m pytest tests/services/test_proposal_review_api.py -q
```

Container-side ledger default: `/var/lib/orion/context-exec/ledger/orion-proposals.json`.

### Migration from `/tmp`

Legacy deployments that used `/tmp/orion-proposals.json` can migrate with:

```bash
sudo mkdir -p /mnt/rlm-nvme/context-exec/ledger
sudo cp -av /tmp/orion-proposals.json /mnt/rlm-nvme/context-exec/ledger/orion-proposals.json 2>/dev/null || true
```

## Verification

```bash
PYTHONPATH=. orion_dev/bin/python -m pytest tests/services/test_proposal_review_api.py -q
PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_health.py -q
PYTHONPATH=. orion_dev/bin/python -m pytest tests/scripts/test_orion_proposal_cli.py -q
```
