# PR: Add Hub read-only proposal review surface

**Branch:** `feat/hub-proposal-review-surface`  
**Title:** Add Hub read-only proposal review surface

## Summary

Adds a read-only Hub proposal review surface.

Hub can now display decision-worthy proposal records from the proposal review API without reading JSON ledger files or owning lifecycle logic. This is a read-only attention surface: it lists pending-review proposals, shows proposal details, and displays execution eligibility. It does not approve, reject, triage, or execute anything.

## Changes

- Add Hub proposal review configuration (`HUB_PROPOSAL_REVIEW_ENABLED=false` default, `HUB_PROPOSAL_REVIEW_API_URL`, timeout).
- Add read-only Hub client/proxy (`proposal_review_client.py`) — GET only with path allowlist.
- Add Hub routes under `/api/proposal-review/*`.
- Add **Pending Decisions** panel on Hub main tab with read-only detail card.
- Add unavailable and empty states (quiet, no error spam).
- Add tests proving Hub does not call review/triage POST endpoints.
- Update docs for Hub as proposal attention router.
- Extend `sync_local_env_from_example.py` with `HUB_PROPOSAL_REVIEW_` prefix.

## Architecture

```text
Hub UI (Pending Decisions)
  → GET /api/proposal-review/pending|health|proposals/{id}|eligibility
  → proposal_review_client (GET allowlist)
  → context-exec proposal review API (:8096)
  → JsonFileProposalLedgerRepository (context-exec only — Hub never reads ledger files)
```

## Safety

- Read-only Hub surface
- No approve/reject/triage actions in UI or routes
- No executor
- No execution receipts
- No memory/repo/runtime mutation
- Hub does not read ledger files directly
- Default filter: `pending_review` only (secondary filters hidden)

## Verification

| Command | Exit | Result |
|---------|------|--------|
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-hub/tests/test_proposal_review_hub.py services/orion-hub/tests/test_proposal_review_ui.py -q` | 0 | 6 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest tests/services/test_proposal_review_api.py -q` | 0 | 21 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-hub/tests/ -q` | 1 | 485 passed, 17 failed (pre-existing/flaky substrate suite; new tests pass) |
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/ -q` | 1 | 119 passed, 14 failed (pre-existing alexzhang/repo eval failures) |
| `ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh` | 1 | Blocked by same pre-existing context-exec failures |

## Local env sync

Updated `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env` (gitignored) with:

```bash
HUB_PROPOSAL_REVIEW_ENABLED=false
HUB_PROPOSAL_REVIEW_API_URL=http://127.0.0.1:8096
HUB_PROPOSAL_REVIEW_TIMEOUT_SEC=10
```

## Manual smoke (operator)

```bash
rm -f /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py seed-demo --store /tmp/orion-proposals.json

PROPOSAL_REVIEW_API_ENABLED=true PROPOSAL_LEDGER_STORE_PATH=/tmp/orion-proposals.json \
  # start context-exec on :8096

HUB_PROPOSAL_REVIEW_ENABLED=true HUB_PROPOSAL_REVIEW_API_URL=http://127.0.0.1:8096 \
  # start Hub on :8080 — Pending Decisions shows pending_review demo proposal
```

## Test plan

- [ ] Hub starts with `HUB_PROPOSAL_REVIEW_ENABLED=false` — panel quiet, no crashes
- [ ] Enable both services — Pending Decisions lists seeded `pending_review` proposal
- [ ] Click proposal — detail shows summary, evidence, eligibility (read-only)
- [ ] Stop context-exec — Hub shows "Proposal review API unavailable."
- [ ] Confirm no POST to `/triage` or `/review` from Hub network tab
