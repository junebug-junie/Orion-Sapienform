# PR: Add proposal ledger review API scaffold

**Branch:** `feat/proposal-review-api`  
**Base:** `main`  
**Create PR:** https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/proposal-review-api

## Summary

Adds a proposal ledger review API scaffold on **orion-context-exec**. This creates the future Hub-facing seam for proposal review without adding Hub UI, approval automation, or execution. The API exposes safe control-plane operations over the proposal ledger: list proposals, fetch proposal details, apply triage decisions, apply review decisions, and inspect execution eligibility.

The proposal ledger remains the system of record. Hub will eventually call this API instead of reading JSON ledger files or owning lifecycle logic.

## Changes

- Add proposal review API scaffold (`services/orion-context-exec/app/proposal_review_api.py`).
- Extend `/health` with `proposal_review_api` store status block.
- Add proposal list endpoint.
- Add proposal detail endpoint.
- Add triage endpoint.
- Add review endpoint.
- Add execution eligibility endpoint.
- Require explicit `PROPOSAL_LEDGER_STORE_PATH`.
- Reuse existing `JsonFileProposalLedgerRepository` and lifecycle validation.
- Return controlled errors for missing proposal IDs and malformed/invalid stores (503/404/403/400).
- Add tests for list/detail/triage/review/eligibility behavior.
- Add tests proving approval creates eligibility only and does not execute.
- Add docs describing the API as the future Hub-facing review seam (`docs/proposal-review-api.md`).

## API surface

```text
GET  /health
GET  /proposals
GET  /proposals/{proposal_id}
POST /proposals/{proposal_id}/triage
POST /proposals/{proposal_id}/review
GET  /proposals/{proposal_id}/eligibility
```

## Safety

- No Hub UI
- No executor
- No execution receipts
- No approval automation
- No auto-approval
- No memory writes
- No repo writes
- No runtime mutation
- Approval only creates future execution eligibility
- Context-exec cannot approve or execute proposals

## Files changed

- `services/orion-context-exec/app/proposal_review_api.py` (new)
- `services/orion-context-exec/app/settings.py`
- `services/orion-context-exec/app/api.py`
- `services/orion-context-exec/app/main.py`
- `services/orion-context-exec/.env_example`
- `services/orion-context-exec/docker-compose.yml`
- `services/orion-context-exec/README.md`
- `docs/proposal-review-api.md` (new)
- `tests/services/test_proposal_review_api.py` (new)

## Verification

| Command | Exit | Result |
|---------|------|--------|
| `PYTHONPATH=. orion_dev/bin/python -m pytest tests/services/test_proposal_review_api.py -q` | 0 | 13 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest tests/scripts/test_orion_proposal_cli.py -q` | 0 | 8 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest orion/schemas -q` | 0 | 9 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_health.py services/orion-context-exec/tests/test_proposal_ledger.py -q` | 0 | 18 passed |
| `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine fake` | 0 | all cases pass |
| `python -m compileall services/orion-context-exec/app/proposal_review_api.py` | 0 | ok |

Not run on this host (pre-existing env failures): full `services/orion-context-exec/tests/` suite (alexzhang/repo organ tests), `context_exec_rlm_eval.py --engine alexzhang`, `context_exec_beta_gate.sh`.

## Env sync (local operator host)

Added to **main workspace** `services/orion-context-exec/.env` (gitignored, not in PR):

```env
PROPOSAL_REVIEW_API_ENABLED=true
PROPOSAL_LEDGER_STORE_PATH=
```

## Notes

This PR does not make Hub consume the API yet. It only creates the review-gate socket Hub can later plug into.

```text
context-exec emits ProposalEnvelopeV1
  → proposal ledger stores ProposalLedgerRecordV1
  → deterministic triage marks stored / blocked / pending_review
  → review API exposes pending_review and detail surfaces
  → operator or future Hub records review decisions
  → future executor may consume approved proposals only later
```

## Commits

1. `85bdbc8c` — Add proposal ledger review API scaffold on context-exec.
2. `1fcf14f7` — Handle invalid ledger schema and drop unused logger in review API.
