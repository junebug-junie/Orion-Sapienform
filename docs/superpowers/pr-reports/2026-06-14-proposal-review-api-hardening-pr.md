# PR: Harden proposal review API boot, safety, and regression coverage

**Branch:** `feat/proposal-review-api-hardening`  
**Base:** `feat/proposal-review-api` (scaffold) → merge to `main`  
**Title:** Harden proposal review API boot, safety, and regression coverage

## Summary

Hardens the proposal review API scaffold on **orion-context-exec** before Hub consumption. Proves the actual FastAPI app boots and mounts proposal routes when enabled, default-safe disabled/missing-store behavior is explicit, health reports ledger store status accurately, malformed stores fail cleanly, missing proposals return 404, approval creates eligibility only (no executor/receipts), and context-exec cannot self-approve.

The API remains a review-gate socket only — no Hub UI, no executor, no memory/repo/runtime mutation.

## Changes

- Default-disable `PROPOSAL_REVIEW_API_ENABLED` (`false`); mount proposal routes only when enabled.
- Harden `/health` `proposal_review_api` block: `enabled`, `store_configured`, `store_path_present`, `ok`, `error`.
- Require store file on disk for `ok=true` (empty in-memory fallback does not look healthy).
- Add `_require_api_enabled()` guard; 503 when enabled but store missing/malformed/invalid; 404 when disabled (routes unmounted).
- Add 21 API regression tests (boot smoke, default-safe, health, malformed store, missing IDs, approval safety, context-exec rejection).
- Strengthen service-local health tests (3).
- CLI coverage unchanged at 14 tests.
- Add `scripts/proposal_review_api_smoke.sh`.
- Update `docs/proposal-review-api.md`, `docs/context-exec-beta-runbook.md`, `services/orion-context-exec/README.md`, `.env_example`, `docker-compose.yml`.

## API surface (unchanged)

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
- No memory writes
- No repo writes
- No runtime mutation
- Approval only creates future execution eligibility
- Context-exec cannot approve proposals

## Files changed (hardening commit range)

- `services/orion-context-exec/app/proposal_review_api.py`
- `services/orion-context-exec/app/settings.py`
- `services/orion-context-exec/app/main.py`
- `services/orion-context-exec/.env_example`
- `services/orion-context-exec/docker-compose.yml`
- `services/orion-context-exec/README.md`
- `services/orion-context-exec/tests/test_health.py`
- `tests/services/test_proposal_review_api.py`
- `scripts/proposal_review_api_smoke.sh` (new)
- `docs/proposal-review-api.md`
- `docs/context-exec-beta-runbook.md`

## Verification

| Command | Exit | Result |
|---------|------|--------|
| `PYTHONPATH=. orion_dev/bin/python -m pytest tests/services/test_proposal_review_api.py -q` | 0 | 21 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest tests/scripts/test_orion_proposal_cli.py -q` | 0 | 14 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_health.py -q` | 0 | 3 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_proposal_ledger.py -q` | 0 | 17 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest orion/schemas -q` | 0 | 9 passed |
| `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine fake` | 0 | all cases pass |
| `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine alexzhang` | 0 | all cases pass |
| `ORION_PY=orion_dev/bin/python bash scripts/proposal_review_api_smoke.sh` | 0 | SMOKE PASS |
| `ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh` | 1 | 14 pre-existing alexzhang/repo-organ failures on this host |

## Env sync (local operator host)

Updated **main workspace** `services/orion-context-exec/.env` (gitignored):

```env
PROPOSAL_REVIEW_API_ENABLED=false
PROPOSAL_LEDGER_STORE_PATH=
```

## Commits

1. `85bdbc8c` — Add proposal ledger review API scaffold on context-exec.
2. `1fcf14f7` — Handle invalid ledger schema and drop unused logger in review API.
3. `5259be45` — Adapt review API to ProposalLedgerStoreError from merged ledger hardening.
4. `a2e45872` — Harden proposal review API boot, safety, and regression coverage.
5. `90472727` — Fix health ok to require store file on disk and clarify disabled-route docs.

## Arsonist note

Boot the socket. Prove it is wired. Prove it cannot swing the hammer.
