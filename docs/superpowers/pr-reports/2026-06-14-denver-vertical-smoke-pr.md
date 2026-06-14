# PR: Add Denver memory-correction vertical slice smoke

**Branch:** `feat/denver-vertical-smoke`  
**Title:** Add Denver memory-correction vertical slice smoke  
**Create PR:** https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/denver-vertical-smoke?expand=1

> **Note:** Branch includes prerequisite Hub read-only proposal review surface commits (`720ff5db`..`5ae729aa`) not yet on `main`. Merge or rebase base as appropriate.

## Summary

Adds a Denver memory-correction vertical-slice smoke.

This proves the proposal control plane is useful end to end: context-exec drafts a `memory_correction_proposal`, the proposal is stored in the ledger, deterministic auto-triage promotes it to `pending_review`, the proposal review API exposes it, and Hub can display it as a read-only Pending Decision.

This smoke does **not** approve, reject, execute, or mutate memory.

## Changes

- Add deterministic Denver fixture in `FakeRLMEngine` for `memory_correction_proposal` mode.
- Add shared fixture `tests/fixtures/denver_vertical_slice.py`.
- Add `scripts/denver_memory_correction_vertical_smoke.sh`.
- Add `test_denver_memory_correction_vertical_slice_persists_pending_review`.
- Add `test_proposal_review_api_lists_denver_pending_review`.
- Add `test_hub_pending_decisions_shows_denver_memory_correction`.
- Enrich proposal review API `inner_artifact_summary` and Hub detail card (belief, correction, rationale, evidence summary, safety flags).
- Fix Hub test module isolation when run after context-exec tests (`app.settings` collision).
- Update docs: Denver vertical slice sections in runbook, proposal-review-api, Hub README.

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

- Read-only Hub surface (GET only)
- No approve/reject/triage actions
- No executor
- No execution receipts
- No memory/repo/runtime mutation
- `mutation_allowed=false`, `requires_human_approval=true`, eligibility `eligible=false` before approval

## Verification

| Command | Exit | Result |
|---------|------|--------|
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/test_proposal_ledger_intake.py -q` | 0 | 25 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest tests/services/test_proposal_review_api.py -q` | 0 | 22 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-hub/tests/test_proposal_review_hub.py services/orion-hub/tests/test_proposal_review_ui.py -q` | 0 | 7 passed |
| `ORION_PY=orion_dev/bin/python bash scripts/denver_memory_correction_vertical_smoke.sh` | 0 | PASS |
| `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine fake` | 0 | all cases ok |
| `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine alexzhang` | 0 | all cases ok |
| `ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh` | 1 | 15 pre-existing alexzhang/repo failures (same as prior PRs) |

## Test plan

- [ ] Run smoke script — one Denver `pending_review` proposal in temp ledger
- [ ] Proposal review API lists/filters/detail/eligibility for Denver proposal
- [ ] Hub Pending Decisions shows Denver card with useful detail (no action buttons)
- [ ] Confirm no POST from Hub to `/triage` or `/review`
