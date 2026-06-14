#682: Harden proposal auto-triage and add ledger intake smoke coverage

## Summary

Hardens proposal auto-triage and adds ledger intake smoke coverage.

This covers the previously untested `CONTEXT_EXEC_PROPOSAL_LEDGER_AUTO_TRIAGE=true` path with deterministic policy tests. Auto-triage can store, block for evidence, or promote decision-worthy proposals to `pending_review`, but it cannot approve, request execution, execute, or mutate anything.

## Changes

- Add/clarify deterministic proposal auto-triage policy via `triage_proposal_envelope`.
- Test AUTO_TRIAGE=false quiet storage path.
- Test high/unknown risk promotion to pending_review.
- Test low-confidence / insufficient-evidence block_for_evidence path.
- Test identity memory correction promotion behavior.
- Prove auto-triage never approves or executes.
- Add CLI visibility for triage_action and attention metadata.
- Add smoke coverage proving CLI can read context-exec persisted proposals.
- Update docs for auto-triage safety.

## Verification

- `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/ -q` — 131 passed, 1 xfailed
- `PYTHONPATH=. orion_dev/bin/python -m pytest tests/scripts/test_orion_proposal_cli.py -q` — 14 passed
- `PYTHONPATH=. orion_dev/bin/python -m pytest orion/schemas -q` — 9 passed
- `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine fake` — pass
- `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine alexzhang` — pass
- `ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh` — BETA GATE PASS

## Safety

- No Hub UI
- No approval endpoint
- No executor
- No LLM triage
- No auto-approval
- No execution
- No repo/memory/runtime mutation

## Base branch

Built on `feat/681-proposal-ledger-intake` (proposal ledger intake scaffold).
