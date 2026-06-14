#681: Harden JSON proposal ledger and add context-exec proposal ledger intake

## Summary

Hardens the JSON proposal ledger and adds opt-in context-exec proposal ledger intake.

This makes real `ProposalEnvelopeV1` outputs from context-exec persistable as `ProposalLedgerRecordV1` records while keeping ledger intake disabled by default and requiring an explicit store path. The operator CLI can then inspect persisted proposals. No approval endpoint, Hub UI, executor, or mutation path is added.

## Changes

- Harden `JsonFileProposalLedgerRepository`: explicit store path required, malformed JSON fails without overwrite, atomic temp+replace writes.
- Harden operator CLI: `--store` required on all commands; clear errors for malformed JSON and missing proposal IDs.
- Add `proposal_ledger_intake.py` with `maybe_persist_proposal_envelope` helper.
- Integrate intake in context-exec runner for `ProposalEnvelopeV1` when ledger explicitly enabled.
- Add settings: `CONTEXT_EXEC_PROPOSAL_LEDGER_ENABLED`, `CONTEXT_EXEC_PROPOSAL_LEDGER_STORE_PATH`, `CONTEXT_EXEC_PROPOSAL_LEDGER_AUTO_TRIAGE` (all safe defaults).
- Add runtime debug ledger metadata and final_text ledger line when persisted.
- Add tests for patch/memory correction intake, investigative mode exclusion, JSON hardening, and CLI smoke.
- Update docs with opt-in intake flow and CLI inspection examples.

## Verification

| Command | Exit | Result |
|---------|------|--------|
| `PYTHONPATH=. orion_dev/bin/python -m pytest services/orion-context-exec/tests/ -q` | 0 | 136 passed, 1 xfailed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest tests/scripts/test_orion_proposal_cli.py -q` | 0 | 12 passed |
| `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine fake` | 0 | All cases pass |
| `PYTHONPATH=. orion_dev/bin/python scripts/context_exec_rlm_eval.py --engine alexzhang` | 0 | All cases pass |
| `ORION_PY=orion_dev/bin/python bash scripts/context_exec_beta_gate.sh` | 0 | BETA GATE PASS |
| `rg "PatchProposalV1|ProposalEnvelopeV1|..." orion/schemas` | 0 | All 7 schemas registered |

## Safety

- Ledger intake disabled by default (`CONTEXT_EXEC_PROPOSAL_LEDGER_ENABLED=false`)
- Explicit store path required when enabled — no repo-local defaults
- Malformed JSON store files are never silently overwritten
- Review approval creates execution eligibility only — no receipt, no executor call
- Investigative artifacts (`belief_provenance`, `trace_autopsy`, `repo_impact_analysis`) do not enter proposal ledger
- No Hub UI, no approval endpoint, no executor, no repo/memory/runtime mutation

## Test plan

- [x] JSON ledger requires explicit store path
- [x] Malformed JSON fails cleanly without overwrite
- [x] Missing proposal IDs fail cleanly (show/review/triage)
- [x] Review approve creates eligibility not receipt
- [x] Ledger intake disabled by default
- [x] Enabled intake requires explicit store path
- [x] Patch and memory correction proposals persist when enabled
- [x] Investigative modes do not persist
- [x] Runtime debug includes proposal_id / ledger_status when persisted
- [x] CLI can show context-exec persisted proposals
- [x] Existing CLI commands still pass
