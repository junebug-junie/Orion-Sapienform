# PR: Add proposal execution dry-run receipt scaffold

**Branch:** `feat/proposal-dry-run-receipt`  
**Title:** Add proposal execution dry-run receipt scaffold

## Summary

Adds a proposal execution dry-run receipt scaffold.

Approved proposals can now be passed through a dry-run execution path that produces a `ProposalExecutionReceiptV1` with `dry_run=true` and `mutation_performed=false`. This proves the executor handoff shape without performing any mutation.

## Changes

- Add `ProposalExecutionReceiptV1` schema with dry-run invariants (`status=simulated`, `mutation_performed=false`).
- Register schema in `orion/schemas/registry.py`.
- Add pure lifecycle helpers: `validate_dry_run_execution_eligibility`, `build_dry_run_execution_receipt`.
- Add CLI command `dry-run-execute` (read-only; receipt to stdout only).
- Add six CLI safety tests plus lifecycle unit tests and registry round-trip.
- Update `docs/proposal-review-api.md` and `services/orion-context-exec/README.md`.

## Architecture

```text
Operator CLI dry-run-execute
  → JsonFileProposalLedgerRepository.get (read-only)
  → derive_execution_eligibility + validate_dry_run_execution_eligibility
  → build_dry_run_execution_receipt
  → stdout JSON (ProposalExecutionReceiptV1)
  → ledger file unchanged; proposal stays status=approved
```

## Safety

- Dry-run only — not real execution
- No memory writes
- No repo writes
- No runtime mutation
- No real executor
- No branch creation
- No git/gh operations from dry-run path
- Proposal status remains `approved` (not `executed` or `execution_requested`)

## Verification

| Command | Exit | Result |
|---------|------|--------|
| `PYTHONPATH=. orion_dev/bin/python -m pytest tests/scripts/test_orion_proposal_cli.py -q` | 0 | 20 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest orion/schemas -q` | 0 | 13 passed |
| `PYTHONPATH=. orion_dev/bin/python -m pytest tests/services/test_proposal_review_api.py -q` | 0 | 22 passed |

## Manual smoke

```bash
rm -f /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py seed-demo --store /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py review <proposal_id> \
  --decision approve --reason "dry run test" --reviewer human:june \
  --store /tmp/orion-proposals.json
PYTHONPATH=. orion_dev/bin/python scripts/orion_proposal_cli.py dry-run-execute <proposal_id> \
  --store /tmp/orion-proposals.json --executor dry-run
```

Expected receipt: `dry_run=true`, `status=simulated`, `mutation_performed=false`.

## Test plan

- [ ] `ProposalExecutionReceiptV1` resolves from registry
- [ ] Dry-run rejects stored/pending_review proposals
- [ ] Dry-run accepts approved + eligible proposals
- [ ] Ledger file byte-identical before/after dry-run
- [ ] `show` still reports `status=approved` after dry-run
- [ ] Receipt has no `changed_targets` field

## Non-goals (confirmed)

- No real memory correction execution
- No patch execution
- No autonomous execution
- No receipt persistence slot yet (stdout only)
