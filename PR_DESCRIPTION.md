# fix(memory-crystallization): make consolidated stance proposals approvable

## Summary

- Consolidation-intake `stance` crystallizations were **structurally un-approvable**: `build_crystallization_from_window()` never set `planning_effects`/`retrieval_affordances`, which the validator hard-requires for `stance`/`procedure`/`decision`. Clicking **Approve** in the hub returned **HTTP 400** (`GovernorError` from `can_activate` → `validate_proposal`).
- `build_crystallization_from_window()` now populates content-grounded `planning_effects` and a kind-keyed `retrieve_when:<intent>` `retrieval_affordances` for those kinds, so proposals pass `validate_proposal()` and can be approved.
- Single source of truth (`_RETRIEVAL_INTENT_FOR_KIND`) with an import-time invariant against `STANCE_PROCEDURE_DECISION_KINDS`; other kinds (episode/semantic/open_loop) unchanged.
- Added regression + direct helper unit tests (TDD: reproduction test failed before, passes after).

## Outcome moved

Operators can now Approve consolidated `stance` memories instead of hitting a permanent HTTP 400. Auto-proposed stances are no longer empty-shell (they carry a real planning effect + retrieval affordance).

## Architecture touched

- `orion/memory/crystallization/intake_consolidation_window.py` — intake producer only. Validator/governor untouched; contract unchanged.

## Files changed

- `orion/memory/crystallization/intake_consolidation_window.py`: enrich stance/procedure/decision at intake.
- `services/orion-memory-consolidation/tests/test_intake_consolidation_window.py`: reproduction test asserting `validate_proposal(...).valid is True` for a stance window, non-stance-stays-empty test, and direct helper tests.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```
PYTHONPATH=. .venv/bin/python -m pytest services/orion-memory-consolidation/tests/test_intake_consolidation_window.py -q
# 6 passed

PYTHONPATH=. .venv/bin/python -m pytest tests/test_memory_crystallization.py -q
# 23 passed, 1 failed (TestMemoryCardBackwardCompat::test_memory_card_v1_unchanged_in_registry_gap)
# ^ pre-existing on main, unrelated to this change (not in this diff)
```

## Review findings fixed

- Finding (Minor): dual source of truth + dead `if intent else []` fallback.
  - Fix: gate on `_RETRIEVAL_INTENT_FOR_KIND` membership; import-time assert it equals `STANCE_PROCEDURE_DECISION_KINDS`.
  - Evidence: commit c291a1a0.
- Finding (Minor): procedure/decision helper branches untested.
  - Fix: added direct `_planning_and_retrieval_for_kind` unit tests.
  - Evidence: commit c291a1a0, 6 passed.

## Restart required

The running `orion-memory-consolidation` service must be rebuilt/restarted to pick up the intake change (existing rows created before this are unaffected; already-approved ones stay active):

```bash
docker compose --env-file .env --env-file services/orion-memory-consolidation/.env -f services/orion-memory-consolidation/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low. Salience stays `0.5` at intake (bypasses `apply_salience`, unlike the operator `/propose` path which yields `0.85` for stance). Not the cause of the 400; suggested as a separate follow-up.
- Severity: low. Pre-existing unrelated failure `TestMemoryCardBackwardCompat::test_memory_card_v1_unchanged_in_registry_gap` on main; separate issue.
