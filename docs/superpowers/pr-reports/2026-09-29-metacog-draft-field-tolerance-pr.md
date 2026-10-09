## Summary

- **The draft schema no longer declares `what_changed`.** The existing sanitizer now strips it like any other unknown key. Publish already computes `what_changed` from the trigger's evidence and overwrites it, so nothing that was used is lost.
- **Validation drops only the fields that fail** (`_validate_draft_patch_per_field`). One wrong-shaped field no longer discards a usable summary and mantra. Dropped fields are recorded as `<field>:invalid_shape` in the draft telemetry.
- **A draft only counts as real if a non-empty summary survives,** checked on every path. Anything less is a genuine fallback: publish uses the summary built from the evidence, and the baseline firebreak still applies.
- **`_postprocess_metacog_draft_summary`** no longer lets the fallback template's `what_changed.summary` take precedence over the draft's own summary.

## Outcome moved

- **Baseline rows had nearly disappeared.** Checked live on 2026-09-29: since #2311 deployed, baseline triggers fire about 22 times a day, but only 5% became `orion_metacog` rows. Before the 24th it was 99%.
- **Cause:**
  1. On baseline there is no event, so the model improvises. It puts the biometrics block into `what_changed.evidence` as an object where a list is expected (see the cortex-exec-background logs: `patch rejected ... what_changed.evidence Input should be a valid list`).
  2. The whole draft failed validation.
  3. The baseline firebreak then dropped the row.
- **Other trigger kinds** hit the same rejection about 40 times a day. They silently lost the model's summary and fell back to the evidence line.
- **Not latency:** the draft LLM call returned in 17.5 s. It was the validation step that threw the draft away.

## Current architecture

- **Who writes what:** the draft LLM writes summary, mantra and tags. Publish (`MetacogPublishService`) builds `what_changed`, severity and touches from `orion/metacog/evidence_map.py`.
- **The mismatch:** `MetacogDraftTextPatchV1` still declared `what_changed`, even though the prompt forbids it and publish discards it.

## Files changed

- `orion/schemas/metacog_patches.py`: `MetacogDraftWhatChangedV1` and the `what_changed` field are removed.
- `services/orion-cortex-exec/app/executor.py`:
  - per-field validation plus the summary-required usability check;
  - the telemetry records dropped fields;
  - `_apply_draft_patch` and `_postprocess_metacog_draft_summary` no longer use the draft's what_changed.
- `services/orion-cortex-exec/tests/test_metacog_draft_field_tolerance.py`: 5 tests, which CI runs through the metacog-capture workflow glob.
- `tests/test_metacog_phase_contract.py`, `tests/test_metacog_prompt_phase_contract.py`: updated for the removed field.

## Schema / bus / API changes

- **Removed:** `MetacogDraftTextPatchV1.what_changed` and `MetacogDraftWhatChangedV1`. These are internal and draft-only. A grep across orion/, cortex-orch and sql-writer finds no other reader. The published `MetacogEntryV1` is unchanged.

## Env/config changes

- None.

## Tests run

```text
services/orion-cortex-exec: pytest tests/test_metacog_*.py -> 72 passed
root: PYTHONPATH=.:services/orion-cortex-exec pytest tests/test_metacog_phase_contract.py -> 12 passed
Mutation checks: re-adding what_changed to the schema, reverting to all-or-nothing validation,
and removing the summary-required check each make the new tests fail.
Pre-existing failures on origin/main, not caused by this change: tests/test_metacog_prompt_phase_contract.py::test_draft_prompt_requires_patch_only,
orion/schemas/tests/test_context_provenance.py::test_static_ctx_assignments_covered.
scripts/check_definition_drift.py --gate -> PASS; scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
None new. The live check after deploy is the real eval: the baseline row rate should return to about 99% of triggers, and the "patch rejected" log lines should fall to about 0.
```

## Review findings fixed

- **Finding (high):** a draft containing only a stripped `what_changed` validated as an empty patch in llm mode. That would have published the fallback template's text as LLM output and slipped past the baseline firebreak.
  - **Fix:** the usability check runs on every path, including a first-try success.
  - **Evidence:** `test_only_a_stripped_what_changed_is_a_fallback_not_an_empty_llm_draft`, mutation-checked.
- **Finding (medium):** a mantra-only survivor kept the fallback template summary.
  - **Fix:** a non-empty summary is required.
  - **Evidence:** `test_mantra_without_summary_is_a_fallback`.
- **Noted, pre-existing, not fixed here:** rows in llm mode still carry the fallback template's `emergent_entity="Fallback Baseline"` and `resonance_signature`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build   # orch loads the same schema and template
```

## Risks / concerns

- **Severity: low.**
  - **Concern:** baseline rows will return, about 22 a day. The trigger itself carries no event evidence (`scheduled_check`, empty upstream), so their value is a "nothing measured" heartbeat.
  - **Mitigation:** whether baseline is worth keeping is a separate decision.
