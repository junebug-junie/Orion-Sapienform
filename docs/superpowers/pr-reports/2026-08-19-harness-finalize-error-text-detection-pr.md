# Detect error-shaped text in voice/reflect finalize steps

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1741
Branch: `fix/harness-finalize-error-text-detection`

## Summary

- Root-caused live, 2026-08-19, during a real circe-worker outage while verifying PR #1739 (endogenous outreach through the real unified-turn pipeline): `orion_voice_finalize`'s own `final_text` came back as the literal string `"[Error: llamacpp timed out after waiting]"` on 3 consecutive test turns — a genuine upstream failure reported only in the text, no different in shape from `ok=False`.
- This is the SAME failure class `services/orion-hub/scripts/endogenous_outreach.py::looks_like_error_text()` was built to catch on 2026-08-14 — but that check only ever lived in outreach's own module. The SHARED finalize chain every real `client_mode=="orion"` turn runs through (`orion/harness/finalize.py::extract_voice_finalize_text`/`extract_finalize_reflection_payload`) had the identical `if text: return text` gap with **no detection at all**.
- A real backend hiccup during a genuine Juniper conversation could have delivered this exact literal error string as Orion's real spoken answer, with no visible red flag except the text itself looking wrong.
- Promoted `looks_like_error_text()` to `orion.cognition.cortex_payload_extract` (the shared text-extraction module both callers already depend on) so there's one definition, not two silently-drifting copies. Applied it at both finalize call sites.
- The downstream failure-handling infrastructure was already correct and is unchanged — raising here routes into `run_orion_voice_finalize`'s existing caller, which already wraps any exception in `emit_finalize_failure_artifacts()`/`HarnessFinalizeFailedError`. This patch only adds the missing detection at the source.

## Outcome moved

Closes a real, currently-live gap in the shared harness finalize chain — every real chat turn, not just outreach, is now protected against an upstream LLM failure being silently delivered as Orion's real answer.

## Current architecture

`extract_voice_finalize_text()` and `extract_finalize_reflection_payload()` both called `extract_cortex_payload_text(result)` and returned any non-empty string unconditionally. The reflect step was *accidentally* partially protected (error text usually fails its own caller's JSON-parse step, degrading gracefully) — fragile, not a real gate. The voice-finalize step had no protection at all.

## Architecture touched

- `orion/cognition/cortex_payload_extract.py`: new `looks_like_error_text()` + constants (canonical home).
- `orion/harness/finalize.py`: both extraction functions now raise on error-shaped text instead of returning it.
- `services/orion-hub/scripts/endogenous_outreach.py`: imports the canonical version instead of keeping its own copy.
- No schema/bus/env changes.

## Files changed

- `orion/cognition/cortex_payload_extract.py`: `looks_like_error_text()`, `_ERROR_TEXT_PREFIXES`, `_ERROR_TEXT_MARKERS` (added `"llamacpp timed out"` to the marker list, matching the exact string observed live — though the `"[error"` prefix match already covered this specific case).
- `orion/harness/finalize.py`: `extract_voice_finalize_text()` and `extract_finalize_reflection_payload()` both check `looks_like_error_text()` before returning; import updated.
- `orion/harness/tests/test_finalize_cortex_payload_extract.py`: 3 new tests (voice-finalize rejects error text, voice-finalize accepts real text, reflect rejects error text).
- `orion/cognition/tests/test_cortex_payload_extract.py` (new): dedicated test file for the promoted shared function.
- `services/orion-hub/scripts/endogenous_outreach.py`: `looks_like_error_text` now imported, not defined; module docstring note.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
rtk proxy /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  orion/cognition/tests/test_cortex_payload_extract.py \
  orion/harness/tests/test_finalize_cortex_payload_extract.py \
  services/orion-hub/tests/test_endogenous_outreach.py -q
101 passed

rtk proxy /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/harness/tests/ orion/cognition/tests/ -q \
  --ignore=orion/cognition/tests/test_packs.py --ignore=orion/cognition/tests/test_planner.py \
  --ignore=orion/cognition/tests/test_reflect_recall_integration.py
279 passed, 4 pre-existing failures verified identical on untouched main
  (test_grounding_capsule_consumers.py x2, test_harness_runner.py, test_projection_builder.py --
  none touch this diff's files)
```

The 3 excluded `--ignore` files have a pre-existing `ModuleNotFoundError: No module named 'orion_cognition'` collection error, confirmed identical on untouched main — unrelated to this patch.

## Evals run

None — this is a correctness/safety fix, not a quality-tunable behavior.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```
Rebuilt and redeployed `orion-harness-governor` with the fix live. Confirmed healthy via logs (`Uvicorn running`, subscribed to `orion:harness:run:request`). Did not attempt a live reproduction of the original failure (would require forcing another real backend outage) — verified via unit tests exercising the exact literal string observed live instead.

## Review findings fixed

Code review (`/code-review medium`) ran clean — no findings across all 8 angles (correctness, removed-behavior, cross-file reference, reuse/simplification/efficiency/altitude, conventions). Verified both new `raise ValueError` sites are caught by their callers' existing exception handling; verified no other file referenced the deleted private constants; verified `looks_like_error_text` stays importable from `scripts.endogenous_outreach` for existing call sites; ran the actual test suites (not just read them) — 18 + 8 passed.

## Restart required

Already deployed during this session:
```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

`services/orion-hub` also depends on `orion/cognition/cortex_payload_extract.py` (via `endogenous_outreach.py`'s updated import) — no functional change there since it re-imports the same logic, but worth a Hub restart too next time one happens for an unrelated reason, to pick up the cleaner import path:
```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: the marker-based detection (`looks_like_error_text`) is a backstop heuristic, not a structural fix to why `llamacpp`/the LLM gateway sometimes reports failure only in text instead of a proper `ok=False`. That deeper inconsistency (confirmed still present at the LLM-gateway/cortex-exec layer) is out of scope here.
- Mitigation: this patch makes the existing, correct failure-handling machinery actually trigger instead of silently shipping bad text — a real, meaningful improvement even without fixing the upstream inconsistency. Kept deliberately narrow (matches error *framing*, not the mere word "error") so genuine reflective prose about errors Orion has encountered is never swallowed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1741
