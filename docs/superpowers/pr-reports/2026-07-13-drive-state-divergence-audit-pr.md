# PR report: drive_state.v1 vs autonomy_state_v2 divergence audit

## Summary

- `drive_state.v1` (`DriveEngine`, `orion/spark/concept_induction/drives.py`) and `autonomy_state_v2` (`orion.autonomy.reducer`) independently compute the same 6-drive taxonomy (`coherence, continuity, capability, relational, predictive, autonomy`) and are explicitly marked `DUPLICATE` of each other in `orion/self_state/inner_state_registry.py`, with the merge-or-keep-separate decision deferred to a later phase, not resolved.
- New `scripts/drive_state_divergence_audit.py` — a report-only, point-in-time snapshot comparison of the two signals' *current* values (neither is historically persisted — `drive_state.v1` lives in a local JSON file that gets overwritten, `autonomy_state_v2` in a single-row-per-subject Postgres UPSERT). Measures, never merges or picks a winner.
- Does not touch either signal's producer, consumer, or the deferred merge decision — pure diagnostic.

## Outcome moved

There was no way to answer "how far apart are these two signals right now" without manually querying two different backends by hand. Now there's a standing, runnable script — the actual evidence input the deferred Phase-4-style merge decision needs, whenever someone picks it back up.

## Current architecture (before this patch)

Two signals, two backends, zero comparison tooling. `drive_state.v1`'s current value: `orion/spark/concept_induction/store.py::LocalProfileStore.load_drive_state(subject)`, a raw unvalidated dict from a local JSON file (`CONCEPT_STORE_PATH`). `autonomy_state_v2`'s current value: `orion/autonomy/state_store.py::load_autonomy_state_v2(subject)`, a pydantic-validated `AutonomyStateV2` from Postgres (`ORION_AUTONOMY_STATE_DB_URL`), fails open (returns `None`) on any error.

## Architecture touched

`scripts/` (new standalone script, matches `check_activation_saturation.py`'s conventions), `services/orion-spark-concept-induction/README.md` (new short section), `tests/`.

## Files changed

- `scripts/drive_state_divergence_audit.py` (new): loads both current signals, computes per-drive-key `abs(pressure_a - pressure_b)` and activation-flag agreement (`DriveStateV1.activations[key]` bool vs. `AutonomyStateV2.active_drives` list membership — the two schemas don't share the same activation representation, handled explicitly). Reports prose or `--json`. Always exits 0 (report-only, no known-good baseline to gate against, matching `check_activation_saturation.py`'s posture before it grew a `--fail-above` flag).
- `tests/test_drive_state_divergence_audit.py` (new, 15 tests): both signals present, one/both missing, activation agreement/disagreement, and a regression test for a corrupted-JSON-value crash the implementing agent found and fixed in its own review pass (`_coerce_float()` guards against the local store's lack of type validation, unlike the Postgres side which fails closed via pydantic).
- `services/orion-spark-concept-induction/README.md`: short section documenting what/why/how to run.

## Schema / bus / API changes

None. Read-only consumer of two already-existing, already-typed signals.

## Env/config changes

None new. Reads existing `ORION_AUTONOMY_STATE_DB_URL` / `CONCEPT_STORE_PATH`.

## Tests run

```text
$ python -m pytest tests/test_drive_state_divergence_audit.py -q
15 passed

$ python -m pytest tests/test_drive_state_divergence_audit.py tests/test_inner_state_registry_gate.py tests/test_check_concept_relation_digest_liveness.py -q
31 passed
```
Independently re-run by the orchestrator (not just the implementing agent) — confirmed.

## Evals run

Not applicable — deterministic diagnostic script, no eval surface. Its own live-data run against a real local store file (see Docker/build/smoke checks) is the closest equivalent to an eval here.

## Docker/build/smoke checks

Live-ran against a real local `CONCEPT_STORE_PATH` file on this host (found real `drive_state.v1` data, `updated_at=2026-07-12T08:50:17`), and confirmed both the DSN-unset and DB-connection-refused paths for `autonomy_state_v2` degrade to a clear "UNAVAILABLE" report with exit 0, no crash — proof beyond the mocked unit tests that the script's degrade-gracefully claims hold against real infrastructure, not just mocks.

## Review findings fixed

- Finding: the implementing agent's own first draft called `float(pressure_value)` unconditionally on the JSON-store side. `LocalProfileStore.load_drive_state()` returns a raw, unvalidated dict (unlike the Postgres side, which fails closed via pydantic validation) — a hand-edited or corrupted store value would crash the script instead of degrading.
  - Fix: `_coerce_float()` helper, non-numeric values reported as unavailable rather than raising.
  - Evidence: `test_compare_drives_corrupted_pressure_value_does_not_crash` (new regression test).

## Restart required

```text
No restart required.
```
Standalone, on-demand script — nothing running to restart.

## Risks / concerns

- Severity: Low — this is a point-in-time snapshot tool, not a trend/history report (neither backing store retains history). If a future need arises for trend analysis, that's a different, larger patch (would require adding persistence to one or both stores) — explicitly out of scope here and not attempted.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...chore/drive-state-divergence-audit?expand=1`
