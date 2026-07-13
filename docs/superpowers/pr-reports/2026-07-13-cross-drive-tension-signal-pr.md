# PR report: cross-drive tension detector (pure function, unwired)

## Summary

- New `orion/spark/concept_induction/drive_tension.py`: a pure function, `detect_drive_tensions(pressures, activations) -> list[DriveTensionV1]`, over `DriveEngine`'s existing 6-key pressure/activation output shape.
- Deliberately scoped to prove the *definition* is sound before touching anything real — zero wiring into `DriveEngine`, `bus_worker.py`, any bus channel, or the schema registry. This round is the "smallest buildable slice" named explicitly in the accepted spec: a typed, documented, unit-tested definition on paper, nothing live.
- Definition (`INVERSE_COACTIVATION`): fires for a pair `(drive_a, drive_b)` when `drive_a` is active and `drive_b`'s pressure is below `DriveMathConfig.deactivate_threshold` (0.42, reused — no new magic number invented). Magnitude = `pressures[drive_a] * (1.0 - pressures[drive_b])`.

## Outcome moved

Nothing live changes. The outcome is a concrete, testable answer to "what would 'two drives are in tension' even mean" — the open question flagged as the riskiest keyword-cathedral candidate in the accepted brainstorm, now closed with a real definition instead of an unspecified concept.

## Current architecture (before this patch)

`DriveEngine` computes each of the 6 drives independently every tick; nothing detects or represents interaction between drives.

## Architecture touched

`orion/spark/concept_induction/` only — one new module, one new test file. No other seam touched.

## Files changed

- `orion/spark/concept_induction/drive_tension.py` (new): `DriveTensionV1` dataclass (`drive_a`, `drive_b`, `tension_kind`, `magnitude`) + `detect_drive_tensions()`. Module docstring states explicitly: correlational co-occurrence, not a causal claim.
- `orion/spark/concept_induction/tests/test_drive_tension.py` (new, 9 tests): no-tension baseline, single-pair tension, multiple simultaneous tensions, boundary-exactly-at-threshold (excluded) vs. just-below (included), empty input, and missing-key-in-either-dict degrading gracefully rather than raising.

## Schema / bus / API changes

None. No bus channel, no schema-registry entry, no producer, no consumer — by design for this round.

## Env/config changes

None.

## Tests run

```text
$ python -m pytest orion/spark/concept_induction/tests/test_drive_tension.py -q
9 passed

# Sibling suite re-run for regression confidence
$ python -m pytest orion/spark/concept_induction/tests/test_drives_leaky.py \
    orion/spark/concept_induction/tests/test_drive_attribution.py -q
15 passed
```
Independently re-run by the orchestrator (not just the implementing agent) — confirmed.

## Evals run

Not applicable — pure, deterministic, fully unit-testable function; the test suite itself is the eval-equivalent (a decidable spec of what "tension" means, per the acceptance bar in the accepted brainstorm).

## Docker/build/smoke checks

Not applicable — no runtime surface, no service, no Docker relevance.

## Review findings fixed

None from a separate review pass — implementing agent's own diff was self-contained (2 new files only, verified via `git status --short` before commit), no touches to `bus_worker.py`, `audit.py`, channels.yaml, or schema registry. Orchestrator independently re-read both files and re-ran tests; no discrepancies found.

## Restart required

```text
No restart required.
```
Nothing running consumes this module yet.

## Risks / concerns

- Severity: Low — a known environment issue was surfaced during this work (not caused by it): the shared `/tmp/orion-test-venv` has `numpy==2.5.1` installed, binary-incompatible with `thinc==8.2.5` (repo's locked `numpy==2.3.0` per `poetry.lock`), which breaks `import spacy` and therefore collection of every test under `orion/spark/concept_induction/tests/` for anyone using that shared venv. The implementing agent built an isolated venv for its own verification rather than touching the shared one (correctly, since other concurrent agents may depend on its current state) — but this is worth fixing separately, since it likely silently blocks other agents' test runs in this same test directory. Flagging for a follow-up, not fixed here (out of scope for this patch, and fixing a shared venv from inside a worktree is exactly the kind of shared-state action that needs its own careful handling, not a drive-by fix).

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/cross-drive-tension-signal?expand=1`
