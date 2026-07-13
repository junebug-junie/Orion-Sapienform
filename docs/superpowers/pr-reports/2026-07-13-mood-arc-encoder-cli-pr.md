## Summary

- Adds `scripts/fit_mood_arc_encoder.py` (new): trains a shallow MLP autoencoder over windows of the mood-arc corpus (Item 1, PR #989) — `train` subcommand only, offline/manual, no bus wiring, no service.
- Fixes this session's spike findings baked into the design: Adam optimizer + mean-initialized decoder bias (the spec's vanilla-SGD/zero-init never converged, scored worse than a trivial mean-repeat baseline) and `hidden_dim=32`/`latent_dim=16` defaults (the spec's `hidden_dim=8`/`latent_dim=4` lacked capacity).
- Adds a two-tier gate: shuffle-baseline floor (hard-gated, unchanged `<0.5` threshold from the spec) plus an AR(1)-surrogate ceiling (diagnostic only, not yet calibrated) — because raw-signal ACF analysis traced most of the corpus's autocorrelation to a known, deliberate decay mechanism (`BIOMETRICS_FIELD_DECAY_RATE=0.92`) that a single shuffle gate could pass without learning anything Orion-specific.
- Adds a purged/embargoed temporal train/held-out split (`purged_temporal_split`) instead of a naive random window split, plus a block-bootstrap 95% CI on the floor ratio.
- Extends `MoodArcEncoderManifestV1` with 5 new `Optional[...] = None` fields for this methodology (kept optional so the sibling schema PR's existing `test_mood_arc_encoder_schema.py` still passes unmodified).
- Registers `mood_arc_encoder.v1` in `orion/self_state/inner_state_registry.py`, closing the exact gap the sibling schema PR (`feat/mood-arc-encoder-manifest-schema`) flagged and correctly declined to fill (no producer existed until this patch).
- Updates both `services/orion-spark-introspector/README.md` and `orion/self_state/README.md` to describe what was actually built (including deviations from the original spec's defaults/gate design).

## Outcome moved

`scripts/check_inner_state_registry.py` now passes (was failing before this patch — `MoodArcEncoderManifestV1` matched the "mood" keyword heuristic with no registry entry). A real, working `train` CLI now exists for roadmap item 2, verified against the live 20,996-row corpus.

## Current architecture

Before this patch, Item 2 of `docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md` was design-only: a schema (`MoodArcEncoderManifestV1`, sibling branch) with no producer, and `check_inner_state_registry.py` failing as a result. No windowing, training, or gating code existed.

## Architecture touched

`scripts/` (new training CLI), `orion/schemas/telemetry/mood_arc.py` (schema extension), `orion/self_state/inner_state_registry.py` (new entry), two service/self-state READMEs, one new test file. No bus channels, no service runtime, no config/env.

## Real training run (live corpus, not synthetic)

```bash
python scripts/fit_mood_arc_encoder.py train \
  --corpus /mnt/telemetry/mood_arc/corpus/mood_arc.jsonl \
  --hidden-dim 32 --latent-dim 16 --epochs 500 --lr 0.003 \
  --purge-gap-windows 6 --out /tmp/mood-arc-encoders/v1-candidate-real2
```

- `rows`: 20,996 (spanning `2026-07-13T06:26:29Z` onward)
- `windows_total`: 1387 (`windows_train`: 1173, `windows_held_out`: 208)
- **`floor_ratio`: 0.439** (need `< 0.5`) → **`floor_pass`: True**
- **`floor_ratio_ci_low`/`ci_high`: 0.394 / 0.485** (95% block-bootstrap CI, entirely under the 0.5 threshold — a robust pass, not a borderline one)
- `ceiling_ratio`: 0.785 (diagnostic only, not gated — real loss beats the AR(1)-surrogate null by ~22%)
- Channel variance reported (not gated): `coherence=0.00094`, `energy=0.00237`, `novelty=0.1196`, `valence=0.098` — coherence/energy are real but low-variance; no threshold invented for this.
- Runtime: ~44s on the full live corpus.

## Files changed

- `scripts/fit_mood_arc_encoder.py` (new, 793 lines): `train` CLI, windowing, purged split, Adam-trained autoencoder, two-tier gate, AR(1) surrogate, block bootstrap, `compute_window_probes`.
- `tests/test_mood_arc_encoder_fit_script.py` (new): end-to-end train run on synthetic periodic-pattern corpus (floor gate passes), negative control (untrained weights on the *same structured* corpus fail the floor gate), `purged_temporal_split` boundary/exclusion/error tests, `build_windows` gap-break test (real recorded 8.09s gap vs. default `max_gap_sec=6.0`), `generate_ar1_surrogate_windows` sanity test.
- `orion/schemas/telemetry/mood_arc.py`: `MoodArcEncoderManifestV1` gains `purge_gap_windows`, `ar1_surrogate_loss`, `ceiling_ratio`, `floor_ratio_ci_low`, `floor_ratio_ci_high` (all `Optional[...] = None`, additive, `extra="forbid"` unchanged).
- `orion/self_state/inner_state_registry.py`: new `mood_arc_encoder.v1` entry + import.
- `services/orion-spark-introspector/README.md`, `orion/self_state/README.md`: document what was actually built, including deviations from the original spec doc.

## Schema / bus / API changes

- Added: 5 optional fields on `MoodArcEncoderManifestV1` (see above).
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing consumers — additive only.
- Compatibility notes: existing `test_mood_arc_encoder_schema.py`'s manifest fixtures still construct valid manifests since new fields default to `None`.

## Env/config changes

None. This is a disk-only offline script; no `.env`/`.env_example`/compose/requirements touched.

## Tests run

```text
.venv/bin/python -m pytest tests/test_mood_arc_encoder_fit_script.py tests/test_mood_arc_encoder_schema.py services/orion-spark-introspector/tests/test_mood_arc_corpus.py -q
=> 17 passed
.venv/bin/python scripts/check_inner_state_registry.py
=> inner_state_registry gate OK (12 entries checked)
.venv/bin/python -c "import yaml"
=> OK
```

## Evals run

The real live-corpus training run above (`floor_ratio=0.439`, `floor_pass=True`, CI `[0.394, 0.485]`) is this patch's eval — there's no separate `evals/` harness for this offline script (mirrors `fit_phi_encoder.py`, which also has no dedicated eval harness beyond its own promote gate).

## Docker/build/smoke checks

Not applicable — no runtime/deployment files touched.

## Review findings fixed

- Finding: `test_negative_control_untrained_weights_fail_floor_gate` used pure i.i.d.-noise windows with no temporal structure, so shuffling destroyed nothing and the test couldn't actually distinguish an untrained from a trained encoder (any model scored `floor_ratio≈1.0` on it).
  - Fix: rewrote the test to use the same structured (real periodic-pattern) synthetic corpus as the end-to-end pass test, so there is real structure an untrained model fails to exploit.
  - Evidence: `tests/test_mood_arc_encoder_fit_script.py::test_negative_control_untrained_weights_fail_floor_gate` now passes against structured data.
- Finding: the AR(1) null-model's training-only cutoff (`cutoff_ts`) was derived from the last training window's *end* timestamp, which — for 50%-overlapping windows — could still overlap the first held-out window's raw ticks if `--purge-gap-windows` were overridden to a small value (e.g. `0`), leaking held-out data into `fit_ar1_per_channel`'s fit regardless of the configured purge size.
  - Fix: window construction now tracks each window's start timestamp too; `cmd_train` derives the AR(1) cutoff strictly from the first held-out window's own start timestamp, correct for any `--purge-gap-windows` value including `0`.
  - Evidence: `cutoff_ts = start_ts[held_start]` in `cmd_train`; full test suite + live run re-verified after the fix.
- Finding: `_per_window_losses` (called every epoch for held-loss tracking, plus several more times per training run) always computed a full backward pass even though gradients were never used there, roughly doubling held-set evaluation cost.
  - Fix: added a forward-only `recon_loss()`; all loss-only call sites switched to it, `recon_loss_and_grads` reserved for the actual training step.
  - Evidence: real live-corpus run time dropped from ~53s to ~44s after the fix; all tests still pass.
- Noted, not fixed (out of scope per this task's hard constraints): `_git_sha()` and `_load_jsonl()` in the new script are near-duplicates of the same helpers in `scripts/fit_phi_encoder.py`. Extracting a shared helper would require touching `fit_phi_encoder.py`, explicitly forbidden as a read-only reference file for this task. Flagged for a future cross-cutting cleanup.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: Low
- Concern: `_git_sha()`/`_load_jsonl()` duplication with `scripts/fit_phi_encoder.py`.
- Mitigation: flagged above; a future patch touching both scripts could extract these into `orion/telemetry/corpus_rotation.py` or a new shared module.
- Severity: Low
- Concern: `ceiling_ratio` has no calibrated pass/fail threshold yet — by design, not an oversight, but means it can't yet catch a regression toward "merely re-learning the known decay filter" on its own.
- Mitigation: recorded in every manifest going forward; calibrate once multiple training runs exist (noted in schema docstring and both READMEs).

## Merge order

This branch is built on top of the not-yet-merged `feat/mood-arc-encoder-manifest-schema` (branches from its tip commit, not `main`). **Merge that PR first**, then this one — rebase onto `main` if needed once the schema PR lands.

## PR link

`gh` is unauthenticated in this environment — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/mood-arc-encoder-cli
