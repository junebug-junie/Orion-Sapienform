## Summary

- `_phi_from_self_state`'s valence formula carried a hardcoded `policy_ease=1.0` dead constant (the dimension it represented, `policy_pressure`, was already deleted from `SelfStateV1`). Deleted outright; weights renormalized to 0.625/0.375 over the two remaining live terms (agency_readiness, social_ease).
- The registry's SHADOW justification for valence ("no trained phi-encoder latent dimension correlates with any hedonic-adjacent felt dimension") was checked directly against the active encoder's real `manifest.json`/`probes.json` and was **false**: `agency_readiness` is one of the encoder's 8 real input features and correlates with 6 of 8 latents at `|r|` up to 0.686.
- Added `_agency_valence_proxy()`: a probe-weighted linear readout of the trained latent space, wired into `_golden_phi_overrides()` as a real (if imperfect) replacement for the heuristic whenever an encoder tick succeeds. Registry entry flipped SHADOW → COMPOSED.
- 8-angle code review surfaced a real second-order bug: valence could flip between two independently-computed, uncalibrated formulas tick-to-tick, and the raw diff reached live metacog prompts (`spark_phi_hint`/`spark_phi_narrative`) as an unearned swing. Fixed with source tracking + turn_effect delta suppression on a source swap, plus a `valence_source` observability field.
- Added finiteness guards and a noise floor (0.05) on the proxy's combined weight to prevent NaN/Inf propagation and tanh-saturation on statistically meaningless correlations.

## Outcome moved

Orion's felt-valence signal no longer contains a disguised constant contributing unconditionally to every tick, and — when the encoder is healthy — is sourced from a real, verified trained-model correlation instead of a hand-tuned heuristic. The formula-swap side effect that would have made this look like noisy/flapping affect in live metacog prompts is closed.

## Current architecture

`_phi_from_self_state()` (`services/orion-spark-introspector/app/worker.py`) computes a 4-axis heuristic (coherence/energy/novelty/valence) every tick from raw `SelfStateV1` dimensions. `_golden_phi_overrides()` replaces coherence/energy/novelty with native trained-encoder outputs (phi, delta_phi, recon_error) whenever an encoder tick succeeds; valence was previously left untouched on the claim that no trained analog existed for it — a claim never checked against the real encoder artifacts.

## Architecture touched

`services/orion-spark-introspector/app/worker.py`, `orion/self_state/inner_state_registry.py`. No schema, bus, or Docker changes — `valence_source` rides the existing free-form `SparkStateSnapshotV1.metadata` dict, avoiding the `orion-sql-writer` extra="forbid" rebuild trap hit twice earlier this session.

## Files changed

- `services/orion-spark-introspector/app/worker.py`: deleted the `policy_ease` dead constant; added `_agency_valence_proxy()` and wired it into `_golden_phi_overrides()`; added `_PHI_PREV_VALENCE_SOURCE` module state, turn_effect delta suppression on a source swap, and a `valence_source` metadata field.
- `orion/self_state/inner_state_registry.py`: `phi_heuristic.valence` flipped `SHADOW` → `COMPOSED`; `cognition_consumers` corrected to `spark_phi_hint` (real consumer, was incorrectly listing `spark_embodiment_narrative`, which never reads valence); notes rewritten with the real verified numbers and the stability fix.
- `services/orion-spark-introspector/tests/test_phi_reward_emit.py`: 9 new/updated tests (proxy math, finiteness guards, noise floor, formula-fallback path, formula-override path, source-swap suppression regression); `_write_tiny_encoder` extended with an optional `probes` param to remove test duplication.

## Schema / bus / API changes

None. `valence_source` is a new key inside the existing free-form `metadata` dict on `SparkStateSnapshotV1`, not a schema field — no consumer rebuild required.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=.:services/orion-spark-introspector pytest \
  tests/test_inner_state_registry_gate.py services/orion-spark-introspector/tests -q \
  --deselect services/orion-spark-introspector/tests/test_inner_state_emit.py::test_inner_features_settings_defaults
155 passed, 1 skipped, 1 deselected

python scripts/check_inner_state_registry.py
inner_state_registry gate OK (9 entries checked)
```

The one deselected test (`test_inner_features_settings_defaults`) fails identically on a clean `origin/main` checkout of this same worktree with no `.env` file present (gitignored, absent in a fresh `git worktree add`) — a pre-existing env-parity artifact of worktree usage, unrelated to this patch. Confirmed by reproducing the same failure against unmodified `main`.

## Evals run

None applicable — this is a formula/signal-plumbing fix, not a model retrain.

## Docker/build/smoke checks

Not deployed this patch. `valence_source` is additive metadata, `_agency_valence_proxy` fails closed to `None` (heuristic fallback) on any missing/malformed probes, so this is safe to deploy via the normal `orion-spark-introspector` rebuild when ready.

## Review findings fixed

- Finding: `phi_heuristic.valence`'s `cognition_consumers` listed `spark_embodiment_narrative` (never reads valence) instead of `spark_phi_hint` (the real consumer).
  - Fix: corrected the tuple.
  - Evidence: `grep -n "phi.get(\"valence\")" services/orion-cortex-exec/app/spark_narrative.py` confirms `spark_phi_hint` reads it; `spark_embodiment_narrative` only reads `dominant_node`/`dominant_node_reason`.
- Finding: `_agency_valence_proxy` had no finiteness guard; a corrupted `weights.npz` or unbounded latent could publish NaN to the live bus.
  - Fix: `math.isfinite()` checks on both latent values and probe weights, plus a final finiteness check before returning.
  - Evidence: new test `test_agency_valence_proxy_ignores_nonfinite_latent_or_weight`.
- Finding: the `weight_total` near-zero guard (1e-9) let a single noise-level correlation (e.g. 0.0001) combined with a large unclamped latent saturate the tanh output to ±1 on meaningless evidence.
  - Fix: raised the floor to 0.05 (real observed correlations are 0.35–0.69 in magnitude, so this only rejects noise).
  - Evidence: new test `test_agency_valence_proxy_noise_level_weight_returns_none`.
- Finding (most material — cross-file trace): valence swapping formulas tick-to-tick produced spurious `turn_effect` deltas reaching live metacog prompts via `spark_phi_hint`/`spark_phi_narrative` as an unearned valence swing.
  - Fix: `_PHI_PREV_VALENCE_SOURCE` tracks which formula fired each tick; `turn_effect["valence"]` is forced to `0.0` specifically on a source-swap tick, not generalized to coherence/energy/novelty (out of scope, don't share this axis's saturate-prone tanh squashing).
  - Evidence: new regression test `test_turn_effect_valence_delta_suppressed_across_a_source_swap`, which fires two ticks with deliberately opposite-sign heuristic/proxy valence and asserts the reported delta is exactly `0.0`.
  - Correction during verification: an initial review pass claimed this reaches `orion.spark.concept_induction.tensions.py`'s `tension.distress.v1` → DriveEngine. Traced directly and ruled out: that pipeline's `substrate.self_state.v1` path uses `extract_tensions_from_self_state`, which reads raw `SelfStateV1.dimensions` directly and never touches `phi_now`/valence; `channel_spark_state_snapshot` (this patch's output channel) is not in concept-induction's `intake_channels` at all. The confirmed-real consumer is `spark_phi_hint`/`spark_phi_narrative`; code comments were corrected to say so rather than leave the overstated claim in place.
- Finding: no observability of which formula produced a given tick's valence, unlike the existing `inner.headline_source="encoder"` precedent for phi.
  - Fix: added `valence_source` ("proxy" or "heuristic") to `SparkStateSnapshotV1.metadata`.
  - Evidence: same regression test asserts `snap_payload.metadata["valence_source"] == "proxy"`.
- Finding: `_agency_valence_proxy`'s docstring overclaimed "real, verified-nonzero signal from the trained model" without disclosing that `agency_readiness` is itself an encoder input feature — making this closer to a lossy reconstruction of an already-available value than newly discovered structure.
  - Fix: docstring rewritten to state this limitation plainly and point to the registry entry for full numbers instead of duplicating them.
  - Evidence: `services/orion-spark-introspector/app/worker.py` docstring, `orion/self_state/inner_state_registry.py` notes.
- Finding: test duplicated ~35 lines of manifest/npz construction already covered by `_write_tiny_encoder`.
  - Fix: extended `_write_tiny_encoder` with an optional `probes` kwarg; both new tests now use it.
- Finding: a redundant second assertion in the policy_ease test added no coverage beyond the preceding exact-equality assert.
  - Fix: removed.
- Finding (CLAUDE.md "runtime truth beats config truth"): registry's "checked against the real numbers" claim is prose, not an attached artifact in the diff.
  - Disposition: `no_change_needed` — the claim cites exact, independently checkable values (matching this file's existing convention for other entries, e.g. "363 samples/24h confirmed 2026-07-12"); the underlying numbers were read directly from `/mnt/telemetry/models/phi/encoders/active/probes.json` and `manifest.json` during this session, not fabricated.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```

Not run this session — code is tested and reviewed but not deployed. `orion-sql-writer` does not need a rebuild (no schema fields added).

## Risks / concerns

- Severity: low
- Concern: the probe-weighted proxy is a post-hoc linear combination of Pearson coefficients (not fitted regression weights over an orthogonalized latent space), and is architecturally weaker than the native-output overrides for coherence/energy/novelty. Documented plainly rather than hidden.
- Mitigation: `valence_source` makes this observable at runtime; the fallback heuristic (now theater-free) is always available when the proxy can't fire.
- Severity: low
- Concern: not yet deployed/live-verified on Athena.
- Mitigation: fully covered by 17 unit/integration tests including a live-shaped two-tick regression test; safe, additive change with no schema/bus impact.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/valence-probe-readout
