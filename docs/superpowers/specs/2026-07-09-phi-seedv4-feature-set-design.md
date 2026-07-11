# φ seed-v4 feature set — excise theater, token-based load, real reasoning

**Mode:** Proposal (cognition-touching: changes what φ perceives). Depends on the reasoning-telemetry adapter spec. No build until sign-off.

## Arsonist summary

Re-version seed-v3 → **seed-v4** with an honest trainable feature set: excise the frozen SelfStateV1 theater trio, drop the two structurally-sparse execution dims, replace `execution_load` with a token-based *work* measure, and source real `reasoning_present`/`reasoning_load` from the new reasoning-activity projection. One re-version, encoder held until it lands.

## Current architecture

`services/orion-spark-introspector/app/inner_state.py`:
- `FELT_DIMENSIONS` (`:75`) — felt/self-state dims (incl. coherence, continuity_pressure, social_pressure, execution_pressure, reasoning_pressure, agency_readiness).
- `INFRA_ONLY_FELT = {reliability_pressure}` (`:95`); `ENCODER_EXCLUDED_FELT`, `DROPPED_DIMENSIONS`.
- `COGNITIVE_FEATURE_NAMES` (`:106`) — recall_gate_fired, reasoning_present, exec_step_fail_rate, execution_friction (from `execution_trajectory` runs, built at `:140`).
- `encoder_trainable_feature_names(features_version)` (`:114`) — seed-v3 = felt (minus excluded/infra) + overall_intensity + COGNITIVE_FEATURE_NAMES.
- `build_cognitive_features` (`:140`) sources the 4 cognitive slots from the substrate execution-trajectory projection.
- Fit: `scripts/fit_phi_encoder.py` — `DEFAULT_FEATURES_VERSION="seed-v3"`, corpus gate (min_rows 500, min_hours 4.0, variance_fraction 0.8), promote gate (degenerate/healthy recon-ratio ≥ 2.0). `INNER_FEATURES_VERSION` env (`services/orion-spark-introspector/.env_example:169`).

## Proposed seed-v4 trainable set

| feature | seed-v3 | seed-v4 | change |
|---|---|---|---|
| coherence | ✓ | ✗ | **excise** (theater, frozen) |
| continuity_pressure | ✓ | ✗ | **excise** (theater, frozen) |
| social_pressure | ✓ | ✗ | **excise** (theater, frozen) |
| execution_pressure | ✓ | ✓ | keep (self-state but LIVE, var 0.59) |
| reasoning_pressure | ✓ | ✓ | keep (self-state but LIVE, var 1.53) |
| agency_readiness | ✓ | ✓ | keep (LIVE) |
| overall_intensity | ✓ | ✓ | keep (LIVE) |
| recall_gate_fired | ✓ | ✓ | keep (LIVE, var 0.27) |
| reasoning_present | ✓ (dead) | ✓ (fixed) | source from `reasoning_activity` projection |
| exec_step_fail_rate | ✓ | ✗ | **drop** (0.3% signal) |
| execution_friction | ✓ | ✗ | **drop** (0.3% signal) |
| execution_load | — | ✓ (new) | `robust_scale(log1p(window completion_tokens_sum))`, step-count fallback |
| reasoning_load | (fake) | ✓ (real) | thinking/reasoning throughput from `reasoning_activity` |

**Resulting seed-v4 trainable dims:** execution_pressure, reasoning_pressure, agency_readiness, overall_intensity, recall_gate_fired, reasoning_present, execution_load, reasoning_load (+ any other already-live felt dims retained). Expected live after fixes: ≥8, clears the 0.8 variance gate robustly.

## Feature sourcing (depends on reasoning-adapter spec)

- `execution_load` and `reasoning_load` and `reasoning_present` come from spark-introspector's new `fetch_reasoning_activity()` → `ReasoningActivityV1`:
  - `execution_load = robust_scale(log1p(completion_tokens_sum))` via existing `RollingRobustScaler` (`inner_state.py:43`); fallback `log1p(started_step_count)` when the projection is dark.
  - `reasoning_load = robust_scale(thinking throughput)` — exact numerator pending the adapter's "separate thinking tokens?" question; 0 (truthful) when thinking disabled, never a 0.05 floor.
  - `reasoning_present = reasoning_present_rate > 0` for the window (or per-run boolean if the projection carries it).

## Implementation surface

- `inner_state.py`: add `features_version == "seed-v4"` branch to `encoder_trainable_feature_names`; add seed-v4 to `ENCODER_EXCLUDED_FELT` (theater trio); redefine cognitive slots for seed-v4 (drop 2, add execution_load/reasoning_load); update `build_cognitive_features` + `build_inner_state_features` to populate the new slots from `reasoning_activity`.
- `services/orion-spark-introspector/app/substrate_reads.py`: `fetch_reasoning_activity()` + base-URL setting.
- `services/orion-spark-introspector/app/settings.py` + `.env_example`: `INNER_FEATURES_VERSION=seed-v4`, orion-thought reasoning-activity URL.
- `scripts/fit_phi_encoder.py`: `DEFAULT_FEATURES_VERSION="seed-v4"`; verify `_feature_maps`/`_row_vector` resolve the new names; keep gates.
- `scripts/backfill_phi_corpus.py`: `--features-version seed-v4`.
- Tests: `tests/test_inner_state_*`, fit-script tests, new tests for execution_load robust-scaling + reasoning-load sourcing.

## Non-goals

- Not enabling the encoder / Step-3 reward.
- Not the full SelfStateV1 excision (execution_pressure/reasoning_pressure/agency_readiness stay; classifier-seam redesign is a separate thread).
- Not coherence redesign (excised for now, revisit under self-state thread).

## Failure modes / rollback

- Sourcing execution_load/reasoning_load from a dark projection → falls back (step-count / 0) truthfully, no crash (fail-open like existing fetches).
- seed-v4 is an additive re-version; seed-v3 corpus rows retained; encoder stays default-off + seed-v2 symlink until a seed-v4 encoder passes the gate.

## Acceptance checks

1. `encoder_trainable_feature_names("seed-v4")` returns the intended set; unit-tested.
2. On a fixture `ReasoningActivityV1`, execution_load/reasoning_load are finite, in-range, and vary across inputs; theater trio absent from the vector.
3. `scripts/diag.py` on ≥4h seed-v4 corpus: ≥8 dims var>1e-6.
4. `fit_phi_encoder.py` corpus + promote gates pass on seed-v4.
5. Flag/version-off path byte-identical to seed-v3 behavior; regression suite green.
6. Code review clean.

## Recommended next patch

After the reasoning adapter is live and accruing, add the seed-v4 branch + consumer, backfill a seed-v4 corpus, run `diag.py`, then a single fit → gate.
