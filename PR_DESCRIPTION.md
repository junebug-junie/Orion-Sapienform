## Summary

- Add `services/orion-spark-introspector/train/evals/eval_phi_encoder_health.py` — offline health report over **every** encoder version under `${TELEMETRY_ROOT}/models/phi/encoders/`
- Primary metrics are **recon** (p50/p95/mean) and **residual-after-headline-fit** (flags φ→headline identity collapse); φ↔headline correlation is reported but labeled as a supervised-target diagnostic, not success
- Each run includes traceable manifest metadata: `git_sha`, `trained_at`, `promoted_at`, train row count, corpus span, train/held-out loss, status, features_version
- Gate test coverage in `tests/test_phi_encoder_health_eval.py` (collapse detection, constant-φ false-positive guard, multi-run report)

## Outcome moved

Stops reading near-perfect φ↔headline “accuracy” as encoder health. Operators can compare all training runs side-by-side and catch identity-collapse (as seen on retired `v20260710-seedv4`) while confirming active `v20260710-seedv4-full` is clean.

## Current architecture

`scripts/fit_phi_encoder.py` trains φ to predict `headline` (`recon + 0.25·(φ−headline)²`). Existing spark `evals/` cover synthetic fit/reward gates only — no corpus-backed multi-run comparison.

## Architecture touched

- Service: `orion-spark-introspector` (train/evals + tests only)
- No schema/bus/API/env changes

## Files changed

- `services/orion-spark-introspector/train/evals/eval_phi_encoder_health.py`: multi-run health eval CLI + report
- `services/orion-spark-introspector/train/evals/README.md`: usage and metric contract
- `services/orion-spark-introspector/tests/test_phi_encoder_health_eval.py`: gate tests

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: read-only eval; does not change runtime inference

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: n/a
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest services/orion-spark-introspector/tests/test_phi_encoder_health_eval.py -q
# 7 passed
```

## Evals run

```text
.venv/bin/python services/orion-spark-introspector/train/evals/eval_phi_encoder_health.py \
  --corpus /mnt/telemetry/phi/corpus/inner_state.jsonl \
  --encoders-root /mnt/telemetry/models/phi/encoders
# ok=True active=v20260710-seedv4-full
# retired v20260710-seedv4 flagged COLLAPSE (near_id≈0.70)
```

## Docker/build/smoke checks

```text
No Docker rebuild required (offline eval + unit tests only).
```

## Review findings fixed

- Finding: low residual_std alone false-positive on constant-φ
  - Fix: collapse requires near_id frac **or** (tiny residual **and** slope≈1)
  - Evidence: `test_constant_phi_is_not_identity_collapse`

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: seed-v2 runs show high recon on today’s mixed corpus (expected feature-set drift); they are informational, not active-gate failures
- Mitigation: ranking and collapse gate scoped to active features_version / active symlink

## Test plan

- [ ] Run gate tests: `pytest services/orion-spark-introspector/tests/test_phi_encoder_health_eval.py -q`
- [ ] Run live report against `/mnt/telemetry/models/phi/encoders` and confirm active is `ok=True`
- [ ] Confirm retired `v20260710-seedv4` still shows `COLLAPSE` while `v20260710-seedv4-full` does not
