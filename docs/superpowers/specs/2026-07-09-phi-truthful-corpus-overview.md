# φ truthful corpus — overview & sequencing

**Mode:** Design/proposal. Umbrella for three specs that make the φ encoder corpus honest before we re-fit. No build until Juniper green-lights.

## Why

Live investigation (2026-07-09) proved the seed-v3 φ corpus is half-garbage: **6 of 11 trainable dims are flat-zero**, so `fit_phi_encoder.py`'s variance gate correctly refuses to promote. Root causes are dead producers and fake features, not clock/accrual:

- `reasoning_present` = True in **1 of 29,165** execution runs — reasoning is only *detected* on the harness FCC lane; 97% brain-mode cortex-exec runs never emit it.
- `reasoning_load` is fake: `0.35 if reasoning_present else 0.05` (`grammar_extract.py:39`).
- `exec_step_fail_rate` / `execution_friction` fire in 0.3% of runs — Orion rarely fails; structurally low-signal.
- `execution_load` = `min(1, started_step_count/8)` — magic constant, saturates, spikes at 0.125.
- `coherence` / `continuity_pressure` / `social_pressure` are frozen SelfStateV1 theater (no channel feeds them).
- The `execution_trajectory` projection is **25 MB / 29,165 runs, unbounded since May 25**.

## The three specs

1. **`2026-07-09-reasoning-telemetry-adapter-design.md`** — the new capability. cortex-exec already computes reasoning diagnostics + completion tokens at `router.py:1389`; emit them per-call, assemble a windowed reasoning-activity projection in orion-thought, expose it for φ to read. Makes `reasoning_present` + `reasoning_load` + token-based `execution_load` truthful.
2. **`2026-07-09-phi-seedv4-feature-set-design.md`** — the φ change. Excise the theater trio, drop the sparse pair, add token-based `execution_load` + real `reasoning_load`/`reasoning_present` sourced from spec 1. Re-version seed-v3 → seed-v4.
3. **`2026-07-09-phi-corpus-hygiene-design.md`** — deterministic "no garbage in": cap/prune the execution_trajectory projection, and add an ingestion-time corpus-health gate so degenerate rows are rejected at write.

## Dependency graph & sequencing (agreed: hold the encoder for ONE re-version)

```
[spec 3 hygiene]  ──(independent, do anytime)──┐
[spec 1 reasoning adapter] ──feeds──> [spec 2 seed-v4] ──> fresh corpus ≥4h ──> single re-fit ──> gate ──> promote
```

- Spec 1 lands first (φ can't get honest reasoning/token features without it).
- Spec 2 depends on spec 1's `reasoning_activity` projection.
- Spec 3 is independent and can land in parallel (pure hygiene).
- Encoder stays `ORION_PHI_ENCODER_ENABLED=false` and the active seed-v2 symlink untouched until a seed-v4 encoder passes the gate. **No interim seed-v4.**

## Non-goals (whole program)

- Not enabling the encoder or the Step-3 Δφ reward loop.
- Not the full SelfStateV1 redesign (classifier-seam replacement of felt dimensions) — separate deferred thread.
- Not adding cortex-orch to the grammar seam (pure dispatch, 1:1 correlated with cortex-exec; revisit only if orch gains its own pre-dispatch deliberation).

## Global acceptance

- `scripts/diag.py` shows ≥8/N seed-v4 dims var>1e-6 on ≥4h fresh corpus.
- `fit_phi_encoder.py` corpus + promote gates pass on seed-v4.
- All three specs' service test suites green; code review clean per spec.
