# Homeostatic Drives — real tensions from the signal substrate, cadence-invariant pressure

> **Status:** Design proposal (proposal mode — changes what Orion feels and wants). **One cohesive spec**, full scope, no phased v-next. Supersedes the four-area arc's dependence on a functioning drive layer: this is the Phase 0 that makes Drives mean something.
>
> **Governing rule for this spec:** *starve the keyword cathedral.* No free-text matching anywhere. Every mapping is on **typed `signal_kind` + numeric dimension deviation**. A tension fires on **change/deviation**, never on signal presence.

## Arsonist summary

Three pieces of theater, burned together:

1. **Pressure is a tick-cadence artifact, not cognition.** `DriveEngine` applies `soft_saturate(x)=1−e^(−1.8x)` to `prev·decay`, and that function has a **stable non-zero fixed point at ~0.731** with `f(p)>p` for all `p<0.731`. With the post-heartbeat frequent cadence (`decay≈1`), all six drives inflate to 0.731 and read "active" — verified live: every drive pinned to 0.7309, identical to four decimals, zero tensions. Pre-heartbeat (sparse ticks, `decay→0`) they instead collapsed to ~0. Either way pressure tracks **tick rate, not tensions.**
2. **Tensions are starved.** They fire in **0.064%** of ticks (284 of 444,943 audits), because the only producers are turn-`turn_effect`, self-state, feedback frames, and world-pulse gaps — all rare.
3. **A rich, real signal bus sits right next to Drives, unused.** `orion:signals:*` carries normalized `OrionSignalV1` with real varying telemetry (biometrics `homeostasis/strain/thermal/power`, mesh health, spark affect) at ~55/s, and Drives consume **none** of it.

The drive-attribution PR already routes *decisions* around the flat pressure (dominant comes from per-tick tension attribution, returns `None` on an all-zero tick). So the fix is not the decision wiring — it is **(a) feed real tensions from the signal substrate, and (b) replace the fixed-point pressure math with a cadence-invariant leaky integrator that rests at zero.**

## Executive summary

Model each drive as a **homeostatic leaky integrator over signal-deviation impulses**:

```
pressure_d(t) = pressure_d(t_prev)·exp(−Δt_wall/τ_d)              # continuous decay, wall-clock
              + Σ impulse(signal)·(1 − pressure_d(t_prev)·decay)   # headroom-scaled accrual
```

- **Impulses come from deviations, not presence.** A per-signal-kind baseline (EWMA) tracks the expected value of each meaningful dimension; a tension is minted only when a dimension deviates past a threshold (homeostasis drops, strain rises, valence goes negative, coherence falls, a failure event occurs). Constant signals (the 55/s `scene_state` flood) produce zero deviation → zero tensions. `is_stub_signal` filters stubs first.
- **Impulse → drive via one structural YAML map** keyed on typed `signal_kind` (never text). Magnitude = deviation × confidence.
- **The math rests at zero and is cadence-invariant.** `Σimpulse=0 → pressure→0`; decay is on real elapsed wall-seconds, so 100 ticks/second and 1 tick/100s converge to the same pressure for the same deviation history. No `soft_saturate` fixed point.
- **Baseline adaptation is the rate-limiter.** A sustained deviation is habituated as the EWMA baseline catches up, so a flapping/persistent source can't restorm — homeostatically correct and O(N)-safe.
- **Attribution is unchanged.** New tensions flow into `compute_tick_attribution` automatically → `dominant_drive` and goals come alive with no decision-path edit.

Sources unified in one spec: **`OrionSignalV1`** (`orion:signals:*`), **failures** (`orion:system:error`, cortex `exec_step_failed` grammar, `rdf:error`, `vision:edge:error`), and **equilibrium health** (`orion:equilibrium:snapshot`).

## Ground truth (verified, live where possible)

- **Pressure engine:** `orion/spark/concept_induction/drives.py::DriveEngine.update` — `raw=prev·decay+Σ(mag·impact)`, `pressure=1−e^(−1.8·raw)`. `DRIVE_KEYS=(coherence,continuity,capability,relational,predictive,autonomy)`. Live payloads: all six = 0.7309, `dominant=None`, `tension_kinds=[]`, `tick_attribution` all-zero. Fixed-point math confirmed.
- **Decision path already fixed:** `orion/spark/concept_induction/drive_attribution.py::dominant_drive_from_attribution` returns `None` when attribution max ≤ 0 → dominant/goal-origin come from tensions, not flat pressure.
- **Flat pressure is still consumed (behavior-change surface):** `orion/autonomy/policy_act.py:66,191` reads `drive_state.pressures.get("predictive")` into the capability context; `orion/autonomy/summary.py:337-339` reads `drive_pressures>=0.6`; `orion/identity/snapshot.py` records `active_drives`. Fixing the math changes these — real, must gate.
- **Current tension producers:** `tensions.py` (`extract_tensions` ← turn_effect; `extract_tensions_from_self_state`; `extract_tensions_from_feedback`; `derive_pressure_competition_tensions` — dead while pressures flat), `substrate_metabolism.metabolize_substrate_signals` (world-pulse gap → predictive). Firing rate 0.064%.
- **Signal substrate is real and well-built:** `orion/signals/models.py::OrionSignalV1` (`signal_kind`, `organ_id`, `dimensions{level,trend,volatility,valence,coherence,novelty,salience,arousal,confidence}`); adapters in `orion/signals/adapters/` (power_guard, vision, chat_stance, social_memory, cortex_gateway, journaler, result, topic_foundry); signal-gateway with dedup + OTEL. **Live 30s tap:** `scene_state` 1661 (~55/s, flat `{0.5,0.5}` — stub), `biometrics_state` rich real telemetry, `mesh_health` real, `spark_signal` 0.5 defaults.
- **Stub filter exists:** `orion/signals/stub_detection.py::is_stub_signal` — flags `dims=={level,confidence}=={0.5,0.5}` w/o `source_event_id` (exactly the scene_state flood) and `"stub adapter"` notes.
- **No self-preservation drive.** Six fixed drives; biometrics somatic signals (homeostasis/strain/thermal/power) have no drive home. Real gap; mapped onto capability/continuity here, not solved with a new drive.
- **Dedup precedent:** `bus_worker.py` keeps `recent_event_seen: Dict[str,datetime]`.

## Core problem

The decision path is ready; the substrate feeding it is starved and its pressure math is a cadence artifact. **Mint honest, deviation-triggered, rate-limited tensions from the signal/failure/health traffic already on the bus, and replace the fixed-point pressure math with a cadence-invariant leaky integrator that rests at zero — reusing the existing stub filter and typed signal envelopes, with zero free-text matching.**

## Design principles / hard constraints

1. **Deviation, not presence.** Tensions fire on a dimension deviating from its adapted baseline past a threshold — never on a signal merely arriving. This is what starves the flood.
2. **Structural, not lexical.** Mapping is keyed on typed `signal_kind` + named dimension. No substring matching. This supersedes and must not reintroduce the `autonomy/reducer.py` `hit()` keyword table.
3. **Cadence-invariant, rest-at-zero.** Wall-clock decay; no input → pressure → 0; identical pressure for the same deviation history regardless of tick rate.
4. **Adaptation is the rate-limiter.** Sustained deviations habituate via EWMA baseline; hard per-(source,kind) caps as a backstop. Collections bounded (`[[feedback_execution_merge_cap]]`).
5. **Reuse the substrate.** Consume existing `OrionSignalV1`; filter with existing `is_stub_signal`; feed existing `compute_tick_attribution`. Invent no new signal bus, no new drive.
6. **Six drives stay.** Somatic/health deviations map onto `capability`/`continuity`. The missing self-preservation drive is named as a gap, not built here.
7. **Proposal mode, flag-gated.** Changes what Orion feels/wants and alters `policy_act` predictive pressure. Default off, clean rollback.

## The unified model

### 1. Pressure: cadence-invariant leaky integrator (replaces `soft_saturate`)
Per drive `d`, on each tick with real elapsed `Δt = (now − prev_ts).seconds`:
```
decay      = exp(−Δt / τ_d)                       # τ_d wall-seconds (default 1800)
base       = pressure_prev_d · decay              # → 0 with no input; cadence-invariant
impulse_d  = Σ_over_signals  impact(signal, d)    # this tick's deviation-driven accrual, ≥0
pressure_d = clamp01( base + impulse_d · (1 − base) )   # headroom-scaled; bounded [0,1]
```
**Properties (all test-asserted):**
- `impulse=0 ⇒ pressure_d = base → 0` as ticks pass. **Rests at zero. No non-zero fixed point.**
- Cadence-invariant: two tick schedules with the same wall-times of the same impulses yield the same pressure (decay integrates over `Δt`, not tick count).
- Headroom scaling bounds to `[0,1]` without an `exp` transform, so there is no `f(p)>p` inflation.
- Per-drive independent: uniform pinning is impossible unless impulses are genuinely uniform.

Activation keeps hysteresis (`activate 0.62 / deactivate 0.42`) but now on an honest pressure.

### 2. Deviation → impulse (the anti-flood gate)
Maintain a per-`(signal_kind, dimension)` **baseline** `μ` and spread `σ` via EWMA (`μ ← (1−α)μ + α·x`, `α` default 0.1). For each meaningful dimension of an incoming (non-stub) signal:
```
z = (x − μ) / max(σ, σ_floor)                     # standardized deviation
deviation = relu( sign_for(dimension) · z − z_threshold )   # only "worse" moves count
impulse_contrib = k · deviation · confidence      # k scale, clamp per-signal
```
- `sign_for`: for `homeostasis`,`coherence`,`stability`,`valence` a **drop** is bad (negative z); for `strain`,`thermal`,`volatility`,`gpu_load`,failure severity a **rise** is bad. Encoded in the YAML map, not in code.
- A steady signal (`x≈μ`) → `z≈0` → **zero impulse**. The 55/s `scene_state` `0.5` flood contributes nothing (and is stub-filtered first anyway).
- A homeostasis drop from 0.82→0.55, or a `mesh_health` down transition, or a `system:error`, produces a real, sized impulse.

### 3. Structural signal→drive map (one YAML contract)
`config/autonomy/signal_drive_map.yaml`:
```yaml
version: v1
# keyed on typed signal_kind; per-dimension direction + drive weights. No text.
kinds:
  biometrics_state:
    dimensions:
      homeostasis: { worse: down, drives: { capability: 0.6, continuity: 0.3 } }
      strain:      { worse: up,   drives: { capability: 0.8 } }
      thermal:     { worse: up,   drives: { capability: 0.5 } }
  mesh_health:
    dimensions:
      level:       { worse: down, drives: { capability: 0.7, continuity: 0.5 } }
  spark_signal:
    dimensions:
      coherence:   { worse: down, drives: { coherence: 1.0 } }
      valence:     { worse: down, drives: { relational: 0.6, continuity: 0.3 } }
      novelty:     { worse: up,   drives: { predictive: 0.7 } }
  failure_event:                       # synthesized kind for §sources.b
    dimensions:
      severity:    { worse: up,   drives: { capability: 0.9, coherence: 0.5 } }
```
Loader validates every drive ∈ `DRIVE_KEYS` and every `worse ∈ {up,down}`. Unmapped `signal_kind` → no tension (graceful `None`).

### 4. Sources (all mint the same `TensionEventV1` shape)
- **(a) `OrionSignalV1`** from `orion:signals:*` → deviation gate → impulse. Primary source.
- **(b) Failures** — `orion:system:error`, cortex grammar `semantic_role="exec_step_failed"`, `orion:rdf:error`, `orion:vision:edge:error` → normalized to a synthetic `failure_event` signal with `severity` dimension → deviation gate (severity vs baseline) → impulse.
- **(c) Equilibrium** — `orion:equilibrium:snapshot`; a service **transition** to `down`/missing is a `mesh_health`-class deviation (edge-triggered, not per-snapshot level) → impulse.

All three feed one adapter path → `TensionEventV1(kind="tension.signal.v1" | "tension.failure.v1" | "tension.health.v1", magnitude=Σimpulse_contrib, drive_impacts=mapped)` → existing `compute_tick_attribution`.

### 5. Rate-limit / dedup / adaptation
- **Adaptation (primary):** the EWMA baseline habituates sustained deviations — a persistent strain becomes the new normal and stops minting tensions. This is the world-class mechanism (a mind stops feeling a constant background pressure).
- **Hard backstop:** per-`(organ_id, signal_kind)` cap `N` tensions per rolling `window` (default 3 / 60s), deduped by signature, reusing the `recent_event_seen` pattern. Prevents a broken adapter from storming before the baseline adapts.
- **Source precedence:** if two sources describe the same underlying event (feedback-frame vs signal), the signal path yields (feedback already mints a tension). One underlying event → one tension.

## Contracts

### Schemas (`orion/core/schemas/drives.py`)
- New tension `kind`s registered: `tension.signal.v1`, `tension.failure.v1`, `tension.health.v1` (reuse `TensionEventV1`; no new fields required — `drive_impacts` + `magnitude` carry everything). Optional additive `provenance` note carries `signal_kind`/`organ_id`/`z` for audit.
- **No new drive key.** `DRIVE_KEYS` unchanged.

### New config
- `config/autonomy/signal_drive_map.yaml` (above).

### Channels consumed (add to `orion/bus/channels.yaml` consumer docs; no new published channels)
`orion:signals:*`, `orion:system:error`, `orion:grammar:event` (filtered to `exec_step_failed`), `orion:rdf:error`, `orion:vision:edge:error`, `orion:equilibrium:snapshot`.

### Registry
Register the three tension kinds in `orion/schemas/registry.py`; producer + consumer tests per §6 contract rules.

## Architecture / data flow

```text
orion:signals:*  ──► is_stub_signal? ─drop─► (55/s scene_state flood dies here)
orion:system:error / exec_step_failed / rdf:error / vision:edge:error ──► failure_event
orion:equilibrium:snapshot ──(down transition)──► mesh_health deviation
        │
        ▼
  DeviationGate (EWMA baseline μ,σ per signal_kind.dimension) ──► impulse (0 if steady)
        │  structural signal_drive_map.yaml (typed, no text)
        ▼
  TensionEventV1(kind=tension.{signal,failure,health}.v1, drive_impacts, magnitude)
        │  rate-limit / dedup / precedence
        ▼
  compute_tick_attribution  [UNCHANGED]  ─► dominant_drive ─► goals
        │
        ▼
  DriveEngine.update  [NEW leaky-integrator math] ─► honest pressures/activations
        ─► policy_act predictive_pressure, summary, identity snapshot (now honest)
```

## Producers & consumers (files)
- **New pure modules:** `orion/autonomy/deviation_gate.py` (EWMA baseline + z + impulse), `orion/autonomy/signal_tension.py` (`signal_to_tension`, failure/health normalizers), `orion/autonomy/signal_drive_map.py` (loader), `orion/autonomy/tension_ratelimit.py` (caps + precedence).
- **Modified:** `orion/spark/concept_induction/drives.py` (leaky-integrator `update`; delete `_soft_saturate` fixed-point), `orion/spark/concept_induction/bus_worker.py` (subscribe new channels; wire adapter + rate-limit before attribution).
- **Reused:** `orion/signals/stub_detection.py::is_stub_signal`, `orion/spark/concept_induction/drive_attribution.py` (unchanged).
- **New config:** `config/autonomy/signal_drive_map.yaml`.

## Env / config
```
ORION_HOMEOSTATIC_DRIVES_ENABLED=false     # master switch (proposal mode)
ORION_DRIVE_LEAKY_MATH_ENABLED=false       # gates ONLY the pressure-math swap (behavior-changing: policy_act predictive_pressure)
DRIVE_DECAY_TAU_SEC=1800
DEVIATION_EWMA_ALPHA=0.1
DEVIATION_Z_THRESHOLD=1.5
DEVIATION_SIGMA_FLOOR=0.02
SIGNAL_TENSION_IMPULSE_K=0.25
SIGNAL_TENSION_CAP_PER_WINDOW=3
SIGNAL_TENSION_WINDOW_SEC=60
```
Two flags on purpose: the source adapter (`ORION_HOMEOSTATIC_DRIVES_ENABLED`) and the math swap (`ORION_DRIVE_LEAKY_MATH_ENABLED`) can be rolled independently, because the math swap alone changes `policy_act`. After edits: `python scripts/sync_local_env_from_example.py`.

## Observability
- Each minted tension logs `{signal_kind, organ_id, dimension, x, μ, z, impulse, drive_impacts}` with correlation id.
- Metrics: `signal_tensions_minted_total{kind,drive}`, `signal_tensions_suppressed_total{reason: stub|steady|cap|precedence}`, `drive_pressure_gauge{drive}`, `drive_active_gauge{drive}`.
- Debug: `GET /autonomy/drives/live` returns current pressures, last N minted tensions, and per-signal-kind baselines.
- **Runtime-truth acceptance:** with the flag on, `drive_pressure_gauge` must show **differentiated, non-pinned** values that move with real signals, and go to ~0 in a quiet window.

## Tests (gate — deterministic, <2s)
`orion/autonomy/tests/` + `orion/spark/concept_induction/tests/`:
1. **Anti-flood:** 1000 identical `scene_state{0.5,0.5}` signals → `is_stub_signal` drops all → **0 tensions**.
2. **Steady non-stub:** 1000 `biometrics_state` at constant homeostasis=0.8 → baseline adapts → **0 tensions** after warm-up.
3. **Real deviation:** homeostasis 0.82→0.55 → one `capability`+`continuity` tension, magnitude ∝ z.
4. **Rest-at-zero:** any pressure, then N ticks with no impulse → pressure → 0 (assert `< 1e-3`). Regression guard against the fixed point.
5. **Cadence-invariance:** same impulse history at 100 ticks/s vs 1 tick/100s → equal final pressure (within ε).
6. **No uniform pin:** distinct impulses per drive → distinct pressures (never all-equal).
7. **Structural map:** every mapped drive ∈ `DRIVE_KEYS`; unmapped `signal_kind` → `None` (no tension); no code path reads signal text.
8. **Rate-limit backstop:** a 100-event failure storm in 1s → ≤ cap tensions; precedence: feedback-frame + signal for same event → 1 tension.
9. **Failure/health normalizers:** `system:error` → `failure_event` severity impulse; equilibrium `ok→down` transition → one tension; `down→down` (no transition) → none.
10. **Attribution liveness:** minted tensions produce non-zero `tick_attribution` → `dominant_drive` non-None (proves the loop comes alive).

## Evals
`orion/autonomy/evals/run_homeostatic_drives_eval.py`: replay a captured 1-hour signal stream (incl. the scene_state flood + a real biometrics strain episode + an injected failure). Assert (a) tension rate rises from ~0.06% to a target band and tracks real deviations, (b) pressures differentiate and rest at zero in quiet spans, (c) the scene_state flood contributes 0, (d) `dominant_drive` distribution reflects the injected events (not alphabetical "autonomy", not constant). Report tension rate, per-drive pressure trajectories, suppression breakdown.

## Failure modes & mitigations
- **Storm replaces starvation** → deviation gate + EWMA adaptation + hard caps (tests 1,2,8). This is the primary risk and the design's center of gravity.
- **Math swap breaks `policy_act`** → separate flag `ORION_DRIVE_LEAKY_MATH_ENABLED`; `policy_act` predictive gating re-validated; with the new math, predictive pressure is 0 unless a predictive tension fired (correct, but changes when readonly fetch is eligible — evaluated in the eval).
- **Baseline cold-start** → warm-up window before impulses count; `σ_floor` prevents divide-by-zero and over-sensitivity early.
- **Double-count with feedback frames** → source precedence (test 8).
- **Somatic signals have no true drive** → mapped to capability/continuity; the self-preservation gap is a named non-goal, not silently conflated.

## Migration / behavior-change note
`ORION_DRIVE_LEAKY_MATH_ENABLED=true` changes `drive_state.pressures` semantics (rest-at-zero vs pinned 0.731), which flows to `policy_act` predictive-pressure, `summary.py` (≥0.6 checks), and identity snapshots. Roll the source adapter first (`ORION_HOMEOSTATIC_DRIVES_ENABLED`) to populate tensions/attribution with the old math (safe — decision path is attribution-based), then roll the math swap once the eval confirms `policy_act` behavior.

## Privacy / safety
Tensions carry `signal_kind`/dimension/z numerics and drive names — no raw private content. Proposal-mode disable: both flags off → exact current behavior, no residue. Somatic telemetry is already emitted on the bus; this consumes it, exposes nothing new.

## Acceptance checks
- [ ] Live (flag on): `drive_pressure_gauge` shows differentiated, non-pinned values that move with biometrics/health/failures and decay to ~0 in a quiet window.
- [ ] The 55/s `scene_state` flood mints **0** tensions.
- [ ] A real homeostasis drop / a `mesh_health` down transition / a `system:error` each mint exactly one correctly-mapped tension.
- [ ] `dominant_drive` reflects real events (not alphabetical "autonomy", not constant None).
- [ ] Rest-at-zero + cadence-invariance tests pass.
- [ ] Both flags off → byte-identical current behavior.

## Non-goals
- **No 7th "self-preservation" drive.** Somatic deviation maps to capability/continuity; the gap is documented for a future proposal, not built here.
- Not touching `orion/signals/*` producers — consume the substrate as-is.
- Not reworking `drive_attribution.py` — it already consumes tensions correctly.
- Not removing the legacy `autonomy/reducer.py` keyword system in this spec (separate cleanup), but this spec must not extend it.
- No LLM anywhere in the tension/pressure path — deterministic substrate math.
