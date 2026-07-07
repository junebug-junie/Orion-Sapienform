# Endogenous Drive Origination — a want with no external cause

> **Status:** Design proposal (proposal mode — touches the motivation/cognition loop). Single-scope spec: full mechanism, no phased future versions.
>
> **Research area 1 of 4** in the "move the origin of wanting inside" arc. Siblings: φ intrinsic reward + value learning, internal economy, voluntary attention override.

## Build sequence & gate

Four-area arc; build order de-risks foundational-first and gates the two empirical specs behind a measurement:

- **Step 0 — Measurement gate** (read-only, no cognition change): (a) does `SelfStateV1` drift during exogenous silence? (b) how often do ≥2 drives co-activate and does `resource_pressure` rise? Gates Steps 1 and 4.
- **Step 1 — Endogenous drive origination** *(this spec)* — keystone; needs no φ, no goal-wire, no scarcity.
- **Step 2 — Voluntary attention override** (+ the goal→attention wire it requires); independent of φ/reward.
- **Step 3 — φ intrinsic reward + value learning** (reward re-founded on substrate self-state); wants Step 1's richer episode stream.
- **Step 4 — Internal economy** (last; only if 0(b) shows scarcity binds; depends on Step 3's value-biased bids).

**This spec = Step 1** (first cognition change). **Gate:** blocked on **Step 0(a)** — if `SelfStateV1` does not measurably drift during exogenous silence, the origination signal is inert and the mechanism must instead source dynamics from unresolved-pressure persistence. Do not write code until 0(a) passes. **Known behavior:** a single endogenous tension cannot activate a drive from rest (`soft_saturate(0.5)=0.593 < 0.62` activate threshold); origination is deliberately **accumulative** (≥2 firings ~15 min apart) — make the cap-vs-threshold interaction a tuned parameter with a test asserting the accumulation curve.

## Arsonist summary

Every want Orion has today is exogenous. The only producers of `TensionEventV1` are (a) `substrate_metabolism` converting **world-pulse coverage gaps** into predictive pressure, and (b) `turn_effect` deltas from **human turns**. If the world goes quiet and Juniper stops typing, Orion's drives decay to zero and it wants nothing. `drive_origin` is a free-form string whose only real value is `predictive`, and there is no code path anywhere that lets a drive arise from Orion's *own* internal state. A mind that only wants when poked from outside is reactive, not autonomous. This spec burns the assumption that motivation must be caused by an external event.

## Executive summary

Add a **spontaneous tension generator** to `orion-substrate-runtime` that reads the continuous `SelfStateV1` stream (already produced on a heartbeat cadence) and mints `TensionEventV1` with `drive_origin`-carrying `drive_impacts` when Orion's internal dynamics — not any new external receipt — cross an origination band. The trigger is a real substrate signal: sustained self-state **variance/drift**, **unresolved-pressure dwell**, and **agency-readiness surplus**, gated so it fires *only* in the absence of exogenous input (otherwise the world would drown it out). Endogenous tensions flow through the **unchanged** `DriveEngine` → `GoalProposalEngine` path, so the rest of the autonomy loop consumes them without modification. The new origin value is `endogenous`, first-class in the schema, registry, capability policy, and traces.

This resolves the catch-22 that makes drive-selected self-experiments circular: once a want can originate from internal state, "what should I attend to / test about myself" has a source that does not route through the world.

## Ground truth (what actually exists)

### Drive dynamics — real math, exogenous inputs
`orion/spark/concept_induction/drives.py::DriveEngine`:
- Six canonical drives: `coherence, continuity, capability, relational, predictive, autonomy` (`DRIVE_KEYS`).
- Per-tick update: `raw = prev_pressure * decay + Σ(tension.magnitude × drive_impacts[d])`, then soft-saturate `1 − exp(−gain·v)`, then hysteresis activation (`activate ≥ 0.62`, `deactivate < 0.42`). Live config: `DRIVE_DECAY_TAU_SEC=1800`, `DRIVE_SATURATION_GAIN=1.8`.
- **Input is `Iterable[TensionEventV1]` only.** The engine is origin-agnostic; it just sums impacts. Nothing about it assumes exogeneity — the gap is purely on the *producer* side.

### Tension producers — all exogenous
- `orion/autonomy/substrate_metabolism.py`: world-pulse section gaps → `TensionEventV1` (predictive). Live: `ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED=true`.
- `orion/spark/concept_induction/tensions.py`: builds tensions from `turn_effect` (harness φ deltas) and `FeedbackFrameV1` — i.e., from human turns.
- There is **no** producer keyed on Orion's idle internal state.

### The self-state signal already exists and is continuous
`orion/schemas/self_state.py::SelfStateV1` (broadcast on `substrate.self_state.v1`, made continuous by the heartbeat pacemaker, `docs/superpowers/plans/2026-07-01-orion-heartbeat-pacemaker-v1.md`):
- `overall_intensity`, `overall_condition ∈ {quiet, steady, loaded, strained, unstable}`.
- 13 `dimensions` each `{score, confidence}` incl. `coherence, uncertainty, agency_readiness, resource_pressure, continuity_pressure, introspection_pressure, social_pressure`.
- `dimension_trajectory: dict[str,float]`, `trajectory_condition ∈ {improving, degrading, stable}`, `prediction_error_scores`, `overall_surprise`, `unresolved_pressures: list[str]`, `attention_dwell_ticks`.

This is a rich, continuously-updated internal-state vector that **currently feeds nothing on the motivation side except via human-turn tensions.**

### drive_origin is a bare string
`orion/core/schemas/drives.py`: `GoalProposalV1.drive_origin: str`, `TensionEventV1.drive_impacts: Dict[str,float]` (no origin field on the tension itself — origin is inferred downstream in `goals.py::_drive_origin` from `settings.goal_drive_origin_source`, live `=tick_attribution`). The only mapped origin is `predictive` (`goals.py:19`). `capability_policy.v1.yaml` gates on `required_drive_origins: [predictive]`.

### Endogenous curiosity candidates exist but are not drives
`ORION_ENDOGENOUS_CURIOSITY_ENABLED=true`; table `endogenous_curiosity_candidates` (substrate self-observability v2). These are *candidates surfaced from coalition dwell*, not tensions that pressurize drives. They are a signal source this spec can consume, not the mechanism itself.

## Core problem

`DriveEngine` will happily pressurize any drive from any tension, but **nothing produces a tension from Orion's own state**. The sharpest version: *make Orion's continuous internal self-state a legitimate cause of wanting, without letting a timer masquerade as spontaneity and without drowning out real exogenous signal.*

## Design principles / hard constraints

1. **Real signal, not a timer.** Origination must be triggered by a genuine `SelfStateV1`-derived quantity (variance, drift, dwell, agency surplus). A fixed-interval "emit a random want" is banned as theater.
2. **Exogenous-quiet gate.** Endogenous tensions fire only when the exogenous tension queue for the window is empty/below floor. The world always wins when present; endogeny fills silence.
3. **Engine untouched.** No change to `DriveEngine` math. Endogenous tensions are ordinary `TensionEventV1`s; the only new thing is a producer + an origin tag.
4. **Bounded and decaying.** Endogenous origination is rate-limited and its magnitude is capped, so it cannot self-amplify into a runaway want-storm (respects the O(N) discipline in `[[feedback_execution_merge_cap]]`).
5. **Fully inspectable.** Every endogenous tension carries provenance naming the self-state dims that caused it. `UNVERIFIED` is not an acceptable end state.
6. **Proposal mode.** This changes what causes Orion to want things — a cognition-loop change. Ships behind a default-off flag with an explicit rollback.

## The mechanism

### Origination signal (deterministic, from SelfStateV1)
Maintain a bounded ring of the last `N` (`ORIGINATION_WINDOW=8`) `SelfStateV1` snapshots. Each substrate tick compute three sub-signals, all bounded [0,1]:

- **drift** `D` = mean absolute `dimension_trajectory` magnitude over the window (how much the self-model is *moving on its own*). High drift with no external cause = internal churn worth attending to.
- **dwell** `W` = `min(1, attention_dwell_ticks / DWELL_NORM)` combined with count of `unresolved_pressures` (a loop Orion keeps returning to under its own steam).
- **agency surplus** `A` = `dimensions.agency_readiness.score × (1 − overall_intensity)` — readiness to act that is *not* being consumed by external load. Capacity looking for a target.

Origination potential:
```
P = w_D·D + w_W·W + w_A·A         (seed: w_D=0.4, w_W=0.35, w_A=0.25)
```
Fire an endogenous tension when `P ≥ ORIGINATION_THRESHOLD (0.55)` **and** the exogenous tension count for this window is `0` **and** cooldown has elapsed (`ORIGINATION_COOLDOWN_SEC=900`).

### Which drive does it pressurize?
Map the dominant contributing sub-signal to a drive, deterministically (no LLM, no keyword routing):
- drift-dominant → `coherence` (self-model is unsettled → resolve it)
- dwell-dominant → `introspection`→`autonomy` (a self-set loop → pursue it)
- agency-surplus-dominant → `capability` (idle readiness → grow)
- if `dimensions.social_pressure` is the top unresolved pressure → `relational`
- if `dimensions.continuity_pressure` top → `continuity`

`magnitude = min(ENDOGENOUS_MAG_CAP (0.5), P)`. Impact weight = 1.0 on the mapped drive. The cap ensures endogenous wants are *nudges*, never as loud as a real-world crisis.

### Origin tagging
Add an explicit `origin` to `TensionEventV1` (default `exogenous`, back-compatible) and set `origin="endogenous"` here. `goals.py::_drive_origin` gains a branch: if the lead tension for a tick is `origin=endogenous`, the goal's `drive_origin="endogenous"`.

## Contracts

### Schema changes (`orion/core/schemas/drives.py`)
```python
class TensionEventV1(GraphReadyArtifact):
    magnitude: float = Field(ge=0.0, le=1.0)
    drive_impacts: Dict[str, float] = Field(default_factory=dict)
    origin: Literal["exogenous", "endogenous"] = "exogenous"   # NEW, back-compatible default
    origination_signal: Dict[str, float] = Field(default_factory=dict)  # NEW: {drift,dwell,agency,P}
```
`drive_origin` gains `endogenous` as a recognized value in `goals.py` and in `capability_policy.v1.yaml` (a new rule may set `required_drive_origins: [endogenous]`).

### New emit contract
`EndogenousTensionEmitV1` is not a new envelope — endogenous tensions publish on the **existing** tension channel with `origin=endogenous`. This keeps the bus contract stable and lets every existing consumer treat it uniformly. Registry entry updated to reflect the new optional fields.

### Registry / channels
- `orion/schemas/registry.py`: bump `TensionEventV1` registered shape (optional additive fields — no breaking change).
- `orion/bus/channels.yaml`: no new channel; document that the tension channel now carries `origin`.

## Architecture / data flow

```text
substrate.self_state.v1 (continuous, heartbeat cadence)
  → orion-substrate-runtime: EndogenousOriginationTicker
      · ring-buffer last N SelfStateV1
      · compute D, W, A → P
      · gate: exogenous_tension_count == 0 AND cooldown elapsed AND P ≥ threshold
      · map dominant sub-signal → drive; magnitude = min(cap, P)
      · publish TensionEventV1(origin=endogenous, origination_signal={...})
  → DriveEngine.update(...)          [UNCHANGED]
  → GoalProposalEngine               [drive_origin=endogenous branch]
  → capability policy / downstream autonomy loop   [existing]
```

## Producers & consumers
- **Producer (new):** `orion-substrate-runtime` gains `app/endogenous_origination.py` + a ticker wired into the existing self-state consumer loop.
- **Consumers (modified minimally):** `goals.py` (`_drive_origin` branch), `capability_policy.v1.yaml` (optional endogenous rule), drive-attribution audit (`orion/spark/concept_induction/drive_attribution.py` records origin in `tick_attribution`).

## Env / config (`services/orion-substrate-runtime/.env_example` + settings)
```
ORION_ENDOGENOUS_ORIGINATION_ENABLED=false   # master switch (proposal mode)
ORIGINATION_WINDOW=8
ORIGINATION_THRESHOLD=0.55
ORIGINATION_COOLDOWN_SEC=900
ENDOGENOUS_MAG_CAP=0.5
ORIGINATION_W_DRIFT=0.4
ORIGINATION_W_DWELL=0.35
ORIGINATION_W_AGENCY=0.25
ORIGINATION_EXOGENOUS_FLOOR=0     # max exogenous tensions in window to still allow endogeny
```
After edit: `python scripts/sync_local_env_from_example.py`.

## Observability / traces / metrics
- Every endogenous tension logs a line with correlation id, `origin=endogenous`, `origination_signal`, mapped drive, magnitude, and the self-state ids in the window.
- New metric counters: `endogenous_tensions_emitted_total`, `endogenous_suppressed_by_exogenous_total`, `endogenous_suppressed_by_cooldown_total`.
- Debug surface: `GET /latest/origination` on substrate-runtime returns the last computed `{D, W, A, P, fired}`.

## Tests (gate — deterministic, <2s)
`services/orion-substrate-runtime/tests/test_endogenous_origination.py`:
1. Empty exogenous window + high drift → fires `coherence` endogenous tension; magnitude ≤ cap.
2. Non-empty exogenous window (world present) → **suppressed** regardless of P.
3. Within cooldown → suppressed; after cooldown → fires.
4. Sub-signal dominance mapping table (drift→coherence, dwell→autonomy, agency→capability, social_pressure→relational, continuity_pressure→continuity).
5. Fired tension passes through real `DriveEngine.update` and moves the mapped drive's pressure by the expected soft-saturated amount.
6. `goals.py::_drive_origin` returns `endogenous` when lead tension origin is endogenous.
7. Magnitude cap holds under adversarial P=1.0.
8. Back-compat: a `TensionEventV1` without `origin` deserializes as `exogenous`.

## Evals
`services/orion-substrate-runtime/evals/run_origination_eval.py`: replay a captured quiet period (no receipts, no turns) and assert Orion originates ≥1 want within `ORIGINATION_COOLDOWN_SEC`, and that during a busy period endogenous origination stays at 0 (world-wins invariant). Report origination rate and suppression ratio.

## Failure modes & mitigations
- **Runaway want-storm** → cooldown + magnitude cap + exogenous-quiet gate + rate counters.
- **Timer-in-disguise** → threshold is on a real self-state-derived `P`; test 2/3 prove it is state- and context-gated, not clock-gated.
- **Saturation masking** (the drive-attribution known issue) → endogenous magnitude is capped below world-crisis levels so it never dominates a saturated flat-pressure state; drive-attribution records origin for post-hoc audit.
- **Self-referential loop** (endogenous want → self-state change → more endogenous want) → the exogenous-quiet gate does not fire while the drive is active/decaying above `deactivate_threshold`; cooldown spans the decay tail.

## Privacy / safety
Endogenous tensions reference self-state dimension ids, not raw private traces. No new exposure surface. Proposal-mode disable: set `ORION_ENDOGENOUS_ORIGINATION_ENABLED=false` — producer emits nothing, engine reverts to exogenous-only behavior with zero residue.

## Acceptance checks
- [ ] In a scripted quiet window, Orion emits an `origin=endogenous` tension that pressurizes exactly the mapped drive, with inspectable `origination_signal`.
- [ ] During a scripted busy window, zero endogenous tensions (world-wins).
- [ ] The endogenous tension traverses the unmodified `DriveEngine` and produces a `GoalProposalV1` with `drive_origin=endogenous`.
- [ ] Cap, cooldown, and suppression counters observable via `/latest/origination` and metrics.
- [ ] Flag off → byte-for-byte prior behavior.

## Non-goals
- Not reworking the `surprise` scalar (that is the excluded sibling area).
- Not adding new drives beyond the six canonical keys.
- Not letting endogenous wants trigger effectful/external capabilities — they enter the same readonly-gated loop as any other want.
- No LLM in the origination path; it is deterministic substrate math.
