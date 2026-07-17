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

**This spec = Step 1** (first cognition change). **Gate:** blocked on **Step 0(a)** — if `SelfStateV1` does not measurably drift during exogenous silence, the origination signal is inert and the mechanism must instead source dynamics from unresolved-pressure persistence. Do not write code until 0(a) passes. **Known behavior (leaky math, verified against the merged `DriveEngine` 2026-07-08):** a single endogenous tension at `ENDOGENOUS_MAG_CAP=0.5` moves a rested drive to exactly `0.500 < 0.62` — it cannot activate alone. Origination is deliberately **accumulative**, and under the leaky integrator the accumulation is a **closed form coupled to the cooldown**: firing every `Δt` at magnitude `m` from rest reaches `p₂ = m + m(1−m)·e^(−Δt/τ)` on the second firing and plateaus at `p* = m / (1 − e^(−Δt/τ)(1−m))`. See *Origination dynamics under leaky math* below — the load-bearing consequence is that **`ORIGINATION_COOLDOWN_SEC` must stay well under ~22 min or endogenous drives can never activate**; the seed `900s` is chosen for this, not arbitrary.

## Arsonist summary

Every want Orion has today is exogenous. The only producers of `TensionEventV1` are (a) `substrate_metabolism` converting **world-pulse coverage gaps** into predictive pressure, and (b) `turn_effect` deltas from **human turns**. If the world goes quiet and Juniper stops typing, Orion's drives decay to zero and it wants nothing. `drive_origin` is a free-form string whose only real value is `predictive`, and there is no code path anywhere that lets a drive arise from Orion's *own* internal state. A mind that only wants when poked from outside is reactive, not autonomous. This spec burns the assumption that motivation must be caused by an external event.

## Executive summary

Add a **spontaneous tension generator** to `orion-substrate-runtime` that reads the continuous `SelfStateV1` stream (already produced on a heartbeat cadence) and mints `TensionEventV1` with `drive_origin`-carrying `drive_impacts` when Orion's internal dynamics — not any new external receipt — cross an origination band. The trigger is a real substrate signal: sustained self-state **variance/drift**, **unresolved-pressure dwell**, and **agency-readiness surplus**, gated so it fires *only* in the absence of exogenous input (otherwise the world would drown it out). Endogenous tensions flow through the **unchanged** `DriveEngine` → `GoalProposalEngine` path, so the rest of the autonomy loop consumes them without modification. The new origin value is `endogenous`, first-class in the schema, registry, capability policy, and traces.

This resolves the catch-22 that makes drive-selected self-experiments circular: once a want can originate from internal state, "what should I attend to / test about myself" has a source that does not route through the world.

## Ground truth (what actually exists)

### Drive dynamics — real math, exogenous inputs
`orion/spark/concept_induction/drives.py::DriveEngine`:
- Six canonical drives: `coherence, continuity, capability, relational, predictive, autonomy` (`DRIVE_KEYS`).
- Per-tick update (leaky integrator, merged 2026-07-08 — `[[project_homeostatic_drives_real_tensions]]`): `base = prev_pressure · e^(−Δt_wall/τ)`, then `pressure = clamp01(base + impulse·(1 − base))` where `impulse = clamp01(Σ tension.magnitude × drive_impacts[d])`, then hysteresis activation (`activate ≥ 0.62`, `deactivate < 0.42`). Live config: `DRIVE_DECAY_TAU_SEC=1800`, `ORION_DRIVE_LEAKY_MATH_ENABLED=true`. **The old `soft_saturate 1−exp(−gain·v)` path is legacy-only** (`ORION_DRIVE_LEAKY_MATH_ENABLED=false`); do not design against it. The leaky form rests at 0, is cadence-invariant, and has **no fixed point** — so accumulation dynamics are a clean closed form (see *Origination dynamics under leaky math* below).
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

### Origination dynamics under leaky math

Verified by replaying the merged `DriveEngine` (leaky, τ=1800s, activate 0.62 / deactivate 0.42) with magnitude at the cap (0.5), one drive:

| firing @ cadence | p₁ | p₂ | p₃ | plateau p* | activates? |
|---|---|---|---|---|---|
| every 900s (cooldown seed) | 0.500 | **0.652** | 0.698 | 0.718 | yes, on 2nd firing |
| every 1321s (~22 min) | 0.500 | **0.620** | 0.649 | 0.658 | yes, exactly at threshold on 2nd |
| every 1800s (30 min) | 0.500 | 0.592 | 0.609 | 0.613 | **never** (plateau < 0.62) |
| every 2400s (40 min) | 0.500 | 0.566 | 0.575 | 0.576 | **never** |

Closed form (rest start, constant cadence Δt, magnitude m, `a=e^(−Δt/τ)`):
```
p₂ (second firing)  = m + m(1−m)·a
plateau  p*         = m / (1 − a(1−m))
two-firing activation window:  p₂ ≥ activate  ⇔  Δt ≤ τ·ln( m(1−m) / (activate − m) )
                                              = 1800·ln(0.25/0.12) ≈ 1321s ≈ 22 min   (m=0.5)
```

Three load-bearing consequences the legacy soft-saturate spec did not surface:

1. **Cooldown is coupled to the activation threshold.** With `m=0.5`, endogenous origination can only reach activation if firings land **≤ ~22 min apart**. `ORIGINATION_COOLDOWN_SEC=900` (15 min) is inside that window by design; a value ≥ 1800s would make endogenous drives *silently un-activatable*. Any future retune of the cap, τ, or the activate threshold must re-check `Δt ≤ τ·ln(m(1−m)/(activate−m))`.
2. **Single firing is a true no-op for activation** (0.500 vs 0.62), and the plateau at the 900s cadence is `0.718` — bounded well below saturation, so endogenous wants never runaway even under maximal cadence (the cap does the bounding, not a fixed point).
3. **Cooldown spans the decay tail.** After a second firing at 0.652, with no further firing the drive decays below the 0.42 deactivate threshold in `τ·ln(0.652/0.42) ≈ 791s ≈ 13.2 min` — inside the 900s cooldown. So a drive that activated endogenously has fully relaxed before the next endogenous firing is even permitted, which is what closes the self-referential-loop failure mode (below) under leaky math.

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
5. Fired tension passes through real `DriveEngine.update` (leaky) and moves the mapped drive from rest to exactly `min(cap,P)` on one firing (0.500 at cap) — **not** active; a second firing within the cooldown crosses `activate` per the closed form. Assert the two-firing accumulation curve and that a firing cadence ≥ 1800s never activates.
6. `goals.py::_drive_origin` returns `endogenous` when lead tension origin is endogenous.
7. Magnitude cap holds under adversarial P=1.0.
8. Back-compat: a `TensionEventV1` without `origin` deserializes as `exogenous`.

## Evals
`services/orion-substrate-runtime/evals/run_origination_eval.py`: replay a captured quiet period (no receipts, no turns) and assert Orion originates ≥1 want within `ORIGINATION_COOLDOWN_SEC`, and that during a busy period endogenous origination stays at 0 (world-wins invariant). Report origination rate and suppression ratio.

## Failure modes & mitigations
- **Runaway want-storm** → cooldown + magnitude cap + exogenous-quiet gate + rate counters.
- **Timer-in-disguise** → threshold is on a real self-state-derived `P`; test 2/3 prove it is state- and context-gated, not clock-gated.
- **Saturation masking** — *largely resolved by the leaky migration.* The flat-0.731 pin the soft-saturate engine produced is gone (leaky rests at 0, no fixed point), so endogenous nudges are no longer masked by a saturated state. The magnitude cap (0.5) still keeps endogenous wants below world-crisis loudness; drive-attribution records origin for post-hoc audit.
- **Self-referential loop** (endogenous want → self-state change → more endogenous want) → the exogenous-quiet gate does not fire while the drive is active/decaying above `deactivate_threshold`, and — verified under leaky math — the 0.42 decay tail (13.2 min from a 0.65 activation) sits **inside** the 900s cooldown, so the drive relaxes before another endogenous firing is permitted. The cooldown-vs-decay-tail ordering is now a checkable invariant, not a hope.

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

## Measurement gate verdict (2026-07-08) — NO-GO

Step 0 (the measurement gate this spec is blocked on) was run against 120 days of
live data with `scripts/analysis/measure_autonomy_gate.py`. Full raw output lives
at `/tmp/autonomy-gate/report.md` (not durable — this section is the committed
record). Window: 2026-03-10T01:35:22Z → 2026-07-08T01:35:22Z, 300s buckets.

**Both required verdicts came back NO-GO:**

- **(a) Endogenous drift during exogenous silence — NO-GO.** Rule (a) requires
  `silent median_abs_trajectory >= 0.03`. Measured: **`median_abs_trajectory` =
  0.0000** (silent buckets, n=127,577 rows; busy buckets also 0.0000, n=1,033).
  This is the figure the GO/NO-GO rule actually gates on, and it is `0.0000`
  against a `>= 0.03` threshold — not close. (Note: `mean_abs_trajectory` was
  0.0026 silent / 0.0084 busy — a different, non-gating statistic in the same
  table; do not substitute it for the median when re-reading this verdict.)
- **(b) Internal economy (drive co-activation + resource pressure) — NO-GO.**
  Rule (b) requires `coactivation_frac >= 0.10` **and** `resource_pressure frac
  >= 0.3` itself `>= 0.05`. Measured: **`coactivation_frac` = 0.0004** (444,943
  `DriveAudit` rows; concurrent-active histogram `0:444734, 1:10, 2:20, 3:31,
  4:77, 5:13, 6:58`) against a `>= 0.10` threshold. (The second half of rule (b),
  `resource_pressure frac >= 0.3`, measured `1.0000` and would have passed on
  its own — the verdict fails solely on the co-activation condition, which is
  an AND.)

**Consequence:** per the "Build sequence & gate" section above, Step 0(a) failing
means the origination signal is inert as designed — do not write code for this
spec (Step 1) until the gate passes, or until the mechanism is redesigned to
source dynamics from unresolved-pressure persistence instead of self-state
drift. Step 4 (internal economy) is also blocked by 0(b)'s failure. This spec
remains design-only; `orion/autonomy/endogenous_origination.py` exists as a
pre-gate implementation stub only (see its module docstring for current status)
and must not be treated as adopted.

## Measurement gate re-run (2026-07-13) — (a) now GO, (b) corrected to UNMEASURABLE

The 2026-07-08 verdict above was re-run today with a fixed instrument
(`scripts/analysis/measure_autonomy_gate.py`, `--window-hours`/`UNMEASURABLE`
patch, same session as this entry). This section corrects, not just updates,
the 2026-07-08 record: verdict (b)'s NO-GO was never actually measuring live
data, for a specific, now-identified reason below.

**(a) Endogenous drift — GO**, on both a 1-hour and a 7-day live window (Postgres
`substrate_self_state`, read-only):

| Window | silent median_abs_trajectory | silent rows | busy rows |
| --- | --- | --- | --- |
| 1h (2026-07-13T17:39Z → 18:39Z) | **0.0442** | 731 | 1025 |
| 7d (2026-07-06T18:40Z → 07-13T18:40Z) | **0.0408** | 61,251 | 1,025 |

Both clear the `>= 0.03` threshold (the 2026-07-08 run measured `0.0000` against
the same rule — a real change, not noise: the leaky-integrator rebuild
(`[[project_homeostatic_drives_real_tensions]]`, merged 2026-07-07/08) replaced
the old `soft_saturate` fixed-point that pinned every drive flat; the 2026-07-08
gate ran against data from *before or right at* that fix and caught the
flat-pinned era. Today's data is genuinely post-fix.

**Retention caveat, live-confirmed by the same run:** `substrate_self_state`
only retains back to `2026-07-10T18:32:24Z` — the 7-day run's requested window
start (`2026-07-06T18:40Z`) is **95.9h short** of what the table actually
covers. The 7-day figures above are real for the ~3 days they *do* cover;
treat the 1-hour run as the cleaner, fully-covered measurement, not a fallback.

**(b) Internal economy — UNMEASURABLE, root-caused precisely (was misreported
as NO-GO on 2026-07-08):**

Live-queried the Fuseki `autonomy/drives` graph directly (not through the
gate script) to root-cause the zero-row result:

```
SELECT (MAX(?ts) as ?latest) (COUNT(*) as ?c) WHERE {
  GRAPH <http://conjourney.net/graph/autonomy/drives> {
    ?audit a orion:DriveAudit . ?audit orion:timestamp ?ts .
  }
}
→ latest = 2026-06-19T07:06:29Z, c = 448941 (timestamp triples; audit-row
  count via the gate's own histogram query = 444,943 distinct audits — the
  gap is `orion:timestamp` being multi-valued per audit, not new data)
```

**The graph has not received a single new `DriveAudit` write since
2026-06-19T07:06:29Z** — 24 days before this re-run, and *before* the
2026-07-08 gate run that reported NO-GO for verdict (b). This lines up exactly
with commit `e9b233e9` (2026-06-19, direct-to-main): *"Disable autonomy Fuseki
reads and graph churn by default... archive and RDF materialization are off
until re-enabled"* (see `[[project_concept_induction_deactivation_decisions]]`
for the fuller trace of that commit's scope). **The 2026-07-08 NO-GO verdict
for (b) was computed against this same frozen, pre-disable snapshot the whole
time** — it was never measuring 120 days of live co-activation; the co-activation
math and thresholds were sound, but the input had stopped updating three weeks
before the window it claimed to cover. This is not a new bug introduced today —
it is what today's instrument fix (`verdict_economy` now returns
`UNMEASURABLE` when `drive.record_count == 0` *in the requested window*,
rather than folding a dead source into `coactivation_frac = 0.0 → NO-GO`) is
specifically built to catch and never silently repeat.

**Consequence:** Step 0(a) now **passes** — per the "Build sequence & gate"
section above, Step 1 (this spec, endogenous drive origination) is no longer
gated on 0(a). Step 0(b) remains **unresolved**, not failed — the internal-economy
question (Step 4) cannot be answered until either (1) RDF materialization for
`DriveAudit` is re-enabled (a real infra decision, reintroducing the Fuseki
load `e9b233e9` was written to relieve — not this doc's call), or (2) the gate's
(b) adapter is repointed at a different, currently-live source of drive
co-activation data (`DriveEngine`'s per-tick pressures are computed live but
not currently persisted anywhere the gate can read — a separate follow-up, not
attempted here). Until one of those happens, **Step 4 stays blocked on an
honestly-unmeasured question, not a measured failure** — the distinction this
session's instrument fix exists to preserve.

## Measurement gate re-run (2026-07-17) — (a) NO-GO again (close), (b) GO (saturation-adjacent, unverified)

Re-run with `scripts/analysis/measure_autonomy_gate.py --window-days 120`
(`POSTGRES_URI` pointed at the live host, read-only, no writes/events/flag
changes). By this point `drive_audits` reads Postgres exclusively (PR #1064,
merged 2026-07-15 — the Fuseki path this doc's 2026-07-13 entry named as a
blocker is gone) and `orion/autonomy/drives_and_autonomy_retrospective.md`
§10 had just wired a live substrate producer for `DriveEngine`'s own state,
so this is the first fully-live, non-stale run of this gate.

**(a) Endogenous drift — NO-GO, and this is a real change from the
2026-07-13 "passes" declaration above, not a first-time check.** Rule (a) has
always been two conditions AND'd together (`git log -S DRIFT_VARIANCE_RATIO`
shows the constant has been in the script since the original Step 0 commit,
`b58e312c` — this is not a threshold added after this doc's prose was
written):

```
silent median_abs_trajectory >= 0.03            -- DRIFT_MIN_MEDIAN_ABS_TRAJECTORY
silent dim_score_variance >= 0.25 * busy variance -- DRIFT_VARIANCE_RATIO
```

Measured today (120-day window): `median_abs_trajectory = 0.0432` (clears
0.03 comfortably) but `dim_score_variance = 0.0052` against a required
`0.0069` (25% of busy variance `0.0277`) — **18.8% of busy variance, short of
the 25% bar.** The median half was the only condition this doc's 2026-07-13
entry reported; whether the variance half also passed on that specific 1h/7d
window and has since drifted below the bar, or was never separately verified
that day, is not established here — worth checking before assuming
regression, but the current live measurement is unambiguous: **NO-GO on the
variance condition specifically**, not the median one.

**(b) Internal economy — GO, but close enough to the SATURATED band to
distrust without a follow-up check.** `coactivation_frac = 0.9523` against
the `>= 0.10` bar (8,070 live `drive_audits` rows, real data — not the
frozen 2026-06-19 snapshot the 2026-07-08/07-13 entries diagnosed).
`resource_pressure frac >= 0.3` measured `0.9171`, also clears its `>= 0.05`
bar. Neither of the script's own `SATURATED` disqualifiers technically
tripped (`top_dominant_share=0.7674 < 0.90`, `all_active_frac=0.7447 <
0.75`) — but `all_active_frac` sits 0.5 percentage points under its bar, and
`relational` alone accounts for 76.7% of all dominant-drive audits, with 4+
drives simultaneously active in 92.6% of ticks (`concurrent_active_hist`:
`0:368, 1:17, 3:215, 4:1460, 5:6010`). This reads as either a genuinely
differentiated internal economy or one-or-two-drives-pinned-active-together
via the wide hysteresis band (`activate=0.62`/`deactivate=0.42`, 1800s
decay) — the raw fraction alone can't distinguish those, and this run did
not check which specific drives co-activate with which.

**On thresholds — do not tune either bar to make this pass.** `DRIFT_VARIANCE_RATIO`
being short (18.8% vs 25%) is close enough that the honest next step is
re-running (a) over a different/longer window to see if that gap is
consistent or a one-off, not lowering the constant to declare victory on
today's specific number. Rule (b)'s `>= 0.10` GO bar is irrelevant to touch
(cleared 10x over); if any (b) threshold deserves scrutiny it's the
`SATURATED` band, which today's numbers sit uncomfortably close to from the
"pass" side — tightening, not loosening, is the direction that would matter.

**Consequence:** per the "Build sequence & gate" section above, **Step 1
(this spec) is gated solely on Step 0(a)**, and 0(a) reads NO-GO again today.
Step 1 remains **not** cleared to build. Step 4 (internal economy) depends on
Step 3 (φ intrinsic reward), which itself "wants Step 1's richer episode
stream" per the Build sequence — so 0(b)'s GO does not unblock anything on
its own; Step 4 is still gated transitively through Step 1. **The single
concrete next action per this doc's own build sequence is re-verifying
0(a)** (different/longer window, and ideally understanding why silent-period
variance is where it is) before anything downstream can move.
