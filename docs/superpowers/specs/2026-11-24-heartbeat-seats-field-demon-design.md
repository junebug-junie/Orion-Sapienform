# Field demon on a tensor network: one trained lattice that forecasts and shows where Orion comes apart

Status: design, rev 3 (2026-11-25), partly built. Juniper asked for this on 2026-11-24 and
approved the rev 3 direction on 2026-11-25 ("sure"). Related: #3296 (integration rung), #3822
(slow test), #3795 (pipeline coupling floor).

**Build state (2026-11-25):**
- Step 0, the heartbeat overhaul: built in #3873, deployed and verified live (#3871, #3838 closed).
- The trained network, its known-answer tests and the preview replay: built in #3888. On about 39 h
  of data it does not beat the linear model (preview only).
- The lasting seat rollup: built in #3896, deployed 11-25 03:45Z. It needs about 2 weeks of rows
  before the network re-runs (#3886).
- Still a plan: the re-run on stable organs (after the ~12-08 slow-test re-run), and the live patch.

**What removing the reheat answered.** It established that the reheat driver was dead (a constant
hum below its calm floor). The 07-28 thermalization question (can the substrate relax at all
without it?) was answered **in replay only**. On the new build, `mean_ratio` falls to 0.000 after
30 minutes with no atoms. Live, the heartbeat never gets 30 silent minutes, because the body and bus
seats tick all night. So whether it relaxes on a real quiet night is UNVERIFIED.

**Series rule.** A seat means the same thing only within one `routing_version`. The rollup stamps
every row with it, and the replay does not mix versions unless asked (`--any-routing-version`).
A routing change starts a new series. Φ and cut costs are never read across a version boundary.
The routing change on 11-24 (orch folded into execution; the harness took site 4) is the first such
boundary. Rows before it exist only in `grammar_events`, recomputed with today's rules, and they age
out in about 3 days.

**Revisions.**
- **Rev 1** used the heartbeat's per-service seats as the demon's organs.
- **Rev 2** made the demon (linear) and a trained heartbeat two separate models that check each
  other's cuts.
- **Rev 3** (this one) merges them. The demon's model **is** a trained tensor network, which is
  the heartbeat rebuilt. The linear forecaster stays only as the baseline the network must beat.

Juniper's objection to rev 2: "how is demon a linear forecaster, that's lame. they were designed
to be lattice!"

## Arsonist summary

There is one model: a matrix-product state (MPS), which is a chain of small tensors and the same
structure the heartbeat runs today. It is **trained** on Orion's real organ states, now and k steps
later. That one trained chain is read three ways:

1. **Forecast (demon).** Condition the chain on "Orion now" and read off a probability for "Orion
   in k steps". Foresight is how much better that forecast is than the simple guesses.
2. **Where Orion comes apart (lattice / Φ).** For any split of the organs into parts, compare the
   whole chain's forecast with forecasts that use only each part's own past. The cost of the
   cheapest split is the integration measure. Exact marginals of an MPS are cheap, so every split
   on the partition lattice can be scored from the one trained chain, with no refit per split.
3. **Tangling (heartbeat).** The entanglement across each cut of the trained chain. For a Born
   machine this bounds the real shared information between the two sides, and it is the
   heartbeat's old headline number, now about something learned.

The linear forecaster (`orion/substrate/demon.py`, VAR(1)) is kept as a **baseline rung**. If the
network does not beat it, Orion's coupling is mostly linear and the cheap model is enough. That is
a real answer, not a failure.

Why this is not lame: a linear model only sees straight-line coupling. Orion's organs switch modes
(idle or busy, a chat starts, the AC cycles). Those are thresholds and gates. A trained tensor
network can represent them, and a linear model cannot.

## History: what each heartbeat spec found (July-November)

| Date | Spec | Finding |
|---|---|---|
| 07-24 | `2026-07-24-spark-field-holographic-lattice-design.md` | Built it: 5 per-service seats, 10-site MPS. Dropped the charter's fidelity test as close to circular; used the entanglement profile. |
| 07-28 | `2026-07-28-precision-weighted-attention-organ-and-heartbeat-discrimination-design.md`, `...spark-introspector-retirement...` | It read "redundant" on almost every tick (it thermalized). Added 8 trajectories and a relax/reheat loop. Called the strongest live "pulse". |
| 07-29 | `2026-07-29-heartbeat-into-ast-hot-design.md` | The verdict became a self-model input. Nothing acts on it. |
| 08-21 | `2026-08-21-phi-v2-design.md` | `heartbeat_mean_ratio` was used as an "independent" Φ input. |
| 09-19/20 | `docs/research/preregistration/2026-09-*heartbeat*` | It passed lattice-vs-thermometer and the exec↔bus link test. It failed "settles during a conversation". |
| 10-01 | `2026-10-01-heartbeat-cabinet-crosswalk-design.md` | It fails the metric gate's independence and theory checks. The design only logs it beside cabinet temperature. |
| 10-07 | `2026-10-07-orion-self-calibration-design.md` | `heartbeat_smear` is broken: a dead seat reads as a huge number, not as absent. |
| 10-30 | `2026-10-30-self-now-capacities-vsm-design.md` | It is a collector only. It replies to nothing. |
| **11-08** | `2026-11-08-prediction-error-suite-rebuild-design.md` | **The 2.5 h replay.** The gates are fixed random unitaries. Memory is about 3 events (the last 3 atoms predict `mean_ratio` with held-out R² 0.88; the last 30 predict nothing). The verdict class is no better than shuffled input. Recommendation: stop treating the verdict as meaningful, and rebuild the heartbeat as a trained Born machine for the integration rung. |
| 11-24 | this doc, #3838 | Dead inputs: stamped constant strengths on 3 of 5 seats; reheat below its calm floor; 13 producers unrouted. |

#3296 already rules that heartbeat entropy is never a Φ *input*. This design keeps that rule. The
trained heartbeat is a second *model* of the inputs, not an input.

## Current architecture

- **Heartbeat** (after #3873, awaiting merge): 5 per-service seats with dead inputs removed. The
  gates are still fixed random unitaries. The verdict is labelled noise, and `mean_ratio` is
  labelled a last-3-events meter.
- **Demon** (#3728): linear VAR(1). Its forecast, horizon and cut-cost code is tested. Its first
  live read (#3796) was on drifting data-derived modes.
- **Organs:** the modes are real but drift. They settle at 2-4 day blocks (#3822). Re-run around
  12-08.

## The model

**State.** n organ variables, n = 5-8. Each is cut into 3 levels: low, usual or high, against the
organ's own 14-day time-of-day profile, with tercile edges set on training data only. One sample
is the pair (organ levels at t, organ levels at t+k): 2n sites with 3 values each.

**Site order.** Interleave each organ's "now" and "later" sites (o1(t), o1(t+k), o2(t), ...) so
each organ is a contiguous block. Contiguous cuts are native. Other splits are scored from exact
marginals (below), so order does not limit the search. It only affects how easily the chain can
fit.

**Fit.** Born machine: p(x) = |ψ(x)|² / Z, with ψ an MPS of bond dimension D. Fit by maximum
likelihood with two-site sweeps (the standard MPS generative-model fit). Choose D from {2, 4, 8, 16}
by held-out log-likelihood. With 16 sites, 3 values and D=8 that is about 3k parameters, against
about 8,000 5-minute samples in 4 weeks. quimb is already a heartbeat dependency; a small numpy fit
is acceptable if simpler.

**Lags.** k ∈ {1, 3, 6, 12} (5-60 min), one chain per k, matching the demon spec's lag set.

**Rolling refit.** Refit every 24 h on the last 14-28 days. Keep the old chain until the new one
beats it on held-out data.

## The three readouts

**1. Forecast skill.** Score by log-loss (bits) on the held-out "later" levels. Compare against:
- **clock:** the time-of-day level distribution;
- **persistence:** the same level as now, with the train-set transition rate;
- **linear:** the VAR(1) Gaussian forecast, integrated into the same 3 bins.

Skill against the best baseline at each k, with a moving-block bootstrap spread, uses the existing
`demon.bootstrap_horizon` machinery. The headline is the skill half-life (the #3296 horizon
question).

**2. Cut cost on the partition lattice.** For a partition P, the cut forecast for part A uses only
A's own "now" sites:

  p_P(later | now) = Π over parts A of p(later_A | now_A)

Each factor is an exact conditional marginal of the one trained chain. Contract out the other
parts; this is cheap for an MPS. Cut cost = held-out log-loss(cut) − log-loss(whole), in bits,
which is a predictive-information cut in the Φ family.
- All Bell(n) partitions: 4,140 at n = 8. Each needs a few marginal contractions, so the full
  search is minutes, not days.
- The minimum-information partition is the cheapest cut, using #3296's normalization.
- Null: the time-shift null already in `demon.cut_cost_vs_null`, refitting the chain on shifted
  data (20 draws).

**3. Entanglement per contiguous cut.** The von Neumann entropy at each bond of the trained chain.
For a Born machine this is an upper bound on, not equal to, the classical mutual information across
that cut. Report both: exact mutual information from marginals, and entanglement. The gap is how
much "quantum-style" capacity the chain uses beyond the classical structure. This replaces the
heartbeat's random-gate `mean_ratio`.

**Cross-check, built in.** The linear Φ_G (closed form from the VAR) and the network's cut cost
are scored on the same partitions. Agreement on the weakest cut is evidence of real structure.
Disagreement shows non-linear coupling where the network finds a weak cut the linear model does
not.

## Missing questions

1. **Organs.** Recommendation: run on per-service seat activity now, labelled as a preview. Move
   to stable data-derived modes at 4-day blocks once #3822's re-run (~12-08) passes. Per-service
   seats are mostly one pipeline (#3795). Report the pipeline block separately.
2. **Discretising.** 3 levels keep the state space fittable. 5 levels would show finer structure
   but need about 3x the data. Start with 3.
3. **Where it runs.** Recommendation: as a replay first. Later, the heartbeat service hosts the
   live chain, which ends the random-gate ensemble, and it publishes to `prediction_integration_
   hourly` (#3296). The forecast belongs in orion-prediction only if a consumer needs it
   (memory: predictions live in the prediction service).

## Proposed schema / API changes

None in the first patch, which is a read-only replay. Later, in the live patch:
- Fields on the integration hourly record: `network_skill_by_k`, `network_cut_cost_by_k`,
  `network_mip_partition`, `entanglement_by_cut`, and `linear_vs_network_bits` (how much the
  network beats the linear model).
- Heartbeat `/h1` reports the trained chain's entanglement profile. The random-gate ensemble is
  retired completely, with no fallback.

## Files likely to touch

- `orion/substrate/born_machine.py` (new, pure): MPS fit, log-likelihood, conditional marginals,
  per-bond entanglement.
- `orion/substrate/demon.py`: a `NetworkForecaster` beside `Var1`, with the same rolling-error
  interface so `skill_curve`, `bootstrap_horizon` and `cut_cost_vs_null` work unchanged.
- `scripts/demon_network_replay.py` (new): the replay and its gate.
- Later: `services/orion-heartbeat/app/substrate/*`, README, and the integration hourly table.

## Non-goals

- No Φ claim about consciousness, in any output (#3296 rule).
- No neural net. If the network loses to the linear baseline, no larger model is tried until the
  direct-vs-iterated misfit check (demon spec) shows missing structure.
- No live wiring in the first patch.
- No hand-made organ list.

## Acceptance checks

- **Known answers (unit tests):**
  - On data from a true VAR(1), the network's skill is within spread of the linear model.
  - On data with a threshold switch (organ B jumps only when A is high), the network beats the
    linear model by a clear margin.
  - Two independent organ blocks give cut cost about 0 at their split, and that split is the
    minimum-information partition.
  - Conditional marginals match brute-force enumeration on a small chain.
  - Entanglement is at least the exact mutual information at every cut.
  - The shifted-data null gives cut cost about 0.
- **Replay (live, read-only):**
  - Skill by k against clock, persistence and linear, with spread and P(real).
  - Cut cost by partition against the null.
  - The minimum-information partition from the network and from the linear Φ_G, with whether they
    agree.
  - The pipeline block reported separately.
  - Entanglement against mutual information per cut.
- **Gate to go live:** the network beats linear on held-out log-loss at k=1 with P(real) ≥ 0.9
  (effect and spread reported, no alpha line), and its minimum-information partition beats the
  null.
- **Mutation check:** replace the network forecast with the linear one. The threshold-switch test
  must fail.

## Recommended next patch

1. Juniper merges #3873 (heartbeat step 0).
2. `born_machine.py` plus its known-answer tests, then the replay on per-service seats as a
   labelled preview.
3. Re-run on stable modes after ~12-08. If the network passes its gate, the live patch retires
   the random-gate heartbeat and runs the trained chain in its place.

