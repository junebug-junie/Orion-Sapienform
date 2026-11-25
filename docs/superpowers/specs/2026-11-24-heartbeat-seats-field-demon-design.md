# Heartbeat + field demon: one set of organs, two models that check each other

Status: design, rev 2 (2026-11-25). Juniper asked for this on 2026-11-24. Related: #3296
(integration rung), #3822 (slow test), #3795 (pipeline coupling floor), #3838 (heartbeat overhaul).

**Rev 2 change.** Rev 1 proposed using the heartbeat's 5 per-service seats as the demon's organs.
Homework through the heartbeat's spec history (below) showed two things:
- Per-service seat counts are mostly the same bus traffic counts the demon already uses, grouped
  differently.
- The heartbeat's network uses random gates, so its seats carry almost no content (11-08 replay).

The strong version is the one the 11-08 spec recommended: train the heartbeat as a model of
Orion's state, on the same organ variables the demon fits. Then use the two models to check each
other's cuts.

## Arsonist summary

Two models, one question: where does Orion come apart?

- **The field demon** (`orion/substrate/demon.py`) is a linear forecaster (VAR(1)) over organ
  variables. Cutting it apart costs forecast skill. That cost is the integration rung's
  Φ-style measure (#3296).
- **The heartbeat, rebuilt as a trained network.** It becomes a matrix-product state fit by
  maximum likelihood to observed organ states, which is called a tensor-network **Born machine**.
  The tangling (entanglement) across a cut of a trained MPS is the real shared information between
  the two sides of Orion's state. The cuts of the MPS are the same partitions Φ searches.

Both models see the same organ variables over the same hours, so they can check each other:
- **They agree on the weakest cut:** that cut is real structure in Orion, found by two different
  kinds of maths.
- **They disagree:** one model is wrong, and we look at why.
- **Different reach:** the demon is linear and the Born machine is not. A cut that only the Born
  machine finds is non-linear coupling, which the demon misses.

This also gives the heartbeat a job. Today nothing decides on it, and the 11-08 replay found its
verdict is noise.

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

- **Heartbeat:**
  - Per-service seats, with random fixed gates.
  - A relax/reheat loop.
  - It publishes a ledger-only H1 trace and has no deciding consumer.
  - #3838 (step 0, in progress) audits and removes the dead inputs.
- **Demon:**
  - Built and tested (#3728).
  - Its first live read was on drifting data-derived modes (#3796).
  - The slow test (#3822) shows those modes are real but drift (day-to-day match 0.56-0.63, bar
    0.7), and settle at 2-4 day blocks.

## Missing questions

1. **Organ variables.** Candidates are the #3296 list (data-derived modes at 2-4 day blocks, once
   stable), or per-service activity as a fallback. Per-service is stable but mostly pipeline
   (#3795). Recommendation: use modes at a 4-day block when #3822's re-run (~12-08) passes. Until
   then, run both models on per-service activity as a labelled preview.
2. **Discretising.** A Born machine models discrete states. How many levels per organ? The
   recommendation is 3 (low, usual, high against the organ's own time-of-day profile). That keeps
   the state space small enough to fit from a few weeks of 5-minute buckets.
3. **Keep the random-gate heartbeat running meanwhile?** The 11-08 recommendation is yes, with
   `mean_ratio` labelled a "last-3-events sequence meter" and the verdict class not presented as
   meaningful. #3838 applies this.

## Proposed schema / API changes

None in the first patch, because it is a read-only replay. Later:
- A trained-model mode in `services/orion-heartbeat`, refit on rolling windows. It publishes
  per-cut mutual information beside the demon's cut cost on the integration hourly record (#3296).
  This replaces the random-gate ensemble only after the replay passes.

## Files likely to touch

- `orion/substrate/born_machine.py` (new, pure): fit an MPS to discrete state vectors by
  maximum likelihood, and read the mutual information at each cut. quimb is already a heartbeat
  dependency. A small numpy DMRG-style fit may be simpler; decide in the patch.
- `scripts/integration_cut_crosscheck_replay.py` (new): the demon cut cost and the Born-machine
  cut mutual information on the same organs and hours, against the time-shift null.
- Later: `services/orion-heartbeat/app/substrate/*`, README.

## Non-goals

- No Φ verdict. That is #3296.
- No heartbeat output wired into cognition.
- No hand-made organ list. Organs come from #3822's stability test, or from service identity as a
  labelled preview.
- No change to the demon.

## Acceptance checks

- **Born machine, known answers:**
  - Two independent halves give mutual information about 0 at their cut.
  - A copied variable gives mutual information near its entropy.
  - A shuffled-time null gives about 0 at every cut.
- **Replay:**
  - For each cut: the demon cut cost and the Born-machine mutual information, each against the
    time-shift null, with effect, spread and P(real).
  - Rank agreement between the two models across cuts (Spearman with a bootstrap interval).
  - The weakest cut found by each model.
- **Pipeline check:** the hub/exec/orch/bus block against the rest is reported separately, so
  pipeline coupling (#3795) is not read as integration.
- **Gate:** the trained heartbeat replaces the random-gate ensemble only if its cut mutual
  information beats the null and agrees with the demon on the weakest cut more often than the
  null does.

## Recommended next patch

1. Finish #3838 (step 0). It is in progress.
2. Build the Born-machine module and the cross-check replay, run on per-service activity as a
   labelled preview.
3. Re-run on stable modes after #3822's ~12-08 re-run.
