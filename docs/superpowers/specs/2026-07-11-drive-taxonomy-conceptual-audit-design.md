# The six drives — conceptual audit (not a wiring bug)

**Mode:** Design / audit. No code change proposed here. This documents a distrust that survives after the math was already fixed.

## Arsonist summary

The `homeostatic-drives-real-tensions` spec (2026-07-07, merged) fixed a real theater bug: the `soft_saturate` fixed point that pinned every drive to 0.7309 regardless of input. That fix is good and confirmed live. But it fixed *how the numbers move*, not *what the numbers name*. The six drive keys — `coherence, continuity, capability, relational, predictive, autonomy` (`DRIVE_KEYS`, `orion/spark/concept_induction/drives.py:10`) — have never had a conceptual grounding document. They were introduced once, early (first commit `7d46b63e`, no docstring, no rationale), and every later spec treats the taxonomy as fixed ("Six drives stay" — hard constraint in the 2026-07-07 homeostatic-drives spec). Distrust in "the drives" is not fully addressed by fixing the pressure math, because the categories themselves were never argued for.

## Current architecture

Two independent computations share the same six-key vocabulary and the same `config/autonomy/signal_drive_map.yaml`, but are not the same state:

- `orion.spark.concept_induction.drives.DriveEngine` — leaky integrator, feeds `GoalProposalEngine`.
- `orion.autonomy.reducer.reduce_autonomy_state` — RDF-persisted, feeds `AttentionItemV1` generation and `capability_policy` drive-origin gating.

Operational semantics of each drive, derived from what actually triggers it (not from any docstring — none exists):

| Drive | What actually fires it |
|---|---|
| `coherence` | drop in self-state/turn `coherence` score, `spark_signal.coherence` dip |
| `continuity` | novelty spikes, uncertainty deltas, biometric volatility, mesh-health drops |
| `capability` | energy/resource/execution pressure, biometric strain, failure severity |
| `relational` | valence drops, social-hazard signals (cooldown loops, self-message loops) |
| `predictive` | coherence deltas, uncertainty, novelty, world-coverage gaps |
| `autonomy` | novelty, uncertainty, low feedback scores |

Package placement (`orion/spark/concept_induction/drives.py`) implies a dependency on concept-induction's `ConceptProfile` output. Verified false by grep — zero references. Naming-proximity artifact only, not a real coupling.

## Missing questions

1. Where did `coherence, continuity, capability, relational, predictive, autonomy` come from? No spec, paper, or design note in the repo argues for this specific set over any other. Was it six because that felt complete, or is there a real theory behind it that just never got written down?
2. `coherence`, `continuity`, and `predictive` draw from largely the same tension inputs (self-state coherence/uncertainty deltas, novelty) with different weight vectors. Is that intentional redundancy (three views on one underlying signal, useful for attribution), or is it evidence the taxonomy is finer-grained than the actual signal supports and should collapse?
3. `autonomy` is named after a capacity for self-initiation, but its actual inputs (novelty, uncertainty, low feedback score) are generic distress signals shared with three other drives. Its one mechanism that would justify the name — endogenous origination — is real code but `ORION_ENDOGENOUS_ORIGINATION_ENABLED=false` by default. Should the drive be renamed to match what it currently measures, or should origination be turned on (proposal-mode change) so the name earns itself?
4. The homeostatic-drives spec names a known gap — no self-preservation drive, somatic biometric signals get mapped onto `capability`/`continuity` as a workaround — and explicitly declines to solve it ("Six drives stay... named as a gap, not built here"). Is six a real constraint (e.g. UI, downstream schema) or just inertia?
5. Two unsynchronized drive-pressure computations (`concept_induction.DriveEngine` vs `autonomy.reducer`) exist under the same names. Should there be one canonical drive-state store, the way φ now has one canonical `_phi_from_self_state()`?

## Proposed schema / API changes

None in this patch. This is a critique to be resolved by a future decision, not an implementation.

## Files likely to touch (future patch, not this one)

- `orion/spark/concept_induction/drives.py` — `DRIVE_KEYS`, if the set changes.
- `orion/autonomy/reducer.py` — second drive-pressure computation, if consolidated.
- `config/autonomy/signal_drive_map.yaml` — signal→drive weights, if drives are renamed/merged.
- `orion/core/schemas/drives.py` — `DriveStateV1`/`DriveAuditV1`, if the taxonomy changes shape.
- Any spec/doc referencing `DRIVE_KEYS` as fixed (`docs/superpowers/specs/2026-07-06-substrate-fed-motivation-design.md`, `2026-07-07-homeostatic-drives-real-tensions-design.md`, `2026-07-07-endogenous-drive-origination-design.md`).

## Non-goals

- Not proposing a specific replacement taxonomy here — that requires a real argument, not a guess.
- Not proposing to merge the two drive-pressure computations in this patch — flagged as a question, not decided.
- Not touching `ORION_ENDOGENOUS_ORIGINATION_ENABLED` here.

## Acceptance checks

This doc is "done" when Juniper has answered the missing questions above and a follow-up spec either (a) writes the conceptual grounding that's currently absent, or (b) revises the taxonomy with a stated rationale, or (c) explicitly decides six-arbitrary-but-good-enough is fine and this stops being a source of distrust.

## Recommended next patch

Smallest useful next step: answer question 5 first (one canonical drive-state store vs. two) — it's the one with a clean precedent (the φ canonicalization patch just shipped) and doesn't require resolving the harder conceptual-grounding question first.
