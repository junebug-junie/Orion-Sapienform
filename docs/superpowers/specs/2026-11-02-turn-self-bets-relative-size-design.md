# Turn self-bets: ask "bigger than usual?", not "how big?" (proposal)

Status: **proposal**, awaiting Juniper's approval. This changes what stance is asked, a cognition loop
(CLAUDE.md, proposal mode). Parent arc: `2026-10-23-unified-turn-self-prediction-design.md` (#2869).
A3 settling and grading: #2976.

## Arsonist summary

Stance cannot guess how big its own turns are, and the 10-29 prompt fix made it worse. Graded
turns since 10-30: 682. Step ranges are right 1% of the time, against a stated 80%, with a median
width of 1 step. Reply-length ranges are right 8% of the time.

Handing stance the usual size per kind of turn would make the ranges look calibrated. Stance could
just copy the hint back: a lookup table, not self-knowledge, and the grade would be leaky if the
hint came from the graded turns. Code can already compute the usual size, and stance adds nothing
there.

So drop the two absolute ranges. Ask two probability bets that only real knowledge of *this* turn
can win:

- **`more_steps_than_usual_p`**: will this turn take more working steps than usual for its kind?
- **`longer_reply_than_usual_p`**: will this reply be longer than usual for its kind?

"Usual" is the median of the same caller's turns over the 7 days **strictly before** the bet. Stance
never sees it. The existing per-caller skill score grades both bets. A constant guess at the
caller's own rate scores exactly 0, so stance only scores by telling long turns from short ones.

## Current architecture

- Producer: stance writes `thought_decision.self_prediction`.
  - Model: `TurnSelfPredictionV1` (`orion/schemas/thought.py:39`, `extra="forbid"`).
  - Normalizer: `normalize_self_prediction` (`orion/thought/stance_react.py:113`).
  - Prompt lines: `orion/cognition/prompts/stance_react.j2:146-149`.
- The motor and the judge never see the bet (`orion/harness/tests/test_self_prediction_never_reaches_motor.py`).
- Settler (#2976): orion-prediction `app/turn_self_bets.py` writes one row per turn × faculty into
  `prediction_turn_self_bet`, with `observed` = the true step count or reply characters.
- Grader: `_grade_p` gives Brier and `skill_vs_caller`, the skill against each caller's own rate.
  `GET /turn-self-bets/live` serves it.

## Live evidence (metric gate)

1. **Where the value comes from.** `observed` is taken from `harness_turn_trace.run_artifact`:
   `step_count` and `len(final_text)`, as today. The reference median comes from those same stored rows.
2. **Independent of each other?** Yes, mostly. Per caller, "above median on steps" agrees with
   "above median on length" only 51–74% of the time (50% would mean no relation). The step/length
   correlation is −0.13 to 0.51. Keep both bets.
3. **Why it measures self-knowledge.** It is a probability forecast of a binary event, scored by
   Brier against the caller's base rate. The same instrument already works for the question and
   fetch bets. Questions went from below the bar to +0.13 skill after the 10-29 prompt fix.
4. **Live data has real spread inside each caller.** Steps, 10th/50th/90th percentile: chat 3/7/28,
   curiosity 13/76/184, world-pulse 8/17/60, stage 4/19/58, self 8/8/37. Above the prior 7-day median,
   computed per row with strictly earlier references: curiosity 50%, world-pulse 48%, stage 58%,
   chat 30%, self 35%. Chat and self sit **at** the median 55–57% of the time, which is why their
   rates are low. Strict "greater than" is fine there, because the bar is the caller's own rate, not 50%.
5. **Existing mechanism.** This reuses the settle loop, the table, `_grade_p` and the live endpoint
   from #2976. No new service and no new grader.
6. **Reversibility.** The change is cheap to undo. Old v1 rows stay as they are, and the settler reads
   both versions. Rolling back means restoring the prompt lines and the producer.

## Missing questions

- **Can stance tell what kind of turn it is in?** Answered: yes. On 11-02, stance inputs
  (`cockpit_turn_sighting`, stage `stance_inputs`) for stage turns that fetched open with "You are
  continuing a deliberate source reading… Stage 1 handoff". The handoff already says "I fetched <url>".
  So the fetch miss on stage turns (stance says 2%, the truth is 65%) is a wrong belief, not missing
  input. Stance assumes stage 1 already fetched, but the motor fetches again. No input fix is needed.
  This is exactly the kind of miss a feedback proposal could later correct.

## Proposed schema / API changes

- `orion/schemas/thought.py`:
  - add `TurnSelfPredictionV2` with `asks_question_p`, `fetches_outside_p`,
    `more_steps_than_usual_p` and `longer_reply_than_usual_p` (all `float|None`, 0..1), and
    `schema_version: Literal["turn.self_prediction.v2"]`;
  - `self_prediction: V1 | V2 | None`, discriminated on `schema_version`.
  - The models are `extra="forbid"`, so consumers go first. Patch 1 ships the union and the settler;
    patch 2 flips the producer and the prompt.
- `registry.py`: register V2.
- Stance prompt: replace lines 148–149 with the two probability lines, worded as "more than is usual
  for this kind of turn". No numbers and no examples, because the last example anchored stance.
- Settler: for v2, settle two new faculties.
  - Faculties: `turn.self.more_steps_than_usual` and `turn.self.longer_reply_than_usual`.
  - `hit` = `observed > reference`.
  - With fewer than 20 reference turns, the row is void as `ungradeable_no_reference`.
  - Migration: add `reference real` (additive, `REQUIRED-BY: orion-prediction`).
  - The reference reads `observed` from both the old range faculties and the new ones, so the history
    carries over.
- `/turn-self-bets/live`: no shape change. The new faculties appear with Brier and `skill_vs_caller`.

## Files likely to touch

- `orion/schemas/thought.py`, `orion/schemas/registry.py`
- `orion/thought/stance_react.py` (normalizer), `orion/cognition/prompts/stance_react.j2`
- `orion/thought/tests/test_stance_self_prediction.py`, `orion/harness/tests/test_self_prediction_never_reaches_motor.py`
- `services/orion-prediction/app/turn_self_bets.py`, `app/store.py`, `tests/test_turn_self_bets.py`
- `services/orion-sql-db/manual_migration_prediction_turn_self_bet_reference_v1.sql`

## Non-goals

- Stance sees no hint, no usual size and no grades. Feeding grades back is a later, separate proposal.
- No change to the question and fetch bets.
- No self-monitor organ (#2827), and no A2 (the judge).

## Privacy boundary, danger, rollback

- **Data touched:** the same turn records #2976 already reads. Nothing new is collected.
- **Dangerous failure:** the bet reaching the motor becomes an order ("I said this would be short")
  and fulfils itself. The existing never-reaches-motor test is extended to the v2 fields.
- **Rollback:** revert patch 2 (the prompt and producer go back to v1). The settler keeps grading
  both, and nothing is deleted.

## Acceptance checks

1. Gate tests:
   - a v2 bet settles both new faculties against a strictly earlier reference;
   - a turn never counts in its own reference;
   - fewer than 20 reference turns makes the row void;
   - a v1 bet still settles exactly as today;
   - the motor never sees the v2 fields.
2. Live, within a day of deploy: new `thought_decision` rows carry `turn.self_prediction.v2`, and
   `/turn-self-bets/live` shows the two new faculties with counts.
3. **Decision, about 2 weeks after deploy** (at least 200 graded turns on the main callers):
   - **Keep** a bet only if `skill_vs_caller > 0` and a paired test of its Brier score against the
     caller-rate guess has a one-sided p below 0.05.
   - **Drop** a bet that fails, removing its prompt line.
   - The result goes in the arc's handoff, either way.

## Recommended next patch

Patch 1, the consumer side: the V2 schema and union, the settler for both versions, the migration,
and tests. It is safe to deploy alone, because nothing produces v2 yet. Patch 2 follows in the same
PR series: the prompt lines and the producer.
