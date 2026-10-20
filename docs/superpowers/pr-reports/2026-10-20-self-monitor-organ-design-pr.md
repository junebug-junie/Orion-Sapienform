# Design: the self-monitor organ (rung 6, metacognition)

## Summary

- Design only, no code. Spec: `docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md`.
- Rung 6 is rebuilt as an **organ**, not a single tuner. It does four things:
  - collects every bet Orion makes that the world settles;
  - keeps a per-faculty track record (reliable, unreliable, broken, miscalibrated, ungradeable);
  - acts in one direction only (more caution, never less);
  - gives Orion the record as self-knowledge.
- Built on a live audit of 10+ bet sources. Only 2 are clean bets today: action claims
  and temperature forecasts. So "I can't tell yet, because Y" is a first-class state.
- Reuses existing pieces instead of new math:
  - orion-prediction's grader;
  - the self-signal statistics;
  - regulation's publish pattern;
  - Temporal Self arcs;
  - the mutation pipeline as one effector.
- Recommends retiring the 10-16 prediction-expectation tuner as one unit.
- Ladder doc rung 6 now points to the spec.

## Outcome moved

No runtime change. This decides the shape of rung 6 before any more code ships.
The live evidence that motivated it:
- 29 painting attempts in 7 days, all abandoned.
- Orion's own estimate fell to 1.7%, but no selector reads it.
- Dreams adopted 1/183 vs random 2/66.
- Forecasts are no better than "same as now" at 5 min.
- None of this is known to Orion.

## Current architecture

See the spec's "Current architecture" section: the bet-source table with live volumes,
the "nothing acts on track records" evidence with file:line, and the reuse list.

## Architecture touched

Docs only.

## Files changed

- `docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md`: new design.
- `docs/plans/substrate/2026-06-27-self-modeling-loop-ladder.md`: rung 6 redesign pointer.
- `docs/superpowers/pr-reports/2026-10-20-self-monitor-organ-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none (proposed in the spec: `SettledBetV1`, `FacultyRecordV1`,
  `SelfMonitorStateV1`, `orion:self_monitor:latest`, `GET /self_monitor/state`, a history
  table).
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
None: docs only.
```

## Evals run

```text
Live read-only audits (2026-10-20 ~23:15 UTC), in the spec:
- bet sources: substrate_action_outcomes, grammar_acts, prediction_forecast/exceedance,
  vision verdicts + orion_ask, vision_percept_expectation, substrate_reverie_thought,
  curiosity_run_outcomes, dream_hypothesis, attention_loop_outcome, chat_response_feedback,
  energy_reconcile, power_intent_settled, substrate_attention_self_model
- scripts/dream_hypothesis_scorecard.py: "recombination not beating random"
- orion-prediction /health: coverage_out_of_band 0.88; 5 min 1.08x worse than baseline
```

## Docker/build/smoke checks

```text
Not applicable (docs only).
```

## Review findings fixed

- Design review: Juniper reviews the open questions (spec "Missing questions").
  No code review: there is no code.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium
- Concern: 8 of 10 sources are ungradeable today, so v1's graded set is small: actions
  and forecasts, with vision and attention self-model to follow.
- Mitigation: parallel source-fix PRs are listed per faculty, and the organ reports each
  ungradeable reason rather than hiding it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2827

🤖 Generated with [Claude Code](https://claude.com/claude-code)
