## Summary

- Design only, no code: `docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md`.
- Stance adds a small self-prediction to each turn: will I ask a question, will I fetch an outside source, how many steps, how long a reply. Orion states a confidence for each.
- The prediction is hidden from the motor and from the judge, so it stays a prediction and never becomes an order.
- Plain code settles each prediction against the turn record. The results feed the self-monitor organ (#2827) as `SettledBetV1` bets, and come back to Orion through the organ's stance cue.
- Tiers: no new model call, so on Juniper's turns the bet rides at Tier 0. Settling is code (Tier 1 upkeep). The standing self (option C) is Tier 1.
- Includes the arc after A (options B–F from the storyboard; G dropped by Juniper).

## Outcome moved

Closes the missing loop "what Orion meant to do vs what Orion did". Today stance writes a plan every turn, nothing checks it, and the judge treats it as an order.

## Current architecture

Traced 2026-10-18; storyboard: https://claude.ai/artifact/71ELFgEA56xpMJSBB76dRy. Live numbers (14 and 30 days) are in the spec.

## Architecture touched

None yet. The proposed patches A1 (producer + persistence), A2 (judge prompt) and A3 (organ adapter) are listed in the spec.

## Files changed

- `docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-design.md`: the design.
- `docs/superpowers/pr-reports/2026-10-23-unified-turn-self-prediction-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none (proposed `TurnSelfPredictionV1` and `thought_decision.self_prediction`).
- Removed / Renamed / Behavior changed: none.

## Env/config changes

None. Proposed: `STANCE_SELF_PREDICTION_ENABLED=true`.

## Tests run

```text
Docs only. git diff --check: clean.
```

## Evals run

```text
None. Live-data queries are in the spec (thought_decision x harness_turn_trace, 14 d; orion-thought / orion-mind logs, 72 h).
```

## Docker/build/smoke checks

```text
Not applicable (docs only).
```

## Review findings fixed

- Not run: design doc. Code review runs on A1.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium.
  - Concern: if the prediction reaches the motor, it fulfils itself.
  - Mitigation: A1 ships a test that every rendered motor and judge prompt is free of it.
- Severity: medium.
  - Concern: stated values may be degenerate (flat p, huge ranges).
  - Mitigation: the metric gate re-runs at day 7, before A3 reads them.
- Severity: low.
  - Concern: depends on #2827 (open) for grading.
  - Mitigation: A1 can ship first. The bets accumulate until the organ grades them.
- Correction (from PR #2879): Juniper's turns were already fully interactive for Mind; the real bug was Orion's own turns riding her tier. #2879 fixed and deployed it. Mind fallback is 92/228 runs in 72 h (stage-1 empty synthesis 49, stage-2 queue timeouts 31, self-sense 8, journal 1); none on Juniper's turns. The spec now treats a shed Mind run as "no reading".

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2869

🤖 Generated with [Claude Code](https://claude.com/claude-code)
