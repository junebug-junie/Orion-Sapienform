## Summary

- Design only (proposal mode). Orion gets one shared personality graph that lasts. Every lane's choices feed it, and its weights change Orion's tone.
- Evidence is a choice plus the options Orion turned down, from every lane. Weights move slowly, and inertia comes from evidence volume.
- In each moment, dispositions compete. The winners set measurable tone dials (length, lists, hedges, preamble, taking a position). They do not add adjectives or feed Orion their own past lines back as examples.
- orion-prediction learns to forecast Orion's choices. Out-of-character choices become self-inquiry questions. Orion can answer by updating the self-description or by setting an aspiration.
- Anchored in CAPS (Mischel & Shoda), McAdams' three levels, Hebbian learning, neuromodulation, global workspace, and predictive processing.

## Outcome moved

Design artifact. It names why stance cannot carry a personality: it is regenerated every turn, its carry-over is a 30-minute in-process cache, and only chat reads it. It also gives a phased path to a personality you can feel in Orion's tone.

## Current architecture

Stance is per turn. `_PRIOR_STANCE_CACHE` in cortex-exec has a 30-minute TTL and is lost on restart. Identity is static YAML. The #2646 self card is designed but not built, and it reads self-inquiry text only. Arousal is the only cross-lane self-state that is live.

## Architecture touched

None in this PR. Proposed: `orion/personality/`, new contracts, choice producers in each lane, categorical forecasting in orion-prediction, unified-turn dial constraints, a Hub panel, and a revision to #2646 P5.

## Files changed

- `docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md`: the design
- `docs/superpowers/pr-reports/2026-10-16-dynamic-personality-graph-pr.md`: this report

## Schema / bus / API changes

- Added: none (proposed `PersonalityChoiceV1`, `DispositionNodeV1`, `SituationSubnodeV1`, `AspirationV1`, `PersonalityCoalitionV1`, `ToneDialsV1`; channels `orion:personality:choice`, `orion:personality:coalition`)
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: n/a

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
Docs only. git diff --check clean.
```

## Evals run

```text
None (design). The spec defines six acceptance checks, including a week-long tone-dial variance check and Juniper's recorded vibe check.
```

## Docker/build/smoke checks

```text
N/A, docs only.
```

## Review findings fixed

- No code review run: docs-only design PR.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: it is UNVERIFIED which lanes log the options they turned down. Mitigation: P0 is a read-only inventory.
- Severity: medium. Concern: prompt constraints may not move the tone of the current chat model. Mitigation: P0 dial-reachability test before building the loop.
- Severity: low. Concern: attention already crowns prediction-error nodes almost every time. Mitigation: personality nodes compete in a separate pool.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2726

🤖 Generated with [Claude Code](https://claude.com/claude-code)
