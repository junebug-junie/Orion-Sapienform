## Summary

- Design only: `docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md`.
- One prediction loop that spans the door and walkway cameras. Orion predicts when a passage reappears on the other camera, points that camera's attention there, grades the prediction (met / late / missed / could not look), and explains real misses with hypotheses it can check.
- Salience for street events comes from who it was, what happened, and how surprising it was. A ring always passes; everything else must be unusual for the hour.
- Adds an event-to-event effect estimator ("does A raise B within a lag") to `orion-prediction` as the general answer to "does X affect Y".
- Found on the way: motion stories use one global 2560x1920 frame size for the doorbell too (likely wrong direction threshold there), and the Ring "ding" is dropped at the retina.

## Outcome moved

No runtime change. This replaces "two unrelated passages" with a buildable plan in 4 phases, each with acceptance checks.

## Current architecture

See the spec's "Current architecture" table (verified live 2026-10-20): per-camera motion stories and individuals, 0 cross-camera individuals, no camera event can win attention, ring kind dropped at `orion-vision-retina/app/main.py:538-546`.

## Architecture touched

None (docs only).

## Files changed

- `docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md`: the design.
- `docs/superpowers/pr-reports/2026-10-20-street-continuity-handoff-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none (proposed only; listed in the spec)
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
None: docs-only change.
```

## Evals run

```text
None: docs-only change. The spec defines the phase-2 replay eval.
```

## Docker/build/smoke checks

```text
None: docs-only change.
```

## Review findings fixed

- None: design review is Juniper's (5 open questions in the spec).

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
  - Concern: the doorbell's real frame size is UNVERIFIED. Stored frames were already pruned.
  - Mitigation: phase 1 reads the frame size from the artifact.
- Severity: low
  - Concern: whether `ring_doorbell` 0.9.14 can open a live view with no event is UNVERIFIED.
  - Mitigation: tested before the self-opened-view part of phase 3.
- Severity: note
  - Concern: the walkway reach queue doc (PR #2752) is not on main yet, so this item is not added there.
  - Mitigation: add it when that PR lands.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2803

🤖 Generated with [Claude Code](https://claude.com/claude-code)
