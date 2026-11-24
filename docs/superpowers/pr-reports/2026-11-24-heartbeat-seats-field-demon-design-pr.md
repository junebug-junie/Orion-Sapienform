## Summary

- Design: use the heartbeat's 5 fixed seats as organs for the field demon, and use the demon's forecast as the heartbeat's surprise.
- It records Juniper's snag: the heartbeat runs on dead metrics. A heartbeat overhaul (step 0) comes first.
- A partial live audit is included. Fixed atom strengths on 3 of 5 seats. The reheat driver sits below its calm floor. 13 grammar producers are routed to no seat. Seat history is 3 days.
- Spec: `docs/superpowers/specs/2026-11-24-heartbeat-seats-field-demon-design.md`.

## Outcome moved

The Φ rung (#3296) gets a path to stable organs now, instead of waiting for data-derived modes to settle (#3822). The heartbeat gets a path to a real forecast.

## Current architecture

The heartbeat routes 5 services to 5 seats of a tensor network and publishes a ledger-only verdict. The demon ran only on drifting data-derived modes.

## Architecture touched

Docs only.

## Files changed

- `docs/superpowers/specs/2026-11-24-heartbeat-seats-field-demon-design.md`: the design.
- `docs/superpowers/pr-reports/2026-11-24-heartbeat-seats-field-demon-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none. Proposed changes are in the spec.

## Env/config changes

None.

## Tests run

```text
Docs only. Live audit queries (grammar_events last 6 h; heartbeat /health) are quoted in the spec.
```

## Evals run

```text
None (design).
```

## Docker/build/smoke checks

```text
None needed.
```

## Review findings fixed

Design doc, no code. No code review run.

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity:** medium. **Concern:** the audit is partial, and Juniper may know of more dead inputs. **Mitigation:** step 0 starts with a full audit.
- **Issues opened:** https://github.com/junebug-junie/Orion-Sapienform/issues/3838 (heartbeat overhaul). **Issues closed:** none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3837

🤖 Generated with [Claude Code](https://claude.com/claude-code)
