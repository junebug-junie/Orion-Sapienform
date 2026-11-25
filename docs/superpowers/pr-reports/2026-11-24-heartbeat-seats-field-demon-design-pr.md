## Summary

- Design, rev 2: the field demon and a **trained** heartbeat (an MPS Born machine) fit the same organ variables and check each other's cuts.
- Rev 1 used the heartbeat's per-service seats as organs. Homework showed that is weak: seat counts are mostly the bus counts the demon already has, and the heartbeat's gates are random (11-08 replay: about 3 events of memory, verdict no better than shuffled).
- It adds a history table of every heartbeat spec from July to November and what each found.
- It records Juniper's snag (dead inputs). Step 0 is #3838, in progress.

## Outcome moved

The heartbeat gets a real job: a second, non-linear model of where Orion comes apart, cross-checking the demon's Φ-style cut cost (#3296). The first patch is a read-only cross-check replay.

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
