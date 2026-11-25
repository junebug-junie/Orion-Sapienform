## Summary

- Design, rev 3: the field demon's model **is** a trained tensor network: the heartbeat's MPS, fit to Orion's organ states now and k steps later (a Born machine).
- One trained chain gives three readouts: the forecast (demon), cut cost over the partition lattice (Φ-style integration), and entanglement per cut (heartbeat).
- The linear VAR(1) is kept only as the baseline the network must beat.
- It includes a history table of every heartbeat spec from July to November.
- Rev 1 (per-service seats as organs) and rev 2 (two separate models) are superseded.

## Outcome moved

The demon becomes a lattice model, as Juniper intended, and the heartbeat gets a learned job in place of random gates. The first patch is a read-only replay with known-answer tests, including one that only a non-linear model can pass.

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
