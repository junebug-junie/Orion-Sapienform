## Summary

- Rewrites the field demon spec (rev 2). The demon is now the transition model behind the integration rung (#3296), read two ways: how far ahead it sees (the horizon), and how much horizon each cut of the organ lattice costs.
- New measure: integration with a timescale. #3296 compares whole and parts one step (5 min) ahead, in-sample. The demon does the same comparison at 5 to 60 minutes and grades it against the real future.
- Baseline ladder: the clock (time-of-day profile), then persistence, then each organ alone (the full cut). Beating the full cut is integration.
- Rev 1's mood-arc v4 latent space, digester-decay baseline and ensembles are dropped (stale encoder, wrong inputs, not needed for a linear model).
- The live fields join #3296's planned table and schema. The producer runs in orion-prediction. No consumer.

## Outcome moved

Design only. When built, Orion gets a self-measurement that can be checked against real events: how far ahead Orion's own state is predictable, and which organ seams that foresight depends on.

## Current architecture

orion-prediction forecasts one series at a time. No joint forecast of Orion's organs exists. #3296 specifies a lag-1 Gaussian VAR(1) over #3290's organ modes, scored in-sample. Rev 1 of this spec built on the mood-arc v4 encoder, which has since been judged stale.

## Architecture touched

None in this PR. Planned: `orion/substrate/demon.py` (new), #3296's `integration_rung.py`, orion-prediction's hourly job, `IntegrationHourV1` and `prediction_integration_hourly` fields, and `config/integration/organs.yaml`.

## Files changed

- `docs/superpowers/specs/2026-09-30-field-demon-forecaster-design.md`: rev 2 of the design.
- `docs/superpowers/pr-reports/2026-11-18-field-demon-spec-rev2-pr.md`: this report.

## Schema / bus / API changes

- Added: none in this PR. Proposed for patch 3: `demon_*` and `phi_g_by_lag` fields on #3296's `IntegrationHourV1` and `prediction_integration_hourly`.
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

- Added keys: none in this PR. Proposed: `DEMON_ENABLED` in orion-prediction (shipped on).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
None; this PR is a doc only.
```

## Evals run

```text
None; this PR is a doc only. The spec defines the patch 1 synthetic tests and the replay gate.
```

## Docker/build/smoke checks

```text
None; this PR is a doc only.
```

## Review findings fixed

No code review ran, because this PR contains no code. One self-review fix before commit: rev 2's first draft said a known stable VAR has an analytic finite horizon against the clock. That is wrong. With a known model, skill against the clock decays but stays positive. The horizon is finite only because of estimation error. The spec now says so, and the acceptance test checks the skill curve instead.

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Severity:** low. **Concern:** the lag-1 horizon cost may restate Φ_G(1). **Mitigation:** an independence check is in the gate. If it is a copy, it is dropped and only the lag curve and the horizon ship.
- **Severity:** medium. **Concern:** a self-measurement fed to Orion as self-knowledge while it is a fitting artifact (the φ-era 0.655 constant). **Mitigation:** no consumer in this design. Any consumer needs its own proposal after 14 days of a passing gate.
- **Issues opened:** #3705 (rev 1's hygiene leftovers: anomaly-flag default mismatch, stale mood-arc docs).
- **Issues closed:** none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2446

🤖 Generated with [Claude Code](https://claude.com/claude-code)
