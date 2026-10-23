## Summary

- This PR answers "does Orion's own load move the cabinet temperature?" with a read-only passive fit, per host. The fit uses outdoor temperature, AC, hour of day and 0/15/30/60-minute lags.
- It adds a deterministic hourly replay of the body-economy price rule over 10-07..10-22, and grades R1, R3 and R4.
- It seeds the Budget organ with `orion/physiology/prices.py`, the pure price math:
  - dual ascent;
  - `no_reading` held, never read as 0;
  - `exogenous_bind` anti-windup;
  - bounded step.
  Nothing live reads it yet.
- It writes up the findings. The findings overturn part of the design's Fact 2 and settle the circe 0.10 vs 0.22 correlation discrepancy.

## Outcome moved

**circe.** circe's load measurably warms the cabinet in the short run. Its typical busy step adds **+1.29 °C within 2 h** (95% CI 0.85..1.75). Over a day, circe adds +0.18 °C per 100 W (CI −0.11..+0.44).

**athena.** athena's 51 W swing is worth about 0.25–0.37 °C.

**Outdoor temperature explains nothing.** The daily correlation is r = +0.05 over 27 days. "Heat is mostly weather" is overturned.

**Something physical changed in the cabinet on 10-16 at about 21:00–01:00Z.** The lidar went 116 → 94 mm, humidity stepped down, and the floor temperature rose about 5 °C. This is the main suspect for the 10-17..22 heat. `UNVERIFIED`: Juniper should confirm what happened.

**The 0.10 vs 0.22 discrepancy** comes from two different 14-day windows, not from chassis vs rollup watts.

**Replay verdicts:**
- R1: FAIL as written. It passes outside the integrator's decay tail.
- R3: FAIL with the literal rule. It passes with hysteresis, a bounded ramp and point sensitivities under the oracle forecast.
- R4: PASS. Worst error −6.1% against $197.17 realized.
- R2, R5, R6, R7: NOT YET EVALUABLE.

## Current architecture

`orion/physiology` did not exist. No replay harness existed. Heat sensitivity was two disagreeing correlation passes in the design doc.

## Architecture touched

- A new pure package, `orion/physiology/` (price math only, no I/O).
- Read-only analysis scripts under `scripts/analysis/`.
- A new CI workflow.

No services, bus, schema or env changes.

## Files changed

- `orion/physiology/__init__.py`, `orion/physiology/prices.py`: the pure price math (`step_price`, `heat_contribution`, `heat_budget`, `usable_sensitivity`, `exogenous_bind_with_hysteresis`, `size_ceiling`, `month_end_forecast`). `bind_ramp` and `slack_decay_floor` are opt-in; the defaults are the design's literal rule.
- `orion/physiology/tests/test_prices.py`: rest-state return, anti-windup, ceiling, `no_reading` never read as 0, bounded step, hysteresis.
- `scripts/analysis/_body_economy_data.py`: the read-only extract (SELECT inside `BEGIN READ ONLY` via `docker exec psql`, with an optional JSON cache).
- `scripts/analysis/fit_heat_sensitivity.py`: the passive fit, model variants, day-block bootstrap and HAC intervals, the correlation audit with exact SQL, and JSON + markdown output.
- `scripts/analysis/replay_body_economy.py`: the heat and energy price replay, plus the R1/R3/R4 graders.
- `scripts/analysis/tests/test_body_economy_replay_and_fit.py`: offline synthetic-data tests (fit recovers a known sensitivity; vacuous-pass regressions).
- `.github/workflows/body-economy-analysis.yml`: runs both test files.
- `docs/superpowers/specs/2026-10-23-body-economy-heat-and-replay-findings.md`: the findings.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none live.
- Compatibility notes: `orion/schemas/registry.py`, `orion/bus/channels.yaml` and sql-writer are not touched; another agent owns them.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced: not needed, because no template changed.
- Skipped keys requiring operator action: none.
- Optional script-only override: `ORION_ANALYSIS_PSQL` (psql command prefix). It is not a service setting.

## Tests run

```text
.venv/bin/python -m pytest -q -p no:cacheprovider orion/physiology/tests scripts/analysis/tests/test_body_economy_replay_and_fit.py
30 passed in 0.77s
```

## Evals run

These are live, read-only runs against Postgres on 10-23 at about 05:00 UTC. They are this patch's eval, and the full numbers are in the findings doc.

```text
fit_heat_sensitivity.py  -> primary (day dummies): athena +0.49 (-0.17..+1.03), circe +0.18 (-0.11..+0.44) degC/100 W, n=2543
                            first differences (2 h): athena +0.73 (+0.14..+1.13), circe +0.35 (+0.23..+0.48)
replay_body_economy.py   -> R1 FAIL (PASS excl. decay tail), R3 FAIL literal / PASS oracle+hysteresis+ramp+point,
                            R4 PASS (worst -6.1%; no-imputation cross-check worst -7.1%)
```

## Docker/build/smoke checks

```text
Not applicable: no service or runtime change.
```

## Review findings fixed

A code review ran in a subagent and reported 2 MUST, 7 SHOULD and several NIT findings. All of them are fixed.

- Finding (MUST): R1 passed with no data. An empty input, or all hours `no_reading`, gave `pass=True`.
  - Fix: R1 now requires coverage ≥ 80% and non-empty under-line and over-line sets. `run()` refuses to replay when any host's sensitivity is unusable.
  - Evidence: `test_r1_with_no_data_is_not_a_pass`.
- Finding (MUST): R3 ignored a price that stayed above 0 after its constraint cleared and then re-bound.
  - Fix: a clear stretch longer than one horizon that never reaches 0 counts as `stuck_positive_past_horizon` and fails R3.
  - Evidence: `test_r3_price_stuck_positive_after_clearing_fails`. Live result: 0 stuck.
- Finding: the R1 exogenous-flag check passed by construction, because it used the same forecast that set the flag.
  - Fix: "unmeetable" is now judged on the realized cabinet temperature, with a recall gate of ≥ 90%.
  - Evidence: live recall 90–96%.
- Finding: the 10-16 cabinet shift landed inside a UTC day dummy.
  - Fix: the transient bins (21:00–01:00Z) are dropped, and models without day dummies get a step regressor.
  - Evidence: the no-dummy athena estimate fell from 3.15 to 0.94.
- Finding: the HAC SE was too narrow.
  - Fix: 96 lags, and it is labelled a lower bound. The replay uses the bootstrap interval.
- Finding: the reference day could be empty, which made the day dummies collinear with the constant.
  - Fix: the reference day is chosen after filtering.
  - Evidence: `test_fit_with_empty_reference_day_keeps_controls_finite`.
- Finding: R4 leaned on gap-filled data.
  - Fix: added a no-imputation athena + circe cross-check, and the fill share is stated in the findings.
- Finding: an empty sensitivity map gave a known 0. `slack_decay_floor > max_step` broke the bounded step. A NaN previous price propagated.
  - Fix: all three now return `None` or raise.
  - Evidence: unit tests.
- NITs fixed:
  - the bootstrap docstring;
  - `or 0` swings now show as n/a;
  - a null `covered_sec`;
  - explicit UTC on date literals;
  - a guard on the primary fit;
  - the baseline look-ahead is noted in the report.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium.
  - Concern: the passive fit is not causal, and athena carries the sensor.
  - Mitigation: these are the passive prior only. Organ 6's randomized holdback pulses should overrule them.
- Severity: medium.
  - Concern: the 10-16 cabinet change is inferred from sensor steps, not confirmed.
  - Mitigation: flagged `UNVERIFIED`. The fit restricted to before the change gives the same sensitivities.
- Severity: low.
  - Concern: the replay is open-loop, and the price ceiling is a placeholder (no wants or costs exist to size it).
  - Mitigation: stated in the outputs. R3's ceiling clause needs patch C's shadow bids.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
