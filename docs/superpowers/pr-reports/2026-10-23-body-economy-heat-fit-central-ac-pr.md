## Summary

- This corrects the heat findings from #2861 (merged). The fit now reuses orion-prediction's own definitions instead of parallel ones:
  - `athena.central_ac_proxy` (#2829), read through orion-prediction's own registry parser;
  - the `prediction_change_event` rows (#2780): the exclusion window is masked, and the 10-17 sensor-geometry break is treated as a **measurement offset**;
  - the portable AC from `home_cooling_sample`.
- It adds a house-meter variant that measures the central AC's load the same way `run_house_heat_balance_eval.py` does.
- It adds a decomposition of the cabinet rise between 09-26..10-10 and 10-18..10-22.
- It rewrites the findings headline and removes the "something physical changed" and "weather explains nothing" claims.
- It reruns the replay, because the sensitivities moved.

## Outcome moved

**New headline.** The cabinet reading rose +4.2 °C (29.1 → 33.3 °C):
- **+2.0 °C** is the sensor reading higher after the 10-17 move. That is a measurement offset, from #2780, not heat.
- **+0.4 °C (CI +0.2..+0.6)** is the central AC no longer running. When it runs, it cools the cabinet by 0.23 °C per °F above 75 °F (3 h earlier). That is about 2 °C on a hot afternoon, but the proxy averaged only about 1.5 °F in the warm period.
- **+0.1 to +0.4 °C** is Orion's measured load.
- **+0.1 °C** is the portable AC, which mirrors the heat rather than causing it.
- **+1.5 to +2.1 °C** is real warming that began on 10-14/15, with the cause unresolved. Four candidates changed within the same 2–3 days, and this data cannot separate them:
  - the T10 swap;
  - circe's load rising on 10-14;
  - the house warming with the AC off (outdoor 9 → 17 °C);
  - house heating.

**This does not support "weather through the central AC is the main driver" of the 10-18..22 rise.** That mechanism is real, but on this data it is about 0.4 °C of the 4.2 °C. The findings doc says so plainly and proposes a controlled check.

**Per-host sensitivity (primary model):**

| host | over a day | within 2 h |
|---|---|---|
| athena | +0.40 °C per 100 W (CI −1.04..+1.47) | +0.62 (CI +0.18..+0.95) |
| circe | +0.28 (CI −0.08..+0.67) | +0.35 (CI +0.24..+0.48) |

circe's typical 366 W step is worth about +1.3 °C within 2 h.

**Replay:**
- R1: FAIL with the literal rule. PASS with hysteresis on the bind flag (oracle).
- R3: FAIL with the upper-bound sensitivities (unflagged ceiling runs of 18–20 h). PASS with point sensitivities (oracle).
- R4: PASS, unchanged (worst error −6.1%).

## Current architecture

#2861's fit had three problems:
- It used a linear outdoor-temperature term.
- It hand-coded a "cabinet shift" constant.
- It took the AC from `orion_energy_hourly`.

orion-prediction already owned the central-AC covariate, the change-events table and the cabinet-cooling series.

## Architecture touched

- Read-only analysis scripts only.
- The fit now **reads** `config/prediction/series.v1.yaml` through `services/orion-prediction/app/series.py`, and **reads** the `prediction_change_event`, `home_cooling_sample` and `energy_usage_interval` tables. It writes nothing.
- No service, schema, bus or env changes.

## Files changed

- `scripts/analysis/fit_heat_sensitivity.py`:
  - proxy loaded through orion-prediction's parser, applied with `frame.py`'s rule;
  - change-event exclusions, break steps and fixed offsets;
  - house-meter variant;
  - `decompose()`;
  - new model set and report.
- `scripts/analysis/_body_economy_data.py`: read-only queries for the portable AC, the house meter and change events. The weather fetch starts 6 h earlier so the lagged proxy has values at the window start.
- `scripts/analysis/tests/test_body_economy_replay_and_fit.py`: synthetic data with a known proxy effect, sensor offset and exclusion window.
  - The proxy definition must match orion-prediction's registry.
  - The fit must recover the proxy effect and the offset, and must mask the exclusion.
  - The decomposition must add up.
- `.github/workflows/body-economy-analysis.yml`: adds pyyaml; triggers on the series YAML and on `series.py`.
- `docs/superpowers/specs/2026-10-23-body-economy-heat-and-replay-findings.md`: rewritten.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none live.
- Compatibility notes: `registry.py`, `channels.yaml` and sql-writer are not touched.

## Env/config changes

- None. No `.env_example` changed, so no `.env` sync was needed.

## Tests run

```text
.venv/bin/python -m pytest -q -p no:cacheprovider orion/physiology/tests scripts/analysis/tests/test_body_economy_replay_and_fit.py
33 passed
```

## Evals run

These ran live and read-only on 10-23.

```text
services/orion-prediction/evals/run_house_heat_balance_eval.py -> +0.303 kWh/h per F above 75 (3 h earlier), se 0.037, p=3.2e-16, 604 h
fit_heat_sensitivity.py -> primary athena +0.40 (-1.04..+1.47), circe +0.28 (-0.08..+0.67) degC/100 W; 2 h step athena +0.62, circe +0.35
  rise decomposition (primary): T10 step +2.09, sensor offset +2.00 (fixed), portable AC +0.14, athena +0.09, circe +0.04, proxy -0.04, residual -0.08
  without free steps: sensor offset +2.00, athena +0.42, residual +1.54
replay_body_economy.py -> sensitivities used (upper bounds) athena 1.47, circe 0.67, hecate 1.67; R1/R3/R4 as above
```

## Docker/build/smoke checks

```text
Not applicable: no runtime change.
```

## Review findings fixed

- Finding (Juniper, through the orchestrator): the heat findings in #2861 were wrong.
  - What was wrong: they claimed "something physical changed" and that "weather explains nothing", and they used a linear outdoor term.
  - The real story: cabinet heat is outside weather acting through the house's central AC. orion-prediction already models this (#2829), and its change events (#2780) record the 10-16/17 exclusion and the ~2 °C sensor-geometry offset.
  - Fix: reused the proxy definition through orion-prediction's own parser, and the change-events rows and windows. Treated the sensor break as a measurement offset. Added the house-meter variant and the rise decomposition. Rewrote the headline. Reran R1/R3/R4.
  - Evidence: the decomposition tables in the findings doc, and the new tests in `test_body_economy_replay_and_fit.py`.
  - Where the data disagrees with the correction: neither AC signal explains the warming when fitted directly. The central-AC attribution is stated as resting on timing (house-meter afternoon load), not as a fitted effect.
- Finding (subagent review, MUST): the doc's central-AC causal claim contradicted the script's own daily table.
  - The facts: the proxy was 0 from 10-11, the cabinet bottomed out on the cold days 10-12/13, and the rise starts 10-14/15, alongside circe's load rise and the T10 swap. The house-meter figures were not reproducible from the script.
  - Fix: the central AC is now one quantified factor (+0.4 °C, from the models where the proxy has the expected sign). The unresolved 1.5–2 °C names all four candidates. "confirms"/"exactly" are dropped. `daily_levels` now emits daily and afternoon house-meter rest kWh, so the meter numbers can be reproduced.
- Finding (MUST): "the extra heat is not Orion's load" ignored that the T10 step is an Orion hardware change.
  - Fix: the doc now says Orion's *measured watts* explain 0.1–0.4 °C, and the T10 step (fans and airflow included) cannot be separated from the other candidates.
- Finding (SHOULD): the doc omitted that the proxy reaches a significant −0.23 °C per °F with a linear outdoor term or day dummies.
  - Fix: reported, and used for the +0.4 °C central-AC share.
- Finding (SHOULD): "reuses … exactly" overstated what was reused.
  - Fix: the doc now separates what was reused (proxy definition, change events, exclusions) from our own choice (the fixed 2 °C offset). orion-prediction's free-step treatment is reported as `sensor_step_estimated`.
- Finding (SHOULD): the house-meter row was mislabelled.
  - Fix: both variants are now listed, each with its own label.
- Finding (SHOULD): the house-meter decompositions run on fewer rows (a 4.03 °C rise, not 4.23 °C).
  - Fix: the doc notes it, and `decompose` now reports the row counts.
- Finding (SHOULD): columns with no variation after filtering (a break step outside the window) stayed in the model.
  - Fix: they are now dropped in `design` and in `first_difference_hourly`.
- NITs fixed:
  - hecate is now averaged over its readings, and is 0 only for a fully missing hour, as in the eval;
  - the docstrings for the house-meter window and hecate are corrected;
  - `decompose` takes base and target means from the raw reading;
  - the CI note on the decomposition is in the doc;
  - circe's rise is corrected to 15 W;
  - hecate's awake hours (23.5 h at > 150 W) are now emitted by the script.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium.
  - Concern: the AC going quiet, the T10 swap and the sensor move fall within about 2 days of each other. Their split rests on timing, plus the T10 card's measured watts.
  - Mitigation: stated plainly in the findings. A controlled central-AC check would settle it.
- Severity: medium.
  - Concern: the sensor offset is fixed at #2780's recorded "~2 C", not estimated. The warming share moves one-for-one with it.
  - Mitigation: the free-step variant is reported beside it (3.3–3.7 °C).
- Severity: low.
  - Concern: the upper-bound sensitivities are now 3–4× the point estimates, which makes R3 fail.
  - Mitigation: flagged as a Patch B design question.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2868

🤖 Generated with [Claude Code](https://claude.com/claude-code)
