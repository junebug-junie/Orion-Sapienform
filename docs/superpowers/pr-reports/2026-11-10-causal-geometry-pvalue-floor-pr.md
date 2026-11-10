# fix(causal-geometry): time-shift null with a tail fit ends the BH floor cliff

## Summary

- Causal geometry found 0 edges in every snapshot since 2026-10-30. The cause was the test, not the data. Its p-value could not go below 0.005 after the within-pair correction for 10 comparisons. Benjamini-Hochberg (BH) at 171 pairs needs 0.00029 for its first pair.
- The p-value no longer has a surrogate-count floor for pairs with enough independent data. In the body of the null it is the plain rank p. Past the last 10 surrogates it comes from a generalized Pareto (GPD) tail fit.
- The null changed. One series is shifted against the other, and r is computed over the overlap only (no wrap-around). Shifts closer to zero than 5% of the window are excluded, so a real link no longer sits inside its own null.
- Some pairs have few independent samples: Bartlett effective n (n_eff) below 100. Examples are trending pairs, slow on/off pairs and pure daily-cycle pairs. These pairs get no extrapolation, and their p never goes below 1/(n_eff+1). So they cannot pile up at a floor and pass BH together.
- BH and the within-pair Bonferroni are unchanged.
- A new CI job runs the causal-geometry tests. No CI job ran them before.
- Spec: PR #3168 (`docs/prediction-error-suite`), integration-rung precondition. Juniper approved it on 2026-11-10.

## Outcome moved

Live replay, read-only. 7-day window ending 2026-11-10 08:08 UTC: 19 channels, m=171 pairs. Both rows use the same data.

| method | edges | pairs at the old floor 10/2001 |
|---|---|---|
| before: circular add-one rank p | 0 | 14 |
| after: time-shift null + GPD tail | 6 | 0 |

```text
* thermal -> power                  lag=0s     r=+0.682 n=2017 p=1.0e-12
* cpu -> power                      lag=0s     r=+0.444 n=2017 p=9.1e-12
* cpu -> thermal                    lag=0s     r=+0.511 n=2017 p=2.8e-10
* cabinet_em_activity -> cpu        lag=0s     r=+0.247 n=2017 p=3.0e-06
* power -> cabinet_em_activity      lag=1800s  r=-0.234 n=2011 p=3.7e-06
* cabinet_em_activity -> thermal    lag=0s     r=+0.224 n=2017 p=4.5e-05
x power -> fan                      lag=600s   r=+0.038 n=2015 p=0.0055   (next below the line)
```

(All channels are `biometrics:*`.)

**Is 6 plausible?** Yes.

- All six edges are inside the cabinet hardware cluster: CPU load, power draw, heat, and the electromagnetic sensor beside them. These links are expected physics.
- The next pair is about 100 times above the last edge. The edges do not sit in a pile at the threshold.
- The result does not depend on the seed. At n=2017 the test uses all 1,818 shifts, so the random seed is never used.
- Shorter windows give fewer edges: 3 at 72 h and 0 at 24 h. A short window has fewer independent samples, so this is expected. Production uses 168 h (`FIELD_PLASTICITY_PRODUCER_WINDOW_HOURS=168`).
- The spec's probe with the Bartlett effective-n test passed 30 pairs. This method is stricter. That agrees with the calibration below: the Bartlett test is liberal on persistent series.

The power/thermal edge changes direction between runs (power->thermal, then thermal->power). At lag 0 the "direction" only tells which side won a tie. The module name says "causal", but the measurement is lagged correlation, not causation.

## Method choice and why

The spec offered two options: a Bartlett effective-n test, or a normal fit to the surrogate distribution. The first version of this PR used a normal fit to Fisher-z circular surrogates. The code review rejected it with synthetic nulls. I tried three more methods before one passed. I scored every method on the same six null shapes (independent pairs, n=2016):

| method | what broke it |
|---|---|
| Bartlett effective-n (analytic) | Liberal on persistent series: 6.6% of null pairs below 0.05 at phi=0.98, and 0.3% below 0.002 at phi=0.95, n=500. |
| Normal fit to circular surrogates (v1 of this PR) | Bursty channels: 1.3% of null pairs below 5e-5, and 34 false edges over 6 seeds of an m=171 null family. Daily cycles: 0% below 0.05 (no power). |
| Circular rank p + GPD tail | Real links were lost to their own near-zero shifts. Live power/thermal: the 15 largest surrogates were all within 25 buckets of zero, up to r=0.60 against an observed 0.68. Result: p=0.0016 and no edge. |
| Circular null with a guard around zero | The wrap-around seam lowers r at every far shift, so persistent and periodic nulls go liberal. Daily-cycle null family: 357 false edges over 5 seeds. |
| **Time-shift overlap null + guard + GPD tail + n_eff gate (chosen)** | Passes all six shapes and both cliff fixtures (see the test results below). |

Why the chosen method is sound:

- **Null.** The null hypothesis is: the two series are independent, and each keeps its own serial structure.
  - Each surrogate is two real, continuous stretches of the data. Autocorrelation, daily cycles and bursts stay intact. Nothing is cut and joined back together.
  - A circular shift adds a seam. This null has no seam, so a guard around zero is safe.
  - The overlap is n/2 to n long. For short-memory series the null is therefore slightly wider than the true null. That makes the test conservative, not liberal.
- **Tail.** Past the last 10 surrogates, the rank p can only say "below about 10/N".
  - The Pickands-Balkema-de Haan theorem says exceedances over a high threshold follow a GPD, whatever the parent distribution is.
  - The fit follows Knijnenburg et al. 2009 (Bioinformatics 25:i161): the top 250 surrogates, with the shape estimated by probability-weighted moments (Hosking & Wallis 1987).
  - The shape is clamped at xi >= 0. A tail that looks bounded is never used to call a value impossible.
  - A heavy-tailed null, such as a bursty channel, gets a heavy fitted tail.
- **Gate.** A tail fit is only valid when many independent values build the tail.
  - Below a Bartlett effective sample size of 100, the p is the rank p, and never below 1/(n_eff+1). That is the floor of a shift test over n_eff exchangeable units.
  - Live, the gate catches the trending pair disk_capacity/mem (r=-0.76, n_eff=8). The circular rank p had put that pair at the floor next to the real edges.

## Current architecture

`orion-field-digester` runs `run_causal_geometry_production_cycle()` on a timer of a few hours. The cycle does these steps:

1. It fetches 7 days of data: biometrics pressures, attention salience, and bus-synaptic node values.
2. It puts the data into 5-minute buckets.
3. For each pair, it picks the best of 2 directions x 5 lags.
4. It tests that winner against 2000 circular shifts with an add-one rank p.
5. It applies Bonferroni x10, then BH across pairs.
6. It publishes the snapshot on `orion:causal_geometry:snapshot`.

## Architecture touched

Only the significance step in `orion/substrate/causal_geometry_engine.py`. There is no schema, bus, env or Docker change.

## Files changed

- `orion/substrate/causal_geometry_engine.py`:
  - Removed `circular_shift_pvalue` and `circular_shift_surrogate_rs`.
  - Added `time_shift_surrogate_rs` (vectorized and exact), `effective_sample_size`, `_gpd_tail_survival`, `surrogate_tail_pvalue` and `time_shift_pvalue`.
  - Added the knobs `GUARD_FRACTION`, `GUARD_MIN`, `MAX_SHIFT_FRACTION`, `MIN_EXCEEDANCES`, `TAIL_FIT_SIZE`, `N_EFF_MIN` and `MIN_REPORTED_PVALUE`.
- `scripts/causal_geometry_report.py`: the docstring describes the new null and both floor incidents. The help text for `--n-surrogates` and `--seed` is updated. The script re-exports `time_shift_pvalue`.
- `orion/substrate/tests/test_causal_geometry_pvalue_calibration.py`: 19 new gate tests.
- `tests/test_causal_geometry_report.py`:
  - The old floor test asserted that the bug still reproduced. It now asserts that the floor is gone at 200 and at 2000 surrogates.
  - Its fixture grew from 50 to 400 points. With 50 points a pair can no longer clear Bonferroni and BH. That is the honest resolution of 50 points.
- `.github/workflows/causal-geometry-tests.yml`: the new CI job.

## Consumers of the engine's edges

What changes when the edge count goes from 0 to about 6:

1. **Bus channel `orion:causal_geometry:snapshot`** -> `orion-sql-writer` -> table `causal_geometry_snapshots`. Rows get a non-empty `edges` list and `insufficient_data=false`. The schema is the same.
2. **Hub `/api/causal-geometry/*` and the Causal Geometry page** (`services/orion-hub/static/js/causal-geometry.js`). The page already renders an edges table and the `insufficient_data` banner, so it shows edges again. `significance` is printed raw, so a value such as `1e-12` appears as written.
3. **`field_topology_plasticity.find_cap_cap_divergence_candidates()`** (HITL weight-patch proposals):
   - It only uses divergence entries with status `both`.
   - An entry is `both` only when an edge endpoint matches a capability channel, such as `pressure` or `reasoning_pressure`.
   - Replay divergence: 17 `designed_only`, 6 `observed_only`, 0 `both`. So today there are **0 new proposals**.
   - Proposals are human-in-the-loop only. Nothing is adopted automatically.
4. **The `orion-field-digester` diffusion learned overlay** (`_load_learned_overlay`). It reads only *adopted* proposals. It is unaffected unless a human adopts one.
5. **Descriptive references only:** the metric-lineage entry `metric://bus_channel/orion-field-digester/orion:causal_geometry:snapshot` (`config/metrics/metric_definitions.lock.json`) and the body-atlas fixtures.

No prompt, reward, attention or curiosity consumer reads these edges. I searched for `causal_geometry`, `causal_geometry_snapshots` and `CausalGeometrySnapshot` across `orion/`, `services/`, `scripts/` and `config/`.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: Python only. `circular_shift_pvalue` -> `time_shift_pvalue`, and `circular_shift_surrogate_rs` -> `time_shift_surrogate_rs`. No other callers exist (grepped).
- Behavior changed:
  - `CausalGeometryEdgeV1.significance` is the Bonferroni-corrected time-shift p.
  - For pairs with n_eff >= 100 it can be as small as 1e-300. For pairs with fewer independent samples it is >= 10/(n_eff+1).
  - `n_surrogates` is now a cap on the number of shifts. At n=2017 there are 1,818 shifts, so the cap does not apply and the seed is unused.
- Compatibility notes: the field type is unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest -s orion/substrate/tests/test_causal_geometry_pvalue_calibration.py      19 passed (44 s)
  surrogates == explicit per-shift overlap Pearson (atol 1e-12); cap draws distinct shifts
  cliff, m=171: n=600  real r=0.581 p=1.2e-06 edges=1 | n=2016 real r=0.547 p=1.8e-07 edges=1
                legacy circular rank p on the same data -> 0 edges, real pair at exactly 10/2001
  raw p, 1000 independent pairs per shape, n=2016   (<0.05 / <0.01 / <1e-3)
    ar1_0.5   0.016 / 0.002 / 0.000      bursty  0.047 / 0.012 / 0.003
    ar1_0.9   0.019 / 0.004 / 0.001      daily   0.000 / 0.000 / 0.000  (n_eff ~7: p >= 1/8 by design)
    ar1_0.98  0.038 / 0.000 / 0.000      regime  0.027 / 0.000 / 0.000
  null family, 19 independent channels (m=171) x 6 seeds, every shape: 0 seeds with any edge
pytest tests/test_causal_geometry_report.py orion/substrate/tests/test_causal_geometry_{fdr_correction,producer,bus_publish}.py \
       orion/substrate/tests/test_field_topology_plasticity.py orion/schemas/tests/test_causal_geometry.py   43 passed
services/orion-field-digester: pytest tests/test_worker_causal_geometry_producer.py                         3 passed
```

## Evals run

```text
Live read-only replay (scratch script, not committed), 168 h window, m=171:
  before 0 edges (14 pairs at the floor); after 6 edges (above)
  72 h: 3 edges; 24 h: 0 edges (fewer independent samples)
  divergence after: designed_only 17, observed_only 6, both 0 -> 0 HITL proposals
```

The calibration tests act as the eval for this change. They measure the false-positive rate and the power on the fixtures, not only the code paths.

## Docker/build/smoke checks

```text
No Docker build: no dependency, env or compose change. numpy only.
```

## Review findings fixed

Round 1 (code-review subagent on 52b924790):

- Finding (blocker): the normal tail is anti-conservative on bursty channels (34 false edges over 6 seeds, m=171).
  - Fix: dropped the normal fit. The new test uses the rank p in the body, a GPD tail past 10 exceedances, a time-shift overlap null with a guard, and an n_eff gate (see "Method choice and why").
  - Evidence: on `bursty`, 0.3% of raw p values are below 1e-3, and the null family has 0 of 6 seeds with an edge.
- Finding (should): the normal fit has almost no power on daily cycles, and the docstring claimed it handled daily cycles.
  - Fix: the body of the null is now the rank p. Pure daily-cycle pairs have low n_eff (about 7), and the docstrings say so. They state what the method handles and what it does not.
  - Evidence: the `daily` null family has 0 edges. The guarded circular variant (357 false edges) is pinned against.
- Finding (should): the docstring of `circular_shift_surrogate_rs` contradicted the code for constant series.
  - Fix: constant series return an empty array (p=1.0), and the docstring says so.
  - Evidence: `test_degenerate_inputs`.
- Finding (should): a length mismatch silently returned p=1.0.
  - Fix: both functions raise `ValueError`.
  - Evidence: `test_degenerate_inputs`.
- Finding (nit): near-zero shifts carry the real effect into the null.
  - Fix: guard = max(12, 5% of n), on a seam-free overlap null.
  - Evidence: the live power/thermal p moved from 0.0016 to 1e-12, and the cliff fixture passes at n=2016.
- Finding (nit): random-walk pairs are liberal.
  - Fix: the n_eff gate keeps their p at or above 1/(n_eff+1).
  - Evidence: `test_low_effective_sample_size_never_extrapolates`.
- Finding (nit): the workflow path filters missed the requirements files, and the push filter missed the tests.
  - Fix: both filters now include them.
- Finding (nit): the tail was only checked down to 0.01.
  - Fix: there is now a 1e-3 bound for each shape, and null-family runs for all six shapes.
- Finding (nit): the Hub prints `significance` raw.
  - Not changed: nothing breaks. Formatting it is a Hub UI change, out of scope here.

ROUND2_PLACEHOLDER

## Restart required

The change takes effect on the next producer cycle after `orion-field-digester` is rebuilt from main. The post-pull rebuild does this, because this repo deploys only from the primary main checkout.

```bash
scripts/safe_docker_build.sh orion-field-digester up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: at m=171, a pair with n_eff below 100 can no longer become an edge, however strong its r is. This covers pure daily-rhythm pairs, slow on/off channels and trending channels.
  - Mitigation: about 10 to 100 independent samples cannot support a smaller p. The docstrings state this rule and tests pin it. Lower `N_EFF_MIN` only after a new calibration run.
- Severity: low
  - Concern: the GPD tail is still an extrapolation, two or more orders of magnitude past the surrogates.
    - It is checked to 1e-3 on six null shapes.
    - It is checked through the full m=171 pipeline on six seeds per shape. That pipeline's BH decision is at about 3e-5 raw.
    - It is not checked point by point at 3e-5.
  - Mitigation: after Bonferroni the six live edges are at p <= 5e-5, and the next pair is about 100 times higher.
- Severity: low
  - Concern: after about 11 days of empty snapshots, edges return on the Hub page and in stored history. The method change causes that step, not the cabinet.
  - Mitigation: this report and the module docstrings date the change (2026-11-10).

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
