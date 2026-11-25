## Summary

- The field demon now has a trained tensor network beside its linear model. It is a Born machine
  on a matrix-product state (a chain of small tensors), the structure the heartbeat runs today, but
  trained on Orion's data instead of fixed random gates. (`orion/substrate/born_machine.py`, new.)
- `demon.NetworkForecaster` forecasts each organ's level (low, usual, high) 5-60 minutes ahead and
  plugs into the existing readouts (`skill_curve`, `bootstrap_horizon`, `cut_cost_curve`,
  `cut_cost_vs_null`) unchanged. It scores by log-loss in bits against clock, persistence and the
  linear model. On network errors, "best baseline" includes the linear model.
- The cut cost of every way to split the organs (all partitions up to 8 organs) comes from the one
  trained chain, with no refit per split, against the existing time-shift null. A closed-form
  Gaussian cut from the linear model is scored on the same splits, and the two weakest cuts are
  compared.
- Known-answer tests for every acceptance check in the spec, plus the mutation check.
- `scripts/demon_network_replay.py`: a read-only PREVIEW on the 5 heartbeat seats. Result: on
  ~39 hours of data the network does not beat the simple guesses and overfits. The data cannot
  support a claim either way.

## Outcome moved

Step 2 of the rev 3 design (PR #3837) exists and is proven on synthetic systems with known
answers: the network matches linear on linear data, beats it clearly on a threshold switch, finds
independent blocks as the weakest cut, and reads about 0 on shifted data. The first live read
says what the next step needs: more rows (#3886), not a bigger model.

## Current architecture

- `orion/substrate/demon.py`: a linear VAR(1) forecaster scored by squared error on continuous
  residuals, with a rolling-refit protocol, horizon bootstrap, cut cost and time-shift null.
- The heartbeat's tensor network uses fixed random gates; nothing is trained (11-08 replay).

## Architecture touched

Pure library code and a read-only script. No service, bus, schema, env or Docker change.

## Files changed

- `orion/substrate/born_machine.py` (new): MPS Born machine. Two-site maximum-likelihood sweeps;
  bond dimension from {2, 4, 8, 16} and sweep count by held-out log-likelihood; exact
  log-likelihood, marginals and conditionals; per-bond von Neumann entanglement; classical mutual
  information (exact, or Monte Carlo with exact terms); exact sampling.
- `orion/substrate/demon.py`: `NetworkForecaster`, `fit_network`, `network_rolling_errors`,
  `skill_spread`, `set_partitions`, `partition_normalization`, `linear_cut_bits`,
  `held_out_lattice`, `lattice_vs_null`, `mip_index`, `cut_information`. `RollingErrors` gains
  two optional fields (`linear`, `lags`, default None); its `baseline` adds linear to the best-of
  only when `linear` is set. `bootstrap_horizon` does the same, and refuses lags that are not
  1..K. `cut_cost_vs_null` gains an optional `errors_fn` (default: the existing
  `rolling_errors`). Var1 results are unchanged (all 32 existing tests pass).
- `orion/substrate/tests/test_demon_network.py` (new): 21 known-answer tests.
- `.github/workflows/field-demon-tests.yml`: runs the new tests; single-threaded BLAS.
- `scripts/demon_network_replay.py` (new): the PREVIEW replay.
- `docs/superpowers/evidence/2026-11-25-demon-network-preview/`: `results.json` + `report.md`.

## Design choices worth checking

- **Same input for both models.** The linear baseline is fitted on, and fed, the same 3-level
  "now" the network sees (each level coded as its training mean). Fed the continuous values, the
  linear model beat the network on true VAR(1) data by 5% only because it knew more about "now".
- **Direct for both models.** The network has one chain per lag, so the linear baseline is the
  direct lag-k Gaussian (`lag_k_model`), not the lag-1 model iterated k times.
- **Null of the same length.** A shifted null copy keeps 75% of the rows. The observed lattice is
  scored on the same number of rows with the same split, so both carry the same fitting noise.
- **Weakest cut against weakest cuts.** `p_real_mip` compares the observed weakest cut with each
  null copy's own weakest cut, not with the null at one fixed partition.
- **Per-organ scoring.** Log-loss is summed over organs' own forecasts, the log-loss twin of the
  demon's summed squared error. A joint score would charge every cut for noise the parts share at
  the same instant, which no part's past can carry.
- **Linear cut is not Oizumi's Φ_G.** Φ_G needs an iterative fit. `linear_cut_bits` is the closed
  form "each part forecasts from its own past" member of the same family, on the same per-organ
  score, so the two weakest cuts can be compared.
- **MIP normalization:** Balduzzi-Tononi, (parts - 1) x smallest part's maximum entropy.

## Schema / bus / API changes

- Added: none (library functions only).
- Removed: none. Renamed: none.
- Behavior changed: none for existing callers.
- Compatibility notes: `RollingErrors` new fields have defaults; `cut_cost_vs_null` new keyword
  has a default.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
OPENBLAS_NUM_THREADS=1 pytest -q orion/substrate/tests/test_demon.py \
  orion/substrate/tests/test_demon_network.py orion/substrate/tests/test_whole_body_structure.py
OPENBLAS_NUM_THREADS=1 pytest -q orion/substrate/tests/test_demon.py orion/substrate/tests/test_demon_network.py
53 passed in 11.4 s   (32 existing demon tests unchanged + 21 new)
```

Known-answer results (fixed seeds):

| Check | Result |
|---|---|
| VAR(1) data: network within spread of linear | skill -0.14%, 10th-90th -0.5% to +0.2% (seeds 1-6: -0.8% to +0.2%) |
| Threshold switch: network beats linear | skill +11.6%, 10th-90th +9.7% to +13.3%, P = 1.00 (seeds 2-7: +11.6% to +15.7%) |
| Independent blocks: split costs ~0 and is the MIP | split -0.028 bits; next cheapest cut +0.51 bits; MIP for both models |
| Conditional marginals = brute-force enumeration | equal to 1e-10 on a 6-site chain, 4 site sets |
| Entanglement >= exact mutual information | holds at every cut, 3 random chains + a trained chain |
| Shifted data: cut cost ~0 | coupled pair +0.60 bits; same pair shifted -0.012 bits |
| **Mutation (spec): network forecast swapped for linear** | **threshold test fails**: skill exactly 0.0, P = 0 |
| **Mutation (review): chain capped at bond dimension 1** | **threshold test fails**: skill -12.7% (seeds 2-7: -12.7% to -21.9%) |

## Evals run

```text
scripts/demon_network_replay.py --out docs/superpowers/evidence/2026-11-25-demon-network-preview
(live, read-only, PREVIEW; ~70 s)
```

Plain-words result (full tables in the evidence report):

- 467 rows (38.9 h) scored. A 3 h ledger gap on 11-24 cut the rest. The hygiene gate passed.
- **Forecast (last 12 h):** the network does not beat linear at any lag. At 5 minutes its skill
  over linear is -0.6% (10th-90th -4.0% to +1.7%), P(network better) 0.30. Gate (P >= 0.9) not
  met. Past 15 minutes the network is worse than the simple guesses.
- **The network's result flips with the window.** On the lattice test window (105 night rows) it
  beats linear at 5 minutes by 0.41 bits. On the forecast window it loses by 0.04 bits.
- **Weakest cut:** the network's weakest cut costs nothing at every lag, so it measures no
  integration here. The two models never agree on the weakest cut.
- **Pipeline block (biometrics against the rest):** at 5 minutes, both models say the split costs
  foresight, above all 20 null copies: network +0.068 bits (null -0.051 ± 0.062), linear +0.135
  (null +0.030). This is the only lead, and it rests on one night.
- **Entanglement** is above mutual information at every cut, about twice it. It describes a chain
  fit on 234 rows, not Orion.
- **What the data cannot support:** any network-vs-linear claim, any weakest-cut or integration
  claim, any entanglement reading. It needs about 20x more rows (#3886).

## Docker/build/smoke checks

```text
No Docker change. The replay ran against live Postgres read-only (docker exec psql, SELECT only).
```

## Review findings fixed

Code review ran in a subagent (read-only). No blockers. It verified the gradient, canonical form,
contractions, entanglement, mutual information, sampling and the absence of leakage. Findings:

- Finding: the weakest-cut null test was post-selection (observed minimum against the null at one
  fixed partition).
  - Fix: `LatticeVsNull.p_real_mip` minimises each null copy too.
  - Evidence: `test_lattice_vs_null_marks_a_real_cut_real` checks it.
- Finding: the observed lattice and the null copies had different lengths (null keeps 75%).
  - Fix: the observed lattice is scored on the same length and split.
  - Evidence: the test asserts 374 scored origins out of 1,125 rows.
- Finding: "best baseline" left out linear.
  - Fix: `RollingErrors.baseline` and `bootstrap_horizon` include linear when it is set. Var1 is
    unchanged.
  - Evidence: the 32 existing tests pass.
- Finding: the linear baseline was iterated while the network is direct per lag.
  - Fix: the linear baseline is now the direct lag-k model.
- Finding: the spec's mutation test is tautological (skill exactly 0).
  - Fix: kept it, and added a second mutation (chain capped at bond dimension 1). The threshold
    test fails with skill -12.7%.
- Finding: `var1_continuous` / `linear_continuous_probs` were unused.
  - Fix: deleted.
- Finding: the replay ignored a hygiene-gate refusal.
  - Fix: it now exits with code 2 and scores nothing.
- Nits fixed:
  - Negative cut costs count as 0 in the weakest-cut choice (`mip_index`, tested).
  - The two linear lattice numbers are documented as different quantities.
  - The gradient step size is capped.
  - D selection leaves a gap of k rows between its fit and validation rows.
  - The Monte Carlo mutual information is seeded from `--seed`.
  - The replay reuses the observed fit instead of fitting twice.
  - The pipeline lookup uses sets.
  - psql runs with `ON_ERROR_STOP` in a read-only transaction and fails clearly on zero rows.
  - `bootstrap_horizon` refuses lags that are not 1..K.
- Not changed: the clock baseline uses the training window's level shares, not per-hour shares.
  The input is already a time-of-day residual, so these are the usual levels for the time of
  day. This is documented on the field.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: the preview reads per-service seats over 39 h; it can mislead if quoted
  as a result. Mitigation: labelled PREVIEW everywhere; the report says what it cannot support.
- Severity: low. Concern: replay runtime grows with lags x bond dimensions x null draws (~70 s
  now). At 8 organs and 4 weeks, the 4,140-partition search with 20 nulls will take minutes.
- Severity: low. Concern: the network's held-out result depends on which hours are scored (it
  flips between the two windows). Mitigation: the report says so; #3886 for more rows.
  Mitigation: single-threaded BLAS (multithreaded was 30x more CPU for the same wall time).
- Issues opened: #3886 (keep a 5-minute seat rollup; grammar_events holds ~3 days).
- Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3888

🤖 Generated with [Claude Code](https://claude.com/claude-code)
