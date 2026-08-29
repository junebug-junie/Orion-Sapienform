# Quota budget reader + the finding that killed its denominator — PR report

**Status:** DONE_WITH_CONCERNS (see Concerns — the feature works and its denominator is refuted)

## Summary

- `orion/autonomy/quota_budget.py` — read-only rolling-window spend budget, mirroring `budget.py`'s contract. Wired into no allocator.
- `scripts/report_quota_budget.py` — replays real spend and reports what a given allowance would have stopped.
- Ran the gate immediately against 9 days of live history instead of waiting a week.
- **Found the denominator does not work**, using ground truth the spec had not thought to look for.
- 33 tests, every expected value hand-computed; 12 mutations applied to the real files, all caught.

## Outcome moved

A decision, not a runtime change: **do not wire the dollar budget into the allocator.** Reached in hours rather than after a week of observation, and reached before anything depended on it.

## The finding

The design spec (PR #1908) passed metric-gate items 1–6 and left one open: whether `cost_usd` on a subscription token tracks the unit the subscription actually meters. Ground truth turned out to be sitting in the transcripts — 66 real rate-limit events, 15 inside the ledger's coverage.

```
limit fired at 5h trailing spend of:   $85.39 ... $289.76     (3.4x spread)
largest 5h window ever observed:       $419.05  -- did not trip it
```

**No threshold separates limited from not-limited.** Likely cause: the subscription meters per-session windows; `dev_economics_ledger_log` is machine-wide across concurrent sessions. That is the right denominator for the contested-resource argument and the wrong one for predicting a per-session ceiling.

How close this came to shipping anyway: at $150/5h the replay refuses 170 asks on observed spend, which reads as a working budget. It is a knob, not a finding. Only the ground-truth check separated "binds" from "binds because I set it low enough to."

## Files changed

- `orion/autonomy/quota_budget.py`: new. Sums per-tick deltas; unknown ≠ zero; fails closed on unknown; discloses undercounts; `None` for unconfigured kept distinct from exhausted; `mode` hard-coded advisory with no enforcing option.
- `scripts/report_quota_budget.py`: new. Fixed clock grid replay, warm-up excluded, no verdict rendered.
- `tests/test_quota_budget.py`, `tests/test_report_quota_budget.py`: new.
- `docs/superpowers/specs/2026-08-27-quota-window-calibration-finding.md`: the measurement.

## Schema / bus / API / env changes

None. No schema, no channel, no env key, no config. Nothing to sync, nothing to deploy.

## Tests run

```text
pytest tests/test_quota_budget.py tests/test_report_quota_budget.py \
       tests/test_motor_budget.py tests/test_motor_allocator.py -q
78 passed in 0.67s

python scripts/check_scripts_dir_no_stdlib_shadow.py     -> clean
```

Note: CLAUDE.md §17 names `check_env_template_parity.py` / `check_schema_registry.py` / `check_bus_channels.py`. **Those scripts do not exist in this repo under those names** — §17 is illustrative. None would apply here regardless (no env, schema, or channel change).

## Evals run

```text
Mutation testing against the real files (not synthetic copies), 12 mutations:
  round 1 (5): max-min instead of sum; would_refuse not failing closed;
               fraction_remaining full-tank on unobserved; is_floor silenced;
               unconfigured collapsing to a zero ceiling            -> 5/5 CAUGHT
  round 2 (7): NaN summed; negative netted; quota_state validation dropped;
               spend_is_floor silenced; tick-driven grid restored;
               warm-up re-included; verdict on zero decision points -> 7/7 CAUGHT
```

No eval harness exists for `orion/autonomy/`; mutation testing stands in, and is the stronger check for this patch.

## Review findings fixed

- **Finding (must-fix): NaN spend defeated the fail-closed guarantee.** `total_estimated_cost_usd` is `double precision`, so Postgres accepts `'NaN'`. Because `nan > 0.0` is False, `max(0.0, nan)` returns 0.0 — the display read "0.0% remaining" while `would_refuse` approved a $1B ask. Silent inversion of the module's purpose.
  - **Fix:** non-finite cost is folded into unpriced (it *is* "we don't know"), routing it through `is_floor`. Negatives raise. `quota_state` also validates a hand-built `WindowSpend`, matching `budget.py`.
  - **Evidence:** `test_nan_cost_is_unknown_not_a_summed_number` asserts `would_refuse(1e9) is True`; mutation reverting the guard fails 2 tests.

- **Finding (must-fix): the script recommended the action this branch's own doc forbids.** It printed `GATE: PASSED ... Wiring it in is defensible` at $150, and on failure advised `lower --allowance-usd ... until it bites` — a literal instruction to tune until it refuses.
  - **Fix:** no verdict is rendered; counts only, with the refutation printed. Docstrings say *not calibratable*, not *UNVERIFIED*.
  - **Evidence:** `test_a_real_replay_renders_counts_but_still_no_verdict` asserts `"PASSED" not in out` and `"until it bites" not in out`.

- **Finding (should-fix): `QuotaState` laundered `is_floor`.** `remaining_usd` / `would_refuse` were derived from a known-incomplete total, unmarked.
  - **Fix:** `QuotaState.spend_is_floor` + `[FLOOR]` markers on derived output lines.

- **Finding (should-fix): `--replay-days N` replayed N+1 days with a biased head.** Warm-up padding was loaded and never excluded, so the first window of points had truncated windows — under-counting spend, biasing toward "never refuses".
  - **Fix:** grid starts one full window after the first tick.
  - **Evidence:** `test_clock_grid_excludes_the_warm_up_window` (6 points, hand-counted).

- **Finding (should-fix): `refused_unknown` was structurally dead.** A tick-driven grid can only sample moments where a tick exists, so it could never land in a producer outage and always printed 0.
  - **Fix:** fixed clock grid (also fixes the warm-up bias and the O(n²) rescan).
  - **Evidence:** live run now finds **36 real producer gaps** across 9 days; `test_a_producer_gap_is_sampled_and_fails_closed` pins 4 on a hand-computed fixture.

- **Finding (should-fix): confident `GATE: FAILED` after replaying zero decision points.**
  - **Fix:** `NO DECISION POINTS` branch. **Evidence:** `test_no_decision_points_renders_no_verdict`.

- **Finding (should-fix): script had no tests.** **Fix:** `replay()` split from `print_replay()`, 10 tests, no DB needed.

- **Nits fixed:** `$0.00` labelled "observed spend" on an unobserved window; leaked socket (`contextlib.closing`); `pace`/`projected_window_usd` degeneracy under a trailing window now documented and never printed.

- **Review claims verified independently:** all 18 original fixtures recomputed by the reviewer and confirmed correct; no place treats ledger rows as cumulative.

## Restart required

```text
No restart required.
```

## Concerns

- **Severity: high (to the design, not the code).** The dollar denominator is refuted. `ORION_QUOTA_ALLOWANCE_USD_PER_WINDOW` should not be set and the reader must not be presented as a quota gauge. PR #1908's recommended patch is superseded by the calibration finding.
- **Severity: low.** `rate_limit_error` is a mixed population — some events are plausibly subscription usage limits, others may be transient API-side throttling. This does not change the conclusion (a proxy that cannot separate the two is not a proxy) but the 3.4x spread is an upper bound on how bad the axis is, not a precise measurement.
- **Follow-up:** add `was_rate_limited_recently()` to the dev-economics ingest and re-run the replay against that signal. It needs no calibrated denominator and cannot be tuned into non-binding. It gives up anticipation, which is a real loss.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1910
