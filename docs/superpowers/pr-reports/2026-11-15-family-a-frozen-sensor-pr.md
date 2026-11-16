# Family A stops reading a frozen sensor as calm (#3370)

## Summary

- A sensor stuck at a normal-looking value (say a GPU temperature frozen at 61.0 °C) used to read as calm in Family A for up to ~12 days. A frozen value is forecast perfectly, and nothing dropped the input until it was one value for 85 % of the 14-day window.
- Now every select stores each reading's **flattest normal day**. That is the largest share of a whole day taken by one value other than the reading's rest floor, counting only days at or under the existing 85 % "one value" line.
- It also stores a **rest floor**: a low reached on at least two normal days (PM2.5's 0). Before, the floor was the window minimum, so a probe dead at its −127 error code became its own "floor" after the next select and read as rest. That gap was already on main, from #3371.
- **Issue** (every 5 min): if the last 24 h are flatter than halfway from that ceiling to fully frozen, the bucket is refused with `flat_beyond_training`. Resting at the floor never counts, so PM2.5 sitting at 0 is rest, not frozen.
- **Select** (every 6 h) runs the same check on its window's last 24 h and drops a frozen input, so Family A goes on serving the other readings. The blackout is bounded to at most 6 h.
- Also fixed, found on the way: floor comparisons mixed a rounded mode (17.1235) with an exact floor (17.1234567), so a reading resting at a non-round floor lost its floor exemption.

## Outcome moved

| | before | after |
|---|---|---|
| real readings frozen at their median (live smoke, 4 tried) | never refused: read as calm | refused after **12-14 h** (gpu1 temp 12:40, office temp 12:10, circe gpu0 temp 13:50) |
| a probe dead at −127 (live smoke) | refused after 23 h (left band), then served as "rest" after the next select | refused after **12 h** |
| false refusals, 72 h of live buckets | 0 | **0** (761 of 761 pass) |

The live loop's stored refusals in that window (155) are all the old PM2.5 `one_value` bug from before #3371 deployed (11-15 06:49Z). There have been none since.

Not caught, by design: a sensor frozen **at its floor**. The light sensor frozen at its median of 2 is its floor, and that looks exactly like rest. Select's `one_value` rule still catches it after ~12 days.

## Metric quality gate (for the new gate signal)

1. **Provenance:** `_flat_share` / `_flat_ceiling` / `_frozen` in `app/family_a.py`. They read the same 5-min bucket means Family A already loads (`app/family_a_load.py`).
2. **Independence:** this is a refusal reason, not a new signal fed to any model. It gates the existing joint surprise.
3. **Theory anchor:** a frozen sensor shows a one-value share no normal training day reaches, while a quiet day of a working sensor stays near its usual flattest day. The floor exclusion follows the existing event-like rule: a series' floor is its rest state.
4. **Live sanity:** 72 h with no false refusals; the closest normal day was circe.gpu0_temp_c at 0.507 (reviewer's 16-day replay, before the margin). Frozen readings trip in 12-14 h.
5. **Existing mechanism:** this extends the existing hygiene gate (`_short_window_reasons`, `screen_series`). No new mechanism.
6. **Reversibility:** set `flat_margin` toward 1.0 in config (its own PR, per the config rule), or revert. No schema change. The ceiling lives in the run manifest, and old manifests without it mean "no check".

## Current architecture

The hygiene gate refused a bucket for: absent, stale, left the trained band, renamed. Degeneracy was decided only at select, over 14 days.

## Architecture touched

orion-prediction only (Family A gate and its eval) and `config/prediction/family_a.v1.yaml`.

## Files changed

- `services/orion-prediction/app/family_a.py`:
  - `_flat_share`, `_flat_ceiling`, `_frozen`, `flat_limit`
  - the `max_day_mode_share` stat
  - the select-time `flat_beyond_training` check
  - the issue-time check (`flat_ceilings`)
  - the rounded-floor comparison fix
- `services/orion-prediction/app/family_a_live.py`: `Served.flat_ceilings` is taken from the selection and passed to the gate; the served floors are now the rest floor, not the window minimum.
- `config/prediction/family_a.v1.yaml`: new required hygiene key `flat_margin: 0.5`.
- `services/orion-prediction/evals/run_family_a_gate_replay_eval.py`:
  - recomputes ceilings for manifests from before this fix
  - `--freeze <input>` / `--freeze-at <value>` live smoke
  - grid-aligned load start
  - `flat_beyond_training` counts as a false refusal on healthy data
- `services/orion-prediction/tests/test_family_a.py`:
  - frozen vs floor rest vs quiet day
  - the dark day does not silence the check
  - order independence
  - a frozen first day
  - the non-round floor
  - the margin
- `services/orion-prediction/tests/test_family_a_live.py`:
  - frozen after select → buckets refused → the next select drops the input → the rest is served
  - the PM2.5-at-rest regression now rests at its floor. Its old "flat at 20.0" was a frozen sensor by this ticket's own definition.
- `services/orion-prediction/README.md`: Family A issue gate.

## Schema / bus / API changes

- None. Run manifests gain `stats.max_day_mode_share` per input (inside existing jsonb).

## Env/config changes

- No env keys. Config key added: `hygiene.flat_margin` (required by `parse_config`; checked-in config only).

## Tests run

```text
pytest services/orion-prediction/tests -q (prediction image) -> 500 passed
mutation checks: floor values counted as flat, frozen days feeding the ceiling, no issue check, no select check,
  live not passing ceilings, 0.85 cut -> 0.99, share over non-floor only, rest floor from all days, rest floor =
  lowest day, live floor = window minimum, no ceiling fallback, new lows not counted -> each fails the suite
```

## Evals run

```text
evals/run_family_a_gate_replay_eval.py --hours 72 --compare-module <main's family_a.py>
  current: pass=761 refused=0; compare (main): pass=761 refused=0
evals/run_family_a_gate_replay_eval.py --hours 24 --freeze <input>  (4 real inputs)
  current: refused after 12:05-13:50 for 3 non-floor inputs; main: never. cabinet_light_raw frozen at its floor: neither.
--freeze athena.cabinet_temp_c --freeze-at -127: current after 12:05; main after 23:20 (left band), and as rest after a reselect
```

## Docker/build/smoke checks

```text
Tests and evals ran in the existing orion-prediction image against the live DB, read-only. No build needed (no deps changed).
```

## Review findings fixed

Round 1 (orion-repo-agent: mutations, 3 probes, and a 16-day live replay with selects every 6 h; no false refusals found):

- Finding: a whole day resting at the floor set the ceiling to 1.0 and silenced the check for exactly the readings that rest at a floor.
  - Fix: flatness counts only values above the floor, so a dark day reads 0.
  - Evidence: `test_floor_rest_is_never_frozen_and_a_dark_day_does_not_switch_the_check_off`.
- Finding: the ceiling depended on day order, so a real steady day after calmer days was thrown out as frozen and could black out buckets.
  - Fix: the ceiling is order-independent, the max over days at or under `max_mode_share`.
  - Evidence: `test_the_ceiling_does_not_depend_on_day_order_and_ignores_a_frozen_first_day`.
- Finding: a frozen first day silenced the check.
  - Fix: the same rule; frozen days never count, wherever they fall.
  - Evidence: same test.
- Finding: two mutations around the floor clause survived.
  - Fix: the clause is gone (floor values are excluded from the share itself); the new tests kill "floor counts as flat".
  - Evidence: mutation run above.
- Finding (low): the per-seed backtest screens only training rows, so its "last 24 h" can differ from select's.
  - Fix: none needed. Backtest targets could already differ from served targets, and select's screen decides what is served.

Round 2 (re-review of the redesign: probes + mutations; it confirmed findings 1-3 closed):

- Finding (high, also on main since #3371): a probe stuck at a new low (−127) became the window minimum, so after the next select it was "resting at its floor" and was served as calm.
  - Fix: a rest floor from normal days, the second-lowest normal-day minimum, so one dying day can't set it. Values below it count as flat. Live serves the rest floor.
  - Evidence: `test_a_probe_dead_at_a_new_low_is_never_its_own_rest_floor`, `test_a_probe_that_died_to_a_new_low_before_select_is_not_served_as_rest`, and the −127 live smoke.
- Finding: two mutations survived (0.85 cut → 0.99; share measured over non-floor values only).
  - Fix: tests pin both: `test_a_day_over_the_one_value_line_never_lifts_the_ceiling`, `test_flat_share_is_over_all_values_so_half_dark_half_stuck_is_one_half`. Over all values is chosen on purpose: conservative, no extra constant.
- Finding (low): no normal day meant no ceiling, so the check was off.
  - Fix: fall back to `max_mode_share`. `test_whole_days_with_none_normal_still_get_a_ceiling`.
- Finding (low, by design): a real steady last day on a continuous sensor is refused for one select cycle.
  - Fix: none; kept, and recorded in Risks.

## Restart required

```text
No manual restart. After merge, the post-pull rebuild rebuilds orion-prediction; the next select (at start) stores the ceilings.
```

## Risks / concerns

- Severity: low. Concern: a working sensor that is legitimately flatter than any normal training day (a fan pinned at one speed all day) refuses buckets for up to 6 h, then is dropped until its days look normal again. Mitigation: the margin; 0 false refusals in 72 h live and 16 days replayed.
- Severity: low. Concern: a sensor frozen exactly at its rest floor still reads as rest until select's `one_value` rule (~12 days). Mitigation: documented; it cannot be told apart from rest within a day.
- Severity: low. Concern: when a sensor dies part-way through a day, that day can become its flattest normal day, which slows the catch to ~15 h (live test). Mitigation: still under a day, the same order as real freezes.
- Issues: Fixes #3370. Opened: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3492

🤖 Generated with [Claude Code](https://claude.com/claude-code)
