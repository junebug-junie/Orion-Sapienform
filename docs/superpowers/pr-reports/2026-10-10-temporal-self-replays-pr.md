## Summary

This PR adds the three read-only checks that Temporal Self rev 4 ([PR #2369](https://github.com/junebug-junie/Orion-Sapienform/pull/2369)) asks for before any regulation code is built. It extends the scripts from #2561 and does not duplicate them. Nothing running changes.

- **R1a, attention hogging:** for each target, it measures how long the target held more than half of every rolling 30-minute window. It also reports the run-length and return-window distributions for the interoception lane of patch 1.
- **R2a, sleep pressure:** it re-runs the dream service's own pressure code at every 10-minute check over 14 days, using the sources as they are stored now. It records how long after each real sleep pressure first crossed each candidate threshold. It then simulates the sleep schedule each threshold would have produced, once with the current all-chat idle rule and once with the Juniper-only rule.
- **R3a, arousal:** it adds hour-by-hour labels as CSV, counts per label, seconds per day, the spec's "no level at 0% or 100%" bar, strict strain exits per day, and a check for an input that never changes.
- Missing inputs are reported as `unknown`, never as idle, calm or 0. Tests cover empty tables, left-censored runs, missing chat, missing GPU state and recorder downtime.

## Outcome moved

The three regulation questions now have measured answers, not guesses. The dream threshold question has a validated replay: it reproduced all 114 saved live checks exactly.

### Live verdicts (all times UTC, read-only exports taken 2026-10-10 ~01:50)

**R1a: does one target hog attention?** Yes in the current data, but the result cannot yet decide whether to build R1. Over 21.0 recorded hours (10-09 04:44 to 10-10 01:45), `node:substrate.bus_synaptic` held more than half of the rolling 30-minute window for four separate stretches: 3.0 h, 2.5 h, 2.3 h and 2.3 h, about 10.2 h in total. Its peak share was 99.4% to 100%. It won 57.5% of all recorded time. No other target held a majority for longer than 50 minutes (biometrics). R1a is supposed to judge the scoring change in #2528 after it has run for 7 days, but #2528 is still an open spec PR, so 0 days of that instrument exist. The verdict is `BASELINE_ONLY`. The data is also thin: only 21 hours of focus runs.

Supporting numbers:
- There were 415 completed runs. The median run was 28 ticks (59 s), the 90th percentile 227 ticks (7.9 min), and the longest 1,024 ticks (36 min).
- 90.1% of runs reach the 3-tick minimum streak.
- No run was longer than 4 h, so there is no stuck reading.
- Return-window probe (the patch-1 rule that merges same-target runs within `R` minutes into one arc with returns):

  | R | arcs | mean returns per arc |
  |---|---|---|
  | 1 min | 264 | 0.57 |
  | 5 min | 142 | 1.92 |
  | 10 min | 87 | 3.77 |
  | 30 min | 38 | 9.92 |

**R2a: does sleep pressure really fall below the threshold and rise across it, or does the clock drive sleep?**

- **Before #2557 the clock drove sleep.** That covers 52 stored cycles, all on the old formula. Pressure at sleep time was 12.9 to 129, always above the threshold of 3. 45 of 51 gaps fell within one check of the 6-hour minimum, and the median gap was 6.001 h.
- **After #2557 pressure really behaves like a drive, but only one full cycle is on record:**
  - 20 minutes after the 10-09 06:33 sleep, the saved checks read 0.0.
  - Pressure stayed below 3 for 18 h. 45 checks had the timer clear and Orion idle, but pressure was below the threshold, so each said "not yet".
  - Pressure rose 2.324 → 3.324 across the threshold at 00:41.
  - Orion slept at 01:01, 18.5 h after the previous sleep and 12.5 h later than the clock allowed.
  - Pressure fell to 0.0 by 01:12.
  - No rise after that fall has been observed yet.
- **The replay shows the threshold of 3 is not starving sleep.** Replaying the new formula over all 14 days, with the 6 h minimum, 45 min idle and 48 h backstop, gives:
  - 24 sleeps instead of the 54 that actually happened;
  - a median gap of 11.2 h and a longest gap of 40.7 h;
  - 0 sleeps forced by the 48 h backstop;
  - 20 of 23 sleeps (87%) later than the clock would allow;
  - 907 checks where the timer was clear and Orion was idle, but pressure held sleep back.

  With the Juniper-only idle rule: 24 sleeps, a median gap of 13.0 h, and 16 of 23 (70%) later than the clock.

  Caveat: the replay matches the saved checks exactly, but those checks cover only the 2 windows after #2557. The 52 earlier windows are an unvalidated counterfactual.

  The spec's bar is "at least one sleep in four fires later than the clock would allow". It is met at threshold 3, and at every threshold from 1 to 21. At 13 and above, the 48 h backstop takes over (5 or more of the 8 sleeps), which is starvation.
- **Missing question 13 does not apply.** A threshold that can say "not yet" exists.

**R3a: what would the engaged / idle / strained / unknown labels have been?**

- **Over 14 days the strict replay is mostly `unknown`:** 312.6 h of 336 h. Saved GPU pool snapshots only start at 10-09 06:46, and missing inputs are counted as unknown, never as idle.
- **In the 18 hours where every input is present** (10-09 07:00 to 10-10 01:00): idle 990 min, engaged 90 min, strained 0 min, unknown 0.1 min. That is 16 idle hours and 2 engaged hours.
- **Strained appears only from cabinet heat:** 5.2 h on 10-08, mostly 14:00 to 19:00, at a time when GPU state was not saved.
- **The spec's "no level at 0% or 100%" bar passes on paper** (engaged 6.4%, idle 70.7%, strained 23.0% of the 23.4 known hours). That pass depends on the heat-only strain from a day whose GPU input is missing. On fully covered time, strained is 0%.
- **The GPU-backlog strain input (S2) cannot fire as specified.** `backlog_depth` is empty in all 11,872 saved snapshots. `backlogged` is a transient lease status, about 10 s before `queued`, so "non-zero for 5 minutes" never happens. Real GPU contention shows up in `queue_depth` instead: for example, 14 agent jobs were queued at 10-09 10:00.
- **Strict strain exits:** 1 per day at most, well under the review threshold of 12.
- **The lease-event reconstruction is unreliable.** It reads 18.1 h of strain on 10-09, while the saved snapshots read 0.17 h. One lease emitted `backlogged` at 03:57 and no exit event until 22:00.
- **Counting every chat row as Juniper's raises engaged time** from 5.0% to 16.5% in the provisional lane.

### Spec drift (spec stale against main)

- **R2 pressure ceiling.** The spec says candidates are capped at 50 per source with a weight of at most 1. On main, the novelty formula (#2557) counts only new distinct keys, uses `KEYS_PER_SOURCE = 5000`, and has no ceiling of 50 (`services/orion-dream/app/cycle.py`).
- **R2 clock.** The spec says "the six-hour minimum decides". Main also has a 48 h `overdue` backstop (`cycle.py:overdue`), which this replay models.
- **R3 S2 input.** The spec cites "9 `backlogged` events in 7 days" as evidence that S2 is live. The saved snapshot field is degenerate: 0 of 11,872 snapshots are non-zero. S2 fails step 4 of the metric gate until it reads `queue_depth`, or a sustained backlog that actually persists.
- **R1a precondition.** #2528 is still open, not live.
- **Focus run rate.** The spec expects about 2,900 `field_dominance_run` rows a day with a median run of 10 ticks. Live there are 415 rows in 21 h with a median of 28 ticks, so the 20-second alternation is gone.
- **Situation history.** Only `orion:situation:latest` exists in Redis, with no history, so R3a cannot use it.
- **Dream idle gate.** It still counts all chat rows (`IDLE_MINUTES_SQL`). The spec's repair 1 has not landed. The replay reports both variants.

## Current architecture

- `field_dominance_run` (from c8e667149) is written by `services/orion-attention-runtime/app/dominance_runs.py`. `replay_focus_runs.py` (#2561) reported exposure and rolling shares but gave no hogging verdict.
- The dream service computes novelty pressure in `services/orion-dream/app/replay.py:compute_pressure`. Rows are loaded by `app/cycle_store.py:SOURCE_QUERIES`, and the gates live in `app/cycle.py:run_cycle_once`. Every check is saved to `dream_pressure_observation` (since 10-09 06:53). `measure_dream_pressure_crossings.py` read the saved cycles and checks but could not replay the past.
- `measure_arousal_replay.py` (#2561) already classified the levels at 5-second resolution. It had no per-day or per-label summary, no acceptance bar, and no strict exit count.

## Architecture touched

Offline analysis only. There are no bus, Redis, HTTP or model calls, and no database writes. The scripts read JSONL exports, and `--print-sql` prints a `BEGIN ... READ ONLY ... ROLLBACK` SELECT transaction.

R2a loads `services/orion-dream/app/replay.py` by file path and reuses its builders, `row_key` and `compute_pressure`. It does not load service settings, does not connect to the database, and does not import the service package.

The source export is text-free: it contains ids, numbers, md5 keys and booleans only. Chat export is timestamps plus a Juniper flag.

## Files changed

- `services/orion-attention-runtime/evals/replay_focus_runs.py`: `hogging_stretches`, `merged_arcs`, `r1a_report`, and the `--r1a` and `--instrument-since` flags.
- `scripts/analysis/measure_dream_pressure_crossings.py`: the text-free source export (`--with-sources`), `prepare_items`, `pressure_at`, and `replay_novelty` (`--replay`, `--formula-boundary`). Upward threshold crossings are added to the saved-check history.
- `scripts/analysis/measure_arousal_replay.py`: `summarize`, `hour_label`, `write_hourly_csv` (`--hourly-csv`), strict strain exits, and a count of non-zero backlog snapshots.
- `scripts/analysis/tests/test_regulation_measurements.py`: 21 new tests.
- `docs/superpowers/evidence/2026-10-10-temporal-self-replays/`: the live outputs and input hashes.
- This report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: offline reports only. Existing CLI modes are unchanged.
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable, because no templates changed
- Skipped keys requiring operator action: none

## Tests run

```text
/tmp/orion-focus-runs-venv/bin/python -m pytest -q \
  scripts/analysis/tests/test_regulation_measurements.py \
  services/orion-attention-runtime/evals/test_focus_run_replay.py
39 passed
git diff --check: clean
```

## Evals run

All three scripts ran against live Postgres exports. The findings above are their output.

```bash
python services/orion-attention-runtime/evals/replay_focus_runs.py --print-sql --start 2026-10-09T04:00:00+00:00 --end 2026-10-10T01:50:00+00:00
python services/orion-attention-runtime/evals/replay_focus_runs.py focus.jsonl --r1a --start ... --end ...
python scripts/analysis/measure_dream_pressure_crossings.py --print-sql --with-checks --with-sources --start 2026-09-26T01:50:00+00:00 --end 2026-10-10T01:50:00+00:00
python scripts/analysis/measure_dream_pressure_crossings.py dream.jsonl --replay --formula-boundary 2026-10-09T06:33:24+00:00 --start ... --end ...
python scripts/analysis/measure_arousal_replay.py --print-sql --gpu-host circe --start 2026-09-26T01:00:00+00:00 --end 2026-10-10T01:00:00+00:00
python scripts/analysis/measure_arousal_replay.py arousal.jsonl --start ... --end ... --hourly-csv arousal-hourly.csv
# each export: docker exec -i orion-athena-sql-db psql -XqAt -v ON_ERROR_STOP=1 -U postgres -d conjourney < X.sql > X.jsonl
```

Validating R2a against the service: 114 saved live checks were compared, 114 matched exactly, and the largest difference was 0.0. 40 of them were nonzero (up to 3.32), all in the 2 windows after #2557.

`--formula-boundary` is set to the first sleep that recorded `new_counts` (10-09 06:33:24). #2557 merged at 03:56. The dream container's current start time (07:01) is a later redeploy for #2565.

## Docker/build/smoke checks

```text
Only read-only `docker exec ... psql` exports were run. No builds, restarts or deploys.
```

## Review findings fixed

A code-review subagent reviewed `feat/temporal-self-replays` at f73c0fb59.

- Finding: R1a counted hogs from before `instrument_since`, so a hog that happened only before the instrument changed could return `BUILD_R1`.
  - Fix: the decision now uses only the judged window (`judged_hog_stretches`). Hogs from before the change are reported as baseline only.
  - Evidence: `test_r1a_hog_before_the_instrument_does_not_decide_and_thin_coverage_is_not_calm`.
- Finding: R1a returned `DO_NOT_BUILD_R1` from almost no data (one 6-minute run over 8 days). It also returned no verdict when every row fell outside the window.
  - Fix: a new `INSUFFICIENT_COVERAGE` verdict when recorded runs cover less than 80% of the judged window. The `NO_DATA` check now keys on runs inside the window. A caveat says a run still open at export is invisible.
  - Evidence: the sparse and outside-window cases in the R1a tests.
- Finding: the R2a mirror test did not pin which row the service keeps per key, the open window bounds, or `KEYS_PER_SOURCE`. The equivalence test hand-picked the kept row.
  - Fix: the test now asserts each ORDER BY, the bounds and `KEYS_PER_SOURCE = 5000` against the service source. The replay refuses a window that exceeds 5000 keys. A new test derives the kept row (critical over a newer degraded row, a rejected kept row, a row exactly at `since`).
  - Evidence: `test_r2a_export_mirrors_...`, `test_r2a_row_kept_per_key_is_derived_not_hand_picked`.
- Finding: a quiet stretch before the export window read as "not idle", the opposite of live behaviour. The latest prior cycle could be a failed one.
  - Fix: the export now seeds the latest chat row and the latest Juniper row before the window, plus the last non-failed cycle.
  - Evidence: `test_r2a_export_seeds_latest_chat_and_last_good_cycle_before_window`. The re-run live result is unchanged: 24 sleeps, 114/114 exact.
- Finding: the simulated "held below threshold" counter mixed checks held by pressure with checks held by chat.
  - Fix: it is split into `idle_timer_clear_checks_held_below_threshold` and `timer_clear_checks_held_not_idle`.
  - Evidence: re-run at threshold 3 gives 907 and 253.
- Nits fixed:
  - The arousal backlog count is limited to rows inside the window and type-checked.
  - The percentile rule is labelled.
  - The assumptions now list salience edits, simulated failed attempts, and the validation scope.

## Restart required

```text
No restart required.
```

## Risks / concerns

- **Medium: R1a has 21 hours of data on the pre-#2528 instrument.** The bus_synaptic hogging is a baseline, not a build decision.
- **Medium: R2a replays sources as they are stored now.** Rows deleted later, or crystallizations deactivated later, are invisible to it. The exact match on 114 checks covers only the last 19 h, so older windows may drift slightly. Only one post-#2557 sleep cycle is complete.
- **Medium: R3a has full input coverage for only 18 h, and S2 is degenerate.** The arousal hysteresis cannot be fitted from this data.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
