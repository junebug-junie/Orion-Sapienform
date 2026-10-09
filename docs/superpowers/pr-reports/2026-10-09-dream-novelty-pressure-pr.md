# feat(dream): sleep pressure counts new distinct things, not rows

## Summary

- Orion's tiredness ("sleep pressure") now counts **things, not rows**: one recurring gateway timeout is one thing, not 222.
- Only **new** things add pressure: anything already seen in the 48 hours before the window adds nothing. A chronic problem adds pressure once and stays available for dreams to replay.
- Crystallizations count when a memory is **activated**, not when it is recalled. Every recall was rewriting `updated_at` on ~100 memories, and the dream read that as new material.
- The sleep record gains `new_counts`, showing exactly which new things drove each sleep.
- Includes the approved design doc (formerly PR #2552) and a backtest eval on a real week of events.

## Outcome moved

Before: pressure read 70-129 against a threshold of 3, so Orion slept on the 6-hour minimum-interval clock (28 sleeps a week, every gap 6 h). After, the same real week replayed through the new code at the same threshold: 11 sleeps, gaps 6 to 40.5 h, and longer gaps follow quiet stretches.

## Current architecture

- `cycle_store.SOURCE_QUERIES` read the four sources with `LIMIT 50` each.
- `replay.build_candidates` made one candidate per row.
- `compute_pressure` summed every candidate weight.
- The crystallization source was `memory_crystallizations.updated_at`, which `retriever._apply_recall_boost` rewrites on all ~100 rendered crystallizations per retrieval (438/438 retrievals over 7 days had exactly 100).

## Architecture touched

`services/orion-dream` plus one optional field on `SleepPressureV1` (`orion/schemas/dream_cycle.py`). No bus channel, env key, or table change.

## Files changed

- `services/orion-dream/app/cycle_store.py`:
  - `SOURCE_QUERIES` return one row per `dedupe_key`, bounded by `[since, until)`.
  - Metacog keys on normalized `trigger_reason`.
  - Crystallization reads activation events from `memory_crystallization_history`.
  - `load_source_rows` takes an `until` bound.
- `services/orion-dream/app/replay.py`: `row_key`, `keyed_candidates` (merge by key at max weight), `prior_keys`, `compute_pressure(keyed, seen_before)` -> (pressure, counts, new_counts).
- `services/orion-dream/app/cycle.py`: reads the lookback before the window through the same loader, fills `new_counts`, adds the `overdue` backstop, `KEYS_PER_SOURCE`.
- `services/orion-dream/app/settings.py`, `docker-compose.yml`, `.env_example`: `DREAM_CANDIDATES_PER_SOURCE` retired.
- `services/orion-hub/static/js/dream-tab.js`, `services/orion-hub/evals/dream_browser.cjs`: show and check new vs distinct counts.
- `services/orion-dream/app/main.py`: wires `load_prior_rows`.
- `orion/schemas/dream_cycle.py`: `SleepPressureV1.new_counts` (optional), docstring corrected.
- `services/orion-dream/tests/test_dream_cycle_v2.py`: 5 new tests, 1 rewritten.
- `services/orion-dream/evals/test_sleep_pressure_backtest_eval.py`, `evals/fixtures/sleep_pressure_events_2026-10-09.psv`: real-week backtest through the real code.
- `services/orion-dream/scripts/backtest_sleep_pressure.py`: the same backtest against fresh live data.
- `docs/superpowers/specs/2026-10-09-dream-sleep-pressure-novelty-design.md`: design and metric-gate record (approved).
- `services/orion-dream/README.md`: "What makes Orion tired".

## Schema / bus / API changes

- Added: `SleepPressureV1.new_counts: dict[str, int]` (optional, default `{}`).
- Removed: none
- Renamed: none
- Behavior changed: `pressure` = sum of max weight per *new* distinct key. `counts` = distinct things per source (was rows read).
- Compatibility notes: `SleepPressureV1` is `extra="forbid"`, and Hub (`services/orion-hub/scripts/dream_routes.py`) validates the dream service's pressure JSON. **Deploy Hub before orion-dream**, or Hub's pressure readout rejects the new field.

## Env/config changes

- Added keys: none
- Removed keys: `DREAM_CANDIDATES_PER_SOURCE` (orion-dream: settings, docker-compose, `.env_example`)
- Renamed keys: none
- `.env_example` updated: yes (orion-dream)
- local `.env` synced: the stale line was removed by hand from `services/orion-dream/.env`, since the sync script only adds keys. `sync_local_env_from_example.py` was run. Until this merges, `check_env_template_parity.py` WARNs that orion-dream's `.env` lacks the key, because it compares against main's `.env_example`.
- skipped keys requiring operator action: none. Unrelated, pre-existing parity FAIL: orion-sql-writer's `.env` lacks `debug.attention.streak_tick.v1` / `orion:debug:attention:streak_tick`.
- Reused: `DREAM_LOOKBACK_HOURS` (48) as both the "seen before" lookback and the backstop. `DREAM_SLEEP_PRESSURE_THRESHOLD` stays 3.

## Tests run

```text
pytest services/orion-dream/tests services/orion-dream/evals -q        -> 128 passed
pytest services/orion-hub/tests/test_dream_routes.py
       tests/test_dream_hypotheses.py -q (the hub-dream CI set)        -> 27 passed
node --check dream-tab.js, dream_browser.cjs                            -> ok
New tests against the pre-change app code                               -> 6 failed, 25 passed
scripts/check_metric_lineage.py --gate     -> PASS
scripts/check_definition_drift.py --gate   -> PASS
scripts/check_scripts_dir_no_stdlib_shadow.py -> clean
```

## Evals run

```text
pytest services/orion-dream/evals/test_sleep_pressure_backtest_eval.py -s
threshold=3: sleeps/7d=11 gaps_h=[6.0, 6.0, 6.0, 6.0, 13.0, 13.0, 19.0, 22.0, 24.5, 40.5]
2 passed

node services/orion-hub/evals/dream_browser.cjs (real Hub template, fixture APIs, Chrome 131)
{"passed":true,"checks":[...,"new vs distinct counts",...], "requests":22}
```

## Docker/build/smoke checks

```text
Each new SOURCE_QUERIES statement run read-only against live Postgres (6 h window):
metacog 35 rows / 35 keys, compaction 8/8, resonance 1/1, crystallization 1/1.
Metacog keys normalized, e.g. telemetry_anomaly:elevated:recon_loss=#:threshold=#:top=compliance_deficit=#
Post-deploy proof: UNVERIFIED. Expect dream_cycle gaps to vary (median not 6 h), "not due"
log lines while idle with new={...}, and pressure 0 right after a sleep.
```

## Review findings fixed

- Finding (medium): nothing guaranteed Orion eventually sleeps. Pressure counts only new things, with no maximum interval.
  - Fix: `cycle.overdue` backstop. When the window reaches `DREAM_LOOKBACK_HOURS` without crossing threshold, Orion sleeps if idle and there is something to replay (note `overdue`).
  - Evidence: three tests (sleeps when overdue; waits when not overdue; never sleeps through a conversation). The backtest eval applies the same rule and bounds every gap.
- Finding (medium): Hub must deploy with or before orion-dream (forbid schema).
  - Fix: stated in the restart command (Hub first).
  - Evidence: `dream_routes.py:66` validates `SleepPressureV1`.
- Finding (low): the per-source cap of 50 sat near real volume (47 metacog keys in 48 h live) and could drop new keys.
  - Fix: retired `DREAM_CANDIDATES_PER_SOURCE`. Both reads use `cycle.KEYS_PER_SOURCE` (5000). The lookback reuses `load_source_rows(..., until=)`, so there is one read path.
  - Evidence: 128 passed. Removed from settings, compose, and `.env_example`, plus the local `.env`.
- Finding (low): the Hub panel showed `counts` as "Candidate counts" and never showed `new_counts`.
  - Fix: `dream-tab.js` shows "New since last sleep (drives pressure)" and "Distinct things to replay".
  - Evidence: `dream_browser.cjs` new check `new vs distinct counts` passed (12 checks), screenshot reviewed.
- Finding (low): the backtest ignored `window_start`'s 48 h clamp.
  - Fix: eval and script both clamp and apply the backstop.
  - Evidence: same result, 11 sleeps, longest gap 40.5 h.
- Finding (note): crystallization replay candidates drop to activations only. Already listed under risks.

## Restart required

From the primary checkout on main after merge, Hub first:

```bash
git pull --ff-only && scripts/safe_docker_build.sh orion-hub up -d --build && scripts/safe_docker_build.sh orion-dream up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: replay material changes. Recall-touched crystallizations no longer appear as candidates, so most windows carry 0-2 crystallizations instead of up to 50.
  - Mitigation: intended (they were not unprocessed material). Watch hypothesis counts in the first week.
- Severity: low
  - Concern: metacog rows with no `trigger_reason` key on their row id, so each counts as new.
  - Mitigation: safe direction (Orion sleeps more, never less). 0 such flagged rows in the fixture week.
- Severity: low
  - Concern: deploy order (forbid schema).
  - Mitigation: Hub first, stated above.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2557

🤖 Generated with [Claude Code](https://claude.com/claude-code)
