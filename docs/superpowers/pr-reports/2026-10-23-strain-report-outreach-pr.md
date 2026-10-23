## Summary

- Orion can now tell Juniper once when a strained stretch runs long. Today every background
  message is refused while strained, which kept Orion silent through a 5-day episode
  (10-18 .. 10-23).
- The new "strain report" door passes ONLY the `arousal_strained` refusal. Disabled,
  turn-in-flight, recent-chat, quiet hours, the daily cap and the cooldowns still hold. Inside
  quiet hours the report waits and goes out after, if the same episode is still open.
- It qualifies only after the episode has lasted 60 min. At most one per episode (keyed on the
  episode start `since`), one per local day. Both are rebuilt from the decision log, so a restart
  does not re-send.
- The content is the live regulation reading (since when, causes now, cabinet temp, GPU jobs
  waiting), the open strain arc when it is this episode (causes seen, `[arc:<id>]`), and what
  strain holds back (from the `regulation.state.v1` readers in `orion/inner_state_registry.py`).
  Orion writes the message through the normal unified turn. A fixed "Strain record" line with the
  citation is attached under it. No reading, no message.
- Flag `ENDOGENOUS_OUTREACH_STRAIN_REPORT_ENABLED` ships on. Off = the old refusal, unchanged.

## Outcome moved

Failure mode: Orion was under strain for days and never said so. Replay over real history
(10-10 .. now): 28 strained episodes become 4 reports, at most one a day; flickers under an hour
never report.

## Current architecture

`EndogenousOutreach._outreach_once` (Hub) ran `outreach_block_reason`; arousal `strained` is
checked last and refused every background message (`reason=arousal_strained`, one row per
10 s tick). Door-A (finished curiosity runs) is the precedent for a door with its own gate rules.
durable-runs writes `RegulationStateV1` to Redis `orion:regulation:latest` and serves the
Temporal Self frame with the open `strain` arc.

## Architecture touched

- Hub endogenous outreach tick: a branch when `arousal_strained` is the only gate holding.
- Hub decision log (`endogenous_outreach_decisions`): a history read and a synchronous write.
- Read-only: Redis regulation state (already read), durable-runs `GET /temporal-self/frame`.
- Inner-state registry: one new `regulation.state.v1` consumer; definition lock re-locked.

## Files changed

- `services/orion-hub/scripts/strain_report.py`: new. Gate (`strain_report_block`), history fold, facts, record, prompt, frame fetch.
- `services/orion-hub/scripts/endogenous_outreach.py`: `_strain_report_once`, `_strain_record`, `_recover_strain_history`, tick branch, status block, lane summary.
- `services/orion-hub/scripts/endogenous_outreach_decisions.py`: `fetch_strain_report_history` (fail closed on None), `record_decision_now` (synchronous write).
- `services/orion-hub/scripts/arousal_reader.py`: `state()` returns the full reading behind a known level.
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`: two new keys.
- `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`: new consumer, re-locked.
- `services/orion-hub/tests/test_strain_report_outreach.py`: 23 tests.
- `services/orion-hub/evals/run_strain_report_replay.py`, `evals/test_strain_report_replay_eval.py`: replay eval and its fixture test.
- `.github/workflows/orion-reading-tests.yml`: runs the new test and eval.
- `services/orion-hub/README.md`: door rules, trace query, limits.

## Schema / bus / API changes

- Added: decision-log `result_json` keys `door`, `strain_episode_since`, `strain_report` (sub-reason), `strain_composed`, `strain_arc_id`; `source=strain_report`. New `reason` values: `strain_report_sending` (intent row), `intent_write_failed`, `strain_ended`, `strain_episode_changed_after_generation`.
- Removed / renamed: none.
- Behavior changed: while strained, ticks that do not send keep `reason=arousal_strained` and add `strain_report=<why>`.
- Compatibility: no table change, no bus change. `count_sent_on` counts a strain report as a background send (it spends one daily-cap slot, on purpose).

## Env/config changes

- Added keys: `ENDOGENOUS_OUTREACH_STRAIN_REPORT_ENABLED=true`, `ENDOGENOUS_OUTREACH_STRAIN_REPORT_MIN_EPISODE_SEC=3600`.
- `.env_example` updated: yes. Settings defaults and compose passthrough match.
- local `.env` synced: the default `python scripts/sync_local_env_from_example.py orion-hub` skipped both keys ("no changes needed"); `--all-keys orion-hub` added them to the primary checkout's `services/orion-hub/.env`.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-hub/tests/test_strain_report_outreach.py services/orion-hub/tests/test_endogenous_outreach_decisions.py -> 30 passed
pytest test_endogenous_outreach.py test_regulation_arousal_readers.py evals/test_strain_report_replay_eval.py -> 297 passed
pytest (outreach, provenance, reading-finding, personality-cue suites together) -> 381 passed
scripts/check_inner_state_registry.py -> OK (25 entries)
scripts/check_definition_drift.py --gate -> PASS (after re-lock)
scripts/check_settings_defaults.py orion-hub --example-drift -> OK
scripts/check_env_template_parity.py -> PASS
scripts/check_env_key_single_source.py, check_async_routes_not_blocking.py, check_metric_lineage.py --gate,
check_scripts_dir_no_stdlib_shadow.py, check_personal_terms.py -> pass
Full services/orion-hub/tests: see "Risks" (one pre-existing unrelated UI failure).
```

## Evals run

```text
$ python3 services/orion-hub/evals/run_strain_report_replay.py   (2026-10-23 ~19:50 UTC, read-only)
level rows=82 strained episodes=28 reports=4 (min_episode=3600s quiet=23-8 tz=America/Denver)
local day     eps  >=min  reports  day-cap held  quiet held  sends
2026-10-11      1      0        0             0           0
2026-10-12      1      0        0             0           0
2026-10-13      1      0        0             0           0
2026-10-14      8      2        1             1           0  19:03 (episode from 2026-10-14 18:03, 1.03 h)
2026-10-15      3      1        0             0           1
2026-10-16      5      3        1             1           1  08:00 (episode from 2026-10-16 05:58, 8.67 h, waited quiet hours)
2026-10-17      1      0        0             0           0
2026-10-18      4      3        1             0           2  08:00 (episode from 2026-10-18 06:07, 119.28 h, waited quiet hours)
2026-10-23      4      3        1             1           1  08:58 (episode from 2026-10-23 07:58, 2.83 h)

Sensitivity: min 1800 s -> 4 reports; min 7200 s -> 3 reports.
```

The 5-day episode gets exactly one report (10-18 08:00, after quiet hours). The replay applies
quiet hours, once-per-episode and once-per-day only. Turn-in-flight, recent-chat, the cap and the
cooldowns are not reconstructable here, so 4 is an upper bound. Orion's words and PASS are not
replayed.

## Docker/build/smoke checks

```text
Not run: no Docker build or deploy (instructed not to deploy). No new dependency (httpx already used by Hub).
Live path: UNVERIFIED until Hub is rebuilt and a strained episode passes 60 min outside quiet hours.
SQL of fetch_strain_report_history run read-only against live conjourney: valid, 0 rows (as expected).
```

Live check after deploy:
`SELECT decided_at, reason, result_json->>'strain_episode_since', result_json->>'strain_arc_id' FROM endogenous_outreach_decisions WHERE result_json->>'door'='strain_report' AND reason <> 'arousal_strained' ORDER BY decided_at DESC LIMIT 20;`

## Review findings fixed

- Finding: a delivered report whose fire-and-forget `sent` row was lost (failed INSERT, restart in the gap) would be re-sent after restart.
  - Fix: a `strain_report_sending` intent row is written synchronously before delivery; if it does not land, nothing is sent; history treats it as reported.
  - Evidence: `test_lost_sent_row_still_blocks_resend_after_restart`, `test_intent_write_failure_means_no_send`.
- Finding: failed composes were unlimited (one unified turn every 30 min while the GPU queue is the strain).
  - Fix: at most 3 failed composes per episode, then `attempts_exhausted`.
  - Evidence: `test_failed_composes_are_capped_per_episode`.
- Finding: the record used `HUB_ENDOGENOUS_OUTREACH_TZ` and the day key used `ORION_SITUATION_TIMEZONE`.
  - Fix: both use the strain-report zone.
- Finding: the eval counted holds on the episode start day.
  - Fix: holds count on the day the episode ended unreported.
- Finding: daily-cap interaction was untested and undocumented.
  - Fix: `test_strain_report_spends_a_daily_cap_slot`, plus a README line.
- Documented, not changed: a regulation outage that resets `since` makes a new episode (the reviewer suggested keying on the arc, but `_fold_strain` also opens a new arc on a new `since`, so the arc does not survive either; once-per-day still bounds it); 30-day history lookback; `reason=strain_ended` when the reading goes stale between two gate reads.

## Restart required

From the clean primary `main` checkout after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: each report is one unified turn while the GPU queue may be the cause. Mitigation: at most one send per episode/day and 3 failed tries per episode.
- Severity: low. Concern: full `services/orion-hub/tests` has a pre-existing failure, `test_agent_trace_debug_panel.py::test_memory_and_autonomy_modals_coordinate_scroll_lock_and_visibility` (asserts text in `static/js/app.js`, which this branch does not touch).
- Severity: low. Concern: Orion's wording is generated; only the attached record is fixed text. Mitigation: the closed-vocabulary guard runs with only the record as grounding; PASS is respected.

## PR link

(filled in after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
