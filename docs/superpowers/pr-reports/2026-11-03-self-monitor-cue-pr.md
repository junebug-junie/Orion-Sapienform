# PR: Orion knows their own track record in chat (self-monitor step 3)

## Summary

- Orion now sees, in their own chat thinking and in reverie, a few short sentences about what they are failing at or misjudging right now. Example from today's live data: "My Prime predictions work 2% of the time (36 tries this week)."
- The sentences come from the self-monitor organ's faculty records (patch 1). A fixed template writes them. No LLM writes them, and every number in them comes from the record.
- Only faculties whose held status is broken, unreliable or miscalibrated are named. Reliable faculties are never named. A held status whose current numbers no longer support it is not named either.
- The cue reads the Redis key `orion:self_monitor:latest` in the background every 5 minutes. A turn never waits on Redis. A missing, stale (older than 3 hours) or broken value shows nothing.
- Kill switch: `SELF_TRACK_RECORD_CUE_ENABLED` (on by default).

## Outcome moved

Before this patch Orion could not say what they are bad at. The organ's numbers existed only in a report a human ran by hand. Now the stance step and reverie get those numbers as plain sentences. Example: Orion's guesses about how many steps a turn takes hold 5% of the time while Orion says 80%. Stance can now weigh that before it promises something.

## Current architecture

- Patch 1 (#3041, merged) added `orion/self_monitor/` (pure reducer, report, and the reader contract `view.py`). It added no publisher and no reader.
- Patch 2 (a parallel PR, not merged yet) makes orion-durable-runs publish `SelfMonitorStateV1` to Redis `orion:self_monitor:latest` every hour.
- Sibling cues in orion-thought already work this way: `self_readings_cue.py` (body signals) and `personality_cue.py`. Both feed `stance_react.j2` and `reverie_narrate.j2`.

## Architecture touched

- orion-thought: a new cue module, wiring in the stance builder and the reverie builder, and a startup warm read.
- Shared contract: three optional fields on `FacultyRecordV1`, and a new local model `SelfTrackRecordCueV1`.
- Prompts: one new block in `stance_react.j2` and one in the coalition branch of `reverie_narrate.j2`.
- No bus channel changed. Orion-thought only reads Redis. It writes nothing.

## Files changed

- `orion/self_monitor/cue.py`: the pure renderer. It maps faculty keys to plain names, picks lines by priority, and enforces the 300-character cap.
- `orion/schemas/self_monitor.py`: adds `SelfTrackRecordCueV1`. Adds `calibrated_share`, `calibrated_lo` and `calibrated_hi` to `FacultyRecordV1` (the share that calibration is actually judged on).
- `orion/self_monitor/record.py`: fills the three new fields. The change is limited to the constructor call, so a merge with patch 2 stays trivial.
- `orion/schemas/registry.py`: registers `SelfTrackRecordCueV1`.
- `orion/schemas/context_provenance.py`: classifies the context key `self_track_record` as `derived_summary`.
- `orion/schema_skew_discovery.py`: names orion-durable-runs as the writer of the self-monitor state, because code cannot see that writer until patch 2 lands.
- `services/orion-thought/app/self_track_record_cue.py`: the background Redis read, the cache, and the fail-open build.
- `services/orion-thought/app/{bus_listener,reverie,main,settings}.py`: wiring, the startup warm read, and the flag.
- `orion/cognition/prompts/stance_react.j2`, `reverie_narrate.j2`: the `MY TRACK RECORD` block.
- `services/orion-thought/{.env_example,README.md}`: the new key, documented.
- Tests: `orion/self_monitor/tests/test_cue.py`, `services/orion-thought/tests/test_self_track_record_cue.py`, `services/orion-thought/tests/conftest.py` (no test touches the live bus), and `orion/thought/tests/test_stance_react_prompt_imperative_discipline.py` (strict render variable).
- Evals: `services/orion-thought/evals/{self_track_record_truth.py,test_self_track_record_truth_eval.py,run_self_track_record_live_eval.py}`.
- `.github/workflows/self-monitor-tests.yml`: a new `cue` job, and path triggers for it.

## Schema / bus / API changes

- Added: `SelfTrackRecordCueV1` (local only, not on the bus). Optional fields `calibrated_share`, `calibrated_lo` and `calibrated_hi` on `FacultyRecordV1`.
- Removed: none.
- Renamed: none.
- Behavior changed: stance_react and reverie prompts get a new block, but only when a fresh state names a notable faculty.
- Compatibility notes: `FacultyRecordV1` uses `extra="forbid"`. A reader with the old schema rejects a state that carries the new fields. Deploy orion-thought (this PR) before or together with patch 2's publisher. Nothing publishes the key today, so today there is nothing to break.

## Env/config changes

- Added keys: `SELF_TRACK_RECORD_CUE_ENABLED=true` (orion-thought).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-thought`: yes. The script added `+SELF_TRACK_RECORD_CUE_ENABLED='true'`. It is now in `/mnt/scripts/Orion-Sapienform/services/orion-thought/.env`.
- Skipped keys requiring operator action: none.
- docker-compose.yml: no change. orion-thought loads `.env` through `env_file`.

## Tests run

```text
pytest orion/self_monitor/tests                                  59 passed
pytest services/orion-thought/tests                              835 passed, 32 skipped (full suite, with evals)
pytest orion/temporal_self/tests orion/thought/tests orion/schemas/tests
       tests/test_attention_self_readings.py tests/test_metric_prompt_semantics.py   560 passed (with self_monitor)
pytest services/orion-cortex-exec/tests/test_chat_stance_context_provenance_hazard.py   8 passed
pytest tests/scripts/test_schema_skew_discovery.py               18 passed
CI "cue" job, run in a clean venv with only its pip installs    15 passed
orion-static-gates.yml, every run step locally                   see Docker/build/smoke checks
```

## Evals run

```text
pytest services/orion-thought/evals/test_self_track_record_truth_eval.py   2 passed
  (every number shown resolves to its faculty record; a forged number is caught)

python -P services/orion-thought/evals/run_self_track_record_live_eval.py   (2026-11-03, live)
orion:self_monitor:latest absent on the bus (patch 2 not deployed?); building from the report
source=report known=True reason=fresh
cue (287 chars, skip_reason=None, shown=4, dropped=6):
My Prime predictions work 2% of the time (36 tries this week). My guesses about how many steps my
turns take hold 5% of the time (I say 80%). My guesses about how long my replies run hold 8% of
the time (I say 80%). My office temperature forecast ranges hold 66% of the time (I say 80%).
truth: ok
```

The Redis key is absent today, so the eval built the same state from the bet tables (`orion.self_monitor.report.collect`). A state built that way has no hysteresis history. Live, every status must hold for 3 hourly evaluations first. The walkway-grade journal was "broken" at 22 of 28 earlier today. It now reads `unknown` (64% of 21 tries did not run, and the lower bound is not above 0.5), so it is correctly left out.

## Docker/build/smoke checks

```text
orion-static-gates.yml run steps, locally: 40 of 40 PASS after the schema-skew declaration
(first run: test_schema_skew_discovery failed with "SelfMonitorStateV1 read by orion-thought:
declare its writer" -> fixed in orion/schema_skew_discovery.py)
```

Docker was not built for this PR. The Dockerfile already copies `orion/` and the cue uses no new dependency (`redis` is already in requirements.txt). The live stance path is UNVERIFIED until deploy and patch 2.

## Review findings fixed

- Finding: miscalibrated lines printed the upheld share over all graded bets. Calibration is judged on the bets that stated a confidence.
  - Fix: `calibrated_share`, `calibrated_lo` and `calibrated_hi` are stored on the record and printed by the cue. The truth eval checks against them.
  - Evidence: `test_miscalibrated_uses_the_calibration_share_not_overall_reliability`, `test_reducer_stores_the_calibration_share_the_cue_prints`.
- Finding: hysteresis lag could print a held "unreliable" or "miscalibrated" line next to numbers that contradict it.
  - Fix: an unreliable line prints only if `reliability_hi < 0.5`. A miscalibrated line prints only if the candidate is still miscalibrated, or if stated_mean is outside the calibration interval.
  - Evidence: `test_hysteresis_lag_never_prints_numbers_that_contradict_the_status`.
- Finding: plain names could come out empty or meaningless ("runs", "messages", `turn.self` with a double space). One name had a mid-sentence capital. Unit suffixes leaked into names.
  - Fix: a host word is kept when it is the only doer word. An empty name falls back to the humanised key. `.capitalize()` is gone. `_c`/`_w`/... suffixes are stripped.
  - Evidence: `test_plain_names` (new cases `act.make`, `act.durable.run`, `act.hub.say`, `turn.self`, `power_w`).
- Finding: "this week" was hardcoded.
  - Fix: the wording comes from the record's own window ("this week" only for 7 days).
  - Evidence: `test_window_wording_follows_the_record_window`.
- Finding: a share such as 0.2% printed as a false "0%".
  - Fix: a share strictly between 0 and 1 is clamped to 1-99%, in both the cue and the eval.
  - Evidence: `test_percent_never_rounds_to_a_false_zero_or_hundred`.
- Finding: the refresh flag could stay stuck on True if the thread failed to start. The startup warm read could also race a first turn.
  - Fix: a failed thread start resets the flag under the lock. warm() claims the flag before it reads.
  - Evidence: `test_thread_start_failure_does_not_wedge_the_refresh`, `test_warm_claims_the_refresh_so_a_racing_turn_does_not_read_twice`.
- Finding: CI path triggers were incomplete, and the cue job installed the hub requirements.
  - Fix: both trigger lists now include the prompts, the wiring files, conftest, the tests and the evals. The job installs only orion-thought's requirements plus pytest, pytest-asyncio and jinja2.
  - Evidence: the job's commands passed in a clean venv with only those installs.
- Nits: the conftest docstring now says how to opt out. The skip log carries the last fetch error.
- Found while checking live data (not from the review): ranking on the raw evidence count let 4 extra bets put "51% vs 38%" ahead of "5% vs 80%". Ranking now uses evidence by order of magnitude, then size of miss. Test: `test_a_few_more_bets_do_not_outrank_a_far_bigger_miss`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
```

Run it from the clean primary `main` checkout after merge. The cue shows nothing until patch 2 is deployed and publishes `orion:self_monitor:latest`. Until then the log line is `self_track_record_skipped ... reason=absent`.

## Risks / concerns

- Severity: medium (follow-up). Concern: the metric prompt-semantics gate (`orion/metrics/semantics.py`: `prompt_sites`, `REQUIRED_FOR_PROMPT`, `PROMPT_INVENTORY_URNS`) does not cover these numbers yet. The `inner_state_registry` entry `self_monitor.state.v1` belongs to patch 2 and does not exist on main yet. Mitigation: when patch 2 adds that entry, give it semantics for status, reliability, calibrated_share and stated_mean. Each one needs `prompt_sites=("orion.self_monitor.cue:project_track_record_cue",)`, and the URNs must be pinned in `PROMPT_INVENTORY_URNS`.
- Severity: low (merge order). Concern: this PR declares durable-runs as the writer of `SelfMonitorStateV1`/`FacultyRecordV1` in `orion/schema_skew_discovery.py`. Once patch 2 lands, code can see the writer, and `test_declared_writers_are_not_stale` fails. Mitigation: whichever PR merges second removes those two entries.
- Severity: low. Concern: the additive `extra="forbid"` fields on `FacultyRecordV1` (see Compatibility notes). Mitigation: deploy orion-thought before or together with patch 2.
- Severity: low. Concern: the line order is strict. When a higher-priority line does not fit, every line after it is dropped, even a shorter one. Mitigation: this keeps the order honest. The cue still used 287 of 300 characters on live data.

## PR link

(added after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
