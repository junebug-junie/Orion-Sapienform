## Summary

Builds Family C Experiment 1, which Juniper approved on #3482 (2026-11-16). The design is `docs/superpowers/specs/2026-11-16-family-c-first-experiment-economy-native.md`.

- When an AI Town session is about to be allowed to start, a runner in orion-prediction takes the next arm from a seeded list for the night. ON: the town plays as it does today. OFF: no new session can start for 40 minutes.
- A list never has more than 2 holds in a row. A running session is never stopped. The 2 h daily cap does not change.
- orion-embodiment obeys a Redis key with a TTL, and only at its start gate. If the runner dies, the town is free again within 90 s plus one 300 s tick.
- The experiment bids as a new tier-2 family, `prediction.experiment`, and its bids bind. Only a funded bid runs a block.
- The effect table pools the nights, with each night as a stratum. It now has lane-cost targets: tier-1 hold and grant wait on `fast`, and Kev latency.
- A finished run becomes one experience for Orion (`experiment_result`, salience `effect_z`).
- **It never starts itself.** A run starts only with `POST /experiments/town-hold/run`.

## Outcome moved

Orion gets its first real, randomised measure of what its own play costs its body: GPU3 watts and heat. It also gets what that play costs its tier-1 upkeep (how much slower tier-1 calls on the `fast` lane get). The live Patch 0 replay says 16 nights is enough: the minimum detectable GPU3 effect is 7.9 W, under the 10 W bar.

## Current architecture

- Family C slice 1 had one actuator (`vision.focus`), fixed clock blocks, one target series per experiment, and a free permutation test.
- The town controller's start gate had only env caps.
- No runtime override of a family's start gate existed.

## Architecture touched

- **orion-embodiment.** The start gate reads `orion:physiology:run_cap:aitown.session` under its start lock, before the town is resumed. The timer pre-checks the same key and places no shadow bid while a hold is on. `/health` gains `start_eligible_at`, `start_in_flight`, `manual_session_open` and `run_cap_override`.
- **orion-prediction.**
  - New `app/town_hold.py`: the night loop, the pending hold, the race read, the stop rules, the binding bid, block checks, the persisted schedule, preview, and `run_result`.
  - `causal.py`: `draw_arm_sequence` and `analyze_stratified_switchback`, with a vectorised restricted permutation.
  - `family_c.py`: event experiments, lane outcomes and the manipulation check.
  - `store.py`: read-only reads of `gpu_pool_leases`, `substrate_system_one_appraisal` and `grammar_acts`.
  - `experiment_runner.py`: key cleanup by action in `recover()`. A vision run is refused while a town-hold run is scheduled.
- **Shared.**
  - `orion/physiology/run_cap_override.py` (pure judge) and `config/physiology/run_cap_overrides.yaml`.
  - `RunCapOverrideV1` schema.
  - `prediction.experiment` family and its binding producer.
  - `experiment_result` source kind, adapter, durable-runs SourceQuery, `effect_z` magnitude rule, and the organs.yaml entry.

## Files changed

- `orion/schemas/physiology_run_cap.py`, `orion/schemas/registry.py`: `RunCapOverrideV1`. Redis only, no channel.
- `orion/physiology/run_cap_override.py`, `config/physiology/run_cap_overrides.yaml`: the bounds and the obey/ignore rules.
- `config/physiology/families.yaml`, `orion/physiology/binding.py`: the `prediction.experiment` family (tier 2, `lanes: []`) and its binding producer.
- `orion/embodiment/town_sessions.py`: the pure `start_eligible_at`.
- `services/orion-embodiment/app/town_sessions.py`, `app/town_self_start.py`, `Dockerfile`, `README.md`: the gate, the pre-check, the `/health` fields, and the YAML copied into the image.
- `services/orion-prediction/app/{town_hold,causal,family_c,store,experiment_runner,main,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`: the runner, the analysis and the endpoints.
- `orion/schemas/temporal_self.py`, `orion/temporal_self/{sources,now}.py`, `services/orion-durable-runs/app/temporal_self_sources.py`, `config/integration/organs.yaml`: the `experiment_result` experience.
- Tests:
  - `services/orion-embodiment/tests/test_town_run_cap_override.py`
  - `services/orion-prediction/tests/test_town_hold.py`
  - `orion/physiology/tests/test_prediction_experiment_family.py`
  - `orion/temporal_self/tests/test_now.py`
- Evals: `services/orion-prediction/evals/run_town_hold_{synthetic,replay,dry_run}_eval.py`.
- CI: the prediction and embodiment workflows get new path triggers, and the synthetic eval is a CI step.

## Schema / bus / API changes

- **Added:**
  - `RunCapOverrideV1` (Redis key `orion:physiology:run_cap:<family>`).
  - `TemporalSourceKind` `experiment_result`.
  - Refusal reason `run_cap_override`.
  - Embodiment `/health` fields.
  - Prediction endpoints:
    - `POST /experiments/town-hold/run`
    - `GET /experiments/town-hold`
    - `GET /experiments/town-hold/preview`
    - `POST /experiments/town-hold/abort`
    - `POST /experiments/town-hold/{run_id}/result`
  - Effect-table targets `lane.fast.*` and `lane.kev.latency_ms_p50`.
- **Removed / renamed:** none.
- **Behavior changed:**
  - A vision.focus run is refused while a town-hold run is scheduled.
  - `recover()` now clears the run-cap key and cancels interrupted town-hold nights.
- **Compatibility:** no bus channel and no migration. A night's `prediction_experiment.design.kind = event_switchback`. A contaminated night is stored as `cancelled` with an abort row that says `contaminated: ...`, because the status CHECK constraint allows only designed, analyzed and cancelled.

## Env/config changes

- Added keys: `PREDICTION_TOWN_HOLD_ENABLED=true` and `PREDICTION_EMBODIMENT_URL=http://orion-athena-embodiment:8130` (orion-prediction).
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The primary checkout `services/orion-prediction/.env` gained both keys.
- Skipped keys: none.

## Deviations from the design (deliberate)

1. Lane costs are computed per block from the raw lease and Kev rows (`family_c.lane_outcomes`), not as new `series.py` source kinds. A per-window mean over leases is the quantity the design defines. Bucket means would weight it wrongly. No series is stored, so removing a target means removing one dict entry.
2. A separate endpoint, `POST /experiments/town-hold/run`, replaces `action=run_cap:...` on `POST /experiments/run`. The parameters (nights, first night) do not fit the fixed-block body.
3. OFF-block shadow wants are not tagged. They are not placed at all: the timer pre-checks the key and skips the bid, the same way it skips a bid during any other hold. The controller still re-checks the key under its lock.
4. Blocks start at the decision moment, not on a 5-min boundary. The outcome window reads every grain bucket that starts at or after block start + 2 min.
5. A missing or stale regulation reading refuses a decision (`strain_unknown` / `strain_stale`). The stop rules fail closed for data quality.
6. Manual sessions are flagged by a `block_check` row (read from the session ledger) after each block's window. An unreadable ledger counts as manual, so the block is excluded.

## Metric gate (lane-cost outcomes)

1. **Provenance.**
   - `gpu_pool_leases`: `granted_at`, `queued_since`, and `updated_at` at release, written by the GPU pool.
   - Kev `latency_ms`: `orion/substrate/system_one_appraisal.py`, stored in `substrate_system_one_appraisal.frame_json`.
2. **Independence.** Tier-1 hold shares a cause with GPU3 watts (the card's load). It is a separate outcome (the cost to other work), not new signal about watts. Wait measures queueing. Hold measures service time. Kev runs a separate process on the same card.
3. **Theory.** Queueing: added concurrent requests on a shared GPU raise both the wait (Little's law) and the per-request service time (batched decode).
4. **Live sanity** (replay, 14 days, read-only):
   - Tier-1 hold: quiet blocks 4.35 s, session blocks 9.48 s. It can rest near its quiet level and is not floored.
   - Grant wait: quiet 0.18 s, session 0.36 s.
   - Kev p50: 229 ms quiet, 274 ms in a session.
   - Missing share (no call in the window, read as None, never as 0): tier-1 hold 73 %, grant wait 57 %, Kev 52 %.
5. **Existing mechanism.** `gpu_wait` (body organ) already reads the same table. No new producer is added.
6. **Reversibility.** These targets are computed at analysis time. Removing one is one entry in `LANE_TARGETS`.

## Tests run

```text
services/orion-prediction (docker orion-prediction-test:local, CI deps): pytest tests -> 553 passed
  (before the perf commit); tests/test_town_hold.py test_experiment_runner.py test_family_c.py -> 76 passed;
  tests/test_town_hold.py after perf -> 33 passed
services/orion-embodiment: pytest tests -> 160 passed
orion/physiology/tests + orion/temporal_self/tests + tests/test_check_integration_seam.py -> 705 passed
services/orion-durable-runs temporal/experience tests -> 68 passed, 21 skipped (need Postgres)
services/orion-sql-writer tests/test_act_downstream_use.py -> 6 passed
static: test_no_personal_terms, test_no_private_eval_artifacts, test_ci_service_image_sources,
  test_compose_logging_journald, test_host_allowlist_buildable -> 33 passed
scripts/check_integration_seam.py --base origin/main -> OK (1 changed exemption: prediction exempt_read)
scripts/check_env_template_parity.py -> PASS; scripts/check_env_key_single_source.py -> OK
```

## Evals run

```text
run_town_hold_synthetic_eval.py (16 reps x 12 nights through the real runner, simulated town with 2 h cap):
  blocks/night on 3.57 off 3.07; sessions started inside an OFF hold: 0; nights with > 2 holds in a row: 0
  circe.gpu3_watts: coverage 0.88, power 1.00, null p<=0.05 0.06
  lane.fast.tier1_hold_s: coverage 0.94, power 1.00, null p<=0.05 0.06  -> PASS
run_town_hold_replay_eval.py --days 14 (live, read-only; Patch 0):
  61 sessions; GPU3 watts SD 15.0 W, MDE 7.9 W at 56/arm -> GO (<= 10 W)
  tier-1 hold MDE 1.5 s (seen +5.1 s); grant wait MDE 0.36 s; Kev MDE 61 ms (seen +45 ms: reported, not claimed)
run_town_hold_dry_run_eval.py (live, read-only): tonight 2026-11-17, run seed 0, arms ON OFF OFF ON ON OFF OFF ON;
  stop rules now: none; projected 4 ON / 4 OFF, 4 sessions; embodiment live image does not obey the override yet
  (expected until this PR is deployed), so a run started now would be refused.
Effect table build at full size (16 nights, 50 targets, 2000 draws): 3.2 s.
```

## Docker/build/smoke checks

```text
No live build or deploy (brief: do not deploy). Live reads only, from inside app-net in the test image.
```

## Review findings fixed

An independent review subagent read the diff and ran the tests. Every must-fix and should-fix finding is fixed.

- **Finding (must-fix):** the quiet period ended one tick after every decision or refusal. `/health` reports an open gate as "eligible now", so `eligible_at > marker` was always true. Blocks ran back to back, and a refusal repeated every 30 s.
  - **Fix:** the gate counts as "moved" only when it moved into the future (more than one tick ahead). The test probe and the simulated town now report "now" while the gate is open, the same as the real `/health`.
  - **Evidence:** regression tests `test_after_an_on_arm_the_runner_waits_for_the_town_instead_of_drawing_again` and `test_after_a_refusal_the_runner_does_not_re_decide_every_tick`. The synthetic eval still passes with the realistic town.
- **Finding:** an OFF hold decided before 08:00 ran past 08:00.
  - **Fix:** no decision (either arm) when the block would end after the night window.
  - **Evidence:** `test_no_block_that_would_run_past_eight`.
- **Finding:** the design says a failed guard read aborts the night. The code only refused the decision.
  - **Fix:** a `*_read_failed` stop rule now aborts the night. A missing reading still only refuses.
  - **Evidence:** `test_a_failed_guard_read_aborts_the_night`.
- **Finding:** "exact" overstated the permutation test, because each night's block count depends on its arms.
  - **Fix:** the docstring and this report now say "approximate, conditional on block count". The synthetic eval simulates that dependence.
  - **Evidence:** with the dependence simulated, null p ≤ 0.05 in 6 % of runs.
- **Finding:** `run_result` was not idempotent. A retry would give Orion a duplicate experience.
  - **Fix:** one result row per run; a retry returns the stored row.
  - **Evidence:** `test_the_run_result_is_stored_once`.
- **Finding:** the cancel and resume paths had no tests.
  - **Fix:** two tests added.
  - **Evidence:** `test_cancel_mid_off_block_deletes_the_hold_cancels_the_night_and_frees_the_lock` and `test_resume_continues_a_scheduled_run_and_drops_a_malformed_one`.
- **Finding:** writing the lock and ACTIVE key outside the `try` could leave a night stuck as `designed`.
  - **Fix:** both writes moved inside the `try`. ACTIVE is deleted only when it names this night. A night that fails to get the lock never touches the other run's keys.
- **Finding:** a cancel during `_finish` could overwrite a night that was already `analyzed`.
  - **Fix:** the cancel handler skips the abort when the night is already analysed.
- **Nits fixed:**
  - A resumed run now keeps the config it was started with.
  - `effect_z` uses the raw SE.
  - The prediction CI now triggers on `families.yaml`.
  - The embodiment README documents the fail-open Redis read.
  - Blocks not aligned to the 5-min grain is a deliberate deviation (listed above).
- **Also fixed before the review:** the effect-table permutation is drawn once per run and vectorised. A full-size build (50 targets, 2000 draws) went from 29 s to 3 s.

After the fixes, the synthetic eval with the realistic town gives:
- blocks per night: 3.20 ON, 2.85 OFF;
- coverage 0.94 for both watts and tier-1 hold;
- null p ≤ 0.05 in 6 % of runs.

That is about 50 ON and 46 OFF blocks over 16 nights. The GPU3-watt MDE is then about 8.5 W, still under the 10 W bar.

## Restart required

After merge, from the clean primary main checkout. Embodiment must go first, so it obeys the key before a run can start:

```bash
scripts/safe_docker_build.sh orion-embodiment up -d --build
scripts/safe_docker_build.sh orion-prediction up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Start a run only when Juniper schedules it:

```bash
curl -fsS -X POST http://localhost:8132/experiments/town-hold/run -H 'content-type: application/json' -d '{"nights":16}'
```

## Risks / concerns

- **Severity: medium. Concern: a persisted schedule resumes after a restart.** That is the design's "a rebuild costs one night". **Mitigation:** only an explicitly started run resumes. `POST /experiments/town-hold/abort` deletes it, and `PREDICTION_TOWN_HOLD_ENABLED=false` stops the resume.
- **Severity: low. Concern: the bid's hunger comes from `opened_at` = the previous decision, capped at one block.** A want opened at the bid time reads `sated` in the real market (a test pins both cases). **Mitigation:** none needed. The tests pin the behaviour.
- **Severity: low. Concern: Kev latency is too noisy to claim anything.** **Mitigation:** it is reported, not claimed, as the design says.
- **Issues:** opened #3549 (Experiment 2, later; follow-up) and #3557 (verify-live: the first night; needs-juniper to schedule the run). None closed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3556
