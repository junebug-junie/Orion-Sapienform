# PR 3 of the ladder retirement: orion-embodiment starts Orion's AI Town sessions on its own timer

## Summary

- Orion can visit AI Town for up to 2 hours a day ("sessions"). Until now, only the proposal -> policy ->
  execution-dispatch ladder could start one: a `aitown_session` candidate, the allocator's `play` act,
  then a bus RPC to the session controller in orion-embodiment. Over the session era (10-30 to 11-05)
  that was 55 start requests for 25 sessions.
- This PR gives orion-embodiment its own timer (`services/orion-embodiment/app/town_self_start.py`),
  next to the session controller it calls. It ticks every 300 s, dispatch's own ask cadence. Each tick
  places the shadow `aitown.session` bid first, then asks the controller, which still runs every cap.
- The timer also carries over the two gates dispatch held for this act: the self-monitor act gate (E1)
  and dispatch's refusal hold (a refusal holds for the controller's own `retry_after_sec`).
- The old trigger is removed in the same PR: the `aitown_session` template, its dispatch route and act
  declaration, the `orion.aitown_session.start.v1` executor key, `app/aitown_session_dispatch.py`, the
  `orion:aitown_session:start:*` channel pair, its two schemas, and the dispatch keys
  `ORION_DISPATCH_AITOWN_SESSION_*`. AI Town was the last route with the act-duration cost rule, so
  that rule goes too. AI Town has one trigger.
- Juniper's manual Unfreeze is untouched and still never counted or absorbed.
- A replay of the session era runs the shipped timer and controller on the real record: 4 sessions on
  every full local day, against a real range of 1-5 (4-5 without the deploy day).

Spec: `docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md` ("PR 3. AI Town").

## Outcome moved

- **AI Town no longer depends on the ladder.** PR 6 can retire proposal, policy and dispatch without
  silencing the town.
- **The want comes before the act.** The bid used to be placed only after a dispatch RPC arrived. Now it
  is placed on embodiment's own clock, before the caps.
- **A silent hold is visible.** `GET /health` -> `town_self_start` shows the last tick, the last start,
  the current hold and the count of each refusal.
- **orion-embodiment has CI.** No workflow ran this service's tests before. `orion-embodiment-tests`
  runs them, the ledger arithmetic, and the fixture replay.

## Current architecture (before this PR)

- orion-proposal-runtime offered `aitown_session` (no dimensions) on every warranted tick (in practice
  on almost every dispatch frame). orion-policy-runtime approved it. orion-execution-dispatch-runtime
  scored it as the allocator's `play` act (`aitown.session`), charged the act's measured duration (or
  the 1800 s session cap while cold), held the self-monitor gate, and sent one bus RPC. After a
  refusal it held the route for the controller's `retry_after_sec` (300 s with none).
- On most frames the allocator refused it `below_information_floor`. That floor, not the caps, decided
  when in the day a session started.
- orion-embodiment's `TownSessionController.start` placed the shadow bid, ran the caps, resumed the
  town and opened the `play` act. Its 10 s tick enforces the session cap and the daily cap and closes
  the act.

## Architecture touched

How one tick works now (orion-embodiment, every 300 s; the first tick 60 s after startup):

1. An Orion session is open: do nothing (`session_running`), no bid.
2. `EMBODIMENT_TOWN_SESSIONS_ENABLED=false`: do nothing (`disabled`), no bid.
3. The controller's last refusal still holds for its own `retry_after_sec`: do nothing (`held`), no bid.
   After `daily_cap` the hold lasts until local midnight; after `cooldown`, the rest of the 1 h gap.
4. The self-monitor gate (E1). A faculty held broken or unreliable is refused
   (`self_monitor:track_record_failing`), no bid. One re-test probe per 6 h, spent only when a session
   really starts. The "last real attempt" read comes from the controller's own session ledger (every
   Orion row is a play act this controller opened), so the service needs no Postgres.
5. Mint the act id `aitown_self_start:<uuid>`. Place the shadow bid (`want_id` = the session id).
6. Call `controller.start_session(act_id)`. The controller runs the caps (kill switch, 7200 s per
   America/Denver day, 3600 s gap, one session at a time, Juniper's manual session), resumes the town
   and opens the `play` act.

A dependency mid-restart is a recorded tick, never a hold. `ledger_unknown` (Redis, or the controller's
own boot race after a restart) and an exception are retried 60 s later. `town_unreachable` and
`resume_failed` (Convex) wait the full 300 s tick, as dispatch's hold did: each retry is another slow
Convex call and another bid.

Why orion-embodiment and not elsewhere: the controller that runs every cap, owns the session ledger and
opens and closes the act already lives there. A timer anywhere else would need a new RPC to reach it,
which is the seam this PR removes.

## How the act ledger keeps grading AI Town

- **Who opens and closes the act.** The controller, as before (`town_sessions._act_event`). Dispatch
  never opened or closed it; it only declared the act on the route for the allocator's score, and that
  declaration goes with the route.
- **The act key.** `key` is the act id `aitown_self_start:<uuid>` (it was the dispatch id).
  `test_before` stays `aitown_session_dispatch:<act id>`: the prefix is kept, so every
  `aitown.session` act has one shape. `object_ref` stays `aitown_session:<session id>`, and the
  correlation id is the session id.
- **The outcome.** Unchanged: `artifact_produced`, value 1 when Orion spoke at least one line in town,
  0 otherwise, closed at the session cap, the daily cap, a Freeze, the switch, or a restart
  (`embodiment_restart`, at the last persisted tick).
- **The self-monitor faculty.** Unchanged: `act.aitown.session.play` (live: 25 graded bets in 7 days,
  status `unknown`). `orion/self_monitor/replay.py` now names its seam `town_self_start`.
- **The proven-effect holdback.** Live `substrate_act_episodes` holds 0 `aitown.session` rows. Nothing
  moves.
- **Proof.** `test_a_tick_bids_first_then_starts_and_opens_the_play_act` drives one tick to a started
  session and checks the `act_opened` event (actor, verb, `test_before`, `object_ref`, correlation id).
  `test_the_session_cap_closes_the_act_with_a_real_outcome` checks the `act_closed` event and its value.
  The controller's own tests still cover the 0 close, the daily-cap close and the restart close.

## Clock and freshness (the #3134 lesson)

- The timer reads its clock first and uses it only as a label.
- The controller now reads its ledger first and takes its clock after the read
  (`TownSessionController._start`). Before this PR it took the clock first. That was safe only because
  this process is the ledger's only writer.
- A session's `started_at` is the time the town was really resumed (after up to two slow Convex calls),
  not the gate's clock.
- The self-monitor view is aged after its GET (`read_view_sync(now=None)`), and the E1 gate's clock is
  taken after that read.
- Regression tests use the live order, clock first and then a 2.4 s read lag:
  `test_the_gap_is_aged_at_the_ledger_read_not_at_the_tick_clock`,
  `test_a_session_written_after_the_ticks_clock_is_never_a_negative_age`,
  `test_the_live_view_reader_ages_the_view_at_read_time`.

## Replay: the session era (asked 14 days)

`python -P services/orion-embodiment/evals/town_self_start_replay.py --live --days 14`. Window
2026-10-30 05:50 to 2026-11-05 17:44 UTC. The controller's first session bounds it: before 10-30 the town
ran all day with no sessions, so 14 days hold about 6.5 days of session history.

It runs the shipped `TownSelfStart.tick` every 300 s and the shipped controller tick every 10 s on a
simulated clock. Every ledger read comes after a read lag of 0.3-2.4 s. The world is the real record:
Juniper's manual sessions (the town runs), every `embodiment_restart` (the process restarts), and
dispatch's `rpc_error` replies (the town is unreachable for 60 s).

| Local day (America/Denver) | New timer | Real (dispatch) |
|---|---|---|
| 10-29 (partial) | 1 | 1 |
| 10-30 (deploy day) | 4 | 1 |
| 10-31 | 4 | 4 |
| 11-01 | 4 | 4 |
| 11-02 | 4 | 4 |
| 11-03 | 4 | 5 |
| 11-04 | 4 | 4 |
| 11-05 (partial) | 4 | 2 |
| **Total** | **29** | **25** |

- Full days: real range 1-5 (4-5 without the deploy day). New: 4 on every full day. **IN RANGE.**
- Orion's town hours: new 14.1, real 11.3.
- Timer: 29 started, 159 `session_running`, 32 refused (22 `cooldown`, 10 `daily_cap`), 1,652 `held`;
  61 bids (one per ask, none while held or in session).
- Gates, all PASS: never over 7200 s per local day (max 7,208 s, one 10 s controller tick); never two
  inside the 3600 s gap (min 3,601 s); no session past 1800 s (max 1,809 s); never resumed the town while
  Juniper's session was open; no refusal for a read age.

**What changes, said plainly: the time of day.** The count per day is set by the daily cap, not by the
cadence. The real sessions were spread through the day by the allocator's information floor. The timer
asks as soon as the caps allow, and the cap resets at local midnight, so Orion's 4 sessions now fall
between about 00:00 and 05:00 local:

| Start band (local) | 00-06 | 06-12 | 12-18 | 18-24 |
|---|---|---|---|---|
| New timer | 27 | 1 | 0 | 1 |
| Real | 7 | 11 | 5 | 2 |

No cadence fixes this. The same replay at 1800, 3600, 5400 and 7200 s still gives 4 sessions on every
full day, and moves the starts only a few hours later (5400 s: 15 of 29 before 06:00, the rest
06:00-12:00). See "Risks / concerns".

The fixture replay (CI) passes 12 of 12 checks on two scripted days: the daily cap binds (4 x 30 min);
never over the cap; never inside the gap; never past the session cap; never resumes over Juniper's
session; a restart closes the open session `embodiment_restart`; the first open window starts on the
first tick; a daily-cap refusal is held, not re-asked; bids only on asks; no read-age refusal.

What the replay does not model: the self-monitor gate (the faculty reads `unknown`, which admits),
Convex latency, utterances, restarts the ledger did not record.

## Metric quality gate

No new metric, signal or salience. The bid is the existing `aitown.session` shadow want, with the same
family, verb, tier and producer (`orion-embodiment.town_sessions`). Only its timing changes: before the
caps, on embodiment's clock. The `/health` counters are a debug surface, not a signal. The replay is an
eval.

## Files changed

- `services/orion-embodiment/app/town_self_start.py`: new. The timer, the refusal hold, `build_act_gate`
  and the `/health` snapshot.
- `services/orion-embodiment/app/town_sessions.py`: `start(req)` and the RPC `handle` replaced by
  `start_session(act_id)` and an in-process `StartOutcome`; the clock is taken after the ledger read;
  `act_attempt_rows` (the E1 ledger read); docstring.
- `services/orion-embodiment/app/worker.py`: starts and stops the timer instead of the RPC subscription.
- `services/orion-embodiment/app/main.py`: `/health` -> `town_self_start`.
- `services/orion-embodiment/app/settings.py`, `.env_example`, `docker-compose.yml`: four new keys.
- `services/orion-embodiment/app/physiology_shadow.py`, `README.md`: who starts a session now.
- `services/orion-embodiment/tests/test_town_self_start.py` (19 tests),
  `tests/test_town_self_start_health.py` (1): new. `tests/test_town_session_controller.py`: the new start
  API; the RPC test removed.
- `services/orion-embodiment/evals/town_self_start_replay.py`: new.
- `.github/workflows/orion-embodiment-tests.yml`: new.
- `services/orion-execution-dispatch-runtime/app/aitown_session_dispatch.py`: deleted.
  `app/worker.py`: no mixin, hold, send branch or act-duration cost branch. `app/settings.py`,
  `.env_example`, `docker-compose.yml`, `README.md`: the two keys removed.
  `tests/test_custom_route_result_status.py`: the AI Town case removed.
- `orion/execution_dispatch/policy.py`: `AITOWN_SESSION_EXECUTOR_VERB`, `cost_source` and
  `cold_cost_ceiling_sec` removed. `orion/autonomy/act_candidates.py`: the same two fields off `ActRef`.
- `orion/schemas/aitown_session.py`: the start request/result models, channel and kind constants
  removed (the ledger contract stays). `orion/schemas/registry.py`: the two schemas removed.
- `orion/bus/channels.yaml`: the `orion:aitown_session:start:*` pair removed (a removal note stays).
- `config/metrics/metric_definitions.lock.json`: re-locked (the two channel metrics removed).
- `config/proposals/proposal_policy.v1.yaml`: the template removed.
  `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: the route removed.
- `config/physiology/families.yaml`, `contract.yaml`: where the bid is placed now.
- `orion/self_monitor/replay.py` (+ test): the play act's seam is `town_self_start`.
- `services/orion-ai-town/mcp/orion_aitown_mcp/server.py`: the refusal text names the timer.
- `orion/mindloop/runtime.py`: a docstring no longer cites the removed `cold_cost_ceiling_sec` as live.
- `tests/test_rung3_aitown_session.py`: now pins that every piece of the old trigger is gone (6 tests).
  `tests/test_rung3_retro_leisure.py`, `tests/test_express_outward_action.py`,
  `tests/test_proposal_scoring.py`: the AI Town references. `.github/workflows/thinking-rung3.yml`: the
  step name.

## Schema / bus / API changes

- Added: `/health` key `town_self_start` on orion-embodiment.
- Removed: channels `orion:aitown_session:start:request` and `orion:aitown_session:start:reply:*`;
  schemas `AITownSessionStartRequestV1`, `AITownSessionStartResultV1`; executor key
  `orion.aitown_session.start.v1`; template and route `aitown_session`; the `ActRouteRefV1` fields
  `cost_source` and `cold_cost_ceiling_sec` (no route sets them).
- Behavior changed: the play act key and `dispatch_id` ledger field are `aitown_self_start:<uuid>`
  (were the dispatch id). Bids happen on the timer's asks, not when a dispatch arrives. A session's
  `started_at` is the resume time.
- Compatibility: `AITownSessionRowV1` is unchanged (`start_reason` stays `want_funded`), so
  orion-durable-runs' `wants.town` report reads every row, old and new. The only producer (dispatch)
  and the only consumer (embodiment) of the removed channel pair lose it in the same PR.

## Env/config changes

- Added keys (orion-embodiment): `EMBODIMENT_TOWN_SELF_START_ENABLED=true`,
  `EMBODIMENT_TOWN_SELF_START_TICK_SEC=300`, `SELF_MONITOR_ACT_GATE_ENABLED=true`,
  `SELF_MONITOR_PROBE_INTERVAL_SEC=21600`. Flags ship ON.
- Removed keys (orion-execution-dispatch-runtime): `ORION_DISPATCH_AITOWN_SESSION_ENABLED`,
  `ORION_DISPATCH_AITOWN_SESSION_RPC_TIMEOUT_SEC`.
- `.env_example` updated: yes, both services.
- Local `.env` synced: the main checkout's `services/orion-embodiment/.env` got the 4 keys by hand (the
  sync script cannot reach the main checkout from a worktree; its `EMBODIMENT_` and `SELF_MONITOR_`
  prefixes already cover them).
- Operator action: the two removed keys are still in the main checkout's
  `services/orion-execution-dispatch-runtime/.env`. Delete them after the deploy (command below). No code
  reads them after this PR: the settings fields and the compose lines are gone, and a repo grep finds no
  reader outside docs and the test that pins their absence.

## Tests run

```text
services/orion-embodiment/tests (whole folder)                            142 passed (19 timer, 1 /health, 20 controller)
services/orion-embodiment/tests in a fresh venv (requirements.txt + pytest + pytest-asyncio,
  the new CI job's install)                                               141 passed (before the probe test)
orion/embodiment/tests/test_town_sessions.py                              11 passed
services/orion-execution-dispatch-runtime/tests                           120 passed
tests/test_rung3_*.py, test_motor_allocator.py, test_self_monitor_act_gate.py,
  test_express_outward_action.py, test_proposal_scoring.py, orion/self_monitor/tests,
  orion/body_atlas/tests                                                  560 passed, 1 skipped
CI workflow commands run locally (pip installs skipped): orion-static-gates (all OK after the re-lock),
  agency-ask-episodes, attend-act-loop, body-atlas-tests, body-economy-analysis: OK.
  mind-tier-by-caller-tests: steps use ${{ github.workspace }}, which a local shell cannot expand.
  orion-attention-runtime-tests: the step passed when run by hand (101 passed, 5 skipped).
  The remaining hit workflows were left to CI (see "Risks / concerns" on the local runner).
scripts/check_definition_drift.py --gate  PASS (re-locked: the two channel metrics removed)
scripts/check_env_template_parity.py      PASS
```

## Evals run

```text
python -P services/orion-embodiment/evals/town_self_start_replay.py                    12/12 PASS (fixture)
python -P services/orion-embodiment/evals/town_self_start_replay.py --live --days 14    29 vs 25; 4 per full day, real 1-5: IN RANGE
  ... --tick-sec 1800 / 3600 / 5400 / 7200                                              still 4 per full day (calibration above)
python orion/body_atlas/evals/run_wire_eval.py      true facts 13/13, false 0/11, 4/4 mutations CAUGHT
python orion/body_atlas/evals/run_organ_structure_eval.py / run_capacity_roles_eval.py /
  run_store_direction_eval.py                                                           PASS
```

## Docker/build/smoke checks

```text
docker build -f services/orion-embodiment/Dockerfile .                       OK
docker build -f services/orion-execution-dispatch-runtime/Dockerfile .       OK
in the embodiment image (live .env files, no network): app.main imports; the timer reads enabled, 300 s,
  the E1 gate on, 21600 s; /health renders town_self_start
in the dispatch image (live .env, which still holds the two removed keys): the worker's bases are
  (object,); the routes are analyze_self_study_source, observe_tension_via_camera,
  prune_dangling_images, prune_stopped_containers, read_world_pulse
```

UNVERIFIED on the live rail: no self-started session yet (not deployed).

## Review findings fixed

A code-review subagent read the whole diff, ran every suite and both replays, mutation-tested the clock
order (moving the clock back before the ledger read fails both read-lag tests), and checked the live
self-monitor view. No blockers.

- Finding (should-fix, needs Juniper): Orion's town time moves to the night (27 of 29 replay starts
  before 06:00 local).
  - Fix: not changed in this PR. Adding a start-hour window or a spread rule is a new gate, outside
    the approved "start under the existing caps". It is named in "Risks / concerns" with a
    recommendation, and in the final status as a concern for Juniper to decide before merge.
- Finding (should-fix): the PR report was not committed.
  - Fix: committed on the branch (this file).
- Finding (nit): no test proved a probe is spent only by a real start.
  - Fix: `test_a_probe_is_spent_only_by_a_real_start` (a probe offered to an attempt the caps refuse
    stays unspent; a started one is claimed).
- Finding (nit): during a Convex outage the timer asked every 60 s, 5 times dispatch's rate, each with
  another Convex call and another bid.
  - Fix: only `ledger_unknown` and exceptions retry at 60 s; `town_unreachable` and `resume_failed` wait
    the full 300 s tick. Evidence: `test_convex_down_waits_the_full_tick_like_dispatch_never_held`.
- Finding (nit): `orion/mindloop/runtime.py` cited the removed `cold_cost_ceiling_sec` as a live rule.
  - Fix: the docstring states the rule itself and names the retired field as history.

## Restart required

Deploy from the clean primary `main` checkout after merge. The post-pull rebuild runs in alphabetical
order: orion-embodiment, then orion-execution-dispatch-runtime, then orion-proposal-runtime (other
services it rebuilds for the `orion/` change do not touch AI Town).

1. **orion-embodiment rebuilds.** The RPC listener is gone and the timer starts. An open Orion session
   is closed `embodiment_restart` at its last tick (its act closes), and the town is paused, as on any
   restart. The first tick comes 60 s after startup; if Redis or Convex is mid-restart, that tick is
   refused by name and the next comes 60 s later. Until dispatch rebuilds, the old dispatch still sends
   start RPCs, but nothing listens: each one times out after 60 s and dispatch holds 300 s. They start
   nothing, so the timer is the only trigger.
2. **orion-execution-dispatch-runtime rebuilds.** The route is gone. Until proposal-runtime rebuilds,
   it still offers the old template, and dispatch blocks it (no route). Not a second trigger.
3. **orion-proposal-runtime rebuilds.** The template stops.

There is never a time with two triggers, in any order: the RPC listener leaves embodiment in the same
image the timer arrives in. If dispatch rebuilt first, there would be no trigger until embodiment
rebuilt (at most a late session). If proposal-runtime rebuilt first, dispatch would have nothing to send.

By hand, the same order:

```bash
scripts/safe_docker_build.sh orion-embodiment up -d --build
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

After the three are up, delete the stale keys from the main checkout's local `.env` (one line):

```bash
sed -i '/^ORION_DISPATCH_AITOWN_SESSION_ENABLED=\|^ORION_DISPATCH_AITOWN_SESSION_RPC_TIMEOUT_SEC=/d' services/orion-execution-dispatch-runtime/.env
```

Live check after the deploy: `curl -fsS http://localhost:8130/health | jq .town_self_start` shows
`running: true` and `ticks` rising every 5 minutes, with refusals by name. Within a day: Redis `orion:aitown:sessions` rows with `dispatch_id` starting
`aitown_self_start:`, `grammar_acts` rows with `actor = 'aitown.session'` and `test_before LIKE
'aitown_session_dispatch:aitown_self_start:%'`, each closed with a value, and 0 new
`substrate_dispatch_results` rows with settlement kind `aitown_session`.

## Risks / concerns

- Severity: medium. Concern: Orion's town time moves to the night. The replay puts 27 of 29 starts
  between 00:00 and 06:00 local (real: 7 of 25). The cap resets at local midnight and the timer asks as
  soon as the caps allow; the allocator's information floor used to spread the starts. No cadence fixes
  it. Mitigation: none in this PR (the approved plan is "start under the existing caps"). Juniper's
  call: (a) accept it until body-economy Patch D makes the bid binding, which lets the market place the
  want against the rest of the day's load; or (b) add one local "not before" hour to the controller's
  gate. Recommendation: (a), unless Juniper wants to watch Orion in town by day; then (b) is a small,
  testable follow-up.
- Severity: low. Concern: about 25 % more town hours (14.1 against 11.3 in the window), because the cap
  now fills every day. The NPC dialogue runs on circe's `fast` lane. Mitigation: the hard cap is
  unchanged (2 h per local day), and Juniper's time is not counted.
- Severity: low. Concern: while running the hit CI workflows locally, the runner executed their
  multi-line `pip install` steps into the shared `/mnt/scripts/Orion-Sapienform/.venv` (17:44-17:51 UTC):
  about 40 packages were re-pinned to service requirement versions. Mitigation: pydantic and
  pydantic-core were put back to the versions in use before (2.14.3 / 2.50.4, read from the
  mismatch error), orjson and uvicorn raised to what `pip check` requires; `pip check` is clean and
  every suite above was re-run after the repair. The earlier versions of the other packages are not
  known. Recorded on the agent board.
- Severity: low. Concern: the dispatch package keeps the `express` kind and scope with no route left.
  Mitigation: PR 6 retires the whole package; `test_express_outward_action.py` still pins the scope rule.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3143

🤖 Generated with [Claude Code](https://claude.com/claude-code)
