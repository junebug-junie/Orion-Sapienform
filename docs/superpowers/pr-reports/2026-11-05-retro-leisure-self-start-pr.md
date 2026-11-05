# PR 1 of the ladder retirement: Hub starts Orion's Zelda leisure on its own timer

## Summary

- Orion can play Zelda alone when nobody is around ("solo leisure"). Until now, only the
  proposal -> policy -> execution-dispatch ladder could start a session. In practice the ladder was a
  5-minute poller: 2,406 start requests in 14 days gave 4 sessions.
- This PR gives Hub its own timer for it (`services/orion-hub/scripts/retro_leisure_self_start.py`). It
  ticks every 300 s, the ladder's real cadence. Each tick asks the leisure runner's own gate
  (`retro_leisure.should_start`), and starts a session only when every gate passes.
- Before the gate, each tick places the shadow bid for `hub.retro_leisure` in the body-economy market.
  The market now sees the want, not the aftermath. The bid is logged, never obeyed.
- The timer also holds the self-monitor act gate (E1) that dispatch held for this act. Without it, the
  move would have dropped that gate silently.
- The old trigger is removed in the same PR: the `retro_leisure_session` template, its dispatch route
  and act declaration, `app/retro_leisure_dispatch.py`, the `orion:retro_leisure:start:*` RPC channel pair,
  its two schemas, and the dispatch keys `ORION_DISPATCH_RETRO_LEISURE_*`. Leisure has one trigger.
- A replay of the last 14 days runs the shipped timer on the real record. It starts 4 sessions, on the
  same days and within 4 minutes of the real 4.

Spec: `docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md` ("PR 1. Retro play").

## Outcome moved

- **Leisure no longer depends on the ladder.** PR 6 can retire proposal, policy and dispatch without
  silencing play.
- **The want comes before the act.** Before, the bid was placed inside Hub only after a dispatch arrived.
  Now it is placed on Hub's own clock, before the gate.
- **A silent hold is visible.** `GET /health` -> `retro_leisure_self_start` shows the last tick, the last
  start, and the count of each refusal reason.

## Current architecture (before this PR)

- orion-proposal-runtime offered the `retro_leisure_session` template (no dimensions) on every warranted
  tick. orion-policy-runtime approved it. orion-execution-dispatch-runtime scored it as the allocator's
  `play` act (`hub.retro_leisure`), held the self-monitor gate, and sent a bus RPC
  (`orion:retro_leisure:start:request`). After a refusal it held the route 300 s.
- Hub's `retro_leisure_listener.py` placed the shadow bid, asked `should_start`, started
  `run_leisure_session`, and opened the play act. `retro_memory_export.py` closed it at session end.
- `retro_leisure.py` said "There is no scheduler here."

## Architecture touched

How one tick works now (Hub, every 300 s; the first tick 60 s after startup):

1. A session this Hub started still runs: do nothing (`session_running`).
2. `RETRO_LEISURE_ENABLED=false`: do nothing (`disabled`). No runner in the image: `runner_unavailable`.
   No bid in either case, and never a fake session.
3. The self-monitor gate (E1). A faculty held broken or unreliable is refused
   (`self_monitor:track_record_failing`), with no bid. One re-test probe per 6 h, spent only when a
   session really starts.
4. Mint the act id `retro_self_start:<uuid>`. Place the shadow bid (`want_id` = the act id).
5. Ask `should_start`. A refusal ends the tick with its own name.
6. Start `run_leisure_session(act_id=)`. A refusal from its own re-check within 5 s is still a refusal.
7. Open the `play` act and follow the session. The export closes the act at the end.

An error (a dependency mid-restart, a gate exception) is a recorded tick, and the next tick comes 60 s
later instead of 300 s.

## How the act ledger keeps grading play

- **Who opens and closes the act.** Hub, as before: `scripts/act_emit.py` (`publish_play_opened`,
  `publish_play_closed`). Dispatch never opened or closed it. It only declared the act on the route for
  the allocator's score, and that declaration goes with the route.
- **The act key.** The act id is `retro_self_start:<uuid>` (it was the dispatch id). `test_before` stays
  `retro_leisure_dispatch:<act id>`. The prefix is kept, so every `hub.retro_leisure` act has one shape.
- **The join to the session.** sql-writer's `act_downstream_use` joins
  `retro_play_sessions.dispatch_id = grammar_acts.correlation_id`. The export writes the act id into
  `dispatch_id`, and the act's correlation id is the act id. The join is unchanged.
- **Proof.** `test_a_session_opens_the_play_act_and_its_end_exports_and_closes_it` drives one tick to a
  started session. It asserts one `act_opened` (actor, verb, `test_before`, correlation id, 2,400 s
  horizon) and, after the session ends, one `act_closed` on the same act with value 1.
  `test_a_session_that_ends_without_data_closes_the_act_at_zero` covers the 0 close.
- **The proven-effect holdback.** Dispatch never drew a control arm for this act (`object_source:
  dispatch`). Live `substrate_act_episodes` holds 0 retro rows. Nothing moves.

## Clock and freshness (the #3134 lesson)

#3134 stopped every painting: the tick took `now` before the read, and a check that refuses a negative
age read every state as stale. Here:

- The tick reads its clock first and uses it only as a label, and to age past attempts in the
  self-monitor gate (always in the past).
- Every age of a read value is taken at the read time: the regulation key (`read_regulation`, after
  its GET), the presence store (after its GET), and the self-monitor view (`read_view_sync(now=None)`,
  new, after its GET).
- Regression tests use the live order: clock first, then the reads, with a real lag.
  `test_a_state_written_after_the_ticks_clock_still_reads_fresh`,
  `test_the_real_gate_passes_with_reads_after_the_tick_clock` (the real `should_start`),
  `test_a_view_published_after_the_ticks_clock_is_fresh_not_unknown`,
  `test_the_live_view_reader_ages_the_view_at_read_time`.

## Replay: 14 days of real history

`python -P services/orion-hub/evals/retro_leisure_self_start_replay.py --live --days 14`, window
2026-10-22 09:02 to 2026-11-05 09:02 UTC.

It runs the shipped `RetroLeisureSelfStart.tick` every 300 s. Each tick calls the shipped gate
(`should_start` -> `gather_gate_inputs` -> `leisure_block_reason`) and the runner's re-check on a
simulated clock. Every read comes after a read lag of 0.3-2.4 s. The world comes from the only real
record of the gate over time: the old route's 2,406 start replies (the runner's own verdict each time).
Each read takes the nearest verdict within 450 s.

| | New timer | Real (dispatch) |
|---|---|---|
| Sessions, 14 days | **4** | **4** |
| Per day | 10-23: 1, 10-27: 2, 10-30: 1 | 10-23: 1, 10-27: 2, 10-30: 1 |
| Start times | 16:52, 06:42, 16:17, 02:07 | 16:53, 06:41, 16:18, 02:10 |
| Gate asks / bids | 4,032 / 4,032 | 2,406 sends |
| retro_session_active | 2,223 | 1,718 |
| arousal_strained | 552 | 431 |
| human_recent | 103 | 82 |
| browser_connected | 101 | 80 |
| cabinet_hot | 89 | 72 |
| cooldown | 8 | 3 |
| arousal_unknown | 3 | 2 |
| no verdict near the tick (`presence_unknown`) | 949 ticks | 14 `rpc_error` |

The refusal reasons are the same set. Acceptance: PASS.

Sensitivity: with `--carry` (a stretch with no verdict keeps the last verdict), the timer still starts
the same 4 sessions.

**What this proves, and what it does not.** The record shows an open gate only 7 times in 2,406
verdicts (4 starts, 3 cooldowns). The replay can start only where the record shows an open gate. So
this is a "no regression in timing" check: the timer loses none of the real starts and adds none. It is
not a "plays more" check.

The fixture replay (CI) passes 9 of 9 checks on two scripted days: starts only in open windows; never
two inside the 3 h cooldown; the 90 min rolling cap binds (3 sessions of 30 min); no start while
Juniper's game runs; no start on an unobserved stretch; the first open window starts at once; a
running session holds the timer (36 ticks, no bid, no second start); the refusals carry the doer's
own names; no refusal for a read age.

What the replay does not model: the self-monitor gate (the faculty reads `unknown`, which admits); a
gate hidden behind the first refusal; the session's own heat.

## Found, not fixed (pre-existing, outside this PR)

- **All 4 real sessions did nothing.** Each one lasted 0.7-1.0 s, ended `policy_ended_by_retro` with 0
  planner decisions, and its play act closed at 0. The trigger is not the problem: Retro drops the
  leisure policy at once.
- **A paused game blocks leisure.** Since 10-25, `retro_session_active` is 71 % of all refusals. A
  read-only gate check from the live Hub container today read `retro paused`. A paused game left in
  Retro blocks every start.

Both need their own investigation. This PR changes neither.

## Metric quality gate

No new metric, signal or salience. The bid is the existing `hub.retro_leisure` shadow want, with the
same family, verb, tier and producer (`orion-hub.retro_leisure`). The only change is when it is placed:
before the gate, on Hub's clock. The `/health` counters are a debug surface, not a signal. The replay is
an eval.

## Files changed

- `services/orion-hub/scripts/retro_leisure_self_start.py`: new. The timer, `build_act_gate` and the
  `/health` snapshot.
- `services/orion-hub/scripts/retro_leisure_listener.py`, `tests/test_retro_leisure_listener.py`: deleted.
- `services/orion-hub/scripts/main.py`: starts and stops the timer instead of the listener.
- `services/orion-hub/scripts/api_routes.py`: `/health` -> `retro_leisure_self_start`.
- `services/orion-hub/scripts/act_emit.py`: `PLAY_OBJECT_PREFIX` and `play_object_ref` (moved from the
  deleted schema module).
- `services/orion-hub/app/settings.py`, `.env_example`: `RETRO_LEISURE_SELF_START_ENABLED`,
  `RETRO_LEISURE_SELF_START_TICK_SEC`, `SELF_MONITOR_ACT_GATE_ENABLED`, `SELF_MONITOR_PROBE_INTERVAL_SEC`.
- `services/orion-hub/scripts/retro_leisure.py`, `retro_memory_export.py`, `retro_session_end.py`,
  `README.md`, `docs/runbooks/retro-browser.md`: who starts a session now.
- `services/orion-hub/tests/test_retro_leisure_self_start.py` (32 tests),
  `tests/test_retro_leisure_self_start_health.py` (2): new.
- `services/orion-hub/tests/test_physiology_shadow_bid.py`: the two leisure bid tests run on the timer.
- `services/orion-hub/evals/retro_leisure_self_start_replay.py`: new.
- `services/orion-execution-dispatch-runtime/app/retro_leisure_dispatch.py`: deleted.
  `app/worker.py`: no retro mixin, hold or send branch. `app/aitown_session_dispatch.py`: now owns
  `_act_duration_or_ceiling_cost` (AI Town is its only user). `app/settings.py`, `.env_example`,
  `docker-compose.yml`, `README.md`: the two retro keys removed.
  `tests/test_custom_route_result_status.py`: the retro case removed.
- `orion/execution_dispatch/policy.py`: `RETRO_LEISURE_EXECUTOR_VERB` removed.
- `orion/schemas/retro_leisure.py`: deleted. `orion/schemas/registry.py`: its two schemas removed.
- `orion/bus/channels.yaml`: the `orion:retro_leisure:start:*` pair removed (a removal note stays).
- `config/metrics/metric_definitions.lock.json`: re-locked (`check_definition_drift.py --update`; the two
  channel metrics are removed).
- `config/proposals/proposal_policy.v1.yaml`: the template removed.
  `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: the route removed.
- `orion/self_monitor/gate.py`: `last_attempt_sql("asyncpg")`; `read_view_sync(now=None)` ages the view
  at read time. `orion/self_monitor/replay.py` (+ test): the play act's gate seam is
  `retro_leisure_self_start`.
- `scripts/sync_local_env_from_example.py`: syncs `RETRO_LEISURE_SELF_START_*`.
- `tests/test_rung3_retro_leisure.py`: now pins that every piece of the old trigger is gone (5 tests).
  `tests/test_express_outward_action.py`, `tests/test_proposal_scoring.py`,
  `tests/test_rung3_aitown_session.py`: comments and allowlists.
- `.github/workflows/retro-memory-tests.yml`: path filters, the new tests, the fixture replay.
  `.github/workflows/thinking-rung3.yml`: the deleted schema path removed.

## Schema / bus / API changes

- Added: `/health` key `retro_leisure_self_start` on orion-hub.
- Removed: channels `orion:retro_leisure:start:request` and `orion:retro_leisure:start:reply:*`; schemas
  `RetroLeisureStartRequestV1`, `RetroLeisureStartResultV1`; executor key `orion.retro_leisure.start.v1`;
  template and route `retro_leisure_session`.
- Behavior changed: the play act id and the want id are `retro_self_start:<uuid>` (were the dispatch id).
  Bids happen every 300 s on Hub's clock, not when a dispatch arrives.
- Compatibility: no `extra="forbid"` model gained a field. The only producer (dispatch) and the only
  consumer (Hub) of the removed channel pair lose it in the same PR.

## Env/config changes

- Added keys (orion-hub): `RETRO_LEISURE_SELF_START_ENABLED=true`, `RETRO_LEISURE_SELF_START_TICK_SEC=300`,
  `SELF_MONITOR_ACT_GATE_ENABLED=true`, `SELF_MONITOR_PROBE_INTERVAL_SEC=21600`. Flags ship ON.
- Removed keys (orion-execution-dispatch-runtime): `ORION_DISPATCH_RETRO_LEISURE_ENABLED`,
  `ORION_DISPATCH_RETRO_LEISURE_RPC_TIMEOUT_SEC`.
- `.env_example` updated: yes, both services.
- Local `.env` synced: the main checkout's `services/orion-hub/.env` got the 4 keys by hand (the sync
  script cannot reach the main checkout from a worktree).
- Operator action: the two removed keys are still in the main checkout's
  `services/orion-execution-dispatch-runtime/.env`. Delete them after the deploy (command below). No code
  reads them after this PR: the settings fields and the compose lines are gone, and a repo grep finds no
  other reader.

## Tests run

```text
services/orion-hub/tests (whole folder)                          4602 passed, 150 skipped
services/orion-hub/tests/test_retro_leisure_self_start*.py       34 passed
services/orion-execution-dispatch-runtime/tests                  124 passed (128 before: 4 retro cases removed)
tests/test_rung3_retro_leisure.py, test_rung3_aitown_session.py, test_express_outward_action.py,
  test_proposal_scoring.py, orion/self_monitor/tests             187 passed
orion/body_atlas/tests                                           229 passed
CI workflow commands run locally (pip installs skipped): orion-static-gates (all OK), body-atlas-tests,
  dig-kit-tests, body-economy-analysis, attend-act-loop (all OK); thinking-rung3, retro-memory-tests,
  retro-preflight, self-monitor-tests: OK except steps that need the CI Postgres service, the real NES
  core, node, or a working directory this runner does not set. CI is the authority there.
scripts/check_env_template_parity.py      PASS
scripts/check_env_key_single_source.py    OK
scripts/check_body_economy_contract.py    OK (102 state-reading modules)
scripts/check_definition_drift.py --gate  PASS
scripts/check_bus_reply_channels.py       19 prefixes resolved, 0 uncovered
```

## Evals run

```text
python -P services/orion-hub/evals/retro_leisure_self_start_replay.py                  9/9 PASS (fixture)
python -P services/orion-hub/evals/retro_leisure_self_start_replay.py --live --days 14  4 vs 4, same reasons (PASS)
  ... --carry                                                                           4 vs 4 (PASS)
python orion/body_atlas/evals/run_wire_eval.py         true facts 13/13, false 0/11, 4/4 mutations CAUGHT
python orion/body_atlas/evals/run_organ_structure_eval.py / run_capacity_roles_eval.py /
  run_store_direction_eval.py                                                           PASS
```

The wire eval's hand-checked facts needed no change. None named the retro RPC. Hub already read
`orion:self_monitor:latest` (the self-monitor question minter), and the scanner still finds it.

## Docker/build/smoke checks

```text
docker build --build-arg INSTALL_DEV_DEPS=true -f services/orion-hub/Dockerfile .          OK
docker build -f services/orion-execution-dispatch-runtime/Dockerfile .                     OK
in the hub image (with the live .env files, no network): scripts.main imports; the runner is found;
  the timer reads enabled, 300 s, the E1 gate on; /health renders retro_leisure_self_start
in the dispatch image: the worker's bases are [AITownSessionDispatchMixin] (no retro mixin)
live Postgres: the asyncpg ledger query prepares and returns the 4 real play acts
live Hub container (old code, read-only): the gate reads arousal strained, Retro paused, 0 browsers
```

UNVERIFIED on the live rail: no self-started session yet (not deployed).

## Review findings fixed

A code-review subagent read the whole diff, ran the suites and the replay, and checked the live
self-monitor view and Postgres. No blockers.

- Finding (should-fix): the 4-vs-4 match is close to guaranteed, because the replay world is open only
  near the real starts. The 4 real sessions did nothing.
  - Fix: the replay prints the ceiling (7 open verdicts in 2,406). The eval docstring and this report
    say it is a no-regression check. The report names the ~1 s sessions and the post-deploy check.
- Finding (should-fix): a timed-out ledger read was left running and held a pool connection.
  - Fix: the future is cancelled on timeout. Evidence:
    `test_a_timed_out_ledger_read_is_cancelled_not_left_holding_a_connection`.
- Finding (nit): the self-monitor view was aged against the tick's clock (safe only by the 300 s skew).
  - Fix: `read_view_sync(now=None)` ages it after the GET; Hub's reader passes no clock. Evidence:
    `test_the_live_view_reader_ages_the_view_at_read_time`.
- Finding (nit): `claim_probe`'s result was ignored. Fix: a lost race is logged.
- Finding (nit): `/health` iterated the counters from another thread. Fix: the loop publishes a copy after
  each tick; `/health` only reads it.
- Finding (nit): the fixture never kept a session running past a tick. Fix: fixture sessions run 30 min
  on the replay clock (36 `session_running` ticks, with no bid on them).
- Finding (nit): the retro-memory `push` filter listed 2 files. Fix: it lists the timer, the runner,
  `act_emit`, the replay and the gate. The health test stays in the full Hub suite only: it imports the
  whole Hub app, which that job does not install.
- Found by the local CI run: the dispatch compose file still passed the two removed keys. Fix: removed.

## Restart required

Deploy from the clean primary `main` checkout after merge. The post-pull rebuild runs in alphabetical
order: orion-execution-dispatch-runtime, then orion-hub, then orion-proposal-runtime.

1. **orion-execution-dispatch-runtime rebuilds.** Dispatch can no longer send a leisure start.
   Until Hub rebuilds, nothing starts leisure (no trigger). Proposal-runtime still offers the old
   template, and dispatch blocks it `no_cortex_route_for_proposal_kind`. The gap lasts as long as the
   rebuilds in between. Leisure started 4 times in 14 days, so the gap costs at most a late session.
2. **orion-hub rebuilds.** The old listener is gone and the timer starts. Its first tick comes 60 s after
   startup. If Redis, Retro or the regulation key is still mid-restart, that tick is refused by name and
   the next comes 60 s later.
3. **orion-proposal-runtime rebuilds.** The old template stops. Between steps 2 and 3 it still offers
   it; dispatch still blocks it. That is not a second trigger.

There is never a time with two triggers, in any order: the RPC listener leaves Hub in the same image
the timer arrives in. If Hub rebuilt first, the old dispatch's RPCs would time out with no listener
(and hold 300 s each). If dispatch rebuilt first, there would be no trigger until Hub rebuilt.

By hand, the same order:

```bash
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

After the three are up, delete the stale keys from the main checkout's local `.env` (one line):

```bash
sed -i '/^ORION_DISPATCH_RETRO_LEISURE_ENABLED=\|^ORION_DISPATCH_RETRO_LEISURE_RPC_TIMEOUT_SEC=/d' services/orion-execution-dispatch-runtime/.env
```

Live check after the deploy: `curl -fsS http://localhost:8080/health | jq .retro_leisure_self_start`
shows `running: true` and `ticks` rising every 5 minutes, with refusals by name. Within a day:
`want_ledger` rows for `hub.retro_leisure` with `want_id LIKE 'retro_self_start:%'`, and 0 new
`substrate_dispatch_results` rows with settlement kind `retro_leisure`. A started session shows as a
`grammar_acts` row with `actor = 'hub.retro_leisure'` and `correlation_id LIKE 'retro_self_start:%'`.

## Risks / concerns

- Severity: medium. Concern: the timer cannot make leisure work. All 4 real sessions ended in about 1 s
  with 0 decisions, and a paused Retro game blocks most starts. Mitigation: both are listed under "Found,
  not fixed" for their own investigation. The timer reports every refusal by name on `/health`.
- Severity: low. Concern: about 1.7x more shadow bids (288 a day against ~172), because the timer asks
  every 300 s and dispatch skipped some ticks. Mitigation: the bids are shadow and logged only. The
  rate is the spec's 300 s cadence.
- Severity: low. Concern: the dispatch act machinery for an `act_duration` route now serves AI Town
  only. Mitigation: PR 3 moves AI Town; PR 6 deletes the machinery.

## PR link

(filled in after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
