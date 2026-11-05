# PR 5 of the ladder retirement: the market decides the cooling act, the pool acts

## Summary

- Orion has one act that changes the world: when the cabinet warms, it stops the GPU pool from starting
  new background work for 15 minutes. Until now the only thing that could start this act was the
  proposal -> policy -> execution-dispatch ladder. This PR gives the act its own loop in orion-durable-runs.
- The loop asks the body-economy market first, and it obeys the answer. Funded: it acts. Deferred: it
  waits and asks again while the warm moment lasts. No answer: it does not act.
- The experiment does not change. The same coin flip puts half of the funded moments in a control group.
  The same ledger row is written before the pool is asked. The same pool request goes out. The same
  settlement runs. The scorer in orion-sql-writer scores the new rows with no change.
- The old trigger is removed in the same PR: the `shed_background_gpu` proposal template, and the
  world-action path in execution dispatch. The act has one trigger, never two.
- A replay over the last 14 days of real cabinet readings runs the shipped loop with the real market rule
  and the real scorer. The eligible count is about the same as the old path's.

Spec: `docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md` ("PR 5", "Decided" 1).

## Outcome moved

- **The act can fire again.** The old path decided nothing after 10-25. In the last 14 days, 2,796 of the
  4,498 shed candidates that reached dispatch were blocked as `world_eligibility_stale`: the hop from
  proposal to dispatch took about 160 s, and the limit was 120 s. The new loop reads the room and acts in
  the same tick, so the reading can never be stale.
- **The market decides.** The act is the first one that binds in the market by design (Juniper,
  2026-11-05). Each decision is in `want_ledger` (producer `orion-durable-runs.cooling_act.binding`) and
  is joined to its episode row by `decision_id`.

## Current architecture (before this PR)

- The cabinet's rise wrote a prediction error on `node:substrate.cabinet`. That node competed for the
  attention workspace. When the cabinet (or biometrics) node won, orion-proposal-runtime read the cabinet,
  hardware-watch and the pool, ran `orion/autonomy/self_shed.py`, and emitted a `shed_background_gpu`
  proposal.
- orion-policy-runtime approved it (`approved_self_reversible`). orion-execution-dispatch-runtime checked
  freshness and the pool's caps, drew treated/control from a hash of the dispatch id, wrote the episode
  row, sent `orion:gpu_pool:shed:request`, and settled it later from the pool's `gpu_pool_orion_shed` row.
- orion-sql-writer scores each episode 20 minutes later on the cabinet sensor
  (`orion/autonomy/world_settlement_store.py`, moved there in #3115).

## Architecture touched

- **orion-durable-runs** gets the act: `app/cooling_act.py` (the loop and its SQL store), wired in
  `app/main.py`, health on `/health` -> `cooling_act`.
- **Pure rules** in `orion/autonomy/shed_act.py`: the event, its salience, the want, the binding rule,
  the pool admission and the episode row (moved from the dispatch worker, not rewritten). The settle rule
  moved unchanged from `orion/execution_dispatch/shed_settlement.py` to `orion/autonomy/shed_settlement.py`.
  The hash draw moved to `orion/autonomy/world_episodes.py` (`stable_draw`); dispatch's act and reading
  holdbacks call the same function.
- **Removed:** the `shed_background_gpu` proposal template and template copy; proposal-runtime's
  workspace-winner reads; dispatch's world-action holdback, admission, shed RPC, settlement reconcile, the
  shed route in `execution_dispatch_policy.v1.yaml` (`allow_self_reversible_dispatch` is now false), and
  their settings and env keys.
- **Untouched:** the pool's thermal reflex (`orion/gpu_pool/shed.py`, `orion_shed.py`, hardware-watch),
  the sql-writer scorer, `orion/autonomy/self_shed.py` (the act ledger still imports it lazily).

How one tick works (every 60 s):

1. Settle any treated shed still `shed_pending` from the pool's row (or as `settlement_timeout`). Emit its
   outcome line once, then record the terminal on the episode.
2. A held want waits until its retry time.
3. Read the cabinet, hardware-watch `/health`, background leases and the ledger. Run
   `evaluate_shed_eligibility`, then the pool's mirrored caps (in flight, 15-minute gap, 1 hour a day, a
   pool that refuses). Not eligible: a held want closes (its event is over).
4. Not warming (no rise >= 0.5 C in 15 min): no new want.
5. Post the event want, and obey `shed_verdict`.

## Files changed

- `orion/autonomy/shed_act.py`: new. The pure rules of the act.
- `orion/autonomy/shed_settlement.py`: moved from `orion/execution_dispatch/`; docstring only.
- `orion/autonomy/world_episodes.py`: `stable_draw` (the one hash draw); docstring names the new writer.
- `orion/autonomy/world_settlement_store.py`: docstring names the new writer.
- `orion/autonomy/attend_act_loop.py`, `orion/autonomy/evals/run_attend_act_loop_eval.py`,
  `orion/autonomy/evals/fixtures/attend_act_loop_chain.json`: the chain check knows the market chain
  (want decision -> episode -> pool shed -> outcome row) and the live eval reads `want_ledger` for it.
- `orion/physiology/binding.py`: the new producer is in `BINDING_PRODUCERS`.
- `config/physiology/families.yaml`: new family `self_regulate.shed` (Tier 1, see the sign-off below).
- `config/physiology/contract.yaml`: the new bid site and its pure rules are registered readers (`bidder`);
  the proposal-runtime store row is gone (it reads nothing now).
- `config/proposals/proposal_policy.v1.yaml`, `orion/proposals/templates.py`: the template is removed.
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`, `orion/execution_dispatch/policy.py`:
  the shed route and `SHED_EXECUTOR_VERB` are removed; the self-reversible scope is closed.
- `config/policy/substrate_policy.v1.yaml`, `orion/schemas/policy_decision_frame.py`,
  `orion/schemas/proposal_frame.py`: comments only (the self-reversible values stay until PR 6).
- `orion/bus/channels.yaml`: the shed request's producer and the reply's consumer are orion-durable-runs;
  orion-durable-runs also produces `orion:autonomy:action:outcome`.
- `services/orion-durable-runs/app/{cooling_act.py,main.py,settings.py}`, `.env_example`,
  `docker-compose.yml`, `requirements.txt` (`requests`, which `orion.substrate` imports), `README.md`.
- `services/orion-durable-runs/tests/test_cooling_act.py`, `tests/test_cooling_act_postgres.py`: new.
- `services/orion-durable-runs/evals/cooling_act_replay.py`: new.
- `services/orion-execution-dispatch-runtime/app/{worker.py,store.py,settings.py}`, `.env_example`,
  `docker-compose.yml`, `README.md`: the world-action path removed.
- `services/orion-proposal-runtime/app/{worker.py,store.py,settings.py}`, `.env_example`,
  `docker-compose.yml`, `README.md`: the workspace-winner reads removed;
  `tests/test_workspace_winner_context.py` deleted.
- `tests/test_cooling_act_rules.py`: renamed from `tests/test_shed_background_gpu_pipeline.py`. The
  signal, eligibility and winner-binding tests stay. The proposal/policy/dispatch pipeline tests are gone.
  New tests cover the event, salience, want, binding rule, admission, draw, scorer compatibility and settle.
- `tests/test_world_action_dispatch.py`: deleted (its settle and draw tests moved to the rules file).
- `tests/test_proposal_scoring.py`: the template left the empty-dimension allowlist.
- `scripts/sync_local_env_from_example.py`: syncs `ORION_COOLING_ACT_*` and `BUS_ACTION_OUTCOME_OUT`;
  drops the removed prefixes.
- `config/metrics/metric_definitions.lock.json`: re-locked (see "Definition drift" below).
- `.github/workflows/attend-act-loop.yml`, `.github/workflows/orion-durable-runs-tests.yml`: path filters,
  the renamed tests, and the new eval step.
- `services/orion-gpu-pool/README.md`, `services/orion-sql-writer/README.md`: name the new sender.

## Schema / bus / API changes

- Added: want family `self_regulate.shed`; producer `orion-durable-runs.cooling_act.binding`;
  `/health` key `cooling_act` on orion-durable-runs.
- Removed: nothing on the wire. `orion:gpu_pool:shed:request` and its reply keep their schemas.
- Renamed: the shed request's only producer (execution dispatch -> orion-durable-runs).
- Behavior changed: a new episode row has no `open_loop_id`, `broadcast_log_id`, `node_id`, `proposal_id`
  or `dispatch_frame_id`. `decision_id` is the market's decision. `eligibility.want` holds the want. The
  scorer reads none of the removed fields except `open_loop_id`, which only gates the attention-loop
  `acted` verdict. So a new episode writes no loop verdict (there is no loop). The episode id is
  `cooling_act:<want_id>`.
- Compatibility: no `extra="forbid"` model changed. Old episodes are untouched; the scorer, the chain
  check and the in-flight check read both kinds. The agency-episode audit (`orion/autonomy/agency_episode*`)
  is keyed on dispatch ids, so it does not list new episodes. It retires with dispatch in PR 6.

## Env/config changes

- Added keys (orion-durable-runs): `ORION_COOLING_ACT_ENABLED=true`, `ORION_COOLING_ACT_TICK_SEC=60`,
  `ORION_COOLING_ACT_HOLDBACK_FRACTION=0.5`, `BUS_ACTION_OUTCOME_OUT`.
- Moved keys, same names and values: `ORION_HARDWARE_WATCH_HEALTH_URL`, `ORION_SHED_RISE_THRESHOLD_C`
  (from proposal-runtime); `ORION_GPU_POOL_SHED_RPC_TIMEOUT_SEC`, `ORION_SHED_TTL_SEC`,
  `ORION_SHED_MIN_GAP_SEC`, `ORION_SHED_MAX_SEC_PER_DAY` (from execution dispatch).
- Removed keys: `ORION_WORLD_ACTIONS_ENABLED`, `ORION_WORLD_ACTIONS_ALLOWED`,
  `ORION_WORLD_ACTION_ELIGIBILITY_MAX_AGE_SEC` (dispatch); `ORION_WORKSPACE_WINNER_PROPOSALS_ENABLED`
  (proposal-runtime).
- `.env_example` updated: yes, all three services. Flags ship ON.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py orion-durable-runs`: yes, 10 keys
  added to the main checkout's `services/orion-durable-runs/.env`.
- Operator action: the removed keys are still in the main checkout's dispatch and proposal-runtime `.env`.
  They are left there ON PURPOSE until this PR deploys: the code running today still reads them, and
  removing them first would switch the old path off before the new one runs. After the deploy, delete
  them (command in "Restart required").

## The metric quality gate: the event and its salience

The want needs a `salience`: how unusual this event is for its own source.

1. **Provenance.** The magnitude is `CabinetHeatReading.rise_c` (`orion/autonomy/cabinet_heat.py`
   `read_cabinet_heat`), which is `orion.hardware_watch.rules.cabinet_rise_c`: the newest reading minus the
   lowest reading of the last 15 minutes. The readings are athena's `cabinet_temp_c` in
   `orion_biometrics_summary` (about every 30 s). The history is `shed_act.warming_minutes`: every minute of
   the last 30 days whose last reading was a warming moment (the thermal gate, folded with its hysteresis
   over the raw readings, reads elevated or hot, and the rise is >= 0.5 C). Salience is the share of those
   minutes with a strictly smaller rise (`orion.physiology.event_wants.source_salience`).
2. **Independence.** It is NOT independent of the attention signal `cabinet_warming_error`: both are the
   same rise (that one is clamped and linear, this one is a percentile). This is by design. The rise is what
   made the cabinet node win attention, which was the old trigger. Nothing new is measured; the existing
   signal is re-expressed as the source's own percentile. It IS separate from the act's outcome. The scorer
   scores the change in `cabinet_heat_pressure` (the level). Over the 854 warming minutes of the last 14
   days, the correlation of the rise with the level was -0.03. That is why salience does not come from
   `cabinet_heat_pressure`, which the spec's sketch named: that number is the outcome, and the cabinet-heat
   module keeps trigger and score different numbers.
3. **Theory.** The cabinet-heat module's own anchor: with the AC healthy, the heat balance predicts a
   steady temperature; the 15-minute change has an SD of about 0.42 C with nothing done. A rise is a
   prediction error. A bigger rise than usual is a more surprising, more urgent warming.
4. **Live data** (last 14 days, 30 days of history before each minute):
   - 854 warming minutes, 4.2% of all minutes (it rests most of the time: no event at all);
   - rise: 0.50 to 2.78 C, median 0.67, 104 distinct values;
   - salience: never unknown in the window; it reads exactly 0 (its rest point, a rise at the threshold)
     on 3.5% of warming minutes; deciles 10/25/50/75/90 = 0.06 / 0.20 / 0.41 / 0.64 / 0.83. It is close to
     uniform, as the patch 2 note says a near-continuous magnitude will be. Not flat, not saturated.
5. **Existing mechanism.** `cabinet_warming_error`, `cabinet_rise_c` and `source_salience` all exist and are
   reused. Only `warming_minutes` (the history fold) is new.
6. **Reversibility.** Cheap. Salience is one float on a want. Nothing stores it except `want_ledger` and the
   episode's eligibility snapshot. It is not in a schema default, a manifest or a training set.

What the market does with it, said plainly: the family holds no lane, and nothing is priced on it, so the
market funds a want whenever its value is above zero. Value is the salience ranked against the family's
own past events. So the market says "not now" only to the least unusual warm moments, and only until the
want's hunger grows (it counts from the want's own `opened_at`, 900 s time constant). In the replay this
deferred 8 bids, and every deferred want was funded on a later bid inside its window.

## Replay: 14 days of real cabinet history

`python -P services/orion-durable-runs/evals/cooling_act_replay.py --live --days 14`, window
2026-10-22 03:21 to 2026-11-05 03:21 UTC (re-run after the review fixes; same counts as the first run). It runs the shipped loop each minute, with the real market rule
on a price-free body (what this family meets live), a fake pool that runs each shed to its TTL, and the real
scorer. Background GPU work is assumed present (live there are hundreds of background leases a day).

| | New loop | Old path |
|---|---|---|
| Eligible minutes, from 10-26 (the oldest proposal frames kept) | **125** (eligible and warming) | **153** (an eligible shed proposal) |
| Eligible minutes, whole 14 days | 365, in 73 different hours | proposal frames before 10-26 are pruned |
| Wants posted / funded / deferred bids | 55 / 55 / 8 | none |
| Episodes drawn (treated / control) | **55 (24 / 31)**, on 10 of the 14 days (10-22 .. 10-31) | **4 (3 / 1)**, none after 10-25 |
| Scored by the real scorer | 55 of 55 (24 treated, 31 control) | -- |
| Shed RPCs | 24 (about 1.7 a day; the pool cap is 4 a day) | 3 |

Acceptance check, "about the same eligible count": 125 against 153 in the span both can see (82%). The old
count is a little higher because it also fired when the biometrics node won attention with no rise (6 of
the 20 lifetime episodes had `warming_error` 0); the new event needs a real rise.

The old path's dispatch frames over the same 14 days, for the record: 4,498 shed candidate rows reached
dispatch. 2,796 were blocked `world_eligibility_stale`, 92 `world_action_in_flight`, 1 was held back as a
control, 3 were sent, and 1,606 were prepared but never sent. That is why the new loop draws many more
episodes from about the same eligible moments: it does not lose them on the way.

What the replay does not exercise: it calls the real decision rule (`choice.decide`) on a synthetic,
price-free body with no act history, not `Market.bid` itself (no Redis body state, no `want_ledger`
publish, no Prime path). So the funded share is built in here; live it should be the same, because this
family holds no lane and meets no price.

The fixture replay (CI) passes 9 of 9 checks: no want outside the warm band; no want while the reflex
covers it (critical, unknown); episodes on the warm afternoon; the treated share inside 0.5 +- 0.35; one
RPC per treated episode; the scorer scores every episode; no two episodes overlap; only funded decisions
become episodes.

## Tier 1 needs Juniper's sign-off

`self_regulate.shed` is proposed as **Tier 1** (critical upkeep), as the plan suggested. Tiers are
Juniper's decision (`families.yaml` header). With no lanes, the tier changes little today: Tier 1 is not
part of the Tier 2 exploration floor, and it would use the Tier 1 reserved share if the act ever held a
lane. **Juniper: please confirm Tier 1, or say Tier 2.**

## Definition drift (lock re-recorded)

`scripts/check_definition_drift.py --update` records five deltas. Three are real: the shed request moved
producer (`removed .../orion-execution-dispatch-runtime/orion:gpu_pool:shed:request`, `added
.../orion-durable-runs/...`) and the reply moved consumer (`routing_changed`). Two are an id artifact: the
lock keys a channel by its alphabetically first producer, so adding orion-durable-runs to
`orion:autonomy:action:outcome` renamed that id. Execution dispatch still produces that channel for its
other acts.

## Tests run

```text
tests/test_cooling_act_rules.py + world settlement + proposal + autonomy + physiology   582 passed
services/orion-durable-runs/tests (whole suite, real Postgres 16 via ORION_ADMISSION_TEST_DSN)  all passed
services/orion-durable-runs/tests/test_cooling_act.py                                   20 passed
services/orion-durable-runs/tests/test_cooling_act_postgres.py (real Postgres)          1 passed
services/orion-proposal-runtime/tests                                                    9 passed
services/orion-execution-dispatch-runtime/tests                                          128 passed
services/orion-sql-writer/tests/test_world_settlement_pass.py                            6 passed
tests/test_execution_dispatch_* tests/test_rung3_* (+ dispatch-adjacent)  same 68 failures as origin/main
  in this environment (they try to reach a database); one main failure is fixed (the removed template).
orion-static-gates.yml commands run locally: all pass after the fixes below
scripts/check_body_economy_contract.py   OK (101 state-reading modules)
scripts/check_env_template_parity.py     PASS
scripts/check_env_key_single_source.py   OK
```

## Evals run

```text
python -P services/orion-durable-runs/evals/cooling_act_replay.py          9/9 PASS (fixture)
python -P services/orion-durable-runs/evals/cooling_act_replay.py --live --days 14   numbers above
python orion/autonomy/evals/run_attend_act_loop_eval.py                    27/27 PASS (6 new market-chain checks)
```

## Docker/build/smoke checks

```text
docker build -f services/orion-durable-runs/Dockerfile .   OK
in the image: import app.main and app.cooling_act; settings flag on; the family loads from the baked
families.yaml (tier 1); the store's SQLAlchemy driver is psycopg (3)
the store's pool join ran read-only against live substrate_world_action_episodes + gpu_pool_orion_shed: OK
```

UNVERIFIED on the live rail: no new episode yet. The cabinet has been cool since 10-31, so a real want, a
real funded shed and a `[WORLD_SETTLEMENT] scored=` line for a new episode wait for the next warm spell.

## Review findings fixed

A code-review subagent read the whole diff, ran the suites and pulled 30 days of live cabinet data.

- Finding (blocker): a funded bid after the event window ended still shed. The market funds a passed
  window (it only warns), a long deferral's retry is clamped to exactly the window end, and the next tick
  bid again.
  - Fix: the tick closes a want whose window has ended before any bid; `shed_verdict` checks the end before
    `funded`.
  - Evidence: `test_a_deferral_clamped_to_the_events_end_never_bids_or_sheds_late` (deferred, then funded
    at end + 40 s: no bid, no row, no RPC) and `test_a_funded_bid_at_or_after_the_events_end_never_sheds`.
- Finding: `families.yaml` said the salience was the temperature's percentile and the event "warm".
  - Fix: it now says the 15-minute rise, and "warming" (elevated or hot and rising >= 0.5 C).
- Finding: the market almost never says no to this act (no lanes, heat unpriced).
  - Fix: said plainly in the gate section and in Risks; the tier waits for Juniper. Not changed in code.
- Finding: a treated row could be precommitted when no bus exists.
  - Fix: the bus is checked before the draw; no bus means no row. Evidence: `test_no_bus_writes_no_row_and_sends_nothing`.
- Finding: two stale comments (`shed_rise_threshold_c`, `is_warming`). Fix: reworded.
- Finding: the lazy DDL ran on the first insert after every boot (lock risk). Fix: it runs only when the
  ledger table is missing. Evidence: the real-Postgres store test still passes.
- Finding: the ledger was read twice per tick. Fix: read once, reused for admission.
- Finding: the replay docstring overstated what it runs. Fix: reworded (and stated above).
- Finding: CI path filters missed `orion/autonomy/contrast.py`, `prediction.py` and
  `orion/substrate/bus_synaptic_surprise.py`. Fix: added to both path blocks.
- Finding (CI, spotted by Juniper): the body-atlas wire eval (`orion/body_atlas/evals/run_wire_eval.py`)
  failed `true facts missed: [('orion-proposal-runtime', 'http://hardware-watch:8131/health')]`. The
  hardware-watch `/health` call moved with the act.
  - Fix: the hand-checked fact now names `orion-durable-runs` (`app/cooling_act.py` `read_hardware_watch`).
  - Evidence: the scanner finds it ("scanner facts: PASS"), all four "mutations must fail" lanes are still
    CAUGHT, `orion/body_atlas/tests` 229 passed, and the organ-structure and capacity-roles evals pass.
- Not fixed (low, same as the old dispatch path): after a failed precommit or control write, the next tick
  can post a new event and draw again. A DB blip can change one moment's arm. Recorded here.
- Not fixed, by plan: the spec this PR cites lives on PR #3121 (docs only); merge #3121 first or with this
  one. `WorkspaceWinnerContext`, the `self_reversible` gate and `approved_self_reversible` have no user now;
  PR 6 (retire the three services) deletes them.

## Restart required

Deploy from the clean primary `main` checkout after merge. Remove the old trigger first, then start the new
one, so there is never a moment with two triggers:

```bash
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

orion-sql-writer and orion-gpu-pool need no restart. After the three are up, delete the stale keys from the
main checkout's local `.env` files (one line):

```bash
sed -i '/^ORION_WORLD_ACTIONS_ENABLED=\|^ORION_WORLD_ACTIONS_ALLOWED=\|^ORION_WORLD_ACTION_ELIGIBILITY_MAX_AGE_SEC=\|^ORION_GPU_POOL_SHED_RPC_TIMEOUT_SEC=\|^ORION_SHED_TTL_SEC=\|^ORION_SHED_MIN_GAP_SEC=\|^ORION_SHED_MAX_SEC_PER_DAY=/d' services/orion-execution-dispatch-runtime/.env && sed -i '/^ORION_WORKSPACE_WINNER_PROPOSALS_ENABLED=\|^ORION_HARDWARE_WATCH_HEALTH_URL=\|^ORION_SHED_RISE_THRESHOLD_C=/d' services/orion-proposal-runtime/.env
```

Live check after the deploy: `curl -fsS http://localhost:8124/health | jq .cooling_act` shows the loop ticking
(`last.result` is `ineligible` or `not_warming` while the cabinet is cool).

## Risks / concerns

- Severity: medium. Concern: about 14x more episodes than the old path drew in the same 14 days (55 against
  4). Mitigation: that is the old path's loss, not new behavior: the pool's caps (15-minute gap, 1 hour a
  day) and the in-flight rule are unchanged and apply before the draw; the replay stays at about 1.7 sheds a
  day. The pool also enforces its own caps.
- Severity: low. Concern: the market almost always funds this want, because the act costs nothing the market
  prices. Mitigation: that is honest for an act that holds no lane; binding still makes "no answer" a no.
  A Prime receptor or a heat price can make it selective later.
- Severity: low. Concern: an "unknown" cabinet (sensor dead) was eligible on the old path. Now it posts
  nothing (no rise). Mitigation: when the sensor is unknown, the reflex itself sheds (`cabinet_unknown`),
  so the old path was ineligible then too (`reflex_active`).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3131

🤖 Generated with [Claude Code](https://claude.com/claude-code)
