# PR 2 of the ladder retirement: Orion starts its own paintings

## Summary

- Orion paints about 9 pictures a day. Until now, only the proposal -> policy -> execution-dispatch
  ladder could start one. This PR gives painting its own trigger in orion-thought
  (`app/painting_self_start.py`).
- The trigger is the schedule that already existed (the "visual baseline"): a painting is due 90 minutes
  (5,400 s) after the last one. It runs every 60 s on the same database row the ladder used. It keeps
  every gate: arousal deferral, the self-monitor act gate, the GPU-pool diffusion hold (the diffusion
  lock), and the thermal gate in the generate step.
- Before it starts a painting, it places a hunger bid on `thought.painting` in the body-economy market.
  The bid is SHADOW: it is logged, not obeyed. Painting is fourth in Juniper's binding order.
- The old triggers are removed in the same PR: the `render_scene` proposal template, its dispatch route
  (with the allocator's `act:` entry), the `express` kind's fallback route (it also pointed at the
  painting verb), and orion-proposal-runtime's baseline scheduling. Painting has one trigger, never two.
- A replay over the last 14 days runs the shipped trigger on real history. It paints 140 pictures against
  the real 129 (+8.5 %, inside the +-30 % check).

Spec: `docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md` ("PR 2. Painting",
"Decided" item 2).

## Outcome moved

- **Painting no longer depends on the ladder.** PR 6 can now retire proposal, policy and dispatch
  without silencing painting.
- **The market sees the want before the act.** Before, the only painting bid was placed in the generate
  step, after the run had already been admitted. Now the bid is placed when the painting becomes due,
  before the run exists.

## Current architecture (before this PR)

- **Allocator path (105 of 129 paintings in 14 days).** orion-proposal-runtime emitted a `render_scene`
  candidate on every warranted tick. orion-policy-runtime approved it (`approved_express`).
  orion-execution-dispatch-runtime scored it on the painting act's own belief (the route's `act:`
  declaration) and sent it to cortex-exec `skills.imagination.render_scene.v1`. cortex-exec submitted a
  `reverie.visual` durable run.
- **Baseline path (24 of 129).** Each proposal tick, orion-proposal-runtime read orion-thought's
  `GET /visual-chain/activity` and ran `orion.reverie.baseline.schedule` on `visual_baseline_checkpoint`.
  A due need replaced the `render_scene` candidate, which then took the same ladder.
- **Downstream of both:** orion-durable-runs drove the run (prepare, then a GPU-pool `diffusion` hold,
  then generate, then caption). orion-thought ran each step. The thermal gate refused a generate in a hot
  room. The steps opened and closed the `thought.painting` `make` act.
- The legacy timer in orion-thought (`ORION_VISUAL_CHAIN_ENABLED`) is off live. It stays off.

## Architecture touched

How one tick works now (orion-thought, every 60 s):

1. The legacy timer is on: issue nothing (`legacy_visual_worker_enabled`).
2. Read the painting activity. Run the baseline schedule on its checkpoint row, under `FOR UPDATE`. Not
   due, an open attempt (a dream hop too), the 600 s retry clock, or arousal `strained`: stop here.
3. The self-monitor act gate (E1). A faculty held broken or unreliable is refused, except one re-test
   probe per 6 h.
4. Place the SHADOW hunger bid (`want_id` = the run's correlation id; `object_ref` =
   `durable_run:<run_id>`, the same `object_ref` the act carries).
5. `POST /runs` to orion-durable-runs: one `reverie.visual` run, diffusion hold, deadline 5,400 s.
   `dispatch_id` = `painting_self_start:<need_id>:<issued_at>`.

From the run onward nothing changed. `/health` -> `painting_self_start` shows the last tick.

## How the act ledger keeps grading paintings

This was the main hidden coupling. The check:

- **Who opens and closes the act.** orion-thought's step handlers (`app/visual_steps.py`), not
  dispatch. The first real generation opens it. The produced caption closes it at 1. A failed terminal
  or an abandon after a generation closes it at 0. The act key is the run id.
- **What dispatch added.** Only the route's `act:` declaration, which the motor allocator read to score
  the candidate. It is removed with the route. The act ledger never read it.
- **The join to the picture.** The act's `test_before` is `visual_dispatch:<dispatch_id>`. sql-writer's
  `act_downstream_use` joins it to `reverie_visual_chain.chain_json.run_request.dispatch_id`. The new
  `dispatch_id` is written to both, so the join still works.
- **Proof.** `tests/test_painting_self_start.py::test_a_self_started_run_opens_and_closes_the_painting_act`
  builds the exact run the trigger submits and drives it through the real prepare, generate and caption
  steps. It asserts one `act_opened` and one `act_closed` (value 1), with the run's `object_ref` and
  `test_before`.

## The proven-effect holdback: it ends for painting

Dispatch's `_proven_act_holdback` wrote painting admissions to `substrate_act_episodes`, but it never
drew a control arm for painting (`holdback_rule("dispatch")` -> `holdback_exempt`). Live: 114 painting
rows, all `treated`, 0 with an object, 0 with an outcome. Nothing reads them. So nothing moves with
painting: those rows simply stop. The holdback stays for the reading pull (20 treated / 6 control in 14
days) until PR 6.

## Rate calibration

The rate comes from the schedule's own cadence. A painting is due `interval_sec` after the last one. The
real cycle is longer: the claim, the diffusion queue (median 456 s), the generation (about 66 s), and the
gates. The replay below ran the shipped trigger with other intervals to show the curve (14 days, same
gates):

| `interval_sec` | paintings | per day | against the real 129 |
|---|---|---|---|
| 3,600 | 197 | 14.1 | +53 % (fails) |
| 4,500 | 163 | 11.6 | +26 % |
| **5,400 (shipped, unchanged)** | **140** | **10.0** | **+8.5 %** |
| 6,300 | 123 | 8.8 | -4.7 % |
| 7,200 | 110 | 7.9 | -15 % |

The shipped interval stays at 5,400 s. It is the value the spec names, it is inside the check, and
changing it would also move the run's deadline and the act's 5,400 s horizon. 6,300 s is a little closer
in count. If a live week reads high, raise `interval_sec` in `config/proposals/visual_baseline.v1.yaml`.

Why the rate stays about the same: the old allocator path was already paced like the baseline. The gap
between paintings in the last 14 days had a median of 5,704 s and a p10 of 4,414 s.

## Replay: 14 days of real history

`python -P services/orion-thought/evals/painting_self_start_replay.py --live --days 14`, window
2026-10-22 04:29 to 2026-11-05 04:29 UTC.

It runs the shipped trigger class with the real schedule, every 60 s. Downstream it uses real rows:

- arousal: 402 `arousal_transition` events (82.9 strained hours in the window);
- the thermal gate: 38,926 cabinet readings, folded with the gate's own hysteresis (10,476 hot minutes);
- the single-flight claim and the 600 s cooldown: 88 real dream-hop attempts;
- the diffusion queue: 152 real claim-to-generation waits (median 456 s);
- the run deadline (5,400 s).

| | New trigger | Real (dispatch + baseline) |
|---|---|---|
| Paintings, 14 days | **140** (10.0 a day) | **129** (9.2 a day: 105 allocator, 24 baseline) |
| Runs submitted / hunger bids | 167 / 167 | -- |
| Runs that ended without a painting | 21 at the deadline (a hot room), 6 `already_satisfied` | 40 abandoned attempts |
| Thermal retries of a generate | 410 | -- |
| Ticks deferred by arousal | 383 | -- |
| Waking runs in flight at once, max | 1 | -- |

Per day, new / real: 10-23 4/4, 10-25 3/2, 10-26 11/9, 10-27 12/14, 10-28 15/15, 10-29 11/12,
10-30 13/15, 10-31 14/8, 11-01 13/11, 11-02 15/8, 11-03 13/14, 11-04 13/15. The heat-lockdown days
(10-22, 10-24) paint nothing in both. Acceptance check: ratio 1.085, inside +-30 %.

What the replay does not model, said plainly:

- the self-monitor gate (the painting faculty read `reliable`, 100 %, through the window);
- a generation that fails after the GPU started (0 of the window's real painting acts closed 0);
- the dream hops' own reaction to our claims (their timings are the real ones);
- the GPU queue reacting to a different painting schedule (the queue waits are replayed in order).

The fixture replay (CI) passes 9 of 9 checks: an ungated day paints about 24 h / (interval + queue +
generation); no submit while arousal reads strained; no generation in a hot room; painting resumes
after the heat; never two waking runs in flight; every painting came from a submitted run; one hunger
bid per submit; a dream attempt holds the claim; 9-16 a day over the scripted world.

## Metric quality gate

No new metric, signal or salience is added. The hunger bid is a clock want with no `event_ref` and no
`salience`. The market fills its intensity from the family's last act close, as it already did for the
generate-step bid. The replay is an eval, not a signal.

## Files changed

- `services/orion-thought/app/painting_self_start.py`: new. The trigger, its live wiring and `/health`.
- `services/orion-thought/app/store.py`: `visual_baseline_eligibility` (moved from proposal-runtime's
  store) and `painting_act_attempt_rows` (the self-monitor gate's ledger read).
- `services/orion-thought/app/main.py`: starts the loop; `/health` -> `painting_self_start`.
- `services/orion-thought/app/settings.py`, `.env_example`: `ORION_PAINTING_SELF_START_*`,
  `ORION_VISUAL_BASELINE_AROUSAL_ENABLED` (moved), `SELF_MONITOR_ACT_GATE_ENABLED`,
  `SELF_MONITOR_PROBE_INTERVAL_SEC`.
- `services/orion-thought/app/visual_steps.py`: the generate step bids only for dream hops now.
- `services/orion-thought/app/physiology_shadow.py`, `app/visual_chain_health_monitor.py`: docstrings;
  the painting-gap alert now points at orion-thought's logs.
- `services/orion-thought/tests/test_painting_self_start.py`: new (17 tests, including the act-ledger
  end-to-end test).
- `services/orion-thought/tests/test_physiology_shadow_bid.py`: runs on the dream-hop generate; a new
  test pins that a waking generate places no bid.
- `services/orion-thought/evals/painting_self_start_replay.py`: new.
- `services/orion-thought/README.md`: who starts a painting now.
- `services/orion-proposal-runtime/app/{worker.py,store.py,settings.py}`, `.env_example`, `README.md`:
  the baseline read, schedule and arousal reader are removed.
  `tests/test_visual_baseline_arousal_worker.py` deleted; `tests/test_visual_baseline_moved.py` new.
- `orion/proposals/builder.py`: no `baseline_eligibility` input; a forged baseline from any producer is
  still refused.
- `config/proposals/proposal_policy.v1.yaml`: the `render_scene` template is removed.
- `config/proposals/visual_baseline.v1.yaml`, `orion/reverie/baseline.py`: `thought_url` and
  `timeout_sec` removed (only proposal-runtime's HTTP read used them); comments.
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: the `render_scene` route (with its
  `act:`) and the `express` kind fallback route are removed. `allow_express_dispatch` stays true for
  retro and AI Town.
- `orion/self_monitor/replay.py` (+ test): painting's gate seam is `painting_self_start` now.
- `config/physiology/contract.yaml`: the new bid site is a registered `bidder`; the proposal-runtime
  worker row is gone.
- `scripts/sync_local_env_from_example.py`: syncs `ORION_PAINTING_SELF_START_*`.
- `orion/body_atlas/evals/run_wire_eval.py`: hand-checked facts moved (see "Review findings").
- `tests/test_express_outward_action.py`, `tests/test_visual_baseline_pipeline.py`,
  `tests/test_visual_baseline_arousal_reader.py`, `tests/test_rung3_act_candidates.py`,
  `tests/test_rung3_reading_pull.py`, `tests/test_self_monitor_act_gate.py`,
  `services/orion-cortex-exec/tests/test_render_scene_durable_kickoff.py`: tests of the removed
  template, route and baseline wiring are removed or now assert they are gone.
- `tests/_dispatch_minted_act_route.py`: new. The dispatch act-machinery tests ran on the painting route.
  That machinery still serves the reading, retro and AI Town routes until PR 6, so those tests now run on
  an in-memory copy of the old route. A separate test pins that the real policy has no such route.
- `.github/workflows/visual-baseline.yml`, `thinking-rung3.yml`, `self-monitor-tests.yml`: path
  filters, the new tests, the proposal-runtime tests and the fixture replay.

## Schema / bus / API changes

- Added: `/health` key `painting_self_start` on orion-thought. orion-thought now calls orion-durable-runs
  `POST /runs` (an existing endpoint).
- Removed: nothing on the wire. No channel changes: the run goes over HTTP; the act events use the
  existing `orion:grammar:event`.
- Behavior changed: a self-started run's `visual_request` has no `proposal_id` or `decision_id`, and its
  `dispatch_id` is `painting_self_start:<need_id>:<issued_at>`. The want for a waking painting comes from
  producer `orion-thought.painting_self_start` (was `orion-thought.visual_steps.generate`).
- Compatibility: no `extra="forbid"` model gained a field. `VisualBaselinePolicy` lost two fields
  together with the yaml in the same image. The `visual_baseline` fields on the proposal, decision and
  dispatch schemas stay until PR 6 deletes those schemas.

## Env/config changes

- Added keys (orion-thought): `ORION_PAINTING_SELF_START_ENABLED=true`, `ORION_PAINTING_SELF_START_TICK_SEC=60`,
  `ORION_PAINTING_SELF_START_DURABLE_RUNS_URL=http://orion-athena-durable-runs:8121`,
  `SELF_MONITOR_ACT_GATE_ENABLED=true`, `SELF_MONITOR_PROBE_INTERVAL_SEC=21600`.
- Moved key, same name and value: `ORION_VISUAL_BASELINE_AROUSAL_ENABLED` (proposal-runtime -> thought).
- Removed keys (proposal-runtime): `ORION_VISUAL_BASELINE_AROUSAL_ENABLED`, `ORION_REGULATION_READER_MAX_AGE_SEC`.
- `.env_example` updated: yes, both services. Flags ship ON.
- Local `.env` synced: the main checkout's `services/orion-thought/.env` got the 6 keys by hand (the
  sync script cannot reach the main checkout from a worktree).
- Operator action: the two removed keys are still in the main checkout's `services/orion-proposal-runtime/.env`.
  Delete them after the deploy (command below). No code reads them after this PR: proposal-runtime's
  settings dropped both fields, and nothing under `orion/` reads either name.

## Tests run

```text
services/orion-thought: tests + evals (whole suite)                              867 passed, 32 skipped
services/orion-thought/tests/test_painting_self_start.py                         31 passed (incl. the act-ledger end-to-end)
services/orion-proposal-runtime/tests                                            8 passed
services/orion-execution-dispatch-runtime/tests                                  128 passed
services/orion-cortex-exec/tests/test_render_scene_durable_kickoff.py            9 passed
tests/ + orion/{autonomy,proposals,policy,execution_dispatch,reverie,self_monitor,physiology}
  branch vs a clean origin/main worktree, same machine:  main 70 failed / branch 67 failed (45 collection
  errors on both: they need a database, FalkorDB or service installs). No new failure; 4 main failures
  now pass (test_maintenance_dispatch_gating x2, test_self_study_analysis_wiring, test_visual_baseline_pipeline).
orion/body_atlas/tests                                                            229 passed
orion/self_monitor/tests                                                          126 passed
CI workflow commands run locally (pip installs skipped): orion-static-gates (all OK after the fix to
  tests/test_check_verb_physiology.py), visual-baseline (all OK), thinking-rung3 (all OK except
  "Reading pull against Postgres", which needs the CI Postgres service), attend-act-loop (all OK),
  body-atlas, dig-kit, body-economy-analysis, temporal-self(-readers), substrate-neighborhood (OK).
  self-monitor, mind-tier-by-caller, self-readings, orion-prime: the same steps fail on clean origin/main
  in this environment (they need installs or live stores); CI is the authority there.
scripts/check_body_economy_contract.py   OK (101 state-reading modules)
scripts/check_env_template_parity.py     PASS
scripts/check_env_key_single_source.py   OK
```

## Evals run

```text
python -P services/orion-thought/evals/painting_self_start_replay.py                 9/9 PASS (fixture)
python -P services/orion-thought/evals/painting_self_start_replay.py --live --days 14  140 vs 129, ratio 1.085 (PASS)
  ... --interval-sec 3600 / 4500 / 6300 / 7200                                        calibration table above
python orion/body_atlas/evals/run_wire_eval.py      scanner facts 13/13, rules PASS, 4/4 mutations CAUGHT
python orion/body_atlas/evals/run_organ_structure_eval.py / run_capacity_roles_eval.py   PASS
python orion/autonomy/evals/run_rung3_replay_eval.py / run_rung3_reading_replay_eval.py  PASS
python orion/autonomy/evals/run_attend_act_loop_eval.py                                  PASS
```

## Docker/build/smoke checks

```text
docker build -f services/orion-thought/Dockerfile .   OK
in the image: app.main imports; the baked visual_baseline.v1.yaml loads (enabled, 5400, 600); flags on;
the durable-runs URL resolves to the default; /health's painting_self_start block renders
from the live thought container: GET http://orion-athena-durable-runs:8121/health -> 200, admission_enabled true
```

UNVERIFIED on the live rail: no self-started painting yet (not deployed).

## Review findings fixed

A code-review subagent read the whole diff, ran the suites, re-ran the live replay (140 vs 130 at its
time, ratio 1.077) and checked the containers live (thought reaches durable-runs on `app-net`; the
image has the baseline yaml). No blockers.

- Finding (should-fix): no test tied the run's deadline to the act's 5,400 s horizon any more (the
  deleted dispatch-route horizon test did).
  - Fix: `test_the_run_deadline_is_the_painting_acts_horizon` builds the run from the shipped yaml and
    asserts its deadline equals `visual_steps.PAINTING_HORIZON_SEC`.
- Finding (should-fix): the trigger settles its shadow want `submitted` at the POST, so a funded
  reservation was never released when the run later ended unpainted (27 of 167 in the replay).
  - Fix: `visual_steps._release_self_start_want` releases the want (Patch C3a `release`) when a
    self-started run ends, painted or not. Dream hops and dispatched runs are untouched.
  - Evidence: `test_the_want_is_released_when_its_run_ends` (7 cases).
- Finding (nit): the self-monitor gate's Redis and ledger reads ran on the event loop. Fix: both in
  `asyncio.to_thread`.
- Finding (nit): a failed submit still spent the 6 h probe. Fix: the probe is claimed only after the run
  is registered. Evidence: `test_a_failed_submit_keeps_the_probe`.
- Finding (nit): the receipt check was weaker than cortex-exec's kickoff, and a 409 read as
  `unreachable`. Fix: `post_run` also checks `workflow_kind` and `requested_resource`, and reports
  `http_<code>`.
- Finding (nit): `physiology_shadow.py` said a painting's want_id is the act's correlation id. Fix:
  reworded (a waking painting joins the act by `object_ref`).
- Found by the local CI run: `tests/test_check_verb_physiology.py` read the painting verb's actor from the
  dispatch route. Fix: it reads `PAINT_ACTOR` from the trigger.
- Found by the body-atlas wire eval: the hand-checked facts `orion-proposal-runtime ->
  http://orion-athena-thought:7155` and `orion-proposal-runtime reads orion:regulation:latest` are no longer
  true. Fix: now `orion-thought -> http://orion-athena-durable-runs:8121` (the `POST /runs`), `orion-thought
  reads orion:regulation:latest` and `orion-thought reads orion:self_monitor:latest` (the moved gate; it
  keeps the "no Redis constructor" mutation caught). The old proposal-runtime call is now a false fact.
- Not fixed, said plainly:
  - Two runs can be submitted for one need when the first waits more than 600 s for its claim (behind
    a dream hop). The later one ends `already_satisfied` and never reaches the GPU (6 in the replay; the
    old ladder had the same behaviour).
  - `GET /visual-chain/activity` and cortex-exec's `CORTEX_EXEC_RENDER_SCENE_*` keys have no live caller
    now. They go with PR 6.

## Restart required

Deploy from the clean primary `main` checkout after merge. The post-pull rebuild runs in alphabetical
order, which is already the safe order: the old trigger goes first, the new one last.

1. `orion-execution-dispatch-runtime` (e) rebuilds: dispatch can no longer send a painting.
2. `orion-proposal-runtime` (p) rebuilds: no more `render_scene` candidates or baseline reads.
3. `orion-thought` (t) rebuilds: the new trigger starts.

The gap between steps 1 and 3 has no painting trigger at all. It lasts as long as the rebuilds in
between (about 10 services). Nothing is lost: a painting due in the gap is issued on the trigger's first
tick (one need, never a burst). Paintings are 90 minutes apart, so the gap costs at most one late
painting. Between steps 1 and 2 the old proposal-runtime may still write the checkpoint; that only sets
the 600 s retry clock, which the new trigger obeys. A `render_scene` candidate still in flight is blocked
`no_cortex_route_for_proposal_kind`.

If you deploy by hand, use the same order and never start orion-thought first:

```bash
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
```

Other services the rebuild touches (they import the shared `orion/` package) need nothing extra.

After the three are up, delete the stale keys from the main checkout's local `.env` (one line):

```bash
sed -i '/^ORION_VISUAL_BASELINE_AROUSAL_ENABLED=\|^ORION_REGULATION_READER_MAX_AGE_SEC=/d' services/orion-proposal-runtime/.env
```

Live check after the deploy: `curl -fsS http://localhost:7155/health | jq .painting_self_start` shows the
loop ticking (`last.result` is `visual_baseline_not_due` between paintings, `submitted` when one starts).
Within a day: `thought.painting` acts with `test_before LIKE 'visual_dispatch:painting_self_start:%'`, and
0 new `render_scene` dispatch frames.

## Risks / concerns

- Severity: medium. Concern: the rate depends on how much of the time arousal is strained or the room is
  hot. This window had 83 strained hours and 10,476 hot minutes. In a calm, cool fortnight the trigger
  paints about 14-15 a day (the fixture's ungated day: 15). Mitigation: the old path did the same on calm
  days (real 10-27..11-04: 14-15 a day on calm days). The live week check after deploy compares
  against the 14-day range; the knob is `interval_sec`.
- Severity: low. Concern: the replay paints 8.5 % more than real. Mitigation: inside the check; the
  calibration table names the knob.
- Severity: low. Concern: the downstream ladder lanes for a baseline candidate (policy evaluator
  validation, dispatch builder/worker `visual_baseline` lanes, the allocator's baseline lane, dispatch's
  render settlement, cortex-exec's durable render kickoff) stay in code, unreachable. Mitigation: nothing
  can reach them now (no template, no route, no kind fallback; a test pins each). They are deleted with
  the three services in PR 6. Dispatch's render settlement keeps running so a run in flight at deploy
  time still settles its dispatch row.
- Severity: low. Concern: the old generate-step bid is gone for waking paintings, so want_ledger shows one
  `thought.painting` row per issued run instead of one per generate. Mitigation: that is the intent (the
  want before the act); dream hops keep theirs.

## PR link

(filled in after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
