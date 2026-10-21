# Thinking rung 3: bind the winner to action, and compete across doers

Design: `docs/superpowers/specs/2026-10-14-orion-thinking-architecture-design.md` (PR #2669), "Recommended
next patch" rung 3, with #2394's attend-to-act design and #2255's stage order (close a loop, world
effector, stakes, holdback). Juniper asked for the implementation directly. This touches autonomy, so
the proposal-mode answers are below.

## Summary

- **Binding fix.** A winner that world-first attention judged eligible is no longer dropped to "no
  action" by the old score ladder. Juniper's chat (`world:chat`) can now bind as a workspace winner.
- **Rung-2 acts compete in the motor allocator.** Painting dispatches are now scored on the
  painting act's own learned belief and measured cost. Before, they were refused as "unmeasurable"
  on every tick. The floor and every scoring constant are unchanged.
- **Holdback treatment arm on.** The per-tick randomized holdback is on at 0.2 (design D3). It only
  draws on ticks that can produce a control observation.
- **Orion-side loop resolver.** A chat reply that Orion opens while Juniper's chat is the workspace
  winner is linked to that loop. When the reply closes, Orion writes its own non-final `acted`
  verdict (`actor='orion'`).
- A replay eval over the real 48 h, new tests, a CI workflow, and README sections.

## Step 0: live numbers (read-only, 48 h ending 2026-10-21 03:18Z)

| What | Number |
|---|---|
| Proposal frames | 80,936 |
| ... with `winner_unbindable:no_selected_action` | 64,421 (80%) |
| Broadcast ticks (the workspace competition) | 4,028 |
| ... selected nothing (`none`), no open loop | 3,210 (80%) |
| ... selected `watch` / `defer` | 798 / 20 |
| ... eligible world-first winner + open loop, but `none` | 48 (49 on the eval's re-run) |
| ... attended `world:chat` (never bindable before) | 64 |
| Dispatch frames | 80,910; **0 dispatched** |
| Allocator refusals (candidate-level) | 162,481: `below_information_floor` 100,330, `unmeasurable` 62,151 |
| ... `render_scene` refused `unmeasurable` | 10,649 (10,785 on the eval's re-run) |
| Allocator log | 1,633 of 1,633 previews since the 10-20 23:36 restart refused everything |
| Holdback | global `ORION_DISPATCH_HOLDBACK_FRACTION=0.0`; shed per-template 0.5 |
| World-action episodes | 48 h: 1 control, 0 treated. All time: 11 control, 5 treated |
| `attention_loop_outcome` | 48 h: juniper `resolved` 5, `system:implicit_decay` 3, **orion 0**. All time orion `acted` 5 (shed only) |
| Outreach decisions (48 h) | 0 sent: `arousal_strained` 10,459, `quiet_hours` 6,122, `recent_chat` 217 |

## Root cause of `no_selected_action`

Three separate facts:

1. **Mostly honest calm, counted 20 times.** 98.5% of the `no_selected_action` frames match a
   broadcast tick that really selected nothing: every world-first candidate read "quiet/usual for
   itself". Proposal frames run about every 2 s and the broadcast about every 43 s, so each calm
   tick is re-reported about 20 times. The proposal reader agrees with the broadcast log on 99.8% of
   frames (100 disagreements, all within 0.5 s of a new broadcast row: a read race, not fixed).
2. **The real defect: two gates disagree.** World-first replaces an eligible loop's salience with
   its rank score, `percentile x event_decay`. That score fades over the 300 s orienting window.
   `select_actions` (`orion/substrate/attention/policy.py:96-117`) still applies its absolute
   0.48 / 0.35 ladder. That ladder was tuned for the old seed-weight score; the file says "disclosed
   not fixed". So once the fade crosses 0.35, a winner the gate still calls eligible becomes `none`.
   Live example: `node:substrate.route`, band high, percentile 0.93, event 252 s old, fade 0.20,
   rank score 0.191, so the action was `none`. That happened 48 times in 48 h.
   Fix: `world_first_action_floor` (`orion/substrate/attention_broadcast.py`) floors an eligible
   loop's `none` at `defer`. The gate's verdict is the bar. The ladder may still grade watch vs
   defer, but it may not drop an eligible loop. Past the window, the tick is calm as before.
3. **World winners could never bind.** `bind_workspace_winner` scanned only `node:` refs, so a
   `world:chat` winner was always `node_not_bound`. Fix: `world:` refs are bindable. Templates
   still bind only to refs they list.

## Outcome moved

- Painting dispatches get an honest verdict. 10,785 render_scene candidates in 48 h moved from
  `unmeasurable` to `below_information_floor`. The frame now says what painting would teach and cost:
  0.0164 nats, 57.1 s, 0.000287 nats/s.
- Orion writes its first loop verdicts from its own doing. Replay: 11 of 20 chat replies opened under
  a bindable `world:chat` winner. Each of those will now write `acted` (actor=orion) when it closes.
  Before this patch, the count was 0 in 48 h.
- 48 eligible-winner ticks in 48 h become bindable (`defer`).

## The proof asked for, and what it shows

**Proof wanted: allocator admissions from real candidates at an unchanged floor, plus treated
episodes.**

- **Admissions at the unchanged floor: 0 (replay over the real 48 h).** This is the honest result, not
  a wiring gap. The cross-doer table below uses one currency. Every rung-2 act has already been
  measured enough that one more run teaches less than the floor charges for the time it takes. The
  floor is 0.02 nats/s. A binary act that has been seen once is worth 0.20 nats, so it clears the
  floor only if it costs under ~10 s. Every doer costs minutes.

| Act | n | Nats | Measured cost | Nats/s | Verdict at 0.02 | Cost needed to clear | Dispatch seam |
|---|---|---|---|---|---|---|---|
| make, thought.painting | 29 | 0.0164 | 57.1 s (route p50, 30 d) | 0.000287 | below floor | 0.82 s | render_scene (all doer gates) |
| look, world_pulse_read | 119 | 0.0041 | 614.8 s (open->close p50) | 0.000007 | below floor | 0.21 s | none |
| make, dream_hop | 15 | 0.0303 | 2394 s | 0.000013 | below floor | 1.52 s | none |
| ask, outreach | 1 | 0.2027 | not measured | - | no cost | 10.1 s | none |
| say, outreach | 7 | 0.0589 | not measured | - | no cost | 2.9 s | none |
| say, chat_reply | 20 | 0.0233 | not measured | - | no cost | 1.2 s | none (reactive) |

- **Treated episodes:** the treated arm exists only for the one world-scored allocator action, the
  shed. It has 5 treated and 11 control all time, but none treated in this window. A painting has no
  meaningful control arm, because a held-back painting produces nothing by construction. With 0
  acting ticks in 48 h, the new per-tick draw also had nothing to take. Live treated episodes from
  this patch: **UNVERIFIED** (0 expected until something is admitted).
- What would move it, and why it needs Juniper's decision: the attend-to-act design's stage 3
  (gate-definition changes). Either (a) score the current state's cell, or (b) admit an action proven
  to help (`confidently_helpful`). Without (b), the allocator only buys information, and every
  well-measured act is refused for being well-measured. That decision is out of scope here.

## Second pass: Juniper's decisions of 2026-10-21

### 1. Stage 3 option B: admit an act proven to help

This is a second, separate admission path. The information floor (0.02 nats/s) and its constants are
unchanged. The new path only reconsiders act-backed candidates that the floor refused
(`below_information_floor`): they are measured, have a cost and are not harmful.

- **Proven.** Every bar comes from code or data that already existed:
  - Enough observations: n >= 20. This is the per-arm minimum in the attend-to-act design, D3.
  - A credible lower bound: LB = mean - 2 sd. The 2-sigma figure is the allocator's existing
    `HARM_CONFIDENCE_SIGMAS`, mirrored.
  - LB must be above a reference rate. Once the act's own randomized control arm has 20 resolved
    rows, the reference is that arm's rate, using the Jeffreys mean (k+0.5)/(n+1). Before that, it
    is the act learner's own prior mean (0.5).
- **Value, in the allocator's currency.** The value is KL(Bernoulli(LB) || Bernoulli(reference)) in
  nats, divided by the measured motor-seconds. In words: how far the act moves its outcome away from
  what happens without it, read at the conservative bound. Candidates are admitted best value first,
  inside the motor allowance that is left.
- **No floor on this path.** The information floor exists to retire acts that have nothing more to
  teach. A proven act is exactly that kind of act, so the floor would refuse it by design. This path
  is still bounded by:
  - the allowance;
  - the in-flight rule: a decision in either arm holds the act for its horizon, 5400 s for a painting;
  - the holdback;
  - every gate in the doer's own path.
- **Holdback 0.2 on this path.** The draw is stable per dispatch. Both arms are precommitted to the
  new table `substrate_act_episodes`, via an additive migration
  (`manual_migration_act_episodes_v1.sql`, `-- ORION-MIGRATION-REQUIRED-BY:
  orion-execution-dispatch-runtime`); the table is also created lazily. If the row cannot be written,
  the act is not sent. For a producer-reported signal, a control row is 0 by construction, because a
  held-back act produced nothing.
  - This needs its own table: the feedback runtime scores every due world-episode row as a cabinet
    shed, so act rows in that table would be scored wrongly.
- **Reason strings.** Admitted candidates carry `admitted:proven_effect:lb=..:ref=..:<source>:nats=..:nats_per_sec=..`.
  Refused ones carry `proven_effect:<why not>:...`.
- **Kill switch.** `ORION_DISPATCH_PROVEN_EFFECT_ENABLED=true`, set in `.env_example`, settings and
  compose. The local `.env` is synced.

### 2. Painting grading: a run that never reached the GPU is not an act

- The `make` act now opens at the first real generation, for both painting and dream hop. That is the
  moment the single-flight lock is held and the stage is marked `generating`. Before, it opened when
  the plan was frozen.
- A close that did not produce an image is published only when the act was opened. Runs refused or
  abandoned before any work record no outcome. A run that started work and then failed keeps its 0.
- Reading (#2733) already works this way: it opens on `run.started`, and closes only if it opened.
  No change was needed there.
- **The existing beliefs are pure outage.** All 44 existing `make` closes (29 painting, 15 dream hop)
  belong to attempts that never reached the GPU: no `generating_started_at` on their
  `reverie_visual_attempt` row. Under the new rule, none of them would have been graded.
- **No production rows were changed.** If Juniper wants the two beliefs reset to cold, this is the one
  line (Postgres, `conjourney`). The grammar_acts and outcome history stays.
  ```sql
  DELETE FROM substrate_action_effect_posterior WHERE dispatch_kind = 'act:make' AND target_id IN ('thought.painting', 'thought.dream_hop') AND signal_id = 'artifact_produced' AND baseline_bin = 0;
  ```
  - Recommendation: reset. The current belief (mean 0.017 from 29 runs that never ran) blocks the
    proven path for painting until at least 20 real closes outvote it. It also wrongly tells the
    information path that paintings are well understood.

### 48 h replay, both paths, unchanged floor

- **Render candidates seen: 10,726** (non-baseline, in time order, with the in-flight rule).
- **Belief as stored:** 0 admissions on either path.
  - Information path: below the floor.
  - Proven path: `not_above_reference`. The lower bound is -0.166; the reference is 0.5.
  - Held back: 0.
- **Belief under the new grading rule** (outage closes excluded, so the cell is cold):
  - 1 admission on the information path, through the existing cold-start rule.
  - The act is then in flight for 452 candidates.
  - After that, the proven path refuses with `too_few:1<20`: one replayed close is not enough.
  - Proven-path admissions: 0. Held back: 0.
  - The scheduled-painting lane, which keeps grading the same belief, is not replayed. In this learner
    the variance does not depend on the value observed, so replayed closes are folded in as spread
    only. Any decision that would rest on an invented mean is reported, never counted.
- **By act, across doers** (one currency; only painting has a dispatch seam):

| Act | Both paths at the unchanged floor |
|---|---|
| look, world_pulse_read (n=119, mean 0.696) | **proven_effect: admitted** (LB 0.604 > 0.5, 0.022 nats / 615 s); no dispatch seam, so informational only |
| make, thought.painting | refused: not proven (outage belief) |
| make, thought.dream_hop | refused: too_few 15 < 20 |
| say/ask, outreach; say, chat_reply | no measured cost |

- **Which would be held back:** at 0.2, a share of the proven-path admissions. In this window there
  were none. Live control and treated rows on the new path: UNVERIFIED (0 expected until the painting
  belief is real or another doer gets a seam).

### Metric quality gate: the "proven effect" signal

1. **Provenance.**
   - The mean, variance and n come from the act's `substrate_action_effect_posterior` cell, written by
     `close_act` in `orion/grammar/act_ledger.py`.
   - The control rate comes from `substrate_act_episodes` control rows, written by
     `_proven_act_holdback` in the dispatch worker.
   - The cost is the route's settled latency.
2. **Independence.**
   - The lower bound reads the same posterior as the information score. The information score reads
     only the variance; this path also reads the mean. So it is a second reading of one belief, not
     new evidence. That is the intent: exploit versus explore on one belief.
   - The reference rate comes from a randomized arm, which is independent by construction.
3. **Theory anchor.**
   - A one-sided 2-sigma credible bound on a Normal approximation of the posterior.
   - KL divergence between the outcome with the act and the outcome without it. That is the expected
     log-likelihood ratio per run, in nats.
   - Weak spots, stated plainly:
     - Before 20 controls, the bar is the learner's prior (0.5): "better than the coin flip it started
       from". This is not an effect against a control.
     - For artifact acts, the control rate is 0 by construction. After 20 controls, the bar falls to
       about 0.024, which is a weak claim.
     - The Normal approximation is poor near 0 and 1.
4. **Live sanity.**
   - Reading reads proven: n=119, LB 0.604.
   - The painting and dream-hop inputs were degenerate: every close was an outage. That is fixed going
     forward by the grading rule, and the reset is Juniper's call.
   - The signal has a real rest state: "not proven", value 0. It is not saturated: 1 of 6 cells is
     proven.
5. **Existing mechanism.** `HARM_CONFIDENCE_SIGMAS`, the D3 n >= 20 rule, and the world-episode
   precommit and stable-draw pattern. `orion.autonomy.contrast` is for field-signal cells and has no
   act arm, so it was not reused.
6. **Reversibility.** The kill switch, and one additive table. The rows can be dropped.

## Current architecture (before)

- Workspace: `orion-substrate-runtime` broadcast (world-first gate) -> `select_actions` ladder ->
  projection. The proposal layer binds only `shed_background_gpu` to the winner.
- Motor allocator (`orion/autonomy/allocator.py`, enforcing): candidates are about 13 housekeeping
  and introspection templates plus render_scene (unmeasurable) and the shed.
- Rung-2 acts (`grammar_acts`, `orion/grammar/act_ledger.py`) graded into EffectPosterior; nothing
  in the allocator read them.
- Doers: painting is dispatched by this runtime. Outreach, reading and dreams self-schedule. Chat
  replies are reactive.

## Architecture touched

- `orion-substrate-runtime` (via `orion/substrate/attention_broadcast.py`): action floor.
- `orion-proposal-runtime` / sql-writer (via `orion/autonomy/self_shed.py`): `world:` binding.
- `orion-execution-dispatch-runtime`: act-backed candidates, in-flight guard, holdback draw rule.
- `orion-sql-writer` (via `orion/grammar/act_ledger.py`): workspace stamping and Orion's verdicts.
- Config: `config/execution_dispatch/execution_dispatch_policy.v1.yaml` (`act:` on render_scene).

## Files changed

- `orion/substrate/attention_broadcast.py`: `world_first_action_floor`.
- `orion/autonomy/self_shed.py`: `BINDABLE_REF_PREFIXES` (`node:`, `world:`).
- `orion/autonomy/allocator.py`: `Candidate.observation_variance` (default unchanged), the
  `act_in_flight` reason, and pass-through in `candidate_from_dispatch`.
- `orion/autonomy/act_candidates.py` (new): act cell -> allocator candidate, expectation reason,
  in-flight refusal.
- `orion/execution_dispatch/policy.py`: `ActRouteRefV1`, `act_refs_by_cortex_verb`, one act per verb.
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: `render_scene.act`.
- `services/orion-execution-dispatch-runtime/app/{worker,store,settings}.py`, `.env_example`,
  `docker-compose.yml`, `README.md`.
- `orion/grammar/act_ledger.py`: `ACT_ANSWERS_WORKSPACE_REFS`, `bind_act_to_workspace`,
  `write_orion_loop_outcome`, savepoint isolation, kill switch.
- `services/orion-sql-writer/app/{main,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`,
  `tests/conftest.py`.
- Tests: `tests/test_rung3_world_first_action_floor.py`, `tests/test_rung3_act_candidates.py`,
  `services/orion-sql-writer/tests/test_act_workspace_binding_integration.py`.
- Eval: `orion/autonomy/evals/run_rung3_replay_eval.py`.
- CI: `.github/workflows/thinking-rung3.yml` (new); `.github/workflows/orion-sql-writer-tests.yml`
  (adds the new must-not-skip integration test).

## Schema / bus / API changes

- Added: `CortexRouteTemplateV1.act` (config model, read only by the dispatch runtime).
- Removed / renamed: none.
- Behavior changed: the `attention_loop_outcome` table now receives `acted` rows with
  `actor='orion'` from sql-writer (shape and verdict already in use by the shed path). Frame candidates
  carry one more `act_expectation:...` string in `reasons`.
- Added table: `substrate_act_episodes` (additive migration `services/orion-sql-db/manual_migration_act_episodes_v1.sql`, REQUIRED-BY orion-execution-dispatch-runtime; fits the post-pull additive allowlist; also created lazily).
- Config model: `ActRouteRefV1.horizon_sec` (required; render_scene 5400).
- Compatibility: no bus payload or registry entry changed. The painting act now opens at generation, not prepare (same event shapes).

## Env/config changes

- Added: `ORION_DISPATCH_ACT_CANDIDATES_ENABLED=true`, `ORION_DISPATCH_PROVEN_EFFECT_ENABLED=true`, `ORION_DISPATCH_ACT_COST_LOOKBACK_HOURS=720`
  (execution-dispatch-runtime); `SQL_WRITER_ACT_WORKSPACE_BINDING_ENABLED=true` (sql-writer).
- Changed: `ORION_DISPATCH_HOLDBACK_FRACTION` 0.0 -> 0.2 (`.env_example`, settings default, compose).
- `.env_example` updated: yes, both services.
- Local `.env` synced: yes. `sync_local_env_from_example.py` added the sql-writer key. For the
  dispatch runtime, `--all-keys --force orion-execution-dispatch-runtime` added the 2 keys and moved
  the holdback 0.0 -> 0.2. That was the only diverged key for that service.
- Skipped keys requiring operator action: none.

## Metric quality gate

Two new signals reach a consumer: the act-backed allocator score for render_scene, and Orion's
`acted` verdict rows.

**Act-backed allocator score**

1. **Provenance.** The belief is written by `close_act` in `orion/grammar/act_ledger.py`. Its inputs
   are painting closes from `painting_act_transition` in `services/orion-thought/app/visual_steps.py`.
   Cost is `substrate_dispatch_results.latency_ms`, settled to the run's real work by
   `orion/execution_dispatch/visual_settlement.py`.
2. **Independence.** It replaces a None. render_scene had no other score. The value and the cost
   describe the same runs: what one painting teaches and what it takes. That is the intended pairing.
   It is not two copies of one signal.
3. **Theory anchor.** The expected information gain of a Normal-Normal update,
   `0.5 ln(1 + s^2/t^2)`. The allocator docstring derives it and checks it against Monte Carlo. Here
   `t^2` = 0.25 is the act learner's own variance: the maximum of `p(1-p)` for a binary event.
4. **Live sanity.**
   - The posterior rebuilt from `grammar_acts` closes matches the stored table for all 6 act cells.
     The painting cell is n=29, mean 0.017.
   - Caveat: the painting belief was learned during the GPU fault, when nearly every painting failed.
     It is confidently pessimistic, and the score has no way to notice the fault is fixed. Information
     gain only rests at 0 as n grows, so it is never "calm". The refusal does not depend on this: even
     a never-run act at 57 s is admitted only once.
5. **Existing mechanism.** It reuses the allocator, `candidate_from_dispatch`'s conventions, the
   rung-2 posteriors and the route cost history. No new learner.
6. **Reversibility.** The kill switch plus one yaml block. Nothing is baked into a schema.

**Orion's `acted` verdicts**

1. **Provenance.** `write_orion_loop_outcome` in `orion/grammar/act_ledger.py`, at act close.
2. **Independence.** They are not read by competition. `verdicts.py` `TERMINAL_VERDICTS` excludes
   them. The decay digest blocks only on terminal verdicts. orion-thought filters `acted` out.
3. **Theory anchor.** None is claimed. This is a trace (act answered loop X, outcome Y), not a score.
4. **Live sanity.** Replay: 11 of 20 replies bind.
5. **Existing mechanism.** The same row shape the shed settle path writes.
6. **Reversibility.** The kill switch. The rows can be deleted by `actor='orion'` plus
   `features_at_close ? 'act_id'`.

## Proposal-mode answers

- **Capability change:**
  - Paintings now compete for motor time on what they would teach.
  - The per-tick holdback can now withhold an acting tick, at most 20% of ticks with a field-signal
    candidate.
  - Orion records which workspace loop its chat reply answered, and how that went.
  - No new doer can be dispatched. The allocator still only admits or refuses.
- **Data touched:**
  - Reads: `substrate_action_effect_posterior`, `substrate_dispatch_results`, `grammar_acts`,
    `substrate_attention_broadcast_log`.
  - Writes: `grammar_acts.workspace_ref`, and `attention_loop_outcome` rows with `actor='orion'`.
- **Privacy boundary:** no message text is read or written. The verdict carries only act ids, the
  verb, the signal and the outcome.
- **Trace that proves it worked:**
  - The dispatch frame reasons show `act_expectation:make:thought.painting:...`.
  - `allocator:below_information_floor` replaces `allocator:unmeasurable` on render_scene.
  - `grammar_acts.workspace_ref` is set on `hub.chat_reply` rows.
  - `attention_loop_outcome` gets rows where `actor='orion'` and `features_at_close->>'act_id'`
    is not null.
  - Blocked candidates show `randomized_holdback` only on ticks with a field signal.
- **Dangerous failure modes and their guards:**
  - Cold-start painting spam. A never-graded act is admitted on every tick until its first close.
    Guard: one in flight per act (an open act or a pending dispatch, 6 h). Read errors fail closed.
  - Self-silencing. Orion writes only `acted`, never `resolved`, and there is a test for it.
  - A lost act. If the binding fails, the act still lands; each side step runs in its own savepoint.
    This was a real defect caught by the integration test.
- **Disable / roll back:**
  - `ORION_DISPATCH_ACT_CANDIDATES_ENABLED=false` puts render_scene back on its old scoring.
  - `ORION_DISPATCH_HOLDBACK_FRACTION=0.0` turns the holdback off.
  - `SQL_WRITER_ACT_WORKSPACE_BINDING_ENABLED=false` stops the stamping and the verdicts.
  - Restart the affected service after any change. No data migration is needed.

## Tests run

```text
tests/test_rung3_world_first_action_floor.py           7 passed  (2 fail on pre-fix code, verified)
tests/test_rung3_act_candidates.py                    20 passed  (holdback test fails on old draw, verified)
attention / broadcast / world_first suites            435 passed
allocator, shed pipeline, world action, holdback, proposal policy/scoring,
  world settlement, action outcome resolution         192 passed
services/orion-execution-dispatch-runtime/tests        107 passed
services/orion-proposal-runtime/tests                   13 passed
sql-writer act ledger + new binding integration (scratch postgres:16 container)   32 passed
services/orion-sql-writer/tests (full)                 14 failed / 962 passed; main: 15 failed / 961 passed (pre-existing, env-sensitive)
tests/test_execution_dispatch_runtime_worker.py        39 failed / 29 passed, identical on main (needs a live DB; not in CI)
scripts/check_env_template_parity.py                   PASS
scripts/check_definition_drift.py                      750 definitions, 0 changed
scripts/check_migration_required_by.py                 OK (0 new migrations)
```

## Evals run

```text
python orion/autonomy/evals/run_rung3_replay_eval.py            6/6 PASS (fixture)
ORION_PG_DSN=... run_rung3_replay_eval.py --live --hours 48     render_scene 10,785: unmeasurable -> below_information_floor;
                                                                admissions at floor 0.02: 0; posterior rebuild 6/6 match;
                                                                eligible winners dropped 49; world: ticks 64; replies that bind 11/20
--hours 336                                                     statement timeout (560k frames); 48 h is the asked window
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, requirements or compose wiring changed beyond env lines, and the task forbids
deploying. Live behaviour after restart: UNVERIFIED.
```

## Review findings fixed

The review was done by hand in a subagent (no code-review skill is installed). Target: the branch
diff against origin/main. It found no blocker.

- Finding: the visual-baseline lane was act-scored: 30-day cost, no typed fallback, a direction claim.
  A quiet month would have refused scheduled paintings forever.
  - Fix: baseline items keep their old cost path and are never act-scored; their act numbers are
    still recorded (commit 1d62c1bfc).
  - Evidence: `test_visual_baseline_lane_is_unchanged_even_with_no_act_cost`.
- Finding: a failed belief read gave `{}`, which read as cold, so the cold-start rule would admit a
  render.
  - Fix: `_posteriors_read_ok`. An act-backed candidate is refused when the read failed.
  - Evidence: `test_unreadable_belief_fails_closed_not_cold`.
- Finding: in-flight missed renders whose submit timed out, and two render candidates in one tick.
  - Fix: count `settlement.state='not_submitted'`, and allow one act per tick.
  - Evidence: `test_one_act_per_tick`.
- Finding: in-flight holds only while the allocator decides the send.
  - Fix: documented in the README. orion-thought's single-flight is the backstop.
- Finding: an `acted` verdict would be written for any `workspace_ref`, whatever the actor.
  - Fix: gated on `ACT_ANSWERS_WORKSPACE_REFS`.
- Finding: the holdback wording read broader than the rule.
  - Fix: README clarifies that the rule only decides whether a tick is drawn.
- Disclosed, not changed:
  - The action floor applies to body winners too. A fading cabinet or biometrics winner now becomes
    `defer`, which the shed template can bind to; the shed still needs its own eligibility. In the
    last 48 h, the rescued winners were only chat, codebase, execution and route.
  - A leading `world:` ref now wins over a later `node:` ref. Live attended lists hold one ref, so
    this is latent.
- Finding: the painting belief is an outage. All 29 closes are `abandoned:run_abandoned` or
  `attempt_missing` (10-16 to 10-18).
  - Not fixed here; it is recorded in the metric gate and the concerns. Whether
    infrastructure-abandoned runs should grade the act at all is a rung-2 question.
  - Note: in this learner, the posterior variance does not depend on the values observed, so 29
    successes would score the same nats. The refusal is not caused by the outage. The belief can
    still move, because the baseline lane keeps producing painting closes.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
./scripts/safe_docker_build.sh orion-sql-writer up -d --build
./scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
./scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
./scripts/safe_docker_build.sh orion-thought up -d --build
```

Run from the clean primary `main` checkout after merge. The post-pull hook applies `manual_migration_act_episodes_v1.sql` (additive). If it has not, the wrapper refuses the dispatch-runtime `up` and prints the apply command. Note: the live dispatch-runtime `.env` already
says `ORION_DISPATCH_HOLDBACK_FRACTION=0.2`. A restart of the old image before this merges would
draw on every acting tick without the new rule. Today that means baseline paintings and sheds would
be delayed a few seconds. Low harm, but restart only after the merge.

## Risks / concerns

- Severity: medium. The proof's admissions half is not met at the unchanged floor (0 admissions).
  The floor cannot be lowered without a gate-definition decision (stage 3). Mitigation: the cross-doer
  table in this report gives that decision real numbers.
- Severity: medium. No doer other than painting has a dispatch seam. Outreach self-schedules on a
  deterministic tension trigger, so a "request" would repeat its own tick. Reading has no arousal gate
  and no daily cap, so making it dispatchable would add uncapped GPU load. Chat replies are reactive.
  None of them were made dispatchable.
- Severity: medium. The painting belief (n=29, mean 0.017) is made only of outage closes
  (`abandoned:run_abandoned` / `attempt_missing`, 10-16 to 10-18). It measures the fault, not
  painting. That fails metric-gate step 4 for the predicted mean. The nats score is unaffected,
  because variance does not depend on the values. Follow-up: decide whether infrastructure-abandoned
  runs grade a `make` act (rung 2).
- Severity: low. The live dispatch-runtime `.env` in the primary checkout already has holdback 0.2.
  A rebuild of the old image before merge would draw on every acting tick. Mitigation: merge, then
  restart; or set it back to 0.0 by hand until then.
- Severity: low. The proposal reader races a fresh broadcast row on about 0.1% of frames. Not fixed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2830

Status: DONE_WITH_CONCERNS. Both paths admit 0 in the 48 h replay with the belief as stored. The painting belief is pure outage; the reset is Juniper's call. Only painting has a dispatch seam.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
