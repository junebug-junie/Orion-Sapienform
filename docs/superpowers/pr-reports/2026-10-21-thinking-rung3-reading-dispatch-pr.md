# Thinking rung 3 follow-up: a reading pull the allocator can admit, and a principled holdback rule

Follow-up to PR #2830 (`docs/superpowers/pr-reports/2026-10-21-thinking-rung3-bind-and-compete-pr.md`).
Juniper approved two changes in one PR. This touches autonomy, so the proposal-mode answers are below.

## Summary

- **Reading can now be dispatched.** The motor allocator may admit a "reading pull": one extra
  world-pulse read, started now, on the seed Hub's own reading lane would claim next. The allocator
  only admits or refuses. Hub runs every gate and the reading pipeline's own path.
- **Reading got the gates a doer needs.** Before it starts a pulled read, Hub checks, in order: a kill
  switch, the lane's own Wallet A gate (window, refund backoff), a daily cap of 8 (derived from data),
  arousal `strained`, and thermal `hot`. Arousal and thermal reuse the painting rules and code.
- **Holdback follows a rule, not a name list.** A control arm is drawn only when the outcome can
  happen without the act. A painting's object is made by its own dispatch, so it is never held back.
  A reading pull picks a seed that already exists and that the lane may read anyway, so it is held
  back, and both arms are scored the same way after the horizon.
- New contract (bus channel, schemas), one additive migration, kill switches ON, env synced, tests
  (unit + real Postgres), a replay eval, CI, README sections.

## Outcome moved

- Reading moved from "proven to help, no dispatch path" (#2830) to "admitted on the proven path, then
  refused by Hub's arousal gate". Replay of the real 48 h: 420 proven admissions, every one refused at
  Hub's `check` with `arousal_strained`. Zero reads started, zero episodes written. That is the gate
  working: arousal has read `strained` (cabinet hot, 33.5 C) since 2026-10-18 12:07 without a break.
- Painting can no longer get a control row that is 0 by construction. #2830 drew one at 0.2.

## Current architecture (before)

- The reading lane (`services/orion-hub/scripts/world_pulse_read_pipeline.py`) drains the seed queue
  on its own every 300 s, up to 3 reads in flight. No daily cap, no arousal or thermal gate.
- The `look` act (`hub.world_pulse_read`, `artifact_produced`) is graded on every read: n=119, mean
  0.696, lower bound 0.604. #2830's replay found it "proven", but no route produced it.
- Holdback on the proven path was drawn for every act. A painting control row was written as 0.0
  (`STRUCTURAL_ZERO_SIGNALS`), and the reference rate ignored it via `max(prior, control)`.

## Design decisions, with the evidence

### Which read to dispatch

- The real selection seam is the queue's own claim order, `CLAIM_SQL` in
  `orion/world_pulse_read/queue.py`: priority, then age. Worth-reading digest items and findings get
  there via `enqueue_from_recent_digests` -> `seeds_from_digest_payload` (`worth_reading` URLs).
- The pull takes the first pending seed in that order that is not already in flight (no unconsumed
  Stage 1 durable turn) and was never pulled (`SELECT_PULL_CANDIDATE_SQL`).
- It **adds** a read. It does not reorder one. This is why it needs a cap and gates. A reorder would
  add no load, and its gates would change nothing.

### Daily cap: 8, derived

- Data: distinct seeds whose Stage 1 durable turn started, per UTC day, last 30 days (25 days with
  reads): `1 12 15 27 5 6 6 8 8 5 5 6 2 2 23 6 4 8 46 49 45 22 13 27 15`.
- Rule (`derive_daily_cap`): the median reading day. Pulls may add at most one more typical day of
  reading. The median, so the 10-14..10-16 backlog burst (46-49/day) does not set the bound.
- Result: 8. The eval re-derives it live and checks it against the shipped defaults.
- In practice the in-flight rule binds first. One decision holds the route for 12000 s, so at most
  7.2 pulls fit in a day. The cap is a backstop.

### Horizon: 12000 s, derived

- p90 of (Stage 1 durable turn submitted -> look act closed), over the 83 successful reads with both
  on record: p50 4205 s, p90 11992 s. Most of it is waiting for a GPU (submission -> work start p50
  57 min).

### Cost: 679 s, measured

- The allocator's cost for a pull is the look act's own median open->close time over 30 days
  (`act_duration_p50_sec`, live 679.3 s). The act opens when the read's work starts, so this is the
  work, not the queue. The dispatch RPC itself takes milliseconds and is not charged.
- Note: the cost is work time; the horizon is measured from submission and includes the GPU queue.
  That is right for each job: the allocator budgets work, the episode waits for the outcome.

### Holdback rule (part 2)

- Rule: `act_episodes.holdback_rule(object_source)`.
  - `dispatch`: the dispatch makes the act's object. Painting: the run id is
    `reverie_visual_run_id(dispatch_id)`. No dispatch, no object, no outcome. Never held back. The
    treated row says `holdback_exempt:outcome_requires_act:object_minted_by_dispatch`.
  - `queue`: the dispatch picks an object that already exists. Held back.
- Declared on the route (`act.object_source`) and pinned by a test on the fact it rests on.
- `STRUCTURAL_ZERO_SIGNALS` is removed, not left beside the rule.
- **Is reading's control informative?** Yes. A held-back pull still names its seed (Hub marks it
  `pull_arm=control` and starts nothing). The lane keeps draining the queue, so it may read that
  seed anyway. Both arms use one outcome: did a graded `look` act with key `read_seed:<seed>` close
  with outcome 1 between the decision and the horizon (`act_episodes.resolve_due`).
- What the contrast measures, plainly: "does an extra read slot get this seed read within 3.3 h more
  often than the lane alone". The pull takes the seed the lane would read next. So the honest
  expectation is a small effect, maybe none. If the control arm reads as well as the treated arm,
  the proven path's reference rises to the control rate and the claim is revoked. That is the design.
- Both arms on one rule (review fix): until the treated arm has 20 resolved episodes, the verdict
  compares the look posterior (P(read succeeds | work started), lane-wide) to the bar. That
  posterior is biased high against the control rate (P(read by horizon | decision), which counts a
  skipped seed as 0). From 20 treated episodes on, the act is judged on its treated arm instead
  (`episode_posterior`, same Jeffreys estimator as the control). The frame reason says which
  (`belief=posterior` or `belief=treated_arm`).

## 48 h replay (both paths, unchanged floor 0.02)

`ORION_PG_DSN=... ORION_BUS_URL=... python orion/autonomy/evals/run_rung3_reading_replay_eval.py --live --hours 48`

Re-run after the review fixes, window ending 2026-10-22 ~06Z. (The first run, ending 2026-10-21
05Z, gave 10,673 ticks / 420 admissions / calm 11 + 4, the same picture.)

- Population: one pull candidate per proposal tick. The ticks are the 7,715 non-baseline
  render_scene candidates the dispatch runtime saw (the same no-dimension, always-emitted template
  shape).
- Information path: refused `below_information_floor` on every tick (look n=119 teaches ~0 nats).

| | As lived (arousal strained) | Counterfactual (arousal calm) |
|---|---|---|
| Proven-path admissions | 289 | 11 |
| Hub `check` RPCs | 289 (one per 360 s hold) | 11 |
| Refused at Hub | 289 `arousal_strained` | 0 |
| Reads started (treated) | 0 | 8 (5 / 2 / 1 on 10-20 / 10-21 / 10-22) |
| Holdback draws (control) | 0 | 3 |
| Cap hits | 0 | 0 |
| Refused before the allocator | 7,426 `doer_refused_hold` | 7,704 `act_in_flight` |

- Per day, calm: about 4-5.5 reads started and 1.5-2 controls, against the lane's median of 8 a day.
  The in-flight rule (12000 s) binds before the cap.
- Painting: 0 control draws possible by the rule. #2830's replay had 0 painting admissions anyway.
- Arousal: Redis `orion:regulation:latest` reads `strained` since 2026-10-18 12:07 (cabinet hot,
  33.2 C now). Outreach decisions agree: every waking-hour decision in the window was
  `arousal_strained`.
- Not replayable, so not applied: thermal (no cabinet temperature history in Postgres), Wallet A
  refund backoff (Redis only), seed availability per tick (25+ pending seeds all window), and the
  treated-arm belief rule (no episodes exist yet).
- Horizon drift: on the re-run the same derivation reads p90 12723 s (n=85, was 11992 s on n=83).
  12000 s is kept as the value derived on 2026-10-21; at most about 5% of reads close in the gap,
  and they count 0 in both arms alike.

## Metric quality gate: the reading-pull episode outcome

1. **Provenance.** Episode rows: `_proven_act_holdback` (dispatch worker), object from Hub's `check`
   then `act` reply (`_settle_pull_episode`). Outcome: `RESOLVE_DUE_SQL` in
   `orion/autonomy/act_episodes.py`, reading `grammar_acts` closes written by `close_act`
   (`orion/grammar/act_ledger.py`) from `_close_read_act` in the reading pipeline (1 =
   `stage1_completed`).
2. **Independence.** The treated arm's outcomes are look closes, so they also feed the look posterior:
   the same evidence, read twice. The control arm is independent of the decision by randomization.
   Until the treated arm has 20 resolved rows the verdict reads posterior vs control; after that it
   reads treated arm vs control arm, one resolver, one estimator (Jeffreys).
3. **Theory anchor.** A randomized controlled trial with one outcome rule for both arms, drawn only
   among decisions that passed every gate (Hub's gates run before the arm matters).
4. **Live sanity.** No live rows yet: arousal has been strained for 3 days, so nothing passes `check`.
   UNVERIFIED. Rest state: a seed that is never read reads 0, not NULL; a voided decision reads NULL
   and does not count; both checked against real Postgres in `tests/test_rung3_reading_pull_pg.py`.
5. **Existing mechanism.** Reuses `substrate_act_episodes`, the stable draw, the in-flight rule, the
   act ledger, `CLAIM_SQL`'s order, `ArousalReader`, `thermal_state`, `wallet_a_block_reason`.
6. **Reversibility.** Two kill switches. Six nullable columns. The episode rows can be deleted by
   `act_label = 'look:hub.world_pulse_read:artifact_produced'`.

## Proposal-mode answers

- **Capability change:** Orion can choose to read one more world-pulse item now, at most 8 a day,
  one decision per 3.3 h. 20% of those decisions are held back to measure whether choosing helps.
  Paintings are no longer held back.
- **Data touched:** reads `grammar_acts`, `world_pulse_read_seed`, `reading_durable_turn`, Redis
  `orion:regulation:latest`, Hub's cabinet sensor file. Writes `world_pulse_read_seed.pull_*` and
  `substrate_act_episodes` (new columns `object_ref`, `void_reason`, `resolved_at`).
- **Privacy boundary:** no text is read or written by the new path. Seeds are public URLs already in
  Orion's queue.
- **Trace that proves it worked:** dispatch frame reasons `admitted:proven_effect:...`, blocked
  `doer_refused:<why>` or `act_holdback`; Hub log `reading_pull op=... state=...`,
  `reading_pull_started`; `world_pulse_read_seed.pull_arm`; `substrate_act_episodes` rows with
  `object_ref`, later `outcome_value` and `resolved_at`; dispatch log `act_episode_resolved`.
- **Dangerous failure modes and guards:**
  - GPU load runaway: daily cap, one decision per 12000 s, arousal and thermal gates, GPU pool.
  - RPC spam while strained: one `check` per hold (360 s), not one per tick (43 s).
  - Biased arms: Hub's gates run before the arm; a refused decision is voided in both arms.
  - A treated seed stuck `claimed` if Hub dies mid-start: the lane's existing stale-claim reclaim
    returns it to `pending`; it keeps its pull mark, so the lane reads it like any other.
- **Disable / roll back:** `ORION_DISPATCH_READING_PULL_ENABLED=false` (dispatch runtime) or
  `HUB_WORLD_PULSE_READ_PULL_ENABLED=false` (Hub), then restart that service. No data migration.

## Architecture touched

- Contract: `orion:world_pulse_read:pull:request` / `:reply:*`, `WorldPulseReadPullRequestV1` /
  `WorldPulseReadPullResultV1`.
- `orion-execution-dispatch-runtime`: route, holdback rule, preflight check, control pulls, sends,
  episode resolution, act-duration cost.
- `orion-hub`: pull gates, pull listener, lane lock.
- `orion-proposal-runtime` (config only): template `read_world_pulse`.
- `orion-sql-db`: one additive migration.

## Files changed

- `orion/schemas/world_pulse_read_pull.py` (new), `orion/schemas/registry.py`, `orion/bus/channels.yaml`: contract.
- `orion/world_pulse_read/pull.py` (new): gates, derivation, SQL.
- `orion/autonomy/act_episodes.py`: `holdback_rule`, `set_object`, `void_decision`, `resolve_due`, columns.
- `orion/autonomy/act_candidates.py`: `ActRef.object_source`, `refuse_held`.
- `orion/execution_dispatch/policy.py`: `ActRouteRefV1.object_source`, `READ_PULL_EXECUTOR_VERB`.
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: route `read_world_pulse`; `render_scene.act.object_source: dispatch`.
- `config/proposals/proposal_policy.v1.yaml`: template `read_world_pulse`.
- `services/orion-execution-dispatch-runtime/app/{worker,store,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`, `scripts/world_pulse_read_pull_listener.py` (new), `scripts/main.py`, `app/settings.py`, `.env_example`, `README.md`.
- `services/orion-sql-db/manual_migration_reading_pull_v1.sql` (new).
- `config/metrics/metric_definitions.lock.json`: re-lock for the two new channels.
- Tests: `tests/test_rung3_reading_pull.py`, `tests/test_rung3_reading_pull_pg.py` (new), `tests/test_rung3_act_candidates.py`.
- Eval: `orion/autonomy/evals/run_rung3_reading_replay_eval.py` (new). CI: `.github/workflows/thinking-rung3.yml`.

## Schema / bus / API changes

- Added: the two channels and schemas above; `ActRouteRefV1.object_source` (config model, default `dispatch`).
- Behavior changed (Hub lane): a treated pull does not take one of `HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT`'s
  slots, and every submitted Stage 1 turn is polled every tick (before: at most max_in_flight).
- Removed: `act_episodes.STRUCTURAL_ZERO_SIGNALS`.
- Behavior changed: proven-path paintings are always `treated`; `substrate_act_episodes` rows carry
  `expected.holdback`; Hub's lane tick now holds a lock (shared with pull starts).
- Added columns (additive migration `manual_migration_reading_pull_v1.sql`, REQUIRED-BY orion-hub,
  orion-execution-dispatch-runtime; fits the post-pull allowlist; also added lazily).

## Env/config changes

- Added (dispatch runtime): `ORION_DISPATCH_READING_PULL_ENABLED=true`, `ORION_DISPATCH_READING_PULL_RPC_TIMEOUT_SEC=15`.
- Added (Hub): `HUB_WORLD_PULSE_READ_PULL_ENABLED=true`, `HUB_WORLD_PULSE_READ_PULL_DAILY_CAP=8`, `HUB_WORLD_PULSE_READ_PULL_AROUSAL_ENABLED=true`.
- `.env_example` updated: yes, both. Compose: dispatch runtime lists its two keys; Hub reads `env_file`.
- Local `.env` synced: `sync_local_env_from_example.py` added the 3 Hub keys; `--all-keys orion-execution-dispatch-runtime` added the 2 dispatch keys. No diverged key for either service.
- Skipped keys requiring operator action: none.

## Tests run

```text
tests/test_rung3_reading_pull.py                         29 passed  (10 review tests fail on the pre-fix commit)
tests/test_rung3_reading_pull_pg.py (scratch postgres:16)  6 passed  (found a real DDL bug: missing ';')
tests/test_rung3_act_candidates.py                       34 passed  (painting holdback test fails on origin/main, verified)
clean venv with the CI job's exact installs: all rung-3 steps  109 passed
services/orion-execution-dispatch-runtime/tests          107 passed
services/orion-proposal-runtime/tests                     13 passed
services/orion-policy-runtime/tests                       12 passed
reading suite (hub reading/world_pulse_read/act ledger, orion/world_pulse_read, unified turn bus catalog)  726 passed, 37 skipped
tests/test_{proposal,policy,execution_dispatch,world_action,...}*.py (each file alone)  all passed
orion-static-gates steps (local)                          pass; check_definition_drift re-locked after the origin/main merge
check_migration_required_by / auto_apply_refusals         OK / []
check_env_template_parity, check_bus_reply_channels, check_single_consumer_channels   PASS
```

## Evals run

```text
python orion/autonomy/evals/run_rung3_reading_replay_eval.py            7/7 PASS (fixture)
... --live --hours 48                                                     see the replay table above
python orion/autonomy/evals/run_rung3_replay_eval.py                     unchanged, PASS (fixture)
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile or requirements changed, and the task forbids deploying. Live behaviour after
restart: UNVERIFIED.
```

## Review findings fixed

The review ran in a subagent, by hand (no code-review skill is installed). Target: the branch
diff against origin/main. No blocker. Every finding below has a test that fails on the pre-fix
commit (10 of them, verified on an extracted copy of that commit).

- Finding: the two arms were scored by different rules, and the resolved treated rows had no reader.
  - Fix: a queue act is judged on its treated episodes once it has 20 (`episode_posterior`,
    `arm_stats`); before that, the frame says `belief=posterior`.
  - Evidence: `test_queue_act_is_judged_on_its_treated_arm_once_it_has_enough_rows`.
- Finding: an RPC error or a lost send left a precommitted row that resolved as if the act happened;
  an unmarked control seed could later be pulled as treated and resolve the control row 1.
  - Fix: precommit with `object_ref` NULL and the checked seed in `expected.checked_object`; set
    `object_ref` only from a confirmed `started` / `control_recorded`; void `unconfirmed_by_doer` at
    the horizon.
  - Evidence: `test_a_control_rpc_error_leaves_the_row_unconfirmed`,
    `test_an_unconfirmed_decision_is_voided_and_old_painting_controls_never_count` (Postgres).
- Finding: a replayed act ran the gates before finding its own mark, so a pull that used the cap's
  last slot would be voided on redelivery.
  - Fix: the prior-mark lookup runs first for `act`.
  - Evidence: `test_a_replayed_act_is_answered_before_the_gates`.
- Finding: several refusals carried no hold (`outside_window`, `disabled`, `queue_unavailable`, an
  RPC failure, a control refusal), so they cost one RPC per proposal tick, up to 15 s each if Hub
  is down.
  - Fix: `outside_window` holds until the window opens; other refusals hold one lane tick (300 s);
    check, act and control failures hold 300 s; both arms set the hold.
  - Evidence: `test_every_refusal_carries_a_hold`, `test_outside_window_holds_until_the_window_opens`,
    `test_a_failed_check_holds_the_route`.
- Finding: a pull admitted outside the proven path (information path, tripwire probe) wrote no
  episode, so nothing stopped a pull every tick until the cap.
  - Fix: a `started` pull result holds the route for its horizon on every path
    (`act_in_flight(started_within_sec=...)`).
  - Evidence: `test_a_started_pull_holds_the_route_on_every_path`,
    `test_a_started_pull_holds_the_route_in_sql` (Postgres).
- Finding: the pulled read took one of the lane's slots (so it only reordered), and with the lane
  full it was not polled (its close could miss the horizon).
  - Fix: the lane counts only its own reads against `max_in_flight`, and every submitted turn is
    polled each tick (a second, poll-only claim loop).
  - Evidence: `test_a_pulled_read_does_not_take_a_lane_slot_and_is_polled`.
- Finding: no test covered the lane lock.
  - Fix: added. Evidence: `test_the_lane_lock_keeps_a_pull_start_and_a_tick_apart`.
- Finding: a pull before the lane's startup reclaim could be reset under it and read twice.
  - Fix: refused `lane_starting` until the first reclaim. Evidence:
    `test_no_pull_before_the_lane_startup_reclaim`.
- Nits fixed: `expected_seed_id` and the unused `already_pulled` refusal removed; the eval asserts
  the exact refusal; gate reads (Redis, bus, sensor file) happen before a DB connection is taken;
  `information_schema` query filtered by schema; #2830's no-object painting control rows are
  excluded from the control rate.
- Not changed, disclosed: a treated pull on a section index or blocked publisher is skipped by the
  lane's own checks and still counts against the cap (and resolves 0 in either arm). The in-memory
  "pulled" set is lost on a Hub restart, so a pulled read still in flight then counts as a lane read
  until it finishes.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-hub up -d --build
./scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
./scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

Run from the clean primary `main` checkout after merge. The post-pull hook applies
`manual_migration_reading_pull_v1.sql` (additive). Restart Hub before the dispatch runtime: an old Hub
never answers a pull, so every `check` would time out and be withheld `doer_refused:check_failed`.

## Risks / concerns

- Severity: medium. No live pull will run while arousal stays `strained` (3 days and counting). The
  new path's live trace is UNVERIFIED until the room cools. Mitigation: the replay shows what it does
  when calm.
- Severity: medium. The pull takes the seed the lane would read next, so the measured effect may be
  near zero. That is a real answer, not a bug: the control arm will revoke the claim if so.
- Severity: low. `read_world_pulse` is added to `tests/test_proposal_scoring.py`'s allowlist of
  empty-dimension templates (the ratchet says adding is a decision). It is a timer by design: its
  pacing is the allocator's score, the 12000 s in-flight hold, and Hub's gates. Caught by CI.
- Severity: low. A pulled seed that is a section index or a blocked publisher is skipped by the lane's
  own checks and resolves 0 in either arm. Not filtered at selection.
- Severity: low. The verdict still compares a conditional posterior to an unconditional control rate
  (as #2830). Treated rows are now resolved, so a real contrast can replace it later.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2840

Status: DONE_WITH_CONCERNS. The path is built, reviewed and tested; no live pull can run while arousal stays `strained`, so the live trace is UNVERIFIED.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
