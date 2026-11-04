# feat(curiosity): curiosity replica patch 3 -- the replica instance and Hub's consumer

Design: `docs/superpowers/specs/2026-11-04-curiosity-replica-design.md` (Juniper approved the design and the
build order on 2026-11-04). Patches 0-2 are merged and live: the value arm is on, the loop library
`orion/mindloop/` runs Prime (#3103), and the contracts exist (#3110). This is patch 3: the replica itself,
and Hub asking it for a plan. **The three-arm test starts the day this deploys.**

## Summary

- **The replica arm is on.** Each regular curiosity run now draws one of three orders with equal chance:
  the value rule, the replica's plan, or the uncertainty rule (one third each).
- **On a replica-arm run, Hub asks for a plan before it shows Orion anything.** Hub sends up to 60 of
  Orion's open beliefs (priors) to orion-mind and waits at most 90 seconds. The replica answers with 3 to 5
  of them in the order it wants them shown.
- **Code checks the plan and never repairs it.** A plan with any bad pick is thrown away whole. A late,
  missing, abstaining or rejected plan means the run gets the uncertainty order. Either way the run gets
  one row in `curiosity_offer_plans` saying what happened and why, and it still counts as a replica-arm run.
- **Orion sees only the order.** The replica's reasons stay on that row. The kickoff prompt is the same as
  on the other arms except for which priors come first.
- **The replica is a second small mind beside Prime** in orion-mind, on the same loop library, with its own
  name, budget family (`curiosity.replica`, tier 2, chat lane), Redis key and settings. It never reads
  Prime's private state. Prime now publishes a small read-only view of its open concerns for it.
- **Its plans are graded on the same number the test scores.** The plan's act closes with 1 when the run it
  ordered moved at least one belief, 0 when nothing moved, the run failed, or the plan was not used.

## Outcome moved

The experiment the replica must win can now run. Before this patch no run could land on the replica arm.
After deploy, about one run in three asks the replica, and every such run leaves a plan row, an act, and a
score. The test needs about 135-300 completed runs per arm (77-170 days at today's 5.3 runs a day; design
section "How long a three-arm test takes").

The live smoke shows the replica makes a real, different choice: on today's live pool it picked four fresh,
never-tested priors at confidence 0.60-0.75 (the uncertainty rule would have put priors nearest 0.50 first).

## What the replica reads before it plans

Every section shows its age. A read that failed is shown as "(unavailable: why)", never as an empty section.

- **The candidates** Hub can offer this run (from Hub's request). At most 60: the 30 most uncertain, the 20
  worth the most under the value rule, every prior tested in the last 7 days, and 10 more by a per-run
  rotation. They are shown in a neutral order, not in either rule's order, so the replica is not handed the
  rule it has to beat.
- **The stale priors** (tested 3 or more times): ids only. The run sees them anyway; they cannot be picked.
- **Recent runs:** Hub's last investigate runs (`/curiosity/api/runs`, the read Prime uses), with each run's
  score and which priors it tested.
- **Prime:** its open concerns, from the new view at Redis `orion:prime:concerns`. The view's age doubles as a
  sign Prime is alive ("STALE, Prime may not be running" past 15 minutes). Any regulate signal on curiosity,
  without its holdback arm, so the replica cannot tell Prime's real signals from its controls.
- **The pending focus**, and who asked for it (Juniper, Prime, or Orion in a conversation).
- **What Juniper has been talking about:** the newest chat-history compactor digest -- the card summary
  (`memory_cards`, provenance `chat_compactor`) and the journal title (`journal_entries`, source_ref
  `chat_history_compactor_pass:<index>`), joined on the compactor index, shown with what day it covers and
  how long ago it was written. Verified live: one a day around 13:00 UTC (11-03 13:01 was the newest when
  checked). No raw chat text.
- **Its own track record:** its last 10 plans -- for each pick, whether Orion tested it and how many nats it
  moved -- plus its pick uptake and fallback count over 30 days. It is not shown the arm comparison or any
  p-value.

## Propensities chosen

| Arm | Key | Value |
|---|---|---|
| value | `HUB_CURIOSITY_VALUE_ORDER_PROPENSITY` | 0.3333 (was 0.5) |
| replica | `HUB_CURIOSITY_REPLICA_ARM_PROPENSITY` | 0.3333 |
| uncertainty | the rest | 0.3334 |

The value and uncertainty arms keep equal shares (0.3333 vs 0.3334). Checked on 30,000 run ids: each arm
lands within 1 point of one third. Lowering the value share from 0.5 to 0.3333 moves some run ids from the
value arm to the replica arm, by design (contracts report). Every offer row records all three numbers in
`arm_propensities`, so the analysis never has to guess.

**Analysis boundary:** compare only runs drawn after the Hub deploy of this patch (design section 5: only
runs randomized at the same time are a fair comparison).

## Stances are recorded only

The replica may answer with a stance on Prime's pending focus or Prime's regulate signal (support, challenge
or refuse). In this patch a stance is sent to Hub and written on the plan row, and nothing else happens:
no focus is refused, no charge is changed, Prime is not told. Delivering stances and the holdback is patch 4.
A signal stance must still name the live signal, or the whole plan is rejected (design section 4).

## Current architecture

- Hub's `_investigate` drew a three-way arm, but passed no replica propensity, so no run was ever on the
  replica arm. `curiosity_offer_plans` existed and was empty.
- orion-mind ran one loop instance, Prime. The loop library could not hold an act open past one tick.
- `PrimeConcernsViewV1` and `concerns_view()` existed with no writer or reader.

## Architecture touched

- **New pure package `orion/curiosity/replica/`:** `candidates.py` (the candidate list), `plan.py`
  (`validate_plan`, `PlanRowV1`), `frame.py` (frame, system prompt, answer parse), `track_record.py`
  (`settle_plan`, record lines), `state.py` (the replica's state and its plan act).
- **orion-mind:** `app/replica_loop.py` (`ReplicaRuntime(MindLoopRuntime)`: listener, bid, one model call,
  answer, act, settle), started in `app/main.py` on its own bus connection. Prime's `save()` now also writes
  its concerns view.
- **Loop library** (`orion/mindloop/runtime.py`): one small hook. `_conclude` may return `produced=None`, which
  keeps the thought act open for the instance to close later. Prime never returns None.
- **Hub:** `scripts/curiosity_replica_plan.py` (`plan_offer`) called in `_investigate` after the worldview
  read and before the offer row; `scripts/curiosity_offer_decisions.py` gains `record_offer_plan`;
  `orion/curiosity/worldview.py` keeps the read's raw rows (`prior_rows`) and gains `fresh_priors` and
  `with_head`, so a plan re-orders the SAME read rather than a second graph read.

## Timing in `_investigate` (the order that matters)

1. draw the arm; 2. load the yield model; 3. read the worldview (uncertainty order on the replica arm, which
is also the fallback); **4. on the replica arm: ask for the plan, wait at most 90 s, check it, re-order or
fall back, write the plan row**; 5. write the offer row (it records the order really shown); 6. everything
else as before. Hub already holds its run lock for the whole turn, so the wait blocks nothing else. The
cooldown stamp and daily count are taken after step 5, unchanged.

The replica answers 5 s before Hub's deadline. If it cannot get a funded model call in time it says so
(`unfunded`, or `model_failed` if the model was reached but failed), so the plan row carries the real reason
rather than a bare `timeout`. Its seat wait is cut so seat wait plus 40 s of generation end in time.

## Metric quality gate: the plan act's grade

The replica's act is graded on "the run this plan ordered moved at least one belief".

1. **Provenance.** Hub's turn end (`_spend_turn_ended`, `services/orion-hub/scripts/curiosity_investigation.py`)
   diffs the start and end prior snapshots with `orion/curiosity/value.py diff_snapshots` and writes
   `curiosity_run_outcomes.realized_nats` (`record_run_outcome`). The replica reads that row joined to its
   plan row (`PLAN_ROWS_SQL`, `services/orion-mind/app/replica_loop.py`) and grades with
   `orion/curiosity/replica/track_record.py settle_plan`.
2. **Independence.** Not independent of the arm score, on purpose: it IS the arm score (the three-arm test's
   primary outcome, made 0/1). It adds no new signal. It exists so the act ledger and the body economy's
   graded mean for `curiosity.replica` agree with the experiment rather than inventing a second measure.
3. **Theory anchor.** KL(after || before) over the priors a run tested is the information Orion gained about
   its own beliefs (Bayesian surprise), the P1 outcome of the attention-with-stakes design. "> 0" reads "at
   least one belief moved".
4. **Live data (2026-11-04).** 202 completed runs: 94 moved (> 0), 108 exactly 0, 0 unknown. The rest state is
   exactly 0.0 (KL of an unchanged belief), not a floor, and it is reached often. The smallest move on record
   is 0.0008 nats, median move 0.031. Weekly share that moved: 0.39, 0.44, 0.43, 0.80, 0.42, 0.69 (not flat,
   not saturated). Failed turns: 32 of 234 rows.
5. **Existing mechanism.** Reuses the P1 spend log. Nothing new is computed.
6. **Reversibility.** No schema, no column: the rule lives in `settle_plan`. Cheap to change.

An unknown score is never graded 0: the act is left to the ledger's horizon (24 h), which is not counted as
an outcome. A plan Hub did not use closes 0 at once (intention to treat). A failed turn closes 0.

## Files changed

- `orion/curiosity/replica/__init__.py`, `candidates.py`, `plan.py`, `frame.py`, `track_record.py`,
  `state.py`: the replica's pure parts.
- `orion/curiosity/replica/evals/run_replica_frame_eval.py`, `fixtures/replica_inputs_live_2026-11-04.json`:
  the frame eval on live-captured inputs (personal names scrubbed with the repo's own privacy matcher; the
  chat digest's words are not in the fixture, only its lengths and times).
- `orion/curiosity/tests/test_replica_plan.py`: 25 tests for the pure parts.
- `orion/curiosity/worldview.py`: `prior_rows` on the snapshot, `fresh_priors`, `with_head`.
- `orion/mindloop/runtime.py`: the hold-the-act-open hook.
- `services/orion-mind/app/replica_loop.py` (new), `app/main.py`, `app/prime_loop.py` (concerns view writer),
  `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-mind/tests/test_replica_loop.py` (new, 14 tests); `test_prime_replay.py`,
  `prime_replay_scenarios.py`, `fixtures/prime_replay_golden.json` (re-recorded, see below);
  `test_prime_loop.py`, `test_rpc_health_publish.py` (lifespan tests turn the replica off; it has its own).
- `services/orion-hub/scripts/curiosity_replica_plan.py` (new), `curiosity_investigation.py`,
  `curiosity_offer_decisions.py`, `main.py`, `app/settings.py`, `.env_example`, `README.md`.
- `services/orion-hub/tests/test_curiosity_replica_plan.py` (new, 15 tests); `test_curiosity_spend_log.py`
  (the "replica at zero" test now tests the rollback switch).
- `config/physiology/contract.yaml`: both new modules registered as `display` readers of Prime's signal.
- `config/physiology/families.yaml`, `orion/bus/channels.yaml`, `orion/schemas/curiosity_replica.py`,
  `orion/curiosity/value.py`, `orion/prime/state.py`: comments now name the producer and consumer.
- `scripts/sync_local_env_from_example.py`: `CURIOSITY_REPLICA_` prefix (without it the default sync would
  have added none of orion-mind's new keys).
- `.github/workflows/orion-prime-tests.yml`: runs the replica's pure tests, patch 2's contract tests (no
  workflow ran `orion/curiosity/tests/test_replica_contracts.py` before) and the frame eval.

## Schema / bus / API changes

- Added: none. Every shape was added in patch 2. This patch adds their producer and consumer.
- Removed / renamed: none.
- Behavior changed: the replica arm is drawn (one third); `curiosity_offer_plans` gets rows; orion-mind
  answers `orion:curiosity:replica:plan:request`; Prime writes Redis `orion:prime:concerns` on every save; new
  act-ledger actor `curiosity.replica` (verb `infer`, no new verb or signal).
- Compatibility: Hub new + orion-mind old means every replica-arm run times out to the uncertainty order
  (counted against the replica). Deploy orion-mind first.

## Env/config changes

- Added keys (orion-mind): `CURIOSITY_REPLICA_ENABLED=true`, `_TICK_SEC=15`, `_REFRACTORY_SEC=0`,
  `_SETTLE_INTERVAL_SEC=300`, `_LLM_ROUTE=prime`, `_SEAT_WAIT_SEC=45`, `_BID_LANE=chat`, `_MAX_TOKENS=500`,
  `_LLM_TIMEOUT_SEC=40`, `_HUB_URL`, `_PG_DSN`.
- Added keys (Hub): `HUB_CURIOSITY_REPLICA_ARM_ENABLED=true`, `HUB_CURIOSITY_REPLICA_ARM_PROPENSITY=0.3333`,
  `HUB_CURIOSITY_REPLICA_PLAN_DEADLINE_SEC=90`.
- Changed: `HUB_CURIOSITY_VALUE_ORDER_PROPENSITY` 0.5 -> 0.3333.
- Not added: the design's `CURIOSITY_REPLICA_CHALLENGE_HOLDBACK_FRACTION` and `_CHALLENGES_PER_DAY`. They
  belong to patch 4; adding them now would be keys with no reader.
- `.env_example` updated: yes, both services. Flags ship ON.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (14 keys added). The script
  reported `HUB_CURIOSITY_VALUE_ORDER_PROPENSITY` as diverged (local 0.5); it was set to 0.3333 by hand,
  since that change is this patch.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion-static-gates.yml, every step locally                         40 of 40 OK
orion-prime-tests.yml, every step locally                          all OK (incl. the new replica steps)
  orion-mind tests                                                 217 passed
  orion/curiosity replica + contracts, orion/prime, orion/mindloop, worldview   479 passed, 29 skipped
services/orion-hub/tests (whole folder)                            4535 passed, 146 skipped, 0 failed
orion-reading-tests.yml pytest steps (Postgres steps skip here)     1992 + 53 + 43 + 3 passed
body-atlas-tests, body-economy-analysis, mind-tier-by-caller-tests,
curiosity-urgent-prompt-tests, agency-ask-episodes, regulation-history: every step OK
check_body_economy_contract.py                                     OK (99 modules, all registered)
```

Prime's no-behavior-change replay was re-recorded once: the only difference from the golden recorded before
the loop library was one `save orion:prime:concerns` line after each `save orion:prime:state` (59 lines).
Every act, bid, model call, saved state and log line was unchanged (checked by diffing the two recordings).

## Evals run

```text
python orion/curiosity/replica/evals/run_replica_frame_eval.py      9/9 PASS
  candidates: 60 of 193 fresh (253 live, 58 stale), top 30 uncertain all in, neutral order
  bounded: ~5,082 tokens (cap 6,000); grounded; honest gaps; digest and concern ages shown
  plan: top 3 = picks, same 8-slot layout and stale bucket as the rule order; stale or made-up pick rejected
python orion/prime/evals/run_prime_frame_eval.py                    all PASS
python orion/prime/evals/run_prime_wake_replay.py                   all PASS
services/orion-hub/evals                                            73 passed, 3 skipped
```

## Live smoke (one real model call)

The real `ReplicaRuntime` with real reads (Hub's runs API, Postgres read-only, Redis read-only, FalkorDB) and
one call on route `prime`. The bus it published to was a fake, so no act and no plan result reached the live
bus or ledger.

```text
route prime, max_tokens 500, generation 40 s, seat wait 43 s (cut to the deadline)
frame 18,128 characters (~4,500 tokens), every section present; one gap:
  "PRIME'S CONCERNS: (unavailable: Prime has published no concerns view)"  (correct: Prime's writer is not deployed yet)
model 13.7 s, of which 2.1 s waiting for a seat
answer: valid JSON, 4 picks, all candidates, all fresh, no stances
  self:record_is_mostly_sampled_machinery_20261104          confidence 0.62, never tested
  self:economy_gate_has_declined_curiosity_at_least_once_20261103   0.60, never tested
  self:hop_paths_are_property_rows_not_edges_20261103       0.75, never tested
  self:curiosity_dies_as_missing_final_frame_20261101       0.70, never tested
result published: status ok; plan act opened and held open for grading
```

## Docker/build/smoke checks

```text
No Docker build: no dependency, Dockerfile or image change (psycopg2 is already in orion-mind's requirements).
docker compose config for orion-mind: the new CURIOSITY_REPLICA_* keys pass through (static gate).
```

## Review findings fixed

(filled in after the review below)

## Restart required

Order matters: orion-mind first, so the replica is listening before Hub starts drawing runs onto its arm.

```bash
# After merge, from the clean primary main checkout synced with origin/main (the post-pull rebuild does both):
scripts/safe_docker_build.sh orion-mind up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

No migration: both tables and columns this uses were applied with patch 2.

## Risks / concerns

- Severity: medium
  - Concern: the replica is tier 2 on the chat lane, where Prime (tier 1) is still deferred about 62% of the
    time. A replica that cannot get funded inside 90 s loses by starvation, not by judgment.
  - Mitigation: every such run records `unfunded` on its plan row. Design rec: if more than 30% of
    replica-arm runs fall back `unfunded` in the first 14 days, bring the number to Juniper.
- Severity: low
  - Concern: the frame is about 4,500 tokens, near twice the design's 2,400 estimate for the candidates alone.
  - Mitigation: bounded by an eval cap (6,000 tokens); ~1.8 plans a day on a warm lane. Smoke: 13.7 s total.
- Severity: low
  - Concern: the replica reads Hub's plan table, which has no `orion_readonly` grant, with the same
    DSN orion-mind already uses for the self-monitor ledger read.
  - Mitigation: read-only session; the reasons it reads are its own and never reach Orion.

UNVERIFIED until deploy: a live replica-arm run with a `curiosity_offer_plans` row, its act pair and its
`want_ledger` row; Prime's concerns view in Redis; the first graded plan act; the 95 s forced-fallback check.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
