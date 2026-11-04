# feat(curiosity): curiosity replica patch 4 -- Prime hears the organ

Design: `docs/superpowers/specs/2026-11-04-curiosity-replica-design.md` (branch `docs/curiosity-replica-design`),
section 9 "Second receiver of Prime's signals: challenge and refuse", and "Recommended next patch" item 4.
Juniper approved challenge AND refuse on 2026-11-04. Patches 0-3 are merged and live.

## Summary

- **Curiosity's planner can now answer Prime, and the answer takes effect.** When Prime asks curiosity a
  question or sends it a "slow down" signal, the planner (the curiosity replica) can agree (support), disagree
  but let it happen (challenge), or say no (refuse). Every answer carries a one-sentence reason.
- **A refusal is real.** A refused question is not shown to the curiosity run. A refused signal stops charging
  curiosity's bids from the next bid on, and the market's decision says who refused it and why.
- **Juniper's own focus can never be refused.** Only Prime's acts are answered at all.
- **Some answers are held back on purpose.** One refusal in five is recorded but not applied, and one challenge
  in five is never shown to Prime. That lets us measure later whether refusing helped and whether a challenge
  changes Prime.
- **Prime hears it.** Prime's thinking frame gains an ORGAN REPLIES section with the reasons. A question the
  organ refused closes as "refused by curiosity.replica: <reason>" and still counts in Prime's treated arm.
- **A guard against an organ that always says no.** The planner sees its own refusal rate. A new readout reports
  it daily; above 50% for 14 days goes back to Juniper.

## Outcome moved

Before: the replica could write a stance on its plan row, and nothing happened. No organ could answer Prime.
After: an answer to Prime is a closed act in the ledger, a refusal changes what the run is shown or what the
market charges, and Prime reads the organ's reason in its next frame. The E4 test ("does Prime's next move
change after a delivered challenge or refusal, compared with a held-back one?") now has the data it needs.

Live today (read-only, 2026-11-04): 10 Prime questions on record (8 treated, 2 held back), 0 Prime regulate
acts, 0 replica acts. So the signal half has nothing to answer yet (UNVERIFIED live until Prime first regulates
curiosity), and the question half has about 2 chances a day.

## How it works, in plain steps

1. **Something to answer.** Prime queues a question (Hub's focus slot, marked as Prime's) or writes a signal on
   curiosity (Redis `orion:prime:regulate:durable.curiosity_investigate`). The replica checks the signal every
   tick (15 s) and Hub's pending focus every 60 s (read-only Postgres).
2. **It wakes once.** A new Prime act wakes ONE short thought, even with no plan request. Or the answer comes
   inside a plan, when Hub asks for one and a Prime focus or signal is live.
3. **Checks before answering.** One answer per Prime act. At most 3 answers in any 24 hours. At least 30
   minutes between answers. Never an answer to an act whose parent is itself an answer.
4. **The holdback draw.** A fixed hash of (Prime's act, stance) decides, the same way every time, whether the
   answer is held back.
5. **It takes effect (when not held back).**
   - Refuse a question: orion-mind asks Hub on the new RPC `orion:curiosity:replica:focus_refusal:request`.
     Hub runs the existing `refuse_focus`, which refuses Juniper's focus by code, by the UPDATE's guard and by the
     table's CHECK. Inside a plan, this happens BEFORE the plan answer, because Hub takes the focus right after.
   - Refuse a signal: the replica writes `PrimeChargeRefusalV1` next to the signal in the same hash. The market
     reads it on each bid and does not apply the charge.
   - Challenge: Prime is told (next step).
6. **The answer is an act.** Actor `curiosity.replica`, verb `ask`, parent = Prime's act, `test_before =
   holdback:<arm>`, opened and closed at once. Its note starts with the stance:
   `refuse: <reason> | focus <id> | delivered`.
7. **Prime hears it.** Delivered challenges and refusals go into a small published view, Redis
   `orion:curiosity:replica:replies`, written on every save. Prime reads only that view (never the replica's
   state): for the ORGAN REPLIES section, and when it settles a question or a signal.

## Holdback and caps chosen

| What | Key | Value | Why |
|---|---|---|---|
| refusals recorded, not applied | `CURIOSITY_REPLICA_REFUSAL_HOLDBACK_FRACTION` | 0.2 | design section 9 |
| challenges not shown to Prime | `CURIOSITY_REPLICA_CHALLENGE_HOLDBACK_FRACTION` | 0.2 | design section 9 (E4) |
| answers in any 24 h | `CURIOSITY_REPLICA_ANSWERS_PER_DAY` | 3 | design section 9 |
| least time between answers | `CURIOSITY_REPLICA_ANSWER_REFRACTORY_SEC` | 1800 | design section 9 (30 min) |
| the whole feature | `CURIOSITY_REPLICA_ANSWERS_ENABLED` | true | ships ON |

The design named `CURIOSITY_REPLICA_CHALLENGES_PER_DAY`. The cap counts every answer (support, challenge and
refuse), so the key is `CURIOSITY_REPLICA_ANSWERS_PER_DAY`. A support is never held back: it changes nothing.
Checked on 4,000 act ids: the draw lands 18-22% control for each stance (`test_replica_answers.py`).

## Prime's recorded replay golden: every diff line explained

`services/orion-mind/tests/fixtures/prime_replay_golden.json` gained 1,414 lines and lost none.

- First I re-ran the recorder on the pre-change tree: its output equalled the committed golden.
- After the change, every one of the 16 older scenarios is byte-identical. They run with the replica's answering
  switch absent (as Prime's own tests always do), so Prime reads no view section and shows none.
- The 1,414 added lines are two new scenarios, recorded LAST so the shared id counter of the older ones does not
  move (an earlier attempt placed them in the middle and shifted two gate scenarios' ids; moving them fixed it):
  - `organ_challenge_in_frame`: answering on, a published challenge. Prime's second frame has the ORGAN REPLIES
    section with the reason, the system prompt has the organ paragraph; the question stays open.
  - `organ_refusal_settles_question`: the same with a refusal. Prime's question act closes 0 with exactly
    `refused by curiosity.replica: three runs died on this kind of question`, in the treated arm, before the
    next thought; the next frame shows "refused it: the question was not shown to the run".

## Metric quality gate: the answer act's grade (`artifact_produced`)

1. **Provenance.** `orion/curiosity/replica/answers.py answer_produced`: 1 when the answer did what its arm says
   (treated: delivered -- Hub confirmed the refusal, the refusal was written next to the live signal, or a
   challenge went into the view; control: recorded held back); 0 when a treated answer could not be delivered.
   Written by `ReplicaRuntime._deliver` (`services/orion-mind/app/replica_loop.py`).
2. **Independence.** Not a new signal about the world. It is a delivery check, like Prime's "treated not queued"
   close. It shares the `curiosity.replica` family with the plan acts, so it does feed that family's graded mean
   in the body economy. See the risk below.
3. **Theory anchor.** A call to another organ is graded on whether it reached its target (the act ledger's own
   rule for `ask` acts, e.g. Prime's nomination closing 0 when Hub refused to queue it). It does NOT claim the
   answer was right: that is the holdback comparison's job, later.
4. **Live data.** None yet: 0 replica acts on the ledger (read 2026-11-04). One isolated live smoke made one real
   answer (below). Rest state: no Prime act -> no answer act at all (not a flat 0). UNVERIFIED on live volume.
5. **Existing mechanism.** Reuses `build_act`, `artifact_produced`, the holdback pattern of Prime's `draw_arm`.
   No new verb, no new signal.
6. **Reversibility.** No schema column, no migration. `CURIOSITY_REPLICA_ANSWERS_ENABLED=false` stops new answers.

The refusal rate (readout, and a line in the replica's frames) is a report, not wired into any model or gate.
Provenance: the answer acts' close notes (`stance_from_note`). It is a share of the same acts, so it adds no
independent signal; it exists for the design's 14-day guard.

## Live smoke (isolated, one real model call)

The real `ReplicaRuntime` with real reads (Hub's runs API, Postgres read-only, live Redis GET/HGETALL only) and
one call on route `prime`. Every write went to fakes: Redis SET/HSET/EXPIRE to a dict, bus publishes to a list,
the Hub refusal RPC faked, the bid faked funded. Nothing reached live Redis, Postgres, the bus or the market.

```text
target: Prime's latest real question (focus 9a476042..., already ran)
  derived Prime act id act_bf0c3c498c363d23a3e5c929  == grammar_acts row for that focus (live)
  derived parent       act_abf80b7b3be41a3e8da085d6  == that row's parent_act_id (live)
reply frame 4,240 chars, system prompt 1,746 chars, seat wait 20 s, route prime, no section unavailable
model 6.7 s: {"stance": "support", "reason": "The question directly addresses Prime's concern about budget
  controls triggering during the forecasted envelope breach.", "reading": ...}
acts (captured, not published): infer opened; ask opened+closed "support: ... | focus 9a476042... | recorded";
  infer closed "woke: Prime's new focus ... | answered ...: support (treated, delivered)"
writes captured (not live): orion:curiosity:replica:state, orion:curiosity:replica:replies
```

Also live and read-only: the watch's pending-focus SQL runs (no focus pending now); the watch dropped that
already-run focus as "gone" on the first try, as it should; the readout runs on live Postgres: `NO DATA: the
replica has not answered Prime yet`.

## Current architecture (before this patch)

- The replica planned the top of the offer on its arm. Its `focus_stance` / `signal_stance` went to Hub's plan
  row only.
- `refuse_focus` and `organ_refusal_blocker` existed in `orion/curiosity/focus.py` with no caller.
- The market applied Prime's charge whenever the signal was treated and in force; nothing could lift it.
- Prime's settle saw a refused focus only as "left the slot", with no reason.

## Architecture touched

- **New pure module** `orion/curiosity/replica/answers.py`: targets, the holdback draw, the answer act, the
  rules, the replies view, the refusal rate and the 14-day alarm.
- **orion-mind, replica** (`app/replica_loop.py`): the watch, the reply thought (bid, one call, stance-only
  frame), `_deliver`, the focus refusal RPC client, the signal refusal write, the replies view publish on save.
  Frames (`orion/curiosity/replica/frame.py`): stance rules in both prompts (with the live caps), YOUR ANSWERS TO
  PRIME with the refusal rate, the new reply frame. State: `answers`, `woken_for`, `pending_targets`.
- **orion-mind, Prime** (`app/prime_loop.py`, `orion/prime/thought.py`, `orion/prime/acts.py`): reads the replies
  view; ORGAN REPLIES section and prompt paragraph while answering is on; refused question settles 0 with the
  reason; regulate settle note names a refusal.
- **Body economy**: `prime_charge.PrimeChargeRefusalV1` (field `refusal`), `choice.decide(prime_refusal=)`, and
  the market (`services/orion-durable-runs/app/physiology_market.py`) reads it and records `unchanged` with what
  the charge would have done.
- **Hub**: `scripts/curiosity_focus_refusal_listener.py` (new), started in `scripts/main.py`.
- **Readout**: `scripts/analysis/curiosity_replica_answers_readout.py`.

## Files changed

- `orion/curiosity/replica/answers.py` (new), `frame.py`, `state.py`: the replica's pure parts.
- `orion/schemas/curiosity_replica.py`: refusal RPC request/result, replies view and reply models, channels, keys.
- `orion/schemas/registry.py`, `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json` (re-lock).
- `orion/physiology/prime_charge.py`, `orion/physiology/choice.py`, `services/orion-durable-runs/app/physiology_market.py`.
- `orion/prime/acts.py`, `orion/prime/thought.py`, `services/orion-mind/app/prime_loop.py`.
- `services/orion-mind/app/replica_loop.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-hub/scripts/curiosity_focus_refusal_listener.py` (new), `scripts/main.py`,
  `scripts/curiosity_replica_plan.py` (docstring), `README.md`.
- `config/physiology/contract.yaml`: the replica and plan-consumer rows say what they now do.
- `scripts/analysis/curiosity_replica_answers_readout.py` (new).
- Tests: `orion/curiosity/tests/test_replica_answers.py` (new, 9), `test_replica_plan.py` (prompt test updated),
  `services/orion-mind/tests/test_replica_answers_loop.py` (new, 15), `test_prime_organ_replies.py` (new, 6),
  `prime_replay_scenarios.py` + golden + `test_prime_replay.py`, `services/orion-durable-runs/tests/test_prime_charge_market.py`
  (+4), `services/orion-hub/tests/test_curiosity_focus_refusal_listener.py` (new, 4),
  `scripts/analysis/tests/test_curiosity_replica_answers_readout.py` (new, 2).
- Workflows: `orion-prime-tests.yml` (answers tests, readout test), `orion-reading-tests.yml` (the Hub refusal
  listener test, and `test_curiosity_replica_plan.py`, which no workflow ran before -- a straightforward follow-up
  from patch 3, fixed here).

## Schema / bus / API changes

- Added: `CuriosityFocusRefusalRequestV1`, `CuriosityFocusRefusalResultV1` (bus), `CuriosityReplicaRepliesViewV1`
  + `CuriosityReplicaReplyV1` (Redis `orion:curiosity:replica:replies`), `PrimeChargeRefusalV1` (Redis field
  `refusal` in `orion:prime:regulate:<family>`). All registered in both registries.
- Added channels: `orion:curiosity:replica:focus_refusal:request` (orion-mind -> Hub),
  `orion:curiosity:replica:focus_refusal:result:*` (Hub -> orion-mind).
- Removed / renamed: none.
- Behavior changed: stances on Prime's acts take effect; `decide` takes `prime_refusal`; the market records a
  refused signal's effects as `unchanged`; Prime's settle and frame read the replies view.
- Compatibility: every old reader ignores the new hash field and the new Redis key. No `extra="forbid"` payload
  gained a field. Hub old + orion-mind new: a focus refusal times out (10 s) and is recorded as not delivered.
  durable-runs old: a signal refusal is written but not applied (no Prime regulate act exists yet, so no effect
  today). Rollback hazard: the replica's persisted state gained three fields; an older replica would refuse it
  and start empty (losing open plan acts, which then expire at the ledger horizon, not graded).

## Env/config changes

- Added keys (orion-mind): `CURIOSITY_REPLICA_ANSWERS_ENABLED=true`, `CURIOSITY_REPLICA_REFUSAL_HOLDBACK_FRACTION=0.2`,
  `CURIOSITY_REPLICA_CHALLENGE_HOLDBACK_FRACTION=0.2`, `CURIOSITY_REPLICA_ANSWERS_PER_DAY=3`,
  `CURIOSITY_REPLICA_ANSWER_REFRACTORY_SEC=1800` (settings.py, .env_example, docker-compose.yml, README).
- Removed / renamed: none.
- `.env_example` updated: yes. Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, all 5
  keys added to `services/orion-mind/.env` (the `CURIOSITY_REPLICA_` prefix was already in the sync list).
- Skipped keys requiring operator action: none.

## Tests run

```text
orion-static-gates.yml, every step locally          41 of 41 OK (definition drift after the re-lock commit)
orion-prime-tests.yml, every step locally           10 of 10 OK
  orion/prime + orion/mindloop + replica pure + readout + refs + orion/physiology   381 passed
  services/orion-mind/tests                         247 passed
orion-reading-tests.yml pytest steps                all OK (its "install local PostgreSQL" setup step is CI-only)
services/orion-durable-runs/tests                   749 passed, 125 skipped (plain run)
  (workflow run: 6 steps fail locally only for "no Postgres at 127.0.0.1:5432", the CI service container;
   none touch this patch)
services/orion-hub/tests (whole folder)             4522 passed, 146 skipped; 3 fail only when run from the
                                                    service folder (cwd-relative paths), all 14 pass from repo root
check_body_economy_contract.py                      OK
```

## Evals run

```text
python orion/curiosity/replica/evals/run_replica_frame_eval.py      all PASS
python orion/prime/evals/run_prime_frame_eval.py                    all PASS
python orion/prime/evals/run_prime_wake_replay.py                   all PASS
services/orion-mind/tests/test_prime_replay.py                      PASS (golden: 2 scenarios added, 16 unchanged)
```

## Docker/build/smoke checks

```text
No Docker build: no dependency, Dockerfile or image change.
Isolated live smoke: see "Live smoke" above.
```

## Review findings fixed

(filled after the review pass)

## Restart required

Order: the market and Hub first, so a refusal is applied and answered the moment orion-mind can make one.

```bash
# After merge, from the clean primary main checkout synced with origin/main (the post-pull rebuild does this):
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

No migration: the focus refusal columns were applied with patch 2 (checked live: the columns exist).

## Risks / concerns

- Severity: medium
  - Concern: answer acts share the `curiosity.replica` family with plan acts. Delivered answers grade 1 almost
    always, so they lift the family's graded mean in the body economy, which can make the planner look more
    useful to the market than its plans are.
  - Mitigation: at most 3 a day. The three-arm test scores plans from the spend log, not from the family mean.
    Watch the family weight in the 14-day review; if it moves, split the answers' grade out.
- Severity: medium
  - Concern: a focus refusal whose reply is lost (Hub applied it, orion-mind timed out) is recorded "not
    delivered", so Prime is not told and sees only "left the slot".
  - Mitigation: the delivery note says "answer unknown"; Prime's existing two-read "gone" rule still closes the
    question 0, in the treated arm.
- Severity: low
  - Concern: a reply thought (seat wait at most 20 s + 40 s generation) can delay a plan request that arrives
    during it.
  - Mitigation: about 2 replies a day; the plan path gives up with the real reason (`unfunded`) if too little
    time is left.

UNVERIFIED until deploy: a live answer act on the ledger; a live refused focus; Prime's ORGAN REPLIES section in
a live frame; the regulate side at all (Prime has never sent a regulate act on curiosity).

## PR link

(added after the PR is opened)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
