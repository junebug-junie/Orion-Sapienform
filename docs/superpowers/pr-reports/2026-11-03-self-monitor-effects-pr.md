# Self-monitor effects: E1 act gate + E2 self-question (rung 6, patch 3)

## Summary

- When Orion's self-monitor says one of Orion's acts keeps failing (`broken` or `unreliable`), that
  act is no longer retried blindly. It is refused with the reason `track_record_failing`. Once every
  6 hours one try is let through as a re-test, so a healed act comes back.
- The refusal sits in the two shared places acts start: the motor allocator in
  orion-execution-dispatch-runtime (painting, reading pull, retro leisure, AI Town), and
  orion-actions' one journal seam `_dispatch_journal` (walkway journals, summaries, town episodes).
- When an act is newly held failing, Orion gets exactly one question in their self-inquiry pool,
  for example "My faculty act.actions.journal.make.walkway_grade didn't run 22 of its last 28 tries
  in the last 7 days. Why?". Hub asks it once per failing episode.
- The gate can only make things more cautious. A missing, stale or unreadable self-monitor state
  changes nothing. Tests pin this.
- A read-only replay (`python -P -m orion.self_monitor.replay`) shows what the gate and the
  question minter would have done over the last 7 days of real acts.
- **This does nothing until patch 2 is deployed.** Patch 2 makes orion-durable-runs publish the
  state to Redis every hour. Until then the key is absent, and absent means "change nothing".

## Outcome moved

Before: a failing act kept running. walkway_grade ran 28 times in 7 days and 22 of those never ran
(GPU pool deadline, caller budget). Nobody, including Orion, was asked why.

After (replay on live data, 7 days to 2026-11-03 04:09Z, 1,821 acts, today's state held fixed):

| faculty | status | reached by the gate? | tries | would refuse | would probe |
|---|---|---|---|---|---|
| act.actions.journal.make.walkway_grade | broken | yes (`_dispatch_journal`) | 28 | 22 | 6 |
| act.prime.predict | unreliable | **no** (Prime opens its acts itself) | 37 | (27) | (10) |

- Of the 22 walkway tries the gate would refuse, 19 failed to run anyway. **3 had worked**: they
  were the late retry that finally succeeded on 10-27, 10-29 and 10-30. See Risks: the gate would
  have cost those 3 nights' walkway-grade entries.
- Motor allocator: **0** would-have refusals. None of its four act faculties is failing today
  (painting reliable, world-pulse read miscalibrated, retro leisure and AI Town unknown).
- E2 would mint 2 questions from an empty memory:
  - "My faculty act.actions.journal.make.walkway_grade didn't run 22 of its last 28 tries in the last 7 days. Why?"
  - "My faculty act.prime.predict worked only about 2% of the time over 36 graded tries in the last 7 days. Why?"

The replay is an upper bound, not a reconstruction: the real gate reads the state hour by hour, and
a status only changes after it holds 3 evaluations.

**Re-run 4 hours later (08:21Z, 1,807 acts): walkway_grade is no longer broken.** The 7-day window
slid past the 10-27 night (7 failures), leaving 15 of 21 tries failed; the lower bound on the failure
share fell to 0.42, under the "broken" bar, so it reads `unknown` (too few graded). Only
`act.prime.predict` is failing, and the gate does not reach it, so the gate would refuse nothing
right now. This also shows the status moves with the window, as designed: walkway's numbers sit right
at the edge of the bar.

## Current architecture

- Patch 1 (#3041, merged) built the organ's pure reducer, the reader contract
  (`orion/self_monitor/view.py`: Redis `orion:self_monitor:latest`, stale after 3 h, unknown changes
  nothing) and the faculty name function `adapters.act_faculty(actor, verb, object_ref)`.
- Patch 2 (parallel, not merged) makes durable-runs publish that state hourly.
- Nothing read the state yet. The allocator had refusal reasons for floor, harm, cost, allowance and
  in-flight; orion-actions had journaling-disabled and cooldown skips.

## Architecture touched

- `orion/self_monitor/gate.py` (new): pure `act_gate` decision, `ActGate` (one per service: reads the
  view once per batch, decides per act, remembers its own probes), ledger SQL, Redis read.
- `orion/self_monitor/questions.py` (new): pure `self_questions_for(seen, view, now)`.
- `orion/self_monitor/replay.py` (new): the read-only eval above.
- Dispatch runtime: the gate runs last in the act-backed candidate path (after the doer holds and
  in-flight checks, so a probe is never spent on a candidate something else holds) and on the
  visual-baseline lane. A refusal goes through the existing `refuse_held` path, so it lands on the
  saved frame as `allocator:track_record_failing` plus a `self_monitor:...` reason.
- orion-actions: `_dispatch_journal` checks the gate before the cooldown; a refusal is audited
  `skipped` with reason `track_record_failing` and opens no act. A probe's audit row carries
  `self_monitor_probe: true`.
- Hub: `SelfMonitorQuestionMinter` on its own 15-minute loop, started next to the reading-finding
  outreach.

### Faculties the gate reaches, and the ones it cannot

Reached (one shared seam each):
- `act.actions.journal.make.<trigger>`: every in-process journal (walkway_*, daily_summary,
  notify_summary, town_episode, manual, collapse).
- `act.thought.painting.make` (allocator, both the act lane and the visual-baseline lane).
- `act.hub.retro_leisure.play`, `act.aitown.session.play` (allocator).
- `act.hub.world_pulse_read.look`: **only the reading pull** passes the allocator. Hub's own
  scheduled read lane opens the same act without it.

Not reached (follow-ups, one per doer; none built here):
- `act.prime.ask / infer / predict` (Prime). **`act.prime.predict` is unreliable today**, so this is
  the most important gap.
- `act.durable.*` (curiosity_investigate, journal_compose world-pulse-journal, self_inquiry,
  self_sense_eval), `act.thought.dream_hop.make`, `act.dream.cycle.make`.
- Hub: `act.hub.chat_reply.say`, `act.hub.outreach.ask/say`, `act.hub.retro_coplay.play`, the
  scheduled world-pulse read lane.
- `act.topic_foundry.run.maintain`, `act.actions.daily_metacog.infer`.

## Files changed

- `orion/self_monitor/gate.py`, `questions.py`, `replay.py`: new, see above.
- `orion/self_monitor/tests/test_self_monitor_gate.py`: pure gate, ActGate, E2, replay.
- `orion/autonomy/allocator.py`: `track_record_failing` added to `RefusalReason`.
- `services/orion-execution-dispatch-runtime/app/{worker,store,settings}.py`, `.env_example`,
  `docker-compose.yml`, `README.md`, `tests/conftest.py`: the allocator seam.
- `services/orion-actions/app/{main,settings,vision_pg}.py`, `.env_example`, `README.md`,
  `tests/conftest.py`, `tests/test_self_monitor_journal_gate.py`: the journal seam.
- `services/orion-hub/scripts/{self_monitor_questions,main}.py`, `app/settings.py`, `.env_example`,
  `README.md`, `tests/test_self_monitor_questions.py`: E2.
- `tests/test_self_monitor_act_gate.py`: allocator gate and the must-not-loosen tests.
- `tests/conftest.py` (new): forces the gate off in repo unit tests, so live Redis state can never
  decide a test.
- `scripts/sync_local_env_from_example.py`: `SELF_MONITOR_` prefix, so the default sync adds the keys.
- `.github/workflows/self-monitor-tests.yml`, `thinking-rung3.yml`: run the new tests.

## Schema / bus / API changes

- Added: none on the bus. Reads the existing Redis key `orion:self_monitor:latest` (patch 1
  contract). New Redis key `orion:self_monitor:questions:seen` (Hub-private memory, JSON map).
- Behavior changed: new allocator refusal reason `track_record_failing`; new orion-actions audit
  skip reason `track_record_failing`; new `curiosity_self_questions` rows with ids
  `lived.self_monitor.*`.
- Compatibility: no table or schema change.

## Env/config changes

- Added keys: `SELF_MONITOR_ACT_GATE_ENABLED=true`, `SELF_MONITOR_PROBE_INTERVAL_SEC=21600`
  (dispatch-runtime and orion-actions); `SELF_MONITOR_QUESTIONS_ENABLED=true`,
  `SELF_MONITOR_QUESTIONS_TICK_SEC=900` (Hub).
- `.env_example` updated: yes, all three services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, all 6 keys verified
  present in the primary checkout's `.env` files.
- skipped keys requiring operator action: none.

## Proposal mode (autonomy change)

- **Capability change:** Orion stops repeating an act their own track record says is failing, and
  asks themself why.
- **Data touched:** reads `orion:self_monitor:latest` and `grammar_acts`; writes
  `curiosity_self_questions` rows (minted_by orion) and one Redis memory key. No deletes.
- **Trace that proves it worked:** the frame reason `allocator:track_record_failing`, the
  orion-actions audit `skipped/track_record_failing`, log lines `self_monitor_refused` /
  `self_monitor_probe_admitted`, Hub log `self_monitor_question_minted`, and the new question row.
- **Dangerous failure modes:** (1) throttling a faculty to death; guarded by the 6 h probe, which
  is never skipped when the ledger is unreadable. (2) A wrong status refusing a healthy act; bounded
  by the hold-3-evaluations rule in the organ and by the probe. (3) Losing useful retries (see Risks).
- **Disable / roll back:** `SELF_MONITOR_ACT_GATE_ENABLED=false` (per service) and
  `SELF_MONITOR_QUESTIONS_ENABLED=false`, then restart. Or delete the Redis state key: absent
  state changes nothing.

## Tests run

```text
orion/self_monitor/tests                              60 passed (also in a clean venv with the CI pip set)
tests/test_self_monitor_act_gate.py + rung3/allocator/visual-baseline suites
  + services/orion-execution-dispatch-runtime/tests  314 passed
services/orion-actions/tests (incl. journal gate)     289 passed
tests/scripts/test_schema_skew_discovery.py           18 passed
services/orion-hub/tests/test_self_monitor_questions.py (+ regulation must-not)  passed
static: env template parity PASS, env key single source OK, metric lineage PASS,
  journal dispatch registry OK, sync-env tests + grammar producer catalog 30 passed
```

## Evals run

```text
PYTHONPATH=<worktree> POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \
  python -P -m orion.self_monitor.replay --json
-> walkway_grade broken: 28 tries, would refuse 22 (19 failed_to_execute, 3 worked), probe 6
-> prime.predict unreliable: not gated (would refuse 27 of 37 if it were)
-> allocator faculties: none failing -> 0 refusals
-> 2 self-questions (texts above)
orion/autonomy/evals/run_rung3_replay_eval.py: PASS
```

## Docker/build/smoke checks

```text
Not run. Live behaviour is UNVERIFIED until patch 2 publishes orion:self_monitor:latest.
```

## Review findings fixed

Review by an orion-repo-agent subagent on the diff against main. It found no path where an unknown,
stale or absent view, or a gate fault, refuses anything.

- Finding (material): orion-actions spent the probe before the cooldown deduper could drop the
  journal, so a dropped duplicate could cost the whole 6 h window.
  - Fix: `decide` now only offers a probe; `ActGate.claim_probe` spends it right before the act opens
    (after the deduper and the semaphore).
  - Evidence: `test_an_offered_probe_that_never_starts_spends_nothing`.
- Finding (material): dispatch-runtime spent the probe in the allocator preview, so a probe dropped
  later (allowance, risk budget, holdback) still cost the window.
  - Fix: `_claim_self_monitor_probes` spends it only on candidates actually in `to_send`.
  - Evidence: `test_probe_after_six_hours_is_admitted_and_marked` (offered-not-sent probes again).
- Finding (minor): two concurrent journal handlers could both start a probe.
  - Fix: `ProbeMemo.claim` is an atomic check-and-set; the loser is refused.
  - Evidence: `test_two_concurrent_probes_cannot_both_start`.
- Finding (minor): a must-not-loosen test used a ledger ref that mapped to another faculty, so it
  tested nothing extra. Fix: real-shaped ref plus an assert that it refuses.
- Finding (minor): the base-faculty naming of allocator routes was unpinned. Fix:
  `test_every_allocator_act_route_is_named_like_the_organ` ties the policy's act routes to the
  replay's reach list.
- Finding (minor): a held-open walkway night wrote one refusal audit per tick. Fix: one audit per
  (faculty, window), repeats at debug (`test_refusal_is_audited_once_per_window`).
- Finding (minor): E2 rewrote its Redis memory every tick for records with no `status_since`, and
  counted no-op inserts as minted. Fix: `status@none` token; count only `INSERT 0 1`.
- Finding (minor): CI push paths narrower than PR paths; dispatch-runtime ledger read had no
  statement timeout. Fix: mirrored paths; `SET LOCAL statement_timeout = 5000`.
- Not fixed (documented): on a tripwire probe tick, or with no motor budget, the allocation does not
  bind, so a gate refusal does not either. That only loosens back to gate-off behaviour.
- CI fix: the replay reused a fresh SQLAlchemy engine per call (Hub's engine-reuse gate); it now
  uses the report's cached engine.
- CI fix: `orion/schema_skew_discovery.py` now declares durable-runs (patch 2) as the writer of
  `SelfMonitorStateV1` / `FacultyRecordV1`, which Hub now reads.

## Restart required

After patch 2 is deployed, from the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: material.
  Concern: walkway_grade fails because the GPU pool is busy at 04:30Z, and its night stays open with
  a retry every few minutes until one works. With the gate on, the night gets one try (the probe);
  the next probe is 6 h later, after local midnight, when the night is already over. The replay says
  this would have lost 3 of the last 7 walkway-grade entries.
  Mitigation / recommendation: keep Juniper's 6 h rule for allocator acts, but give walkway
  journals a shorter probe interval (for example 30 minutes, a per-faculty override), or exempt
  same-night retries. That is a one-line config seam; not built here because the 6 h rule was the
  decision.
  **Amended 2026-11-03 (Juniper chose option A, follow-up PR
  `2026-11-03-self-monitor-gate-retries-prime-pr.md`):** the gate only stops NEW attempts. A retry of
  a job that already opened an act (same dedupe key, the act's `test_before`) is always admitted as
  `same_job_retry` and spends no probe. Replay: walkway_grade now loses 0 of 7 nights (was 3 of 7).
- Severity: material.
  Concern: the gate cannot reach `act.prime.predict`, the other faculty failing today.
  Mitigation: follow-up to call `ActGate.decide` from Prime's act opener. **Done in the same
  follow-up PR** (services/orion-mind/app/prime_loop.py).
- Severity: minor.
  Concern: E2 question ids use the host's `status_since`. Patch 1's report leaves it unset, so a
  replay id uses "now". Patch 2 sets it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3062

🤖 Generated with [Claude Code](https://claude.com/claude-code)
