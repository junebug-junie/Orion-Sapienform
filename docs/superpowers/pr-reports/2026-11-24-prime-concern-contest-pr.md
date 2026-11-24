# Prime's concern slots decided by the concern contest (#3574 patch 2, #3656)

Fixes #3656. Refs #3574, #3610, #3649.

## Summary

- A new promise always gets a slot. With all 3 slots full it pushes out the weakest held concern that is not a
  promise and not taken on. A promise never pushes out a promise. If every slot holds a promise (or something Prime
  took on), the promise waits on a list and the turn record says `waiting: ...`. It is never dropped silently.
- Letter threads, dream concerns, and concerns Prime names while full now wait on a capped list (`waiting`, at most
  10) instead of being forgotten. They take a free slot on the next tick (a letter at most 2).
- The contest: a waiting concern takes the slot of a held one that nothing touched for half its life (12 h), with no
  question open, and that Prime has thought about since it went quiet and since it got its slot. Score = own-source
  salience x fade; unscored ranks below scored; newer evidence breaks an unscored tie; ties keep the incumbent; at
  most 2 per tick.
- Every push-out closes the concern with end `displaced`; the note gives both scores and the reason. Prime's frame
  shows a `contest:` standing under each concern and a WAITING FOR A SLOT section.
- Rules are pure, in `orion/attention/concern_contest.py` (Q1: attention's rule, Prime's tick). Flag
  `PRIME_CONCERN_CONTEST_ENABLED` (on) and `PRIME_CONCERN_CONTEST_MARGIN` (0.1).

## Outcome moved

Replay of the live record 11-17 20:01Z..11-24 02:00Z (`orion/prime/evals/run_concern_contest_replay.py`):

| | Live (contest off) | Contest on |
|---|---|---|
| Letters with 0 threads held | 3 of 6 | 0 of 6 |
| Threads held per letter | 0/2/0/1/1/0 | 2/1/2/1/1/2 |
| Promises opened (of 10 the hand-back saw) | 3 | 5 (4 at once, 1 later) |
| Promises lost | 7 | 0 (5 still waiting at the end, all behind the 1-turn-concern loop guard) |
| Displacements | 0 | 16 (4 by promise intake, 12 by the contest) |
| Contest displacements under 12 h quiet, with a question open, or unseen by Prime | - | 0 |
| Promises displaced | - | 0 |
| Waiting threads that expired unseated | (all forgotten at once) | 12 |

The OFF arm reproduces the live record exactly (threads per letter, which promises opened, the same 5 fades).

Honest cost: 4 concerns the ON arm displaced were later asked about and settled live (197c7a88, 68bb10d1, c3586d86,
10d6b315). The replay cannot show what Prime would have asked about instead. One promise intake pushed out a letter
thread 16 minutes after it got its slot (promise intake has no quiet rule, per #3656). Both are tracked in #3778.

## Current architecture

Prime held 3 concerns. A letter thread with no room was counted and forgotten. A promise with no room was refused
"already holding 3 concerns" and lost (#3656: 11-17 21:55Z; six more on 11-23). Patch 1 (#3649) made concerns fade
24 h after their last evidence, but nothing could take a quiet concern's slot.

## Architecture touched

orion-mind only (Prime's tick and `orion/prime/`), plus a new pure module in `orion/attention/`. No bus, table or
endpoint change. `orion:prime:state` gains `waiting`, `PrimeConcernV1.evidence_salience`, and the `displaced` end.

## Files changed

- `orion/attention/concern_contest.py` (new): score, displaceable, beats, displacements, intake_victim, standings.
- `orion/prime/slots.py` (new): applies the rules to `PrimeStateV1`: seat_or_wait, queue, seat_promise, run_contest,
  standing and waiting lines.
- `orion/prime/state.py`: `evidence_salience`, `waiting`, `displaced` end, track record count.
- `orion/prime/handback.py`: a promise goes through `seat_promise`; loop-guard holds wait; `note_opened_later`.
- `orion/prime/day_letter.py`, `orion/prime/day_dream.py`: no room = wait.
- `orion/prime/thought.py`: a concern named while full waits; event concerns record their want's salience; frame
  standings and waiting section; `CONTEST_PROMPT`.
- `services/orion-mind/app/prime_loop.py`: contest on the tick and at letter close; flag wiring.
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the two keys.
- `orion/prime/evals/run_prime_report.py`: counts `displaced`.
- Tests: `orion/attention/tests/test_concern_contest.py`, `orion/prime/tests/test_prime_concern_slots.py`,
  `services/orion-mind/tests/test_prime_concern_contest_loop.py` (new); hand-back, dream, report tests updated.
- `orion/prime/evals/run_concern_contest_replay.py` + fixture; `.github/workflows/orion-prime-tests.yml` runs both.

## Schema / bus / API changes

- Added: `PrimeStateV1.waiting`, `PrimeConcernV1.evidence_salience` (both omitted when empty), `ConcernEnd`
  `displaced`.
- Removed / renamed: none.
- Behavior changed: no room no longer means lost. `PrimeTurnV1.thread_refused` may start with `waiting:`.
- Compatibility: Hub checked (the design marked this UNVERIFIED). `orion/prime/block.py parse_prime_redis` (Hub's
  Prime block, Claude tools in harness-governor and room-companion) and `orion-durable-runs` dream day validate
  `PrimeStateV1` with `extra="forbid"`. Consumer-first: rebuild orion-hub, orion-durable-runs, orion-harness-governor
  and orion-room-companion before orion-mind. Until then those readers see Prime's state as unreadable once a
  `waiting` entry or `displaced` record exists.

## Metric gate

`evidence_salience` reuses S3 (event-want `source_salience`), which passed the gate in the design. No new signal.
S2 (letter-thread birth salience) is not wired: it failed coverage; #3779 holds the ~12-01 re-check.

## Env/config changes

- Added keys: `PRIME_CONCERN_CONTEST_ENABLED=true`, `PRIME_CONCERN_CONTEST_MARGIN=0.1` (orion-mind).
- `.env_example` updated: yes. Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (both
  keys present in `services/orion-mind/.env`). Skipped keys: none.

## Tests run

```text
pytest orion/prime orion/attention/tests        375 passed
pytest services/orion-mind/tests                448 passed
pytest orion/claude_tools/tests/test_registry_and_readers.py   20 passed
pytest services/orion-durable-runs/tests/*dream*               75 passed, 3 skipped
check_env_template_parity, check_integration_seam, check_definition_drift --gate, check_metric_lineage --gate: PASS
```

## Evals run

```text
run_concern_contest_replay.py        7/7 PASS (fidelity, threads_held, promise_3656, no_promise_lost, no_churn,
                                     never_a_promise, notes)
run_concern_evidence_clock_replay.py, run_prime_frame_eval.py, run_prime_block_eval.py,
run_prime_hunger_replay.py, run_intention_take_on_replay   all exit 0
```

## Docker/build/smoke checks

Not run. No dependency, port or compose wiring change beyond two env keys. Live behavior is UNVERIFIED (#3778).

## Review findings fixed

- Finding: the `CONCERN_BASIS` text change broke the orion-mind replay golden and changed every concern act's basis.
  - Fix: reverted; the basis text is unchanged.
  - Evidence: `services/orion-mind/tests` 448 passed.
- Finding: a fresh concern's standing told Prime it could be displaced after this thought.
  - Fix: `unseen_quiet` is set only once the concern is quiet. Test `test_a_fresh_concern_reads_protected...`.
- Finding: a concern seated from the waiting list with old evidence could be displaced before Prime saw it held.
  - Fix: the thought must come after both the quiet moment and the seating. Test added.
- Finding: a waiting promise that opened later still read "waiting" in the turn record.
  - Fix: `handback.note_opened_later`, called by the runtime and the replay. Test added.
- Finding: promise intake can push out a concern with a question open; not stated.
  - Fix: kept (Juniper: a promise waits only if every slot is a promise), ranked last, and now said in the prompt
    and README.
- Nits fixed: dream never re-queues an ended concern; prompt says a waiting concern cannot be dropped; README says
  flag off is not a rollback; `no_churn` also checks open questions and the thought rule; replay prints
  displacements within 1 h of seating.
- Not changed: a letter thread is skipped when its own letter holds 2 slots, even if the victim is from the same
  letter (rare, conservative).

## Restart required

Consumer-first, after merge, from the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-room-companion up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: medium. Concern: the contest is almost pure recency today (15 of 16 replay displacements were "both
  unscored"); quiet concerns Prime would later have asked about can lose their slot (4 in the replay). Mitigation:
  quiet + thought + no-question rules; #3778 watches it live; #3779 adds real scores.
- Severity: medium. Concern: promises still wait behind the 1-turn-concern loop guard (5 of 6 on 11-23 in the replay).
  Mitigation: they wait, not lost; #3777 asks Juniper whether a promise should bypass the guard.
- Severity: low. Concern: deploy order. Mitigation: consumer-first rebuild above; rollback note in the README.
- Issues opened: #3777 (needs-juniper), #3778 (verify-live), #3779 (follow-up). Closed: #3656 (by this PR).
