# PR: intentions Orion takes on or lets go (choice-as-EFE Step -1, promise adapter)

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/3329
Design: PR #3317, `docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md`, section "Step -1".
Signed off by Juniper 2026-11-13: all defaults, with one change, the intention family is tier 1.

## Summary

- Orion can now carry out a promise it made in chat. Today a promise becomes a Prime concern and nothing can act
  on it (live: concern `d8db9d7b1bba4cf7`, held a day, "I'll take a lazy browse through the last 24 hours").
- Prime gets one new move, `take_on`. It is offered only on a concern whose source has an adapter. Prime opens a
  `make` act (`intention:<concern_id>`), holds the order until its own `not_before`, then orders ONE carry run.
- "Let go" is the existing concern `drop` (an open take-on act closes 0 with the why). "Not yet" is the existing
  `keep`; a promise concern never taken on ages out at its 72 h horizon.
- Hub reads the promise (exact quote from `harness_turn_trace`, the one exchange from `chat_history_log`) through
  the promise adapter and submits a durable run `intention.carry`. The run binds in the body economy as family
  `durable.intention`, TIER 1, so the economy picks the time inside the horizon.
- The reply goes back into the chat thread where the promise was made: one outreach attempt through the normal
  gates (quiet hours, shared daily cap, economy); if held, the reply is still written into that thread's history.
- The act closes 1 only when the reply is stored. Loop guard: a carry turn's own promises open no concern.

## Outcome moved

- Failure mode closed: "Orion promises things and then cannot do them." Before: no path from a promise concern to
  an act. After: take on -> act -> run -> reply in the thread -> act graded -> concern settled.
- A graded act per take-on (`grammar_acts`, actor `prime`, verb `make`, `object_ref intention:<id>`,
  `test_before intention_source:<source>`): the row Step 3 trains on, and the per-source follow-through rate
  (S-1.10) after 30 days.

## Current architecture

- Chat promise -> Prime concern `turn:<corr>` (hand-back, no bet). Prime's only moves: `curiosity_focus` (ask) and
  `regulate`. A concern settles only when a turn whose PROMPT cites `prime_concern:<id>` completes.
- Outreach `offer_message` always delivers into the newest connected socket's session.
- `BINDING_FAMILIES` = curiosity investigate and consult turns. Nothing reads `WantWindowV1.earliest`.

## Architecture touched

- orion-mind (Prime): new move, own Redis key `orion:prime:intentions`, order/follow/deliver/close on the tick.
- orion-hub: new desk + two HTTP routes; `offer_message(session_id=...)`; new `record_in_thread`.
- orion-durable-runs: workflow `intention.carry` on the existing single-step harness graph (no new graph file),
  run act family, binding family.
- Shared contracts: `orion/schemas/intention.py`; `DurableWorkflowV1` + brief union; schema registry;
  `config/physiology/families.yaml`.

## Where the design doc and the live code disagreed (code wins)

- Doc: "a new `intention_graph.py`". Code: consult turns already run one harness turn on
  `build_single_step_harness_graph`. Reused it (`build_intention_carry_graph`, ~10 lines): thinner, same retries.
- Doc: "Hub writes the reply into the thread" at the run's end. Code: durable-runs has no Hub call-back. So Prime,
  which already follows the run, asks Hub to deliver (`POST /api/intention/deliver`) and grades on Hub's answer.
- Doc: the take-on act carries Prime's `predicted` value. Code: `build_act` has no predicted field; the act carries
  source (`test_before`), horizon and outcome. `predicted` is left for Step 3.
- Doc: one family per source is avoided "because the family weight needs ~10 acts". The family's history joins by
  act-ledger actor, so the carry run also opens the usual run act (actor `durable.intention`, verb `make`,
  `durable_run:<id>`, `app/run_acts.py`), like a consult turn. Two acts per take-on: Prime's commitment
  (`intention:<id>`) and the run's own completion (`durable_run:<id>`).
- Doc: the adapter is "pure". It is one file with a pure `adapt` and a one-query `fetch` (Hub calls it).
- `letter_thread` is already in the `IntentionSource` Literal, so the letter adapter is one file plus one
  `ADAPTERS` entry (S-1.9). Hub refuses delivery routes other than `chat_thread` until then.

## Files changed

- `orion/schemas/intention.py`: new contract (order, result, deliver, result, carry brief, adapter output).
- `orion/schemas/durable_run.py`: `intention.carry` workflow value, brief in the union, validator.
- `orion/schemas/registry.py`: the six intention models.
- `orion/prime/intention.py`: the primitive (adapters map, take-on check, not_before, act, keep-alive, age-out, frame).
- `orion/prime/intention_sources/turn_promise.py`: the promise adapter (+ package `__init__`).
- `orion/prime/thought.py`: `TakeOnMoveV1`, prompt paragraph, frame marks and section, apply/summary.
- `orion/prime/handback.py`: session kind `intention`; loop guard.
- `services/orion-mind/app/prime_loop.py`: the runtime (take on, order, follow, deliver, close, let go).
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: the flag.
- `services/orion-hub/scripts/intention_order.py`: order + delivery desk and routes; `scripts/main.py` wiring.
- `services/orion-hub/scripts/endogenous_outreach.py`: `offer_message(session_id=)`, `record_in_thread`.
- `services/orion-durable-runs/app/{harness_smoke_graph,admission_runtime,run_acts,run_bids}.py`: the workflow.
- `config/physiology/families.yaml`: `durable.intention`, tier 1, 86400 s, agent lanes.
- Tests: `orion/prime/tests/test_prime_intention.py`, `test_prime_handback.py` (+2),
  `services/orion-mind/tests/test_prime_intention_loop.py`, `services/orion-hub/tests/test_intention_order.py`,
  `test_endogenous_outreach.py` (+2), `services/orion-durable-runs/tests/test_intention_carry.py`,
  `test_run_bids_binding.py` (binding set).
- Eval: `orion/prime/evals/run_intention_take_on_replay.py` + fixture; CI step in `orion-prime-tests.yml`.

## Schema / bus / API changes

- Added: `IntentionOrderV1`, `IntentionOrderResultV1`, `IntentionDeliverV1`, `IntentionDeliverResultV1`,
  `IntentionCarryBriefV1`, `IntentionSourceV1` (HTTP and nested payloads, not bus kinds). Hub routes
  `POST /api/intention/run`, `POST /api/intention/deliver`. `DurableWorkflowV1` value `intention.carry`.
  Want family `durable.intention`. `TurnWho` value `intention`.
- Removed / renamed: none.
- Behavior changed: `offer_message` takes an optional `session_id` (None = unchanged).
- No new bus channel, no new table, no migration.
- Compatibility: `intention.carry` is additive on `Literal`/`extra="forbid"` models. Deploy consumers first (below).
  `PrimeTurnsStateV1.recent[].who` may now hold `intention`: an older orion-mind image loading that record fails the
  load and starts the hand-back record empty (it already logs and continues), losing its replay guard and the
  4-a-day count until it refills. Prime's main state and `orion:prime:intentions` are unaffected.

## Env/config changes

- Added keys: `PRIME_INTENTION_TAKE_ON_ENABLED=true` (orion-mind; settings default true; compose default true).
- `.env_example` updated: yes (`services/orion-mind/.env_example`).
- local `.env`: the key is not on main yet, so the sync script cannot add it from the primary checkout. Added by
  hand to `/mnt/scripts/Orion-Sapienform/services/orion-mind/.env` after `PRIME_DAY_LETTER_RUN_DEADLINE_HOURS`:
  `PRIME_INTENTION_TAKE_ON_ENABLED=true` (with a one-line comment).
- No Hub or durable-runs env key (Hub reuses `HUB_CURIOSITY_DURABLE_RUNS_URL`).
- Skipped keys requiring operator action: none.

## Tests run

All from the worktree root with `PYTHONDONTWRITEBYTECODE=1 <venv>/python -m pytest -q -p no:cacheprovider`:

```text
orion/prime orion/physiology/tests orion/mindloop              (PYTHONPATH=.)                       528 passed
services/orion-mind/tests                                       (PYTHONPATH=.:tests:../..)           372 passed
services/orion-durable-runs/tests                               (PYTHONPATH=.:../..)                 898 passed, 136 skipped (Postgres DSN tests)
services/orion-hub/tests/test_intention_order.py test_endogenous_outreach.py                         212 passed
services/orion-hub/tests -k "intention or outreach or orion_day or consult"   666 passed, 4 errors
  (the 4 errors are test_consult_belief_store_sql.py, which needs a live test Postgres DSN; not touched here)
Every python command in .github/workflows/orion-static-gates.yml                                    all pass
scripts/check_env_template_parity.py                                                                 PASS
```

Acceptance checks covered by tests: S-1.1 (refused take-on shown in the next frame), S-1.2 (live concern replay),
S-1.3 (tier-1 held run rebids and is admitted at hold end), S-1.4 (not_before), S-1.5 (carry turn settles the
concern), S-1.6 (reply in the thread when outreach is held), S-1.7 (loop guard), S-1.8 (no quote in Prime state or
the Prime block). S-1.9: Prime side tested; a new ROUTE also needs Hub delivery (stated in the code). S-1.10 needs 30
days of live acts.

## Evals run

```text
PYTHONPATH=. python -m orion.prime.evals.run_intention_take_on_replay          -> ok (6/6 checks)
PYTHONPATH=. python -m orion.prime.evals.run_intention_take_on_replay --live   -> ok (6/6 checks, live Redis + Postgres, read-only)
  adapter on the live concern: exact quote, session 9c91d646-cbd0-4977-8cdd-d2c82045c7f9,
  horizon 2026-11-15T07:08:40.440369+00:00 (promise 2026-11-12 07:08:40 UTC + 72 h)
```

## Docker/build/smoke checks

```text
Not run: no image build needed for config (no new dependency); deploy order below.
Live smoke (first real take-on trace): UNVERIFIED until deployed.
```

## Review findings fixed

Code review ran in a subagent against `git diff origin/main...HEAD`. Five material findings, all fixed:

- Finding M1: a failed act-close publish re-ran the whole lifecycle on the next check (a let-go could be ordered and
  delivered, a refusal re-ordered every 60 s, a delivery written twice).
  - Fix: the decided end is saved on the record (`pending_close`); the next check retries only the close.
  - Evidence: `test_a_failed_close_publish_retries_only_the_close_never_the_lifecycle`.
- Finding M2: Prime dropping a concern the carry turn had already settled would "let go" a finished reply.
  - Fix: a drop after the hand-back settled it (or after the run completed) detaches; the reply is still delivered.
  - Evidence: `test_dropping_a_concern_its_turn_already_answered_still_delivers_the_reply`.
- Finding M3: "stored" meant "published on the bus" (the publish swallows errors): the act could close 1 on nothing.
  - Fix: Hub answers `published`; only `already_delivered` (the `chat_history_log` row is visible) closes 1; a row
    that never appears closes 0 at the horizon. A repeat call never sends or writes again.
  - Evidence: `test_a_published_reply_whose_row_never_appears_closes_zero_at_the_horizon`,
    `test_a_repeat_call_after_publishing_never_sends_or_writes_again`.
- Finding M4: the history fallback wrote replies that outreach's CONTENT guards refused (PASS, ungrounded signal claim).
  - Fix: fallback for every hold except the content refusals (`CONTENT_REFUSALS`: PASS, ungrounded signal, empty); a content refusal is `failed`, nothing written, act 0.
  - Evidence: `test_a_content_guard_refusal_is_never_written_into_the_thread`.
- Finding M5: a taken-on concern held the only turn-concern slot for up to ~72 h, so every new promise was untracked.
  - Fix: concerns with an open intention are left out of the turn-concern count (`fold_turn_signals(taken_on=)`).
  - Evidence: `test_a_concern_taken_on_frees_the_turn_concern_slot_for_the_next_promise`.

Minor findings fixed: age-out note names how a taken-on concern ended (m1); no "take this on" mark on a concern
already taken on (m2); Hub follows an existing run before the horizon check (m3); let-go/expiry cancel the run (m4);
a run unreadable 30 checks in a row closes not_carried (m5); a 422 from an older durable-runs is `unavailable`, not
final (m6); the S-1.9 claim is narrowed in code (m9); Hub and durable-runs READMEs updated (m10).
Not fixed, stated here: m7 (rollback, see compatibility note), m8 (a held outreach want lapses on its own; the per-
concern lock dict grows by one small entry per take-on).

## Restart required

Deploy order (consumer-first; `intention.carry` is additive on forbid/Literal models):

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

(From the clean primary `main` checkout after merge; the post-pull rebuild does the same when the files change.)

## Risks / concerns

- Severity: medium. Concern: this reverses intention ledger D3 ("no outreach trigger") for the carry path only.
  Mitigation: the quote enters only the carry turn's prompt (Hub-built); Prime's state never holds it (test); the
  ping obeys quiet hours, the shared cap and the economy.
- Severity: medium. Concern: hand-back hops are pub/sub (not durable); a missed carry-turn hop would not settle
  the concern. Mitigation: Prime settles the concern itself when the act closes 1.
- Severity: low. Concern: "stored" means the chat-history envelope was published on the bus; sql-writer writes the
  row. Mitigation: delivery is idempotent on `chat_history_log (correlation_id, session_id)`.
- Live smoke: UNVERIFIED. Expected trace: concern `turn:<corr>` -> act `make intention:<id>` -> run
  `intention-<id>` (`run.economy_held`/`run.economy_admitted`) -> harness turn with `prime_concern:<id>` ->
  `chat_history_log` row in the thread -> act closed -> concern settled.

## Status

DONE_WITH_CONCERNS: built, tested, reviewed, fixes applied; the live smoke (first real take-on) is UNVERIFIED until
deployed, and so is the carry turn's hand-back classification (`orion_intention` ingress hops), which follows the
same seam the day letter uses (verified live for `orion_day_letter`).
