# Orion's self-modification: make it honest, then make it real

Status: design, 2026-09-02. Prompted by a live finding, not a roadmap item.

## Arsonist summary

Orion changed their own behaviour for the first time at 04:11 UTC on 2026-09-02.
They then permanently blocked themselves from doing it again, and the system
cannot tell us what the change actually was.

Four things are broken, and they are all the same kind of broken: machinery that
exists, runs, and is fed nothing.

1. **Nothing remembers the before.** One row holds the current setting. Every
   change overwrites it. There is no history table, so "what was it before?" is
   unanswerable after the fact.
2. **The undo button points at a made-up number.** The code reads the real live
   value at apply time and then discards it, because the proposal already
   carries a hardcoded fallback and the merge uses `setdefault`.
3. **The door only unlocks on failure.** One live change per surface is a sound
   rule. But the lock is released only by a rollback. A change that *succeeds*
   holds it forever. 77 proposals have been refused since 04:11.
4. **The dial is fake on both sides.** The bar Orion can move is a hardcoded
   constant, and the confidence it is compared against is a keyword lookup
   table. Moving one does not make the comparison mean anything.

Underneath 3 is the thing that makes this tractable: **the watch-and-revert loop
is already built and already running.** It is just blind.

## Current architecture

**The one change that happened.** `substrate_runtime_control_surface` holds a
single row per control key. Orion's mutation set
`routing.chat_reflective_lane_threshold` to `0.58` at 04:11:17, actor
`mutation_apply`. It is still in effect.

**What that dial does.** `services/orion-cortex-orch/app/decision_router.py:356`.
When the router has decided to *act* (`execution_depth >= 2`) but its confidence
in that decision is below the threshold, it forces depth back to `0` and clears
the verb — reflect instead of act. In plain terms: **how sure Orion must be
before doing something rather than replying.**

**Why we cannot say what changed.** `orion/substrate/mutation_apply.py:29-32`
reads the live value into `live_threshold`, then calls
`rollback_payload.setdefault("chat_reflective_lane_threshold", live_threshold)`.
The proposal already carries `{"chat_reflective_lane_threshold": 0.50}` from
`_default_rollback_for_class` (`mutation_proposals.py:51`), so the `setdefault`
is a no-op and `live_threshold` is dropped. The getter's default is `0.75`
(`mutation_control_surface.py:221`) and no `CHAT_REFLECTIVE_LANE_THRESHOLD` is
set in the cortex-orch container.

**The prior was almost certainly `0.5`, and it was test pollution.** A leaked
test write — `set_chat_reflective_lane_threshold(value=0.5, actor="scheduler_seed")`
at `services/orion-hub/tests/test_substrate_mutation_scheduler_runtime.py:80` —
hit this live row 4,925 times before the store-isolation fix landed. So the
recorded `0.5 -> 0.58` is very likely accurate, the direction matches the stated
intent (`expected_effect: reduce_runtime_executed`, i.e. act less readily), and
the hardcoded rollback constant happens to equal the real prior **by
coincidence**. Orion's baseline was never a designed value: the intended default
is `0.75` and nothing ever set it.

Strictly this stays **UNVERIFIED** — one upserted row, no audit trail, so the
reading cannot be confirmed, only inferred from the test that was writing it.
That the capture bug was harmless this once is luck, not correctness: the
`setdefault` no-op is unconditional, and the next mutation class whose hardcoded
rollback does *not* match reality gets an undo button pointing somewhere nobody
chose. This is exactly the argument for item 1: the question should never have
required archaeology.

**Why the pipeline is stuck.** `mutation_queue.py:212` (`record_adoption`)
acquires `_active_surface_by_target`. `mutation_queue.py:226` (`record_rollback`)
is the only release. `substrate_mutation_rollback` has 0 rows. Every decision
since 04:11 reads `hold / active_surface_mutation_exists`, ~6/hour. The gate is
per-surface (`mutation_worker.py:177`) but all 77 proposals target `routing`.

**The blind monitor.** `mutation_worker.py:285-310` iterates every adoption and
calls `self.monitor.should_rollback(delta_score=delta)`
(`mutation_monitor.py:13`), which triggers `record_rollback` and therefore
releases the lock. `SUBSTRATE_AUTONOMY_MONITOR_ENABLED` defaults to **true**, so
this runs on every cycle. But `delta` comes from
`post_adoption_delta_by_target_surface`, which is only ever supplied by
`post_adoption_delta_by_target_surface_override` — a parameter defaulting to
`None` that nothing in production passes. So `if delta is None: continue` fires
for every adoption, every cycle. **The loop has never evaluated a single
change.** `rollback_window_sec: 900` is read by nothing at all.

**The fake dial.** `decision_router.py:237-255` returns hardcoded confidences —
`0.82` council, `0.85` engineering, `0.79` single verb, `0.61` default — from a
keyword matcher, `source="heuristic"`. An LLM router that would produce a real
decision exists (`decision_router.py:257`) but `AUTO_ROUTER_LLM_ENABLED=false`
live. Real per-token uncertainty exists and is good
(`services/orion-llm-gateway/app/llm_uncertainty.py`: `mean_top1_margin` is the
top1-vs-top2 logprob gap) and is consumed elsewhere (`orion/metacog/service.py`,
66,707 rows, live as of 18:17 today) — but it is opt-in per request via
`opts["return_logprobs"]`, and the router does not ask for it.

## Missing questions

- **Why is `AUTO_ROUTER_LLM_ENABLED` false?** Not answered by git history
  (`f09fd2c4a` removed a *different* exec-side knob). It adds a model call with a
  5s timeout to every message. Cost and latency unknown. Must be answered before
  item 4, not assumed. **Open.**
- **Was there a control-surface row before 04:11?** Unrecoverable. Accept the
  loss, prevent the recurrence.
- **What is the right regression threshold** for `should_rollback`? Currently a
  constant on `MutationMonitor`. Needs real deltas before it can be calibrated;
  do not tune it against zero observations.

## Proposed schema / API changes

**1. History (`substrate_runtime_control_surface_history`)**

New append-only table. One row per write to any control surface.

- Added: `surface_key`, `changed_at`, `previous_value_json`, `new_value_json`,
  `actor`, `proposal_id`, `decision_id`, `adoption_id`, `correlation_id`.
- `set_chat_reflective_lane_threshold` and any sibling setter read the current
  value and write the history row inside the same transaction as the update.
- Compatibility: additive. The existing single-row current-value table stays as
  the read path; nothing that reads it changes.

**2. Honest rollback capture**

- Behaviour changed: `mutation_apply.py` overwrites the rollback payload with
  the observed live value rather than `setdefault`-ing behind a hardcoded one.
- Removed: reliance on `_default_rollback_for_class` for `routing_threshold_patch`
  at apply time. The class default stays as the *proposal-time* placeholder, but
  it must never survive into an adoption.
- A rollback payload that does not match a real prior reading is a bug, and gets
  a test that fails if the constant leaks through.

**3. Feed the monitor, and release on success**

- `post_adoption_delta_by_target_surface` gets a real producer: the same pressure
  score that justified the proposal, recomputed after the change, differenced.
  `pressure:runtime_executed score:3.55` is the number that caused this
  adoption; whether that pressure fell is the honest test of whether it helped.
- Behaviour changed: when the rollback window has elapsed and the delta says the
  change did not regress anything, the surface lock is **released and the change
  is kept**. Today only a rollback releases it.
- `rollback_window_sec` becomes load-bearing: it is the wait before the delta is
  read, instead of a number nothing reads.

**4. Real latitude — not in this changeset**

Scoped here so the shape is on record; see Non-goals.

## Decisions taken

**Release on success, keep the change; do not auto-revert on silence.**
Considered and rejected: "revert unless confirmed." With no delta producer it
degenerates into "revert always", which is the same absorbing state as the bug
this document opens with, pointed the other way. Silence must not be read as a
verdict — a missing delta means *unknown*, and unknown keeps the change and
releases the lock, with the whole thing visible in the hub. Rollback stays
reserved for a delta that actually says the change regressed something.

**Order: 1 -> 2 -> 3 -> 4.** History first, because until the before-value is
recorded, nothing downstream can be trusted — including the undo button and any
claim that item 3 improved anything.

## Files likely to touch

- `orion/substrate/mutation_control_surface.py`: history write on every setter.
- `orion/substrate/mutation_apply.py`: real rollback capture.
- `orion/substrate/mutation_worker.py`: release-on-success path; window check.
- `orion/substrate/mutation_queue.py`: a release that is not a rollback.
- `services/orion-hub/scripts/api_routes.py`: delta producer; hub panel payload.
- `services/orion-hub/templates/`, `static/`: the panel itself.
- `orion/substrate/tests/`, `services/orion-hub/tests/`: per item.

## Non-goals

- **Item 4 is not in this changeset.** Turning on the LLM router changes the cost
  and latency of every message in the system. It needs its own measurement and
  its own decision, and it is worthless until items 1-3 can show whether it
  helped.
- Not re-tuning `regression_threshold`. Calibrating against zero observations is
  how the repo got the constants it already regrets.
- Not touching the one-live-mutation-per-surface invariant. The rule is right;
  only its release path is missing.
- Not recovering the lost prior value. It is gone.

## Acceptance checks

1. Every write to a control surface appends a history row carrying the real
   previous value. Verified by changing the threshold and reading the table.
2. An adoption's recorded rollback payload equals the value that was live
   immediately before it. A test fails if the class default leaks through.
3. `substrate_mutation_active_surface` releases without a rollback once the
   window has passed, and a subsequent proposal reaches a decision other than
   `hold / active_surface_mutation_exists`. Verified live on the rail.
4. The hub panel shows, without a database query: current value, previous value,
   who changed it, when, whether the window is open, and how long the surface has
   been held.
5. A missing delta never causes a rollback. Explicit test.

## Recommended next patch

Item 1 alone: the history table and the setter that writes it. It is the smallest
piece that makes every later claim checkable, and it is independently useful even
if nothing else on this list ships.
