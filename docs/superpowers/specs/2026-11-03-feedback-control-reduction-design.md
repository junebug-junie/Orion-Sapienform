# Feedback control reduction and causal consolidation pairing

## Arsonist summary

Layer 10 currently turns every execution-dispatch tick into a full `FeedbackFrameV1`, even
when the dispatch contains no candidates, blocks, sends, or other action evidence. Those
structurally empty frames are not cognition; their only live value is the untreated field
delta folded into `substrate_signal_control_cells`. They consume the same serial 2-second
worker slot and the same bus/storage path as meaningful outcomes, so the control population
can make the consequence pipeline fall behind the event stream it is supposed to observe.

Reduce a truly empty dispatch directly into the existing control cells and write one compact,
idempotent SQL receipt. Keep full feedback for every dispatch with candidate evidence,
including randomized holdbacks and blocked/refused candidates. Drain at most three pending
items serially per poll so backlog capacity rises without adding ordering or concurrency.
Finally, make Layer 11 select feedback by the source dispatch's window, not by the feedback
worker's later processing timestamp.

This changes a cognition-adjacent measurement path, not Orion's action policy. No allocator,
proposal, dispatch, or model prompt changes.

## Current architecture

- `orion-execution-dispatch-runtime` persists one `ExecutionDispatchFrameV1` and sets
  `feedback_pending=true`.
- `orion-feedback-runtime` selects the oldest pending dispatch, waits for its scoring window,
  builds and stores one `FeedbackFrameV1`, publishes `feedback.frame.v1`, then sleeps for the
  configured poll interval.
- The feedback transaction folds untreated observations into
  `substrate_signal_control_cells` and clears `feedback_pending`. A reconciler requeues a
  cleared marker when no feedback child row exists.
- Randomized holdback is represented by blocked candidates carrying
  `randomized_control_holdback`; it is causally stronger than an ordinary idle tick and is
  intentionally retained as a full inspectable feedback frame.
- `orion-consolidation-runtime` currently loads dispatch and feedback independently by each
  table's `generated_at` hour. Feedback `generated_at` is processing time, so delayed feedback
  is consolidated beside unrelated dispatches rather than its source dispatch.
- Live inspection on 2026-11-03 found 2,009 of 2,651 pending dispatches structurally empty,
  while the worker's one-item-plus-sleep capacity was approximately equal to dispatch input.
  The control table held 12,462,509 `no_action` observations but only 37 randomized-holdback
  observations. This patch adds no new metric; these numbers are diagnostic provenance for the
  reduction decision.

## Missing questions

- How much historical Layer 11 state should be rebuilt after causal pairing is fixed? Existing
  consolidation frame IDs are stable and inserts are `DO NOTHING`; rewriting them and their
  derived tensor slices is a separate, explicitly destructive operation.
- Should ordinary `no_action` controls remain an input to future causal decisions? This patch
  preserves them exactly. It does not promote their weaker quasi-experimental interpretation to
  the status of randomized holdback.
- Should feedback publication eventually use an outbox? The present save-then-publish failure
  window predates this change and is not widened conceptually by the serial batch.

## Proposed schema / API changes

### Compact completion receipt

Add `substrate_feedback_control_reductions`, keyed by
`source_execution_dispatch_frame_id`, with only:

- source dispatch identity and generation time;
- count of folded control observations;
- count of touched aggregate cells;
- receipt creation time.

The receipt contains no prompts, responses, field snapshots, reasoning, or source payload. It is
the durable proof that the dispatch was reduced, and the pending-marker reconciler treats either
a feedback frame or this receipt as completion.

For a structurally empty dispatch, one transaction must:

1. insert the receipt idempotently;
2. fold the control cells only when that insert is new;
3. clear `feedback_pending`.

Any failure rolls the transaction back. A replay that finds the receipt clears a stale marker
without folding the observation twice.

### Runtime classification

`structurally_empty` means all candidate arrays and warnings are empty,
`dispatch_attempted=false`, and both dispatch/block counts are zero. If any field disagrees,
preserve the full feedback path. Compact
reduction is allowed only after the normal settlement window closes and only when at least one
real control observation can be folded. Missing field evidence falls back to a full feedback
artifact instead of masquerading as a successful reduction.

### Bounded catch-up

Each feedback poll may process at most three completed items, serially and oldest-first. A defer,
empty queue, or other no-progress result ends that poll immediately. Meaningful frames are still
published individually in FIFO order. There is no parallel scoring and no unbounded drain loop.

### Causal Layer 11 pairing

Execution dispatch writes `substrate_execution_dispatch_inflight` before assigning a frame's
event time or starting slow work. The database timestamp returned by that insert becomes
`ExecutionDispatchFrameV1.generated_at`; saving the real frame deletes the receipt in the same
transaction. Consolidation therefore has a real lower bound for producer visibility, including a
paused process or blocked result write. A 900-second lag remains defense in depth, with route,
global-fallback, and GPU-shed timeout models bounded to 720 seconds. The oldest
`feedback_pending` marker then prevents closure until every visible earlier dispatch has a durable
full-frame or control-reduction completion. The feedback query
chooses the same bounded set of dispatch
rows whose `generated_at` lies in that closed window and joins feedback by
`source_execution_dispatch_frame_id`. Feedback processing time no longer decides which cognitive
window owns the consequence. Control-reduction receipts are not feedback frames and do not enter
Layer 11.

### Privacy boundary

The compact receipt stores identifiers, timestamps, and integer counts only. Full feedback keeps
the existing privacy surface. No payload is added to health, logs, or the bus.

### Dangerous failure and rollback

The dangerous failure is clearing a marker without either persisting meaningful feedback or
folding the intended control observation exactly once. Transactional receipt semantics and
pending-marker reconciliation are the gate against it. Disable by deploying a follow-up that
turns off the worker's compact classification while retaining receipt-aware reconciliation.
A blind revert to the prior worker is unsafe: its reconciler cannot see receipts and would
requeue already-folded controls. Full rollback therefore requires stopping feedback and either
retaining the small receipt-compatibility seam or explicitly migrating every receipt to another
durable completion form before the old reconciler starts. The additive table itself need not be
dropped.

## Files likely to touch

- `services/orion-feedback-runtime/app/worker.py`
- `services/orion-feedback-runtime/app/store.py`
- `services/orion-feedback-runtime/tests/`
- `services/orion-feedback-runtime/README.md`
- `orion/substrate/pending_marker_reconcile.py`
- `orion/substrate/tests/pending_marker_fake.py`
- `services/orion-consolidation-runtime/app/store.py`
- `services/orion-consolidation-runtime/tests/`
- `services/orion-consolidation-runtime/README.md`
- `services/orion-sql-db/manual_migration_feedback_control_reduction.sql`
- `services/orion-execution-dispatch-runtime/app/{worker,store,settings}.py`

## Non-goals

- Changing action allocation, proposal generation, holdback fraction, or control estimators.
- Deleting historical feedback frames or control cells.
- Treating randomized holdback as empty or reducing it out of the inspectable artifact stream.
- Rebuilding historical consolidation frames in this patch.
- Adding a new bus schema, taxonomy, service, metric, or health signal.
- Parallelizing feedback scoring.

## Acceptance checks

- A truly empty dispatch with a closed, measurable window advances the same control cells as
  before, writes no `substrate_feedback_frames` row, publishes no `feedback.frame.v1`, writes one
  compact receipt, and clears its marker atomically.
- Reprocessing that source dispatch does not advance any control cell a second time.
- A holdback, blocked candidate, dispatched action, inconsistent count, missing field window, or
  failed resolution stays on the full feedback path. A warning-only unevaluable/stale frame also
  stays full evidence.
- The pending reconciler does not requeue a dispatch with either completion form and still
  requeues a dispatch with neither.
- A poll processes no more than three items, stops on no progress, and preserves frame publish
  order.
- Execution dispatch exposes an in-flight receipt before event-time assignment and atomically
  replaces it with the real frame; consolidation waits for that producer watermark, the
  defense-in-depth lag, and the feedback-completion watermark, then
  selects feedback whose source
  dispatch is in the window even when feedback was generated later, and excludes feedback whose
  source dispatch is outside the window.
- Focused tests, migration gates, service test suites, and Docker builds pass.
- Post-deploy live proof must show receipt rows increasing, empty feedback artifacts no longer
  increasing, `feedback_pending` oldest age draining, meaningful feedback still publishing, and
  a new consolidation frame containing causally matched dispatch/feedback IDs. Until checked,
  runtime behavior is `UNVERIFIED`.

## Recommended next patch

After this patch is deployed and observed for one full consolidation window, build a dry-run-first
historical Layer 11 repair tool that reports the exact frame/tensor rows it would replace, requires
an explicit apply flag, and verifies source-dispatch pairing before any destructive write.
