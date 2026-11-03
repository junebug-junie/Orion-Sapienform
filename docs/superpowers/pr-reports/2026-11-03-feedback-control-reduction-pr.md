# Feedback control reduction and causal consolidation pairing

## Summary

- Reduce truly empty execution-dispatch ticks directly into the existing untreated control cells
  instead of manufacturing full `FeedbackFrameV1` rows and bus events.
- Persist a payload-free, per-dispatch completion receipt in the same transaction as the control
  fold and pending-marker clear, making crash recovery and replay exactly-once.
- Drain at most three feedback items serially per poll while preserving FIFO publication for
  meaningful frames.
- Keep warning-only, blocked, dispatched, inconsistent, and randomized-holdback frames on the
  full inspectable feedback path.
- Close Layer 11 windows only after the producer in-flight receipt, defense-in-depth delay, and
  feedback completion watermark cross the hour, then pair
  feedback to the source dispatch ID rather than feedback processing time.

## Incident evidence

Live inspection on 2026-11-03 found feedback-runtime approximately capacity-matched to its input:
one item per poll plus a fixed two-second sleep. Of 2,651 pending dispatches, 2,009 had no
candidates, blocks, sends, or warnings. The existing control table held 12,462,509 ordinary
`no_action` observations while the randomized holdback arm held 37. Feedback lag was roughly
92–94 minutes, and consolidation frames contained feedback processed in the hour but zero source
dispatches from that same hour.

## Current architecture

Execution dispatch sets `feedback_pending=true`. Feedback either creates a full consequence frame
or, for a truly empty tick with a real closed field window, folds the untreated observation and
writes `substrate_feedback_control_reductions`. Either child clears the marker transactionally;
the reconciler recognizes both. Consolidation treats the oldest pending dispatch as its closure
watermark. Execution dispatch first writes an in-flight receipt and uses its database time as the
frame event time; saving the real frame atomically deletes it. A 900-second visibility lag remains
defense in depth. Consolidation then joins full feedback through
`source_execution_dispatch_frame_id`.

## Files changed

- Feedback worker/store, pending-marker reconciler, README, and focused tests.
- Consolidation worker/store, README, causal-pairing/watermark tests.
- Execution-dispatch timeout models and docs: one enforced 720-second producer-visibility bound.
- Additive SQL migration for compact reduction receipts.
- Design spec: `docs/superpowers/specs/2026-11-03-feedback-control-reduction-design.md`.

## Schema / bus / env changes

- Adds `substrate_feedback_control_reductions`, keyed by source dispatch ID, with timestamps and
  integer observation/cell counts only.
- Adds transient `substrate_execution_dispatch_inflight` receipts so Layer 11 can see producer
  work before the RPC/result path and durable frame insert.
- No new bus event or channel. Empty reductions intentionally publish nothing; meaningful
  `feedback.frame.v1` behavior is unchanged.
- No environment key changed, so local env sync is not required.
- Consolidation policy adds `window.source_visibility_lag_sec=900`; a static test pins it above
  every configured execution route RPC timeout, while route/global/shed timeout models refuse
  operator overrides above the shared 720-second producer bound.
- Migration declares `ORION-MIGRATION-REQUIRED-BY` for feedback, execution-dispatch, and
  consolidation runtimes, and passes the additive DDL auto-apply classifier.

## Tests and evals

```text
orion-feedback-runtime full suite:       71 passed
orion-consolidation-runtime full suite:   6 passed
orion-execution-dispatch full suite:    130 passed
shared pending-marker tests:             24 passed
root feedback/consolidation tests:        46 passed
consolidation worker/window/lag tests:     5 passed
producer timeout/visibility contract:      2 passed
migration/liveness/deploy static tests:  211 passed
migration REQUIRED-BY gate:             PASS
git diff --check / py_compile:           PASS
feedback + execution + consolidation Docker builds: PASS
```

No latent/model eval applies. This patch changes deterministic reduction, queue, and SQL pairing
contracts. Neither service has a service-local `evals/` directory.

## Acceptance checks

- Empty measurable tick: control cells advance once, compact receipt lands, marker clears, no
  feedback artifact/event is emitted.
- Replay: receipt conflict skips the control fold and only repairs a stale marker.
- Warning, holdback, block, dispatch, inconsistent count, missing window, or failed resolution:
  full feedback path remains.
- Pending-marker reconciliation treats either child as completion and requeues neither.
- Batch is serial, FIFO, stops on no progress, and is capped at three.
- A dispatch materialization first lands an exact in-flight producer receipt; once replaced by the
  real row, a pending dispatch holds its hour open. The visibility lag remains defense in depth.
  After completion, delayed feedback is selected through its source dispatch window.

## Risk and rollback

The primary risk is acknowledging an empty tick without folding it, or folding it twice. The
receipt/cell/marker transaction and receipt-key replay guard close both paths. A blind revert to
the old feedback reconciler is unsafe because it cannot recognize receipts and would requeue
already-folded controls. Disable compact classification while retaining receipt-aware
reconciliation, or stop feedback and migrate receipts to a completion form the old code recognizes
before rollback. The additive receipt table does not need to be dropped.

## Deployment

After merge, deploy only from clean synchronized primary `main`. The post-pull migration path may
apply the additive migration; otherwise use the exact command printed by the migration gate. Then:

```bash
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-feedback-runtime up -d --build
scripts/safe_docker_build.sh orion-consolidation-runtime up -d --build
```

Live runtime remains `UNVERIFIED` until deployment. Verify receipt count and newest timestamp,
pending queue depth/oldest age, continued meaningful feedback publication, and a newly closed
consolidation frame whose feedback source IDs resolve to dispatches inside its own window.
