# Stop a review re-seed from resetting the prune clock

## Summary

- `GraphReviewQueue.upsert` rebuilt a region-key match from the **incoming** seed and named four fields to carry forward, so `created_at` reset to now and `last_review_at` reset to `None` on every re-seed.
- `prune_finished` ages an item against `last_review_at or created_at`, and the review scheduler re-seeds precisely when nothing is usable — which is exactly when every item is suppressed. Each 7-minute tick refreshed the clock meant to retire the items it had just failed to use.
- Inverted the merge: build from the existing item, refresh only the fields describing the incoming proposal. A field added to the schema later now defaults to **preserved** rather than silently wiped.
- Evict suppressed/terminated items ahead of live ones at `max_items`, so the now-honest age tiebreak cannot drop an active region in favour of a dead one.
- Added `items_merged` to the bootstrap execution and the tick log — the field whose absence hid this bug for a full day.

## Outcome moved

The unattended graph-review loop went from **permanently absorbing** back to cycling. PR #2024 removed a 42-minute absorbing state; this removes the unbounded one that replaced it.

Live evidence, 2026-09-02 (control plane `conjourney` @ `127.0.0.1:55432`):

- The loop genuinely ran **08:21 → 15:04 UTC** — 7 items, clean `cycle_count` 0→1→2, suppression flipping on cycle 2, 20 real `eligible_item_selected` telemetry rows.
- It then stopped dead. 14 consecutive ticks 16:01→17:36 UTC, every one byte-identical:

```json
{"status":"seeded_none_due","queue_total":8,"usable_before":0,"pruned":0,
 "due_now":0,"bootstrapped":true,"items_enqueued":0,"store_kind":"postgres"}
```

- All 8 rows: `suppression_state=true`, `last_review_at=NULL`, `created_at` rewritten to recent tick times. Two rows shared a `created_at` to within 69 ms (`17:36:18.866` / `17:36:18.935`) — one tick's two seeds merging, the clock reset caught in the act.
- Two rows carried `next_review_at == created_at + 7200s`, the *post-review reschedule* cadence (`review_schedule.py:155`), not the bootstrapper's `now - 1 day` (`review_bootstrap.py:32`) — yet still read `last_review_at = NULL`. Direct proof the merge wiped `last_review_at` in the same call chain that had just set it (`review_runtime.py:80` → `:102-107` → `review_schedule.py:65` → `upsert`).

Simulating the exact 8 live rows on a 7-minute prune→reseed tick:

```
origin/main (pre-fix merge) : NO RECOVERY after 80 ticks; queue_total=8
this branch                 : 19:42 pruned=1 -> 20:03 pruned=1
                              20:03 re-minted a fresh active item
                              20:03 RECOVERED usable=1 queue_total=7
```

## Current architecture

`execute_substrate_review_scheduled_cycle` (PR #2024) ticks every 420 s: prune finished items, and if `usable_items()` is empty, re-seed the frontier via `GraphReviewBootstrapper`. `GraphReviewQueue` dedups by region key `(focal_node_refs, target_zone)`, so a re-seed of a known region merges into the existing item rather than adding one.

The `prune_finished` docstring already named the resurrection half of this interaction — `upsert` copies `suppression_state`/`termination_state` forward, so re-seeding resurrects a dead item — and the prune was written to break it by ageing the item out. What it did not account for is that the same merge also resets the timestamps the ageing reads.

## Architecture touched

`orion/substrate` only; no service boundary, contract, bus channel, or schema crossed. `services/orion-hub/scripts/api_routes.py` changes are two added log/response keys.

## Files changed

- `orion/substrate/review_queue.py`: invert the region-key merge; add `_RESEED_REFRESHED_FIELDS`; add a liveness term to the `max_items` eviction key.
- `orion/substrate/review_bootstrap.py`: count `upserted_count`; report `items_merged`.
- `services/orion-hub/scripts/api_routes.py`: emit `items_merged` in the scheduler tick log and the bootstrap route payload.
- `orion/substrate/tests/test_review_queue_pruning.py`: 5 tests — the ordering regression, the live `last_review_at=NULL` shape, refresh-still-works, eviction liveness, and a partition/schema mirror.
- `orion/substrate/tests/test_review_bootstrap_merge_counting.py`: new; `items_merged` against the real bootstrapper/scheduler/queue.

## Schema / bus / API changes

- Added: `GraphReviewBootstrapExecutionV1.items_merged` (internal dataclass, single constructor); `items_merged` key in the tick log and bootstrap response.
- Removed / renamed: none.
- Behavior changed: a region-key re-seed preserves `created_at` and `last_review_at`; `max_items` eviction prefers dead items.
- Compatibility: additive. `items_merged` is a new response key; no consumer reads the tick log structurally.

## Env/config changes

None. No `.env_example` touched, so no sync required.

## Tests run

```text
pytest orion/substrate/tests -q                                    -> 653 passed
pytest services/orion-hub/tests/test_substrate_review_scheduler.py \
       services/orion-hub/tests/test_control_plane_isolation_guard.py -q -> 13 passed
```

Mutation-checked against the real file (not a synthetic copy), restored by file copy — never `git stash`, which is shared across worktrees here:

| Mutation | Result |
| --- | --- |
| pre-fix merge restored | `test_reseed_after_suppression_preserves_the_prune_clock` FAILED |
| `created_at` refreshed alone (the live shape) | 3 FAILED, incl. the `last_review_at=NULL` test |
| eviction liveness term dropped | `test_eviction_drops_a_dead_item_before_a_live_one` FAILED |
| `items_merged` hardcoded to 0 | `test_reseeding_a_known_region_reports_merges_not_enqueues` FAILED |

## Evals run

```text
none
```

`orion/substrate/evals/` exists but has no review-runtime harness. Not added here: the loop's remaining behaviour is a multi-hour duty cycle (2 cycles per ~8 h per region), which a gate-speed eval cannot observe honestly. Follow-up below.

## Docker/build/smoke checks

```text
none — pure library change, no dependency, port, or boot-config surface touched
```

## Review findings fixed

- **Finding: `max_items` eviction now targets the longest-lived item, unremarked.** With `created_at` genuinely preserved, the age tiebreak drops an active long-surviving region while a young suppressed one survives. Confirmed by the reviewer running both merges on identical input.
  - Fix: liveness term first in the eviction sort key.
  - Evidence: `test_eviction_drops_a_dead_item_before_a_live_one`; fails when the term is removed.
- **Finding: the prune assertion was insensitive to a single-field regression.** The headline test set both timestamps, so `prune == 1` passed if either survived — and the live rows have only `created_at`.
  - Fix: `test_reseed_preserves_created_at_for_an_item_never_reviewed`, the exact live shape.
  - Evidence: refreshing `created_at` alone now fails 3 tests; previously 0.
- **Finding: 4 refreshed fields had no behavioural coverage** (`focal_edge_refs`, `originating_decision_id`, `originating_request_id`, `anchor_scope`/`subject_ref`) — protected only by a test mirroring the implementation list, which proves a choice was made, not that it was right.
  - Fix: assertions added to `test_reseed_still_refreshes_the_incoming_proposal_details`.
  - Evidence: 3 of the 4 now covered behaviourally; see Concerns for the fourth.
- **Finding: the tick log cannot distinguish "seeded into an existing region" from "seeded nothing".**
  - Fix: `items_merged`.
  - Evidence: new test file; fails when hardcoded to 0.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Deploy before **19:37 UTC** to catch the first prune window; otherwise the queue simply recovers at a later row's cutoff, no later than ~23:43 UTC.

Confirm on the live rail: watch for a tick logging `pruned: 1`, then one with `usable_before > 0` and `status` off `seeded_none_due`.

## Risks / concerns

- **Severity: low.** `anchor_scope`/`subject_ref` are refreshed but are not in the region key (`_region_key` is `zone|nodes`), so a match can legitimately arrive with different values — `review_bootstrap.py:159-160` takes them from the first node with an `anchor_scope`, so node ordering alone can flip them. They are not decorative: `review_runtime.py:219-220` filters eligible items by them, and `consolidation.py:59`/`:207` filter which nodes the next review examines; `subject_ref` refreshing to `None` widens that from one subject to all. **Not a regression** — the pre-fix merge adopted the newest values identically — and inert live (all 8 rows are `orion`/`orion`). Real fix is to put both in the region key. *Mitigation: follow-up.*
- **Severity: low.** `cycle_budget` is preserved wholesale, including the policy-derived `max_cycles` and `suppress_after_low_value_cycles`, so a policy-profile change reaches an already-queued item only after a prune. Unchanged here, but more visible now that items live their full lifetime.
- **Severity: low.** `bootstrap_notes: ["seed_skipped:contradiction_region"]` on all 14 logged ticks — one of three seed specs never fires, narrowing recovery to two region kinds. Pre-existing; own investigation.
- **Severity: informational.** Region-key collision on commas: `_region_key(["a","b"]) == _region_key(["a,b"])`. Unlikely with UUID-shaped ids.
- **Severity: informational.** Post-fix duty cycle is ~2 cycles per ~8 h per region across 2 regions. This converts a dead loop into a slow one; it does not produce a busy one.

## Follow-ups

1. Put `anchor_scope`/`subject_ref` in the region key so a differently-scoped raise is a distinct item.
2. Investigate the permanent `seed_skipped:contradiction_region`.
3. A review-runtime eval harness under `orion/substrate/evals/` that can assert a multi-tick duty cycle.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2044
