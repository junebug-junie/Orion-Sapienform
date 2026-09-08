# Wire `prior_cycle` into graph consolidation's decision loop

## Arsonist summary

`GraphConsolidationEvaluator._decide_outcomes()` can only reach `reinforce`
(a region got more stable) or a real `retire` (an evidence gap actually
closed) by comparing this review against the *last* review of the same
region. That comparison is the `prior_cycle` parameter. Nothing has ever
passed it a real value — `review_runtime.py`'s one call site never supplies
it, so it silently defaults to `None` every single time, for every zone,
forever. The record needed for the comparison (`GraphReviewCycleRecordV1`)
is built fresh on every call and thrown away; nothing persists it, nothing
looks it up. A field that looks like it was meant to carry the lookup key
(`prior_cycle_refs`) is set on the request and never read back by anything.

Confirmed live, 2026-09-08 (direct Postgres query, `substrate_review_telemetry`):
every `consolidation_outcomes` value ever recorded, across all 243 rows in
the table's entire history, is `keep_provisional`. Zero `retire`, zero
`reinforce`, zero `damp`, zero `requeue_review`. Not a recent regression —
the pattern that shows up in the very first rows too.

## Current architecture

`GraphReviewRuntimeExecutor.execute_once()` (`orion/substrate/review_runtime.py`)
selects a queue item, then calls `self.consolidation_evaluator.consolidate(request=consolidation_request, ...)`
with no `prior_cycle` argument. `GraphConsolidationEvaluator.consolidate()`
(`orion/substrate/consolidation.py`) computes real, varying inputs every
call — `contradiction_count`, `evidence_gap_count`, `mean_activation`,
`mean_pressure`, `isolated_frontier_count` — from the live semantic-graph
region it selects. It passes those, plus `prior_cycle`, into
`_compare_with_prior()`, which produces a `GraphStateDeltaDigestV1`:

```python
if prior_cycle is None:
    return GraphStateDeltaDigestV1(
        node_persistence_ratio=0.0,
        edge_persistence_ratio=0.0,
        activation_delta=mean_activation,      # raw value, not a real delta
        pressure_delta=mean_pressure,          # raw value, not a real delta
        contradiction_delta=contradiction_count,
        evidence_gap_delta=evidence_gap_count,
        isolated_frontier_delta=isolated_frontier_count,
    )
```

`_decide_outcomes()` then requires `node_persistence_ratio >= 0.65` (among
other conditions) to ever return `reinforce`. Since `prior_cycle` is always
`None`, `node_persistence_ratio` is always `0.0` — `reinforce` is
**structurally unreachable**, not merely rare. Every review that doesn't
hit an early branch (contradiction/evidence-gap/isolated-frontier all >0,
which per the live histogram also never happens) falls through to the
`keep_provisional` catch-all at the bottom of `_decide_outcomes()`.

**A second, real consequence downstream, same root cause:**
`GraphReviewRuntimeExecutor._no_change_cycle()` treats `keep_provisional` as
"nothing meaningful happened" (it's not in the `{"reinforce", "damp",
"retire", "requeue_review", "maintain_priority"}` set). So every review
reports `no_change=True`, which `GraphReviewQueue.apply_cycle_feedback()`
uses to increment `cycle_budget.no_change_cycles` on *every single cycle*.
Once that counter crosses `suppress_after_low_value_cycles`, the item gets
suppressed — permanently (suppression is `or`'d forward, never cleared).
Regions are plausibly being suppressed as "low value" purely because this
wiring gap makes every cycle look unchanged, not because they actually are.
Not confirmed by data in this pass (would need a `suppression_state`
correlation check against `no_change_cycles` history), but named here as a
real, plausible, and testable consequence of the same bug, not a separate
one.

The record type needed for the comparison already exists and is already
built and returned every call — `GraphConsolidationExecutionV1.cycle_record:
GraphReviewCycleRecordV1` — carrying exactly the fields `_compare_with_prior`
needs (`focal_node_refs`, `focal_edge_refs`, `mean_activation`,
`mean_pressure`, `contradiction_count`, `evidence_gap_count`,
`isolated_frontier_count`). It is discarded by every caller.

`GraphReviewQueueItemV1` is a stable identity across its whole review
lifecycle — `mark_reviewed()` and `apply_cycle_feedback()` both operate on
the same `queue_item_id` every cycle (confirmed by reading
`review_runtime.py`'s call sites); the item is rescheduled via
`next_review_at`, never deleted and recreated. `queue_item_id` is therefore
a safe, stable join key across consecutive reviews of the same region.

`GraphReviewTelemetryRecordV1` (`orion/core/schemas/substrate_review_telemetry.py`,
Postgres-backed via `GraphReviewTelemetryRecorder`, already holds
`queue_item_id` per row) is the obvious place to persist what's needed for
this comparison — it's already the durable, already-queried record of "what
happened on this review," just missing the five numeric fields the
comparison needs. Building a second, parallel persistence layer for
`GraphReviewCycleRecordV1` itself would duplicate a store that already does
this job for everything else about a review.

`GraphReviewRuntimeExecutor` already holds a `telemetry_recorder:
GraphReviewTelemetryRecorder | None` field — it can query its own store
before calling `consolidate()` without a new constructor argument or a new
wiring point in Hub's `api_routes.py`.

## Missing questions

- Should the lookup be "the single most recent telemetry row for this
  `queue_item_id`," or should it also require that row's `execution_outcome
  == "executed"` (skip a row from a `failed`/`terminated` cycle whose
  numbers might not reflect a real consolidation pass)? Leaning toward
  filtering to `executed` outcomes only, since a failed cycle's
  `mean_activation`/`contradiction_count` may not have been meaningfully
  computed at all — needs confirming by reading what `execution_outcome`
  values actually accompany a real `consolidation` result versus an early
  return.
- `telemetry_recorder` is `Optional` on `GraphReviewRuntimeExecutor` — what
  should happen when it's `None` (degraded/misconfigured deployment)? Almost
  certainly: fall back to `prior_cycle=None`, i.e. today's behavior, not an
  error. Consistent with `_resolve_policy()`'s existing fallback pattern for
  a missing `policy_profiles` store.
- Is one telemetry row's snapshot always the *immediately preceding* review
  of this exact region, or could `focal_node_refs`/`focal_edge_refs` have
  drifted enough between two reviews of "the same" queue item that reusing
  the old row's counts as a baseline is misleading? (`_select_region()`
  re-runs a live semantic query each time, not a literal replay of the
  queue item's static refs — see Current architecture above.) The
  persistence-ratio computation already exists specifically to measure and
  report this drift, so this isn't a blocker, just worth being explicit
  that "prior cycle" means "prior *resolved region*," which can legitimately
  shift cycle to cycle.
- Confirm the suppression-cascade consequence above with a real query
  (correlate `suppression_state=true` items against their
  `cycle_budget.no_change_cycles` history) before or alongside this fix, so
  the PR can state plainly whether it also unsticks already-suppressed
  regions or only prevents new ones from being wrongly suppressed going
  forward.

## Proposed schema / API changes

- `orion/core/schemas/substrate_review_telemetry.py`,
  `GraphReviewTelemetryRecordV1`: add five new `Optional[...] = None` fields
  — `mean_activation`, `mean_pressure`, `contradiction_count`,
  `evidence_gap_count`, `isolated_frontier_count` — plus `focal_node_refs:
  List[str] = Field(default_factory=list)` and `focal_edge_refs: List[str]
  = Field(default_factory=list)`. All additive with safe defaults, so
  existing persisted rows (`model_config = ConfigDict(extra="forbid")`,
  loaded via `model_validate()` from stored `payload_json`) keep validating
  unchanged.
- `GraphReviewTelemetryQueryV1`: add `queue_item_id: Optional[str] = None`
  filter field, honored the same way the existing `subject_ref`/`outcome`
  filters are in `GraphReviewTelemetryRecorder.query_with_attrition()`.
- `orion/substrate/review_runtime.py`: populate the five new fields (plus
  `focal_node_refs`/`focal_edge_refs`) on the `GraphReviewTelemetryRecordV1`
  built in `_record_telemetry()`, from `consolidation.cycle_record` when a
  real consolidation ran.
- No bus/channel changes — this is a same-process read/write against the
  existing telemetry store, not a new event.

## Files likely to touch

- `orion/core/schemas/substrate_review_telemetry.py` — new fields (above).
- `orion/substrate/review_telemetry.py` — `GraphReviewTelemetryRecorder.query_with_attrition()`
  (or a small new `latest_for_queue_item()` helper) honors the new filter;
  Postgres/SQL/in-memory backends all read the new columns via the existing
  `payload_json` blob (no new SQL columns needed, matching the current
  schema-in-JSON pattern for this table).
- `orion/substrate/review_runtime.py` — `execute_once()` looks up the prior
  telemetry row for `reviewed_item.queue_item_id` before calling
  `consolidate()`, builds a `GraphReviewCycleRecordV1` from it (or passes
  `None` if none found / recorder unavailable / a validation question from
  above resolves to skip it), passes it as `prior_cycle=`; `_record_telemetry()`
  populates the new fields going forward.
- `orion/substrate/consolidation.py` — likely untouched; `_compare_with_prior()`
  and `_decide_outcomes()` already do the right thing once given a real
  `prior_cycle`.
- Tests: `orion/substrate/tests/` (or wherever `GraphConsolidationEvaluator`/
  `GraphReviewRuntimeExecutor` are already tested) — a regression test
  building two consecutive `execute_once()` calls against the same queue
  item with a stable, unchanging region and asserting the second call's
  `node_persistence_ratio` is high (not the always-0.0 it is today), plus a
  test that `reinforce` is reachable at all through this path.
- Docs: this file's own findings should be referenced from wherever the
  graph-review/consolidation architecture is documented (likely
  `docs/architecture/` — not yet located precisely; check during
  implementation) with a short "prior-cycle comparison was structurally
  dead until 2026-09-08" note, mirroring how other retirement/fix banners
  are handled in this repo.

## Non-goals

- Not fixing the "graph consolidation trial metrics are inconclusive"
  problem directly in this patch — that was the investigation that led
  here, but once `reinforce`/real `retire` become reachable,
  `queue_resolution_delta`/`requeue_rate_delta` still need their own
  evaluator (mirroring `RoutingReplayEvaluator`) built on top of this fix,
  not as part of it. Keep these as two separate patches: this one makes the
  underlying signal capable of varying at all; a follow-up derives the
  mutation-trial metrics from it once it does.
- Not touching `_decide_outcomes()`'s thresholds (`0.65` persistence,
  `0.2`/`0.15` activation bands, etc.) — those are calibration questions
  that can't be evaluated honestly until real (non-zero) persistence data
  exists to calibrate against. Revisit only after this fix has been live
  long enough to produce real distributions.
- Not building a new dedicated store/table for `GraphReviewCycleRecordV1`.
  The existing telemetry store already persists one row per review and
  already has a Postgres/SQL/JSON-fallback backend; extending it is a
  smaller, more consistent seam than a parallel table (see Current
  architecture above).
- Not addressing the plausible suppression-cascade side effect beyond
  naming and flagging it for a data check (see Missing questions) — fixing
  already-suppressed items, if the correlation confirms the theory, is
  either part of this same PR's acceptance check or an explicit, named
  follow-up, decided once the data check comes back.

## Acceptance checks

- Two consecutive `execute_once()` calls against the same queue item, same
  region, same graph state: the second call's `GraphStateDeltaDigestV1.node_persistence_ratio`
  is `1.0` (or close to it, depending on region drift), not `0.0`.
- A region that is genuinely stable (high persistence, no contradictions,
  no isolated frontier nodes, `mean_activation >= 0.2`) reaches `reinforce`
  through two real, consecutive `execute_once()` calls in a test — today
  this is impossible to construct as a passing test at all, since
  `prior_cycle` can't be supplied through the public entry point.
- `telemetry_recorder=None` (or a degraded store) falls back to today's
  `prior_cycle=None` behavior without raising.
- Existing `GraphConsolidationEvaluator`/`GraphReviewRuntimeExecutor`
  test suites still pass unmodified where they test single-cycle behavior
  (this change only activates a second-cycle code path that was previously
  unreachable in practice, not present tests' happy path).
- Live check post-deploy: query `substrate_review_telemetry` for
  `consolidation_outcomes` distribution over the following days/weeks;
  confirm outcomes other than `keep_provisional` start appearing for
  regions reviewed more than once. (Cannot be same-day — needs real repeat
  reviews to accumulate; name this explicitly as a delayed verification in
  the PR, not a same-PR check.)

## Recommended next patch

Implement the schema additions and `review_runtime.py` wiring above in one
focused PR: extend `GraphReviewTelemetryRecordV1`/`GraphReviewTelemetryQueryV1`
additively, populate the new fields in `_record_telemetry()`, look up and
thread `prior_cycle` through `execute_once()` before `consolidate()`, with
the two acceptance-check tests above as the regression tests that would
have caught this. Leave the mutation-trial metric derivation
(`queue_resolution_delta`/`requeue_rate_delta`) and the suppression-cascade
correlation check as explicitly named follow-ups, not folded into this
patch.
