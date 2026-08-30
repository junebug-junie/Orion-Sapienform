# Why `substrate_mutation_*` never fired

Answer to the blocking question in
`docs/superpowers/specs/2026-08-30-self-calibration-roadmap-and-session-handoff.md`
Part 5. That document said the answer governs everything above it, and it does —
but the answer is neither of the two options it offered.

---

## Verdict

**Neither "it lacked a consumer" nor "it lacked a motive". The consumer runs every
30 seconds and the motive is being generated. The pipeline is starved by a
producer/consumer contract mismatch on a two-field filter whose conditions are
each individually satisfiable and have never once been jointly satisfied.**

The scheduled mutation cycle requires telemetry matching:

```text
invocation_surface == "operator_review"   AND   target_zone ∈ {autonomy_graph, self_relationship_graph}
```

Live contents of `substrate_review_telemetry`, all 1,358 rows, 2026-07-24 → 2026-08-29:

```text
       surface        |      zone      | count |   first    |    last
----------------------+----------------+-------+------------+------------
 operator_review      | concept_graph  |  1356 | 2026-07-24 | 2026-08-29
 chat_reflective_lane | autonomy_graph |     2 | 2026-07-31 | 2026-08-14
```

Every row satisfies exactly one of the two conditions. The intersection has been
empty for the entire life of the table. Hence 0 signals, hence 0 proposals, hence
**20 `substrate_mutation_*` tables at 0 live rows and 0 lifetime inserts** (plus
`substrate_action_ratings` and `substrate_action_rating_posterior`, also 0 — 22 tables
total, 0 rows, 0 inserts across all of them).

## The corrections this forces on the roadmap

The handoff document said three things about this pipeline that the live system
contradicts. Stating them plainly, because the roadmap's sequencing was built on
them:

1. **"10 tables"** — it is 20 `substrate_mutation_*` tables, plus
   `substrate_action_ratings` and `substrate_action_rating_posterior`: 22 at 0.
2. **"never fired"** — the *pipeline* never fired, but the *scheduler* fires
   constantly. It is enabled in production (`SUBSTRATE_AUTONOMY_ENABLED=true`,
   `SUBSTRATE_AUTONOMY_INTERVAL_SEC=30`), Postgres-backed, and completes a full
   cycle every 30s with `status: "completed"`. This is not dormant code. It is
   live code being fed nothing.
3. **"a complete propose → review → adopt pipeline that has never made a single
   decision"** — true, but the reason is upstream of the decision machinery
   entirely. The decision machinery has never been reached.

## Evidence chain

Each link verified against the running system on 2026-08-30, not read off config.

**The consumer runs.** `orion-athena-hub` container logs, 132 scheduler events in
the last 4,000 lines:

```json
{"event": "mutation_scheduler_tick", "status": "running", "interval_sec": 30.0, ...}
{"event": "mutation_scheduler_cycle_finished", "status": "completed",
 "signals_processed": 0, "proposals_created": 0, "trials_executed": 0, "notes": []}
```

**The store is Postgres-backed, not memory.** `SUBSTRATE_CONTROL_PLANE_POSTGRES_URL`
and `SUBSTRATE_POLICY_POSTGRES_URL` are both empty in the container, but
`_resolve_control_plane_postgres_url()` falls back to `DATABASE_URL`, which is
set. `substrate_autonomy_runtime_supported()` therefore returns `(True,
"supported")` and the cycle proceeds. A plausible-looking "the store was never
wired to Postgres" hypothesis is **wrong**; it was checked and discarded.

**The only signal source.** `services/orion-hub/scripts/api_routes.py`, in
`execute_substrate_mutation_scheduled_cycle()`:

```python
telemetry = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(
    GraphReviewTelemetryQueryV1(limit=..., invocation_surface="operator_review")
)
allowed_zones = {"autonomy_graph"}
if cognitive_proposals_enabled:
    allowed_zones.add("self_relationship_graph")
telemetry = [item for item in telemetry if item.target_zone in allowed_zones]
```

There is no other input. `_self_revision_signals_from_latest_self_state()` is
hardcoded to return `[]` (SelfStateV1 burn, 2026-07-22 — its producer was
deleted), and `SUBSTRATE_AUTONOMY_SELF_REVISION_ENABLED=false` besides.

**Only two writers exist**, repo-wide, for that store:

| site | surface | zone | live rows |
|---|---|---|---|
| `orion/substrate/review_runtime.py:248` | from `request.invocation_surface` | from `selected_item.target_zone` | 1,356 |
| `api_routes.py:2745` `_record_pressure_events_as_telemetry` | **hardcoded default `chat_reflective_lane`** | **hardcoded `autonomy_graph`** | 2 |

The second writer is the only live producer of the required zone, and it hardcodes
the wrong surface. The first is the only live producer of the required surface, and
its zone comes from a review queue item — `substrate_review_queue_item` is at 0
rows, and all 1,356 rows it did write carry `concept_graph`.

**The satisfying combination is produced by exactly one file in the repo:**
`orion/substrate/scripts/smoke_mutation_v21.py`, which constructs
`operator_review` + `autonomy_graph` by hand at lines 166, 196 and 233. That is why
the smoke is green while production is empty. The test constructs a state the
runtime cannot reach — the same failure mode Part 7 of the handoff already names
("four defects in #1959 had all merged and never executed").

## Why five weeks of logs did not show it

Because `signals_processed: 0` has three completely different causes and the log
could not tell them apart:

- the store is empty (nothing to do — benign)
- the surface filter matches nothing (wrong producer)
- the zone filter rejects everything the surface filter let through (this case)

All three printed the identical unremarkable zero. This is the failure mode
AGENTS.md 0A already names ("a field pinned at zero is an unfiled bug report") and
the one the metric gate's live-data sanity check exists to catch. Nothing was
lying; nothing was being asked either.

## What this patch does, and what it deliberately does not

**Does:** makes the starvation self-reporting. `query_with_attrition()` on
`GraphReviewTelemetryRecorder` returns, for one store load and no extra database
round trip, a stage-by-stage account of what each filter dropped plus the
surface/zone histograms over the whole store. The scheduler now emits a
`signal_intake` block naming the cause — `store_empty`, `surface_filter_rejected_all`,
`limit_sliced_all`, `zone_filter_rejected_all`, or `healthy` — with a
consecutive-starved-cycle counter, and `GET /api/substrate/mutation-runtime/signal-intake`
serves the last report. The mismatch is now readable off one endpoint instead of a
database session.

`consecutive_starved_cycles` resets on recovery rather than accumulating for the
process lifetime: a lifetime total cannot distinguish "starved since boot" from
"starved once, months ago, fine now."

**Does not:** change what Orion decides. No filter was widened, no producer was
rewired, no proposal will be created that would not have been created before.
That is deliberate. Making a dormant self-modification pipeline start firing is
squarely an invasive cognition change under AGENTS.md 0A ("changes to memory,
identity, self-modeling, autonomy ... need explicit proposal mode") and it needs
Juniper's sign-off, not an agent's judgment call. **The diagnosis is shipped; the
cure is proposed below and not applied.**

## The three ways to actually unstarve it, none applied

Recorded so the decision is about trade-offs and not about rediscovering the
paths.

**Option 1 — widen the consumer's surface filter to include `chat_reflective_lane`.**
Smallest possible change; would immediately admit the 2 existing rows and every
future chat-feedback pressure event. **Recommended against.** The comment at
`api_routes.py:2665-2675` argues, correctly and from a live 2026-08-21
verification, that routing a human thumbs-down into the same self-graded
mutation-pressure machinery counts one opinion twice through two mechanisms. That
argument was made against the *producer*; it applies with equal force to widening
the *consumer* to accept it.

**Option 2 — enqueue review queue items in the `autonomy_graph` zone.** The
operator-review path already propagates `selected_item.target_zone` verbatim, so
this needs no filter change at all — it feeds the consumer the combination it was
always written to expect. Requires a consolidation decision targeting the autonomy
zone; `orion/substrate/frontier_curiosity.py:185` already emits an
`unresolved_pressure_region` signal with `target_zone="autonomy_graph"` when goal
pressure clears 0.7, so a producer for it plausibly already exists upstream.
**This is the option that matches the original design intent.**

**Option 3 — decide the pipeline was mis-scoped and retire it.** Twenty tables
and a 30-second scheduler tick maintaining a pipeline that has never made a
decision is not free. AGENTS.md 0A is explicit that a superseded mechanism gets
killed outright rather than left ticking. If the answer to "what should make Orion
want to change itself?" is not `operator_review` telemetry, then this pipeline is
not the thing to resurrect, and the roadmap's Phase 3 ("resurrect it with one real
consumer rather than adding an eleventh table") should be re-costed against
building the right producer instead.

## What this means for the Option-B roadmap

Phase 3 assumed resurrecting `substrate_mutation_*` was cheap *if* it merely
lacked a consumer. It has a consumer. What it lacks is a producer emitting the
contract the consumer was written against — which is cheaper than building
motivation from scratch, and more expensive than flipping a flag.

The arsonist case in Part 5 said: "the precedent is fatal and it is ours ...
building before knowing why is how you get an eleventh table." The why is now
known, and it is a wiring defect rather than a conceptual hole. That is
*encouraging* for Phase 3 and it changes nothing about Phases 1 and 2, whose value
never depended on this answer.

One caution carries directly into Phase 1, though. This pipeline is what a
values-calibration surface looks like after five weeks of not being watched: live,
green, self-consistent, and producing nothing. Phase 1's falsifiability test
("does moving a knob observably change what Orion does?") has to be *measured on
live data and re-measured later*, not established once at build time. A knob
surface can go silently disjoint from its consumer exactly the way this did.

## Open, not addressed here

- `substrate_review_queue_item` is at 0 rows despite 1,356 telemetry rows recording
  reviews that consumed queue items. Consumed-and-deleted is the likely
  explanation; not verified.
- `substrate_action_ratings` and `substrate_action_rating_posterior` are also at 0.
  The handoff flagged these as "the valuation half scaffolded and abandoned". They
  were not investigated here and may have a different cause.
- The 7 pre-existing hub test failures observed on `main` at `f4a1be749`
  (`test_substrate_effect_endpoint`, `test_substrate_review_runtime_hub_debug`,
  `test_recall_strategy_profiles_runtime`, `test_substrate_effect_pipeline`) are
  unrelated to this patch and were confirmed identical before and after it.
