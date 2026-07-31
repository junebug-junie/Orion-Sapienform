# Metacog temporal episodes: "what happened now, what happened next" — design spec

Status: **design mode, not implemented.** Touches metacog, the episodic/self-modeling ladder, and
(prospectively) reverie — cognition-loop surfaces CLAUDE.md §0A requires explicit proposal mode for.
This document proposes; it does not build.

Supersedes nothing. Sits underneath
`docs/superpowers/specs/2026-07-29-stream-of-consciousness-hop-chain-design.md` as the concrete
"hop 2" input format that doc defers, and picks up the thread
`docs/superpowers/specs/2026-07-28-metacog-turn-scoped-trend-reducer-design.md` opened at hop 0.

## Arsonist summary

Juniper's ask, stated directly: *"we needed a what happened now, what happened next set of reducers
so those can feel across time and not point in time in the current shape of metacog."* Followed by:
*"each agent just keeps fucking off on this."*

That second sentence is the more useful one, and it is visible in the code. `orion/metacog/trend_reducer.py`
(hop 0) is **built, wired into `orion-equilibrium-service`, given checkpointed resumable state — and
left at `ENABLE=false`.** Its own module docstring still says "does NOT wire into any live poll loop
yet"; someone wired it and never updated the doc. The pattern is: build the pure computation, defer
the live wiring as "a separate, smaller patch," never do it.

**The blocker is not a missing reducer. It is that metacog has no thread to reduce along, and no
consumer to reduce *for*.** Both measured live this session:

- `orion_metacog`: 5,958 rows. `correlation_id` is 100% populated and **100% unique** — 5,958 rows,
  5,958 distinct values, every group size exactly 1. Nothing links any row to any other.
- Nothing reads the table. The only `SELECT ... FROM orion_metacog` in the repo is an eval script
  that counts rows to prove the pipeline runs.

**The thing that makes this tractable, and that I did not expect to find: the episode layer is not
terminal.** `substrate_episode_summaries` has **690 live rows spanning 8 days**, and a real consumer
chain — `orion/substrate/felt_state_reader.py` exposes it as the `episode_summary` ctx lane, which
`services/orion-equilibrium-service/app/substrate_metacog_gate.py` calls live, and
`ReverieProposalV1.episode_summary_refs` (`orion/schemas/reverie.py:78`) exists to cite it.

So the recommendation is **not** to build a new consumer path for metacog. It is to route metacog's
events into the primitive that already has consumers, including the reverie surface this arc was
always aiming at.

The gap is that the existing episode is the wrong *kind* of episode: `_episodic_tick` rolls a
**clock-aligned fixed window** of reduction receipts into bookkeeping counts (accepted/rejected/
merged/noop, organ counts, reducer counts). It is time-boxed, not event-boxed. It cannot express
"the bus went anomalous at 21:13 and recovered at 22:15", which is exactly the unit "what happened
now, what happened next" needs.

## Current architecture

### metacog (the point-in-time layer)

Eleven trigger kinds. One gate module per kind evaluates real evidence, builds a `MetacogTriggerV1`,
`_publish_metacog_trigger()` applies a per-kind cooldown lane, publishes to
`orion:equilibrium:metacog:trigger`; `orion-cortex-orch` dispatches a `log_orion_metacognition` plan;
`orion-cortex-exec` LLM-drafts a `MetacogEntryV1`; `orion-sql-writer` persists to `orion_metacog`.

Live volume, 7 days, `trigger_kind='transport'` broken out by evidence branch:

```
bus_synaptic   1248        telemetry_anomaly  3247 (all kinds)
cortex-exec     810        transport          2553
cortex-orch     523        baseline            103
rpc_timeout     118        chat_turn            15
```

Two properties of the persisted row matter for anything downstream:

- `upstream` **does not survive**. It feeds the LLM draft prompt only and is dropped under budget
  pressure — verified 0 of 1,248 `bus_synaptic` rows carry it. The fields a reducer can actually read
  are `trigger_kind`, `trigger_reason`, `summary`, `mantra`, `severity`, `touches`, `timestamp`.
- `correlation_id` is a per-row UUID, not an episode key.

### The signal layer underneath it

`substrate_field_state` carries every Active-Inference domain's `prediction_error` in
`field_json->'node_vectors'` at **2-second resolution** — 42,169 samples in 24h for one node, 15x
finer than the 30s metacog poll. Retention is exactly 24h.

Reconstructing contiguous above-threshold runs for `node:substrate.bus_synaptic` over that window
gives **17 real episodes**:

```
started    ended      duration   samples
12:11:46   13:15:19     3813s     1866
21:13:30   22:15:57     3746s     1834
22:55:31   23:09:58      867s      423
22:49:03   22:52:29      206s      102
17:28:04   17:29:20       76s       38
...  (17 total)
```

17 episodes against 1,812 metacog rows for the same period. **Episode boundaries and durations are
already recorded** — they do not need a new producer, only a reader.

### The episode layer (the part that already has consumers)

- `EpisodeSummaryV1` (`orion/core/schemas/substrate_episodes.py`) — proposal-marked, explicitly
  "derived autobiographical memory, never authoritative truth", excluded from execution context by
  default, review-gated. Good precedent for how a derived temporal artifact should be marked.
- `_episodic_tick` (`services/orion-substrate-runtime/app/worker.py:1588`) — clock-aligned windows,
  idempotent by derived `episode_id`, retention-pruned. Live: 690 rows over 8 days.
- `felt_state_reader.py` — seven ctx lanes, including `episode_summary` and `latest_reverie_thought`,
  behind a freshness gate. Called live by `substrate_metacog_gate.py`.

Note the direction of the existing arrow: **episodes feed metacog** (via the dense/pulse gate), not
the reverse. metacog is the sink.

### What hop 0 does and does not do

`orion/metacog/trend_reducer.py` answers *"has this kept happening"* for a **single** series —
EWMA-z intensity, sustained-run detection, four floats of checkpointed state. That is intensity over
time on one channel. It is **not** a partial version of "what happened now, what happened next",
which is succession across heterogeneous kinds. Building hop 2 does not mean finishing hop 0.

### What just changed underneath this

PR #1533 (merged, deployed) made the `bus_synaptic` transport branch rising-edge triggered. Its
projected output is ~17 rows/day where it was 1,812 — i.e. one metacog row per real episode. That is
what makes a sequence reducer viable at all; before it, "sequence" meant 2,880 restatements of one
fact per day.

## Missing questions

1. **Does a condition-scoped episode belong in `EpisodeSummaryV1`, or is it a sibling schema?**
   The existing one is a window rollup with receipt-count fields that make no sense for a
   condition-scoped episode (`accepted_event_count`, `merged_event_count`, …). Extending it risks a
   schema whose fields are half-meaningful depending on which producer made the row. A sibling
   (`ConditionEpisodeV1`?) reusing the same proposal-marked/review-gated discipline is probably
   cleaner, but that is a real fork worth Juniper's call, not mine.
2. **What is the actual stitch across domains?** Time-overlap ("transport went hot at 21:13,
   execution followed at 21:19") is available today and is a genuine relation, not the synthetic
   time-adjacency I floated earlier in the conversation. But co-occurrence is not causation, and
   this repo has a documented history of exactly that confusion. Does the first slice claim only
   *succession*, or does it attempt *relation*?
3. **Does the 24h retention on `substrate_field_state` bound the reducer, or does the reducer
   persist its own episodes and escape it?** The latter is normal reducer work and is what makes
   arcs longer than a day possible — but it means a new durable table, i.e. a real schema change.
4. **Should metacog rows be joined to episodes, or should episodes be the thing metacog fires on?**
   These give very different systems. Joining is additive and reversible. Inverting the arrow
   (episodes become a trigger source, metacog reflects on completed episodes rather than instants)
   is a bigger, more interesting change and is arguably what "feel across time" actually means.
   **Not resolved here — this is the central design fork.**
5. **What happens to the other 10 trigger kinds?** This design is grounded in `bus_synaptic` because
   that is where the measured data is. `telemetry_anomaly` is the largest producer (3,247 rows) and
   has not been examined for episode structure at all.
6. **Is `rpc_health` (Option A) going to swamp this?** Measured 820 rows/24h vs `bus_synaptic`'s 466
   — it is now the largest transport contributor and still level-triggered. Any sequence reducer
   built now inherits that noise unless it gets the same edge-trigger treatment first.
7. **Does reverie actually want this?** `ReverieProposalV1.episode_summary_refs` exists, but nothing
   was traced this session that populates it from real episodes. Whether the reverie surface would
   consume condition-episodes or ignore them is unverified.

## Proposed schema / API changes

None proposed as final — Missing Question 1 and 4 gate the shape. The *candidate* shape, recorded so
the next pass starts from something concrete:

- A condition-scoped episode record: `domain`, `started_at`, `ended_at`, `duration_sec`, `peak_value`,
  `mean_value`, `sample_count`, `threshold_used`, plus `metacog_entry_ids` linking the reflections
  that fired inside its window. Proposal-marked, review-gated, retention-bounded — same discipline as
  `EpisodeSummaryV1`.
- No change to `MetacogTriggerV1` or `MetacogEntryV1`. No new trigger kind. No new bus channel in
  the first slice.

## Files likely to touch

- `orion/core/schemas/substrate_episodes.py` *or* a sibling schema module — pending MQ1.
- `orion/substrate/episodic_consolidation.py` — the existing evaluator is the closest precedent for
  a pure, testable consolidation function.
- `services/orion-substrate-runtime/app/worker.py` — a second tick alongside `_episodic_tick`, or an
  extension of it.
- `orion/substrate/felt_state_reader.py` — a new ctx lane, only once there is something real to read.
- `scripts/analysis/` — a measurement script, first, before any of the above.

## Non-goals

- **Not building a resolution/falling-edge metacog trigger.** I raised it as a gap and then measured
  it away: duration is already recoverable at 2s resolution from `substrate_field_state`. A
  resolution trigger would buy information we already have at the cost of an LLM-drafted reflection
  on "it stopped", which is not a metacognitive moment.
- **Not touching the dream aggregator's `observer='juniper'` filter.** That is a stated design
  decision — only Juniper's collapse mirrors enter dreams, deliberately, to stop dreams becoming
  shit blocks. I misread it as a dropped read earlier in this conversation; recording the correction
  here so the next agent does not "fix" it.
- **Not flipping hop 0's flag as part of this.** Different axis (single-series intensity), and
  turning it on is its own verification, not a side effect of this arc.
- **Not adding trigger kinds.** The README is explicit that shipping a new kind is not progress on
  the consumer question, and this arc is entirely about the consumer question.
- **Not wiring reverie in the first slice.** MQ7 is unverified; wiring an unproven consumer is how
  the last several of these ended up write-only.

## Acceptance checks

1. A measurement pass reconstructs cross-domain episode sequences from the existing 24h and reports
   them, **before** any production code is written. If the arcs look like noise, that is a real
   result and the arc stops there having cost one script.
2. Any episode record produced is non-degenerate against real data: episodes have varied durations
   (not all one window length), and the count is materially smaller than the metacog row count for
   the same period.
3. Whatever is built has a **named live consumer before it ships**, not after. This is the specific
   failure this whole design exists to avoid — `orion_metacog` (5,958 rows, 0 readers) is the
   cautionary case, and the README already warns that adding producers is not progress.
4. The reducer's output survives the `substrate_field_state` 24h retention boundary — i.e. an arc
   spanning more than a day is representable.
5. No new write-only table. If acceptance check 3 cannot be met, the correct outcome is to stop, not
   to ship the table and hope.

## §0A proposal-mode disclosures

- **What capability changes**: Orion gains the ability to represent an internal condition as a
  bounded event with a start, end, and duration, rather than only as instantaneous readings —
  and, prospectively, to relate such events in sequence.
- **What data is touched**: `substrate_field_state` (read), `orion_metacog` (read), a new derived
  episode store (write). All machine-generated telemetry and Orion's own machine-generated
  self-observations.
- **Privacy boundary**: none crossed in the proposed slice. Notably, Juniper's manually-authored
  `collapse_mirror` entries are **not** an input, and the dream aggregator's Juniper-only filter is
  explicitly out of scope. If a later slice proposes mixing those, that is a separate proposal.
- **What trace proves it worked**: episode records whose boundaries reconcile against the raw 2s
  `substrate_field_state` series they were derived from, plus a named consumer reading them.
- **Dangerous failure mode**: a derived episode treated as authoritative. `EpisodeSummaryV1`'s
  existing proposal-marked / excluded-from-execution-context / review-gated discipline is the
  mitigation and should be inherited, not re-litigated.
- **Disable / rollback**: flag-gated tick, default off, following `_episodic_tick`'s own
  `enable_episodic_tick` precedent. The derived store is droppable without affecting any producer.

## Recommended next patch

**A measurement script, and nothing else.** `scripts/analysis/measure_metacog_episode_arcs.py`:
reconstruct condition-scoped episodes for all five Active-Inference domains from the existing 24h of
`substrate_field_state`, join them against `orion_metacog` rows falling inside their windows, and
report: episode counts and durations per domain, cross-domain overlap/succession patterns, and how
many metacog reflections land inside versus outside any episode.

That is read-only, uses data already on disk, needs no schema decision, and answers Missing Questions
2 and 5 with evidence. It also directly tests whether this arc is worth continuing — if reflections
turn out to be uncorrelated with episodes, the join that this whole design rests on is not there, and
better to learn that from a script than from a merged reducer.

Explicitly **not** recommended as the next patch: extending `EpisodeSummaryV1`, adding a tick, or
touching `felt_state_reader`. Those all depend on MQ1/MQ4, which the measurement informs.
