# Curiosity operations analytics

## Arsonist summary

Curiosity has real lifecycle and output evidence, but it is split between
`substrate_durable_run_state`, deterministic journal metadata, and Orion's
FalkorDB world-view graph. Lightdash currently exposes none of it. This patch
adds a privacy-safe relational projection of the evidence that already reaches
PostgreSQL; it does not ingest prompts, findings, journal prose, graph claims,
or raw JSON.

## Current architecture

- `orion-durable-runs` emits one `DurableRunStateV1` event per durable graph
  transition.
- `orion-sql-writer` persists those events in
  `public.substrate_durable_run_state`.
- The durable workflow is
  `harness_turn -> read_turn_result -> publish_attention_row -> journal -> finish`.
- Curiosity journal records use `source_ref=curiosity:<run_id>`. Their
  deterministic footer records the material counts offered to the run,
  observed harness usage, and the graph elements written by that run.
- The footer is built by
  `orion/curiosity/journal.py::build_investigation_journal_entry` from the
  `CuriosityRunBriefV1` material counts, harness debug fields, and
  `read_run_footprint` output. The prose body itself is private.

## Missing questions

- The configured daily caps, cooldowns, outer timeout, governor timeout, and
  stream-stall timeout are runtime configuration, not persisted run facts.
  Historical "budget remaining" cannot be reconstructed honestly.
- FalkorDB may contain graph writes from a run that died before journaling.
  PostgreSQL can only prove the footprint recorded in a completed journal.
- A latest non-terminal database event is not proof that a process is running
  now. The model reports it as a latest recorded state with age, not as live
  process truth.
- `substrate_durable_run_state` is retained for 90 days by default. Lifecycle
  measures are retained-window evidence; a journal-only row may be older than
  that window and is not classified as pre-durable.

## Proposed schema / API changes

The role bootstrap creates four security-barrier views in a non-writable
`analytics_source` schema. They expose only bounded structural fields:

- `curiosity_run_transitions`: lifecycle identifiers, timestamps, node/status,
  type/attempt metadata, and an error-present flag; no correlation ID, error
  text, finding text, or raw detail JSON.
- `curiosity_run_journals`: one structural metadata row per curiosity journal;
  no title or body.
- `curiosity_run_graph_writes`: one row per journal and graph element type.
- `curiosity_run_material_pool`: one row per journal and approved material
  kind available at kickoff.

dbt then publishes:

- `fct_curiosity_runs`: one row per run ID observed in either lifecycle events
  or a journal.
- `fct_curiosity_run_transitions`: one row per persisted transition event.
- `fct_curiosity_graph_writes`: one row per run/journal/graph-element type.
- `fct_curiosity_material_pool`: one row per run/journal/material kind.

The one-to-many facts remain separate Lightdash explores. They are not joined
onto the run fact, so their measures cannot fan out run counts.

## Metric quality gate

### Run and lifecycle counts

1. **Provenance:** `DurableRunner._emit_state` constructs
   `DurableRunStateV1`; SQL Writer's `DurableRunStateSQL` persists it.
2. **Independence:** observed runs, runs with retained lifecycle rows, runs
   with retained completion evidence, and
   latest non-terminal rows are overlapping populations, not independent
   cognitive signals. They are deliberately labelled operational counts.
   Failed/resumed transitions are event counts and are not presented as run
   counts.
3. **Theory anchor:** event-sourcing state-machine accounting: distinct entity
   IDs measure observed entities; persisted transitions measure recorded state
   changes. No success-quality claim is attached.
4. **Live sanity:** 2026-09-14 inspection found 640 transitions across 50
   durable runs: 50 completed events, 186 failed events, 245 resumed events,
   and 159 running-node events. All 50 durable run IDs had a matching journal.
5. **Existing mechanism:** Hub Surface already summarizes a short window but
   does not provide a governed historical semantic model. This patch reuses the
   same table and status semantics.
6. **Reversibility:** analytics-source views and dbt views are additive and can
   be dropped without changing the producer, operational table, or graph.

### Actual usage and offered material

1. **Provenance:** `build_investigation_journal_entry` formats selected and
   available material counts plus `harness_step_count`, whole-turn elapsed
   seconds, and FCC/harness elapsed seconds from the real turn debug record.
2. **Independence:** whole-turn and harness time share the same run clock path;
   they are related, and the dashboard names the latter as a component of the
   former. Step count is a separate discrete execution trace but not a quality
   score. Available pool counts and offered sample totals are input inventory,
   not output quality.
3. **Theory anchor:** resource accounting: elapsed wall time, executed steps,
   and supplied item counts directly measure consumed time/steps and presented
   input volume. They do not measure curiosity quality.
4. **Live sanity:** 94 of 96 final journal footers parsed material counts and
   steps, and 65 parsed both time fields. Investigation runs offered 12 concepts
   and 6 relations in the current format; self-inquiry runs offered zero of
   each. Missing legacy metadata remains null and is excluded from averages.
5. **Existing mechanism:** Curiosity Atlas renders this footer as prose but has
   no aggregate or run-type comparison. No existing analytics model exists.
6. **Reversibility:** parsing is isolated in security views; a future typed
   telemetry field can replace it without changing Lightdash fact grains.

### Graph-write counts

1. **Provenance:** `read_run_footprint` counts FalkorDB nodes/edges carrying the
   run ID; `format_footprint` serializes those exact counts into the journal
   footer. The safe source view parses only element type and integer count.
2. **Independence:** per-type counts sum to total graph elements and therefore
   are intentionally components, not independent signals. Graph-write coverage
   is a presence measure derived from the same footprint and is labelled so.
3. **Theory anchor:** artifact accounting: persisted graph nodes/edges bearing
   a run ID are inspectable outputs of that run. Count is not correctness,
   novelty, or value.
4. **Live sanity:** 68 journals recorded non-empty graph footprints and 26
   explicitly recorded no graph writes. Parsed types included Hop (113),
   Finding (46), TurnOutcome (40), SUPPORTS edges (31), Prior (29),
   PriorRevision (21), SelfDefinition (15), ABOUT edges (5), Concept (3), and
   CONTRADICTS edges (3).
5. **Existing mechanism:** Curiosity Atlas already shows per-run graph growth;
   this patch aggregates the same evidence by type and time without duplicating
   graph reads.
6. **Reversibility:** the parser and child fact are isolated and carry no raw
   graph content. Removal does not affect FalkorDB or Curiosity execution.

## Files likely to touch

- `services/orion-analytics/scripts/bootstrap_analytics_roles.sql`
- `services/orion-analytics/models/{sources,staging,marts}`
- `services/orion-analytics/tests` and `evals`
- `services/orion-analytics/lightdash/{charts,dashboards}`
- `services/orion-analytics/README.md`

## Non-goals

- No raw prompt, journal, finding, claim, hop note, error text, correlation ID,
  graph JSON, or checkpoint payload enters analytics.
- No claim that a completed run was correct or valuable.
- No configured-budget or remaining-budget metric until those values are
  persisted with the run that actually received them.
- No write to Orion's FalkorDB graphs and no Graphify graph refresh.
- No union between Curiosity and Reverie facts.

## Acceptance checks

- Source/fact counts and distinct grains reconcile with zero deltas.
- Every child fact run ID resolves to `fct_curiosity_runs` without fanout.
- The analytics reader can query marts but cannot query operational tables or
  `analytics_source` views.
- dbt parse, compile, run, and test pass; the read-only live reconciliation
  reports no deltas.
- Lightdash compiles the new explores, uploads the dashboard, and executes every
  new chart successfully.
- Focused static tests prove private fields do not enter dbt models or semantic
  metadata.

## Recommended next patch

Add typed, non-narrative run-budget telemetry at admission/kickoff: configured
daily-cap lane, outer deadline, governor deadline, stream-stall deadline, model
route, and selected material counts. Validate it live before adding remaining-
budget or budget-headroom metrics.
