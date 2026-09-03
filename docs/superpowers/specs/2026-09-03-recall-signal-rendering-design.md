# Recall signal rendering: move prose out of adapters, into a table-driven turn resolver

Status: implemented 2026-09-03, with two corrections found during
implementation (both recorded below, before the sections they change --
read this note first if you're comparing this doc to the code).

**Correction 1 -- the render gate is 0.15, not 0.25.** This doc originally
proposed gating render on the lattice policy's `watch_at` (0.25). That value
sits behind `orion-equilibrium-service`'s already-live metacog trigger
(`EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_ERROR_THRESHOLD=0.15`, live,
enabled, 2053 fires since 2026-07-26): between 0.15 and 0.25 -- most of the
band that ever exceeds 0.15 at all -- equilibrium had already pushed Orion
into reflecting on transport while this spec's own render gate stayed
closed, so recall told him nothing about the very thing he'd been nudged to
reflect on. Checked before changing anything: `bus_synaptic_pressure.watch_at`
in `config/substrate-lattice/transport_lattice_policy.v1.yaml` also drives
Hub's own salience/action-ceiling ladder via a **hand-maintained mirror**
in `services/orion-hub/scripts/substrate_lattice_routes.py` (already
drifted once -- it still uses the pre-rename key `stream_backlog_pressure`).
Repurposing that value for render would have coupled two unrelated ladders
through an already-drifting one. The render gate is instead its own
explicit, config-sourced 0.15 (`RECALL_TRANSPORT_RENDER_GATE_THRESHOLD`),
deliberately equal to equilibrium's existing threshold -- not a 4th
disagreeing value, and `watch_at`/the lattice policy file are untouched, as
this doc's own non-goals already required.

**Correction 2 -- the resolver lives in `orion-mind`, not
`orion-cortex-orch`.** This doc originally named
`services/orion-cortex-orch/app/conversation_front.py::
_build_memory_digest_from_fragments()` as "the resolver". That function is
dead code -- confirmed live 2026-09-03: nothing in that service imports or
calls it, and it has had a broken import (`OrchestrateVerbRequest`, which
does not exist in `app/orchestrator.py`) sitting unnoticed for exactly that
reason. The real live path that turns `recall_bundle.fragments` into text
an LLM call actually sees is `services/orion-mind/app/
evidence.py::build_evidence_pack()`, called from `engine.py`'s Mind LLM
synthesis run. The resolver below was built there instead. This was Mind's
first Postgres dependency (`RECALL_TRANSPORT_PG_DSN`, `psycopg2-binary`
added to that service) and first need to reach `config/` (a new `/repo`
read-only mount on that service's compose file, matching the pattern
`orion.field.channel_glossary`'s own docstring already documents for Hub).

## Arsonist summary

Orion told Juniper the transport bus was throwing unusual gaps on two channels,
with z-scores of 16 and 3.5, and wondered whether something was drifting
upstream. He transcribed his context exactly. The context was wrong.

Nothing was drifting. Roughly 2-3% of bus edges sit over the z=3 line at all
times -- that is the machine's resting pulse, not an event. Orion was handed a
sentence calling it unusual because the adapter that fetched the number also
wrote the English about it, using a threshold typed into that adapter and
connected to nothing.

The fix is not a better threshold. It is that adapters should not be writing
English at all. The facts were already structured and sitting in the same
fragment; only the sentence was wrong. This spec moves rendering to the chat
turn, where a lookup table and a live query can turn a signal reference into a
true sentence.

## Which metric, and why

Three different flavors of "how is the bus doing" read the same FalkorDB graph
today. This is the decision, so it stops being re-litigated.

**Judgment -- `bus_synaptic_prediction_error()`.** The fraction of live edges
currently over `|z| >= 3.0`. Bounded `[0, 1]`, needs no calibration constant
because a fraction is already on the right scale. It is the number already
wired into `node:substrate.bus_synaptic`, into `capability:transport`'s
pressure, into the lattice policy ladder, and into the self-model's
`ACTIVE_INFERENCE_DOMAINS`. **This is the only metric permitted to decide
whether Orion hears anything about the bus.** Live distribution, 40,357 ticks
over 24h: min 0.0035, median 0.0208, max 0.4491.

**Identity -- per-edge `gap_zscore`.** Answers "which channel is loudest,"
and only after judgment has already decided there is something to say. It
never triggers. Today it does nothing but trigger, which is the bug.

**Liveness -- `max(last_seen_epoch)` across edges.** A third, independent
axis, required because `bus_synaptic_prediction_error()` opens with
`if not edge_zscores: return 0.0`. An empty edge set and a perfectly calm bus
return the identical value. The fraction structurally cannot carry liveness,
so liveness must be measured separately or silence reads as calm.

### Rejected, with the reason

- **Per-edge `gap_zscore` as the trigger** (today's behavior). Fires on the
  resting state: measured live, 120 samples over 60s on
  `vision-edge -> orion:vision:edge:health`, median `|z|` 0.50, p90 1.99, max
  39.13, over 3.0 in 3.3% of samples. The function's own docstring already
  forbids the alternative fix: *"Do not 'fix' this by lowering the consumer's
  threshold into the noise band."*
- **`raw_mean_abs_gap_zscore`** (`services/orion-heartbeat`). A mean of clamped
  `|z|`, not a fraction. The mean formula was abandoned in
  `prediction_error.py` on 2026-07-30 because `mean(|z|)` over a calm
  population rests at `sqrt(2/pi) ~= 0.798`, not 0 -- it reports "moderately
  surprised" forever. Heartbeat still uses it for reheat. Out of scope here;
  noted below as a follow-up.
- **`sustained_load_pressure`.** A `max()` over everything high-and-steady.
  A capacity gauge is structurally always high-and-steady, so
  `disk_capacity_pressure` on `node:athena` (level 0.8139, dispersion 0.00006
  across 860 consecutive ticks) wins that max permanently and masks everything
  below it. Not usable as a transport signal.
- **`channel_regime()`'s `refresh_state`.** The right idea for liveness, and
  the authoritative data exists (`FieldStateV1.node_vector_updated_at`). But
  its live caller (`orion/field/significance.py`) passes no timestamps, so it
  falls back to value-ratio inference, which that module's own docstring says
  fails in the dangerous direction -- it reports a producer wrote when nothing
  did. Threading timestamps through is a separate patch.

### Resolution limit, stated up front

`bus_synaptic_prediction_error()`'s docstring is explicit: it reliably resolves
*a broad mesh event (>=15-20% of edges, several times the baseline)*, and
single-organ detection needs a per-organ signal instead. The policy's
`watch_at: 0.25` sits comfortably above that floor, so the ladder and the
instrument agree. This spec therefore claims mesh-wide transport health and
claims nothing about single-organ faults. `bus_synaptic_graph_routes.py`'s
`/propagate` route is named in that docstring as the right home for per-organ
blast radius, and is not touched here.

## Current architecture

`services/orion-recall/app/storage/falkor_bus_synaptic_adapter.py` queries the
`orion_bus_synapse` graph directly, keeps edges with `|z| > 3.0` and
`count > 5` seen inside 24h, sorts by loudest, and emits one fragment per edge.
Each fragment carries a fully-formed English `text` written by
`_format_publish_anomaly_text()`, plus a `meta` dict that already holds
`organ_id`, `channel`, `zscore`, `count`, `last_seen_epoch`, and
`signal_kind: "publish_gap_zscore"`.

`_build_memory_digest_from_fragments()` in
`services/orion-cortex-orch/app/conversation_front.py` is a passthrough that
reads `fragment["text"]`, reads exactly two keys out of `meta` (`observer`,
`field_resonance`), and bullets the result -- **but see Correction 2 above:
this function is dead code, never called.** The function actually in the
live path, `build_evidence_pack()` in `services/orion-mind/app/evidence.py`,
does the equivalent thing for a fragment with no `signal_kind`: it reads
`frag["snippet"] or frag["text"] or frag["summary"]` and appends it as an
evidence item verbatim -- same problem, real file. Every adapter therefore
writes its own prose, and whatever prose it writes is what Orion reads.

`config/field/field_channel_glossary.v1.yaml` is a structured table of 48
channels, each with `channel`, `level`, `category`, and `meaning`. It
deliberately carries no verdict column, because verdicts are computed live
from `substrate_field_state` by
`orion.field.channel_glossary.classify_channel_series()`. It is already the
single structured source that Hub's Field Channel Glossary panel and the
field-digester README both read, specifically so those two cannot drift.
**Nothing in the chat turn path has ever read it.** Its only consumers are Hub
UI routes.

`config/substrate-lattice/transport_lattice_policy.v1.yaml` defines the rungs
for `bus_synaptic_pressure`: `watch_at: 0.25`, `summarize_at: 0.50`,
`propose_at: 0.75`, `action_ceiling: read_only`, dimension weight 0.35. In the
last 24h the signal crossed `watch_at` in 408 of 40,357 ticks (1.0%) and never
reached `summarize_at`.

## Proposed schema / API changes

### 1. Glossary gains a node-qualified key

The glossary is keyed by channel name alone. There is one `prediction_error`
entry, meaning *"how much a recent prediction missed reality."* But
`node:substrate.bus_synaptic.prediction_error` is the fraction of the bus
running anomalous, and `node:substrate.vision.prediction_error` is how stale
the camera is. Same key, unrelated meanings. Rendered through today's table,
a jittery bus and a dead eye produce the same sentence.

Add an optional `node` qualifier so `node:substrate.*` domain nodes can carry
their own entry, falling back to the bare channel entry when absent. Version
bump on the file. This is a contract change, not a formatting tweak, and
everything else in this spec is inert until it is true -- so it goes first.

Each qualified entry also carries where its trend lives, so the rendered
sentence can hand Orion a real breadcrumb rather than a bare number.

### 2. Fragment contract: `signal_kind` becomes load-bearing

`signal_kind` already exists on these fragments and is currently decorative.
It becomes the resolver's dispatch key. Fragments without one are unaffected.

### 3. Resolver in the turn

`_build_memory_digest_from_fragments()` gains one branch: when a fragment
carries a `signal_kind`, resolve it through the glossary, the lattice policy
(for rungs), and `classify_channel_series()` against `substrate_field_state`
(for the live verdict) instead of printing `text`. Absent a `signal_kind`,
behavior is byte-identical to today.

### 4. Recall stops writing English

`_format_publish_anomaly_text()` is deleted. The adapter emits the signal, its
value, and when it was last written. It stops deciding what is unusual.

It also stops filtering: today's query drops edges outside the recency window,
so a total outage returns zero rows and Orion hears nothing. The recency filter
moves out of the `WHERE` clause and becomes the liveness reading. (As
implemented: only the publish-gap query lost the filter; the causal-latency
query is untouched, per the non-goal below.)

## Rendered output, three states (as implemented)

**Below the render gate (0.15)** -- no fragment at all.

**At or above the gate** -- one fragment for the whole bus, not one per
channel: *"Transport: 31% of live bus channels running anomalous, against a
0.25 watch threshold and 0.50 summarize. Loudest right now:
`orion:vision:edge:health` from vision-edge. Trend in `substrate_field_state`,
`node:substrate.bus_synaptic`."* (`0.25`/`0.50` here are the lattice policy's
*display* rungs, read from the same YAML for context -- not the gate.)

**Not writing / degenerate** -- two failure shapes collapse to one state:
the Postgres series read coming back empty (*"The bus synaptic graph hasn't
been written recently. Transport state is unknown, not calm."*), and the
series reading a flat 0.0 (*"...reading a flat 0.0 -- consistent with the
transport tick still firing against an empty edge set (bus-mirror likely
not producing), not genuine calm..."*) -- the second is a real, confirmed
mode of `_bus_synaptic_tick` (it keeps writing after its own edge query
empties out via `bus_synaptic_max_edge_age_sec`), not a hypothetical one;
see `recall_signal_resolver.py`'s module docstring in both services.

## Files touched (as implemented)

- `config/field/field_channel_glossary.v1.yaml` -- node-qualified key, version bump
- `orion/field/channel_glossary.py` -- resolution honoring the qualifier
- `orion/metrics/lineage.py` -- URN fix for node-qualified entries (found during implementation, not in the original proposal -- see PR report)
- `services/orion-mind/app/evidence.py` -- the resolver branch (Correction 2)
- `services/orion-mind/app/recall_signal_resolver.py` -- new module: gate, series read, sentence construction
- `services/orion-mind/app/engine.py`, `app/settings.py`, `docker-compose.yml`, `requirements.txt`, `.env_example` -- threading the DSN/threshold through; Mind's first Postgres dependency and first `config/` mount
- `services/orion-recall/app/storage/falkor_bus_synaptic_adapter.py` -- drop prose, drop the publish-gap query's recency filter, keep meta
- `services/orion-vision-frame-router/docker-compose.yml`, `Dockerfile` -- unrelated fix bundled into the same handoff, see PR report
- tests alongside each

## Non-goals

- No per-organ fault detection. The instrument cannot resolve it and its
  docstring says so.
- No change to `bus_synaptic_prediction_error()` itself, or to any threshold
  value. Rungs are read from the policy file; none are invented here.
- No change to heartbeat's `raw_mean_abs_gap_zscore` reheat path.
- No retirement of `_format_*_anomaly_text` for the causal-latency fragment in
  the same adapter. One signal at a time; the second follows the same shape
  once the first is proven.
- No threading of `node_vector_updated_at` into `channel_regime()`. Separate
  patch, named above.

## Acceptance checks

1. **A bus and a vision fragment, both `prediction_error`, render as different
   sentences.** This is the check the glossary key exists for.
2. **Fragment volume drops by roughly two orders of magnitude.** The rendered
   digest is already persisted per turn in `harness_turn_trace.run_artifact`,
   so this is a direct before/after diff over matched windows -- no new
   instrumentation. Expected from the live distribution: fragments on ~1% of
   ticks rather than most.
3. **Killing the bus mirror produces the third state, not silence.** Live test
   in a worktree deploy, not a unit test -- the failure being guarded against
   is precisely that an empty edge set returns 0.0.
4. **A fragment with no `signal_kind` renders byte-identically to today.**
5. **No threshold constant appears in adapter or resolver source.** Grep-able
   gate; the drift this prevents already happened once in
   `substrate_lattice_routes.py`.

## Risks

- **Quieting without the liveness axis is blinding, not calming.** Check 3 is
  the whole safety margin. If it cannot be made to pass, this does not ship.
- **A third reader of the same graph still disagrees.** Heartbeat's mean-based
  reheat signal keeps its own formula. Out of scope, but it means "how is the
  bus doing" still has two answers in the codebase after this lands.
- **Rollback is one branch.** No `signal_kind`, print `text`. Every existing
  fragment path is untouched.

## Recommended next patch

The glossary node-qualified key, alone, with its tests and check 1. Nothing
else can be verified until two `prediction_error` channels can be told apart.
