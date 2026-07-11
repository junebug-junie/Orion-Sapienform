# Cognition metric lineage & liveness registry — ideas spec

**Mode:** Design/ideas. No build until Juniper picks a starting point. This
doc itself is tooling/observability scaffolding, not a cognition change — but
any *fix* to a dimension it flags as dead (`coherence`, `resource_pressure`,
`policy_pressure`, etc.) is proposal-mode territory per AGENTS.md 0A and
needs its own sign-off.

## Arsonist summary

Three times this session (φ corpus, tissue-viz novelty, tissue-viz arousal),
a new consumer was built against a SelfStateV1 dimension, and only then did
we discover the dimension was theater — hardcoded, structurally pinned, or
saturated by one stuck upstream channel. Each time the investigation was
redone from scratch by hand (curl-sampling `/latest`, hand-deriving formulas
from `scoring.py`). No record of any prior finding survived to help the next
consumer. The fix is not a taxonomy of metric names — that's a keyword
cathedral. The fix is turning the manual investigation into a rerunnable
tool, and recording its output somewhere the next consumer actually checks
before building.

## Current architecture

What already exists and is real, not aspirational:

- `orion/schemas/registry.py` — working registry convention:
  `SchemaRegistration(model, kind)` dataclass, dict-keyed, imported by
  callers. Closest existing precedent for "a registry of typed things with
  metadata" in this codebase.
- `orion/bus/channels.yaml` — declarative YAML registry already tracking
  `producer_services` / `consumer_services` / `stability` / `since` per bus
  channel. Real lineage tracking, scoped to channels not metric fields.
- `config/self_state/self_state_policy.v1.yaml` — already contains upstream
  lineage for every SelfStateV1 dimension as live data: `channel_dimension_map`
  (26 entries, channel → dimension_id), `stabilizing_channels`,
  `pressure_channels`. This is current and machine-readable; it has just
  never been read as "lineage" by anything.
- `scripts/fit_phi_encoder.py::_variance_gate` + `scripts/diag.py` — a
  working, already-proven empirical liveness classifier (`variance > eps`
  over a corpus of rows), used today only for phi corpus features. This is
  the exact mechanism I ran by hand three times this session via
  `curl .../latest` sampling — it already exists in code, just scoped to one
  consumer (`scripts/diag.py`, phi-corpus-only).
- `orion/telemetry/corpus_gate.py::is_corpus_row_healthy` — a second,
  narrower liveness check (`source.endswith(".none")`) at ingestion time.
- `orion/self_state/policy.py` — `dominant_channel_threshold: 0.25`,
  `unresolved_pressure_threshold: 0.60` already exist as policy constants,
  used in `builder.py` for `unresolved_pressures` / `dominant_channels`
  fields on the built SelfStateV1 — i.e. the builder already computes
  *some* per-tick provenance, it's just not persisted or aggregated across
  ticks anywhere.
- No existing file or endpoint answers, in one place: for dimension X, what
  formula produces it, what upstream channels feed it, who downstream reads
  it, and is it actually moving right now. That question gets re-derived
  from scratch every time a new consumer needs it.

This session's completed audit (13 SelfStateV1 dimensions, hand-classified,
evidence via `curl /latest` sampling + source tracing) is the seed data for
whichever idea below gets built first — it does not need to be re-derived.

## Sharpest problem statement

Liveness classification for a cognition metric already has a working
mechanism (variance-gate) and a working data source (`/latest` or a bus
sample) — it just never runs until a new consumer breaks in production. The
question is how to make that check something a new consumer runs *before*
building on a dimension, instead of discovering it live in Hub after
shipping.

## Ideas

### 1. `scripts/diag_self_state.py` — generalize the variance-gate to SelfStateV1 directly

- **What:** Sample `/latest` (orion-self-state-runtime, port 8118) or a
  short window of `substrate.self_state.v1` bus events N times over N
  seconds, compute per-dimension variance, classify using the same
  `_variance_gate` math already proven in `fit_phi_encoder.py`.
- **Why it matters:** Automates the manual investigation from this session
  (curl-sampling + hand math) into a rerunnable tool. Every future consumer
  gets the classification for free instead of re-deriving it.
- **Smallest buildable version:** read-only script, no new schema, mirrors
  `scripts/diag.py`'s existing shape. Output: JSON
  `{dim: {variance, classification, n_samples, window_s}}`.
- **Files:** `scripts/diag_self_state.py` (new); reuse `_variance_gate` from
  `scripts/fit_phi_encoder.py` if importable without dragging in corpus-only
  deps, else duplicate the ~5-line formula.

### 2. `MetricRegistration` dataclass, following the existing `SchemaRegistration` convention

- **What:** A small dict-based registry (`orion/self_state/metric_registry.py`)
  with one entry per dimension: `dimension_id`, `formula_ref` (function name
  in `scoring.py`), `upstream_channels` (pulled from `channel_dimension_map`
  in the policy YAML, not re-authored), `known_consumers` (file paths),
  `liveness_class`, `evidence_ref` (path to a diag report), `last_verified`.
- **Why it matters:** Reuses a pattern that already exists in this exact
  codebase instead of inventing new ceremony. A test can assert every entry
  in `ALL_DIMENSION_IDS` has a registration — deterministic gate, not vibes.
- **Smallest buildable version:** hand-populate 13 entries using this
  session's completed classification as seed data
  (2 live, 1 derivative, 6 dead/saturated, 4 sparse-legitimate), plus a test
  that fails if `builder.py`'s `ALL_DIMENSION_IDS` grows without a matching
  registration.
- **Files:** `orion/self_state/metric_registry.py` (new); `orion/self_state/builder.py`
  (import for completeness check); `tests/test_metric_registry_completeness.py` (new).

### 3. Downstream-consumer discovery via grep, not hand-maintained lists

- **What:** A script that greps for `dimensions["X"]`, `dimensions.get("X")`,
  `.score` access patterns per dimension name across `services/` and
  `orion/`, and emits the consumer list mechanically.
- **Why it matters:** Hand-maintained "who consumes this" lists go stale the
  moment a new consumer is added without updating the registry — the exact
  failure mode this initiative exists to fix, just moved up one level.
  Mechanical discovery can't rot the same way.
- **Smallest buildable version:** `scripts/find_dimension_consumers.py --dim
  uncertainty` → list of `file:line` hits. Feeds idea 2's `known_consumers`
  field at generation time instead of by hand.
- **Files:** `scripts/find_dimension_consumers.py` (new).

### 4. Pre-build gate: block/flag new consumers of a known-dead dimension

- **What:** A grep-based (not AST) check that a PR adding a new file reading
  a dimension marked `dead`/`saturated` in the registry fails with a message
  pointing at the registry entry and the known issue, instead of shipping
  silently.
- **Why it matters:** The actual leverage point — converts "discovered live
  in Hub" into "caught before merge." Matches AGENTS.md's "deterministic
  gates over repeated yelling."
- **Smallest buildable version:** one function folded into whatever
  `make agent-check` runs, scoped only to `orion/self_state/` dimension
  names for now.
- **Files:** `scripts/check_self_state_consumer_liveness.py` (new); wired
  into `Makefile`'s `agent-check` target.

### 5. Debug endpoint on self-state-runtime: explain-per-dimension

- **What:** Extend the existing `/latest` endpoint with an `explain=1` param
  returning, per dimension, its formula name, raw contributing
  `channel_pressures`, and a rolling variance stat computed server-side over
  the last N ticks.
- **Why it matters:** Makes liveness classification available *at runtime*
  to anyone building a new consumer, without reading `scoring.py` source or
  hand-sampling. This is the AGENTS.md-sanctioned "UI/debug surface" seam,
  and the one piece of infrastructure genuinely missing (field-digester has
  none either — see idea 7).
- **Smallest buildable version:** reuse idea 1's classifier logic, keep a
  small ring buffer of the last N self-state ticks in-process, compute
  variance on read.
- **Files:** `services/orion-self-state-runtime/app/main.py` (or wherever
  `/latest` is served — not yet confirmed this session).

### 6. `evidence_ref` staleness gate tied to source diffs

- **What:** Every registry entry's liveness classification must point at a
  committed artifact (a diag-style JSON snapshot with a timestamp), not an
  assertion in a doc. If `last_verified` is >90 days old, or `scoring.py` /
  `self_state_policy.v1.yaml` changed since, flag the entry stale.
- **Why it matters:** The difference between a real registry and a keyword
  cathedral — AGENTS.md's "runtime truth beats config truth" rule applies to
  the registry's own claims about other things, not just to Orion's runtime.
- **Smallest buildable version:** a `git log -1 --format=%cI -- <file>`
  check against each entry's `formula_ref` file, run by idea 4's gate
  script.
- **Files:** folds into `scripts/check_self_state_consumer_liveness.py`.

### 7. Narrow, standalone: field-digester debug/snapshot endpoint

- **What:** The one concrete unresolved mystery from this session — which
  capability keeps re-saturating the generic `pressure` / `contract_pressure`
  channels that dominate `resource_pressure` (MAX) and `transport_integrity`
  (MIN) — needs a debug endpoint on `orion-field-digester` to inspect the
  raw capability graph. It currently has none.
- **Why it matters:** Without this, `resource_pressure` /
  `transport_integrity`'s "aggregation-saturated" classification stays
  permanently un-actionable (diagnosed, not fixable) — no registry entry can
  carry real evidence for these two dims until this exists.
- **Smallest buildable version:** a read-only `/graph/snapshot` endpoint
  dumping current capability vectors and their channel values.
- **Files:** `services/orion-field-digester/app/` (route location not yet
  confirmed this session).

### 8. Phi corpus manifest: surface per-feature liveness at train time

- **What:** `fit_phi_encoder.py` already knows which seed-v4 dims passed the
  variance gate at training time (`SEEDV4_OPTIONAL_VARIANCE_DIMS`). Surface
  that classification as a field in the trained encoder's promoted manifest
  JSON.
- **Why it matters:** Closes the loop for the one consumer family (φ) that
  already has the most mature gating — smallest possible extension of
  existing, working code, no new mechanism.
- **Smallest buildable version:** one new key in the manifest dict already
  written by `--promote`.
- **Files:** `scripts/fit_phi_encoder.py`.

## Tensions and risks

- **Hand-authored registries rot.** The moment someone adds a channel to
  `self_state_policy.v1.yaml` or a new consumer file without updating the
  registry, it becomes exactly the stale keyword cathedral AGENTS.md 0A
  bans. Idea 3 (mechanical consumer discovery) and idea 6 (staleness gate
  tied to git diff) are not optional extras — without them this becomes a
  fourth place for facts to go stale, on top of the three that already
  exist.
- **"Dead" vs "quiet" is genuinely ambiguous from sampling alone.** A short
  live sample can't distinguish "structurally incapable of moving"
  (`policy_pressure` hardcoded 0.0) from "legitimately alarm-only and
  nothing's failing right now" (`reliability_pressure`). Pure runtime
  variance-sampling (idea 1) will misclassify sparse-but-real signals as
  dead. Correct classification needs both a variance sample *and* a static
  read of whether the formula can vary given non-degenerate inputs — the
  hand-tracing done this session — which is hard to automate well and is
  the biggest open risk in this whole effort.
- **Fragmentation instead of consolidation.** If the registry becomes a
  fifth file competing with `channels.yaml`, `registry.py`, and
  `self_state_policy.v1.yaml` instead of reusing/extending them, it adds a
  new place to keep in sync rather than reducing the count. Idea 2 leans on
  reuse (pulling `upstream_channels` straight from the existing policy YAML)
  deliberately for this reason.
- **Scope creep into "redesign SelfStateV1."** The registry surfaces the
  theater; it doesn't fix it. Any actual fix to `coherence` /
  `resource_pressure` / `policy_pressure` touches shared cognition machinery
  feeding multiple consumers — that's explicit proposal-mode territory per
  AGENTS.md 0A, separate from this initiative.
- **Field-digester debug access (idea 7) is a real service change**, not a
  metadata/tooling change — higher cost, its own scoping, shouldn't be
  bundled into "build the registry" as if it were free.

## Missing questions

- **Scope:** does this need to cover only SelfStateV1's 13 dimensions, or
  also phi corpus features (already has a working diag tool) and tissue-viz
  stats and whatever's next? Changes whether this is one small module or a
  cross-cutting convention.
- **Primary consumer:** is the main use case (a) a human running a check
  before wiring a new consumer, (b) a CI/pre-merge gate, (c) a live debug
  endpoint, or (d) corpus metadata for φ training? These want different
  artifact shapes (static file vs. generated report vs. HTTP endpoint vs.
  manifest field) — picking the wrong one first wastes the "smallest slice."
- **Consumer definition:** does "downstream consumer" mean direct field
  access only, or anything transitively derived (e.g. `agency_readiness`
  counts as a consumer of `coherence`)? Affects whether idea 3's grep alone
  is sufficient or real call-graph tracing is needed.

## Non-goals

- Not a redesign of SelfStateV1's formulas or aggregation strategy (MAX/MIN
  saturation fixes for `resource_pressure` / `transport_integrity` are
  separate, proposal-mode work, already scoped in
  `project_tissue_viz_novelty_arousal_theater` memory).
- Not a general-purpose metrics/observability platform. Scoped to Orion's
  "felt"/cognition metric families (SelfStateV1, phi corpus, tissue-viz),
  not infra metrics (CPU, latency, etc.) which already have their own
  channels.
- Not a continuously-running classifier service. Point-in-time
  classification with an explicit staleness gate (idea 6), not a new
  always-on component.

## Acceptance checks (for whichever idea is picked first)

- If idea 1: `scripts/diag_self_state.py` run against the live deployment
  reproduces this session's hand-derived classification (2 live dims, 1
  derivative, 6 dead/saturated, 4 sparse) without manual curl/math.
- If idea 2: `pytest tests/test_metric_registry_completeness.py` fails when
  a dimension is added to `ALL_DIMENSION_IDS` without a matching
  `MetricRegistration` entry; passes on current `main`.
- Either way: no changes to `orion/self_state/scoring.py` or `builder.py`
  formulas themselves — this phase is read-only tooling over existing
  behavior.

## Recommended next patch

Idea 1 (`scripts/diag_self_state.py`) first, then idea 2
(`MetricRegistration` seeded from this session's completed audit). Idea 1
needs no new abstractions — it points already-proven variance-gate code at a
new data source and has an immediately checkable output. Idea 2 then gets
real seed data for free (this session's 13-dimension classification) instead
of requiring a second investigation, and follows an existing codebase
convention (`SchemaRegistration`) rather than inventing one.
