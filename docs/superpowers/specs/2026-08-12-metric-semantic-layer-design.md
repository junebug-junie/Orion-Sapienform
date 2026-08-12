# Metric semantic layer: lineage, liveness, and edit-time compliance gates

**Mode:** Design/proposal. Tooling and observability scaffolding, not a
cognition change — but any *fix* to a metric this layer classifies as dead is
proposal-mode territory per CLAUDE.md §0A and needs its own sign-off.

**Date:** 2026-08-12

## Arsonist summary

CLAUDE.md §0A already contains the correct rule. The "Metric quality gate"
spells out six mandatory steps — trace provenance to real code, check
independence, name a theory anchor, sanity-check live data, search for an
existing mechanism, assess reversibility — and demands they be re-run *every
time*, even for a metric that "seems obviously fine."

That gate is prose in a contract file, enforced by an agent choosing to
remember it. The failure record shows what that buys:

- `bus_synaptic_prediction_error()` shipped with a mathematically permanent
  ~0.27 floor. Caught only by hand-recovering the pre-aggregation statistic.
- `node:substrate.route`'s `prediction_error` decayed to subnormal because a
  generic staleness loop multiplied it by 0.92 per tick for 48+ hours. Looked
  identical to "genuinely calm at zero" until the exact geometric ratio was
  checked by hand.
- `transport_prediction_error()` was excluded from one consumer's active-domain
  set as known-dead, yet its tick kept running and its node kept winning real
  budget slots in `endogenous_curiosity.py` — a generic consumer nobody checked.
- A recorded session claimed a decay-artifact check that was never run, while
  the disproving 0.92 ratio sat in the data.

Every one of these is the same shape: **a consumer was built or modified
against a metric whose real upstream behavior nobody traced.** The gate did not
fail because it was wrong. It failed because nothing mechanically ran it, and
nothing put the answer in front of the agent at the moment of the edit.

The fix is not another taxonomy of metric names. Five registries already name
things. The fix is a **join across what already exists**, a **computed**
liveness verdict instead of a declared one, a **mechanically discovered**
consumer list, and a gate that fires **at edit time, in the agent's context**,
not in a doc the agent won't open.

## Current architecture

Five registries exist today. Each is real and load-bearing. None of them join.

| Registry | Granularity | Count | Carries | Gate |
|---|---|---|---|---|
| `orion/bus/channels.yaml` | bus channel | 261 | producers, consumers, `schema_id`, stability, since | `check_bus_channels.py`, `check_single_consumer_channels.py` |
| `orion/schemas/registry.py` | schema class | hundreds | `SchemaRegistration(model, kind)` name→class | `check_schema_registry.py` |
| `orion/inner_state_registry.py` | inner-state signal | 13 | producer service, cadence, composition status, cognition consumers | `check_inner_state_registry.py` |
| `orion/signals/registry.py` | organ | 30 | signal kinds, canonical dimensions, causal parent organs, bus channels | none found |
| `config/field/field_channel_glossary.v1.yaml` | field channel | 38 | level, category, meaning, `self_state_dimension`, `evidence_dimension` | Hub panel + live classifier |

Plus 11 `scripts/check_*.py` deterministic gates and 5 `scripts/platform/audit_*.py`
auditors, wired into `make` targets.

### What is already right, and should be reused rather than rebuilt

`orion/field/channel_glossary.py::classify_channel_series()` is the single best
asset here. It is a battle-tested liveness classifier whose verdict vocabulary
directly encodes the incident history above:

- `never_produced` — absent entirely, not even a reconciled default
- `dead` — present but all values subnormal (covers folded-away,
  unproduced-past-reconcile, and subnormal-noise alike)
- `ratchet_suspect` — monotonically non-decreasing with real net climb, and
  ≥ `RATCHET_MIN_SAMPLES` points; catches a `mode=add` channel missing from the
  decay channel sets
- `quiet` — wired and present, low variance this window
- `live` — real variance

The glossary YAML also gets a design decision exactly right, and says so in its
own header: it **deliberately omits a `verdict` field**, because a static
verdict column is precisely what went stale once already. Liveness is computed
live from `substrate_field_state`, not hand-maintained.

That principle — *verdicts are computed, never declared* — is the load-bearing
idea of this whole design. It exists today for 38 of the metric surfaces and
for none of the others.

### The actual gap

Three things are missing, and they are specific:

**1. No shared ID space.** A field channel `cpu_pressure`, an organ dimension
`biometrics/gpu_load#level`, and an inner-state field
`self_state.v1#reasoning_pressure` are the same *kind* of thing — a named
scalar with an upstream and downstream — but nothing can address them
uniformly, so nothing can ask a question that spans them.

**2. No reverse edge, and graphify structurally cannot supply it.** The
knowledge graph carries 81,046 links, including 7,984 `references` and 7,039
`uses`. None of them answer "who reads `cpu_pressure`", because
**metrics are string dict keys, not symbols** — verified: every access is of
the form `vector["cpu_pressure"]` / `.get("cpu_pressure")` / a YAML key. Symbol-level
extraction cannot see them. Consumer lists in `inner_state_registry.py` and
`ORGAN_REGISTRY` are therefore hand-maintained, and go stale silently.

**3. No edit-time surface.** Every gate today is a CI/`make` check that runs
*after* an agent has already written the code. Nothing intercepts the moment an
agent opens a file that reads a metric and tells it what that metric actually is.

### Confirmed staleness in the prior design

`docs/superpowers/specs/2026-07-10-cognition-metric-lineage-registry-design.md`
proposed 8 ideas against this same problem. None were built. It is now partly
stale: it centers on `config/self_state/self_state_policy.v1.yaml` and
`orion/self_state/`, both of which **no longer exist** — removed in the
2026-07-22 SelfStateV1 burn. Its ideas 1, 2, 3, and 4 survive conceptually and
are absorbed below; its `config/self_state`-specific mechanics do not.

## Missing questions

Answered by inspection, not posed to Juniper:

- *Does a metric registry already exist?* Five do, at four different
  granularities. A sixth would be a keyword cathedral. → **join, don't author.**
- *Can graphify supply consumer discovery for free?* No. String-key access is
  invisible to symbol-level extraction. → purpose-built scanner needed, and its
  cost is justified rather than assumed.
- *Is there a proven liveness classifier to reuse?* Yes,
  `classify_channel_series()`, scoped to field channels only. → generalize the
  data source, keep the classifier.
- *Is there a working edit-time hook precedent?* Yes. The graphify PreToolUse
  nudge fires on every Bash/Read call in this repo today. → the mechanism is
  proven; this design reuses its shape.

Genuinely open, and the reason this stops for a decision — see **Recommended
next patch**.

## Proposed schema / API changes

### Metric URN — the ID space

```
metric://<surface>/<producer>/<name>[#<field>]
```

```
metric://field_channel/orion-field-digester/cpu_pressure
metric://inner_state/orion-self-state-runtime/self_state.v1#reasoning_pressure
metric://organ_signal/biometrics/gpu_load#level
metric://bus_channel/orion-substrate-runtime/orion:substrate:brain_frame
```

This is a *projection key*, not a new registry. No URN is hand-authored. Every
URN is derived from an entry that already exists in one of the five registries.
A metric that resolves to no URN is unregistered — and that is itself the
finding.

### `MetricNode` — the resolved join (computed, not stored)

```python
@dataclass(frozen=True)
class MetricNode:
    urn: str
    surface: str                      # field_channel | inner_state | organ_signal | bus_channel
    producer_service: str
    producer_ref: str | None          # file:line of the function that computes it — §0A step 1
    upstream: tuple[str, ...]         # parent URNs — §0A step 2 independence
    downstream: tuple[str, ...]       # file:line consumers, MECHANICALLY discovered
    liveness: str | None              # computed via classify_channel_series, never declared
    liveness_window: str | None       # what window produced that verdict
    sampled_at: datetime | None
    registry_source: str              # which of the 5 registries this came from
    theory_anchor: str | None         # §0A step 3; None is a legitimate, visible answer
```

`liveness` is **never persisted into a config file.** It is computed at
resolve time or read from a dated report artifact. The glossary already
established this rule; this design does not break it.

### No bus/channel/schema contract changes

Nothing here adds a channel, changes a payload, or alters a schema. It is a
read-only projection over existing registries plus a live sample. That keeps
reversibility high (§0A step 6): if the layer is wrong, deleting it removes a
script, a hook, and a doc — nothing is baked into a schema, manifest, or
training default.

## Files likely to touch

**New:**

- `orion/metrics/lineage.py` — the resolver. Reads all five registries, emits
  `MetricNode`s. Zero hand-authored metric content.
- `orion/metrics/consumers.py` — mechanical downstream discovery. AST-based
  scan for string-literal dict-key access (`Subscript` with a `Constant` slice,
  `.get("...")` calls) over `orion/` and `services/`, cross-referenced against
  known metric names. Explicitly **not** a regex sweep (§0A "no regex swamp").
- `orion/metrics/liveness.py` — generalize `classify_channel_series()`'s data
  source beyond `substrate_field_state`. The classifier math is imported, not
  reimplemented.
- `scripts/check_metric_lineage.py` — the CI gate.
- `scripts/hooks/metric_lineage_nudge.py` — the edit-time PreToolUse hook.
- `tests/test_metric_lineage.py`, `tests/test_metric_consumers.py`.

**Modified:**

- `Makefile` — `check-metric-lineage` target, following the 11 existing ones.
- `.claude/settings.json` — register the PreToolUse hook.
- `CLAUDE.md` §0A — point the metric quality gate at the command that now runs it.

**Read-only inputs, unchanged:** all five registries.

## Non-goals

- **Not a sixth registry.** No new hand-authored list of metric names. If it
  can't be derived from an existing registry, it doesn't get a URN.
- **Not a static verdict column.** Liveness is computed. The glossary's header
  already explains why, from a real incident.
- **Not a metric taxonomy or ontology.** No categories, tiers, or semantic
  classes beyond what the five registries already carry.
- **Not a fix for any dead metric it finds.** Detection only. Every fix is a
  separate, individually signed-off patch.
- **Not a replacement for the existing gates.** It joins them; it does not
  subsume `check_bus_channels.py` et al.
- **Not blocking on unknown theory anchors.** `theory_anchor: None` is a
  legitimate, visible state — not a build error.

## Acceptance checks

Each is a runtime fact, not a config fact (§0A "runtime truth beats config truth").

1. `python scripts/check_metric_lineage.py --json` resolves ≥ 38 field
   channels + 13 inner-state signals + 30 organs without a hand-authored URN
   list, and prints the count per registry source.
2. For a known-live metric (`cpu_pressure`) and a known-dead one
   (`node:substrate.route` `prediction_error`, decay-to-zero, 2026-07-26), the
   computed verdicts are `live`/`quiet` and `dead` respectively — verdicts
   recovered from real stored history, not fixtures.
3. Consumer discovery finds `endogenous_curiosity.py` as a consumer of the
   generic node-metric path — the consumer that the `transport_prediction_error`
   retirement missed. This is the regression test for the incident that
   motivated the design.
4. The gate exits non-zero when a test fixture adds a new consumer of a
   metric whose latest computed verdict is `dead` or `never_produced`.
5. The PreToolUse hook, on an `Edit` to a file containing a registered metric
   key, emits that metric's lineage card (producer ref, upstream, downstream
   blast radius, verdict, sample age) into the agent's context — verified by a
   real hook fire with a log line, not by the hook merely being registered.
6. `make check-metric-lineage` is green on `main` at time of merge, with any
   pre-existing dead metrics listed as **reported, not fixed**.

## Recommended next patch

Build in this order. Each phase is independently useful and independently
reversible; phases 2 and 3 are where the user-visible payoff is.

**Phase 1 — resolver + URN join (no gate, no hook).**
`orion/metrics/lineage.py` + `tests/`. Read-only, additive, zero runtime blast
radius. Deliverable: `check_metric_lineage.py --json` dumps the joined graph.
This alone answers "where does this come from and what's upstream" in one place
for the first time.

**Phase 2 — mechanical consumer discovery.**
`orion/metrics/consumers.py`. This is the highest-value single piece: it
produces the **blast radius** — the thing an agent editing a widget currently
has no way to see. It also mechanically re-derives the hand-maintained consumer
lists in `inner_state_registry.py` and `ORGAN_REGISTRY`, and any disagreement
between hand-list and discovered-list is an immediate finding.

**Phase 3 — edit-time PreToolUse hook.**
The actual answer to "agents don't trace upstream." Puts the lineage card in
front of the agent *before* the edit lands, using the same mechanism as the
graphify nudge that already works in this repo.

**Phase 4 — CI gate + `make` target + CLAUDE.md §0A pointer.**
Turns §0A steps 1, 2, and 4 from prose into a failing check.

**Phase 5 — liveness generalization beyond field channels.**
Deferred deliberately: it needs a per-surface decision about where the live
sample comes from (Postgres history, bus window, or a service `/latest`
endpoint), and that is a bigger question than the first four phases combined.

### Decisions taken (2026-08-12)

- **Scope of the first pass: phases 1+2.** Built and merged as one PR.
- **Phase 3 hook behavior: always show the full lineage card**, on every edit
  touching a registered metric — not only on dead/suspect verdicts. Juniper's
  call. The nudge-blindness risk (the agent-board Stop banner a live session
  ignored across ~20 turns) is accepted deliberately: a card that only appears
  for already-known-bad metrics cannot prevent the failure where nobody knew
  the metric was bad yet.

## Build findings (phases 1+2, 2026-08-12)

Two things the build changed about the design, both found by running the
scanner against the real repo rather than reasoning about it:

**1. `collection_member` is the access kind that matters most, and it was
missing from the original design.** The first scanner draft classified only
subscript / `.get` / dict-key / attribute access as high confidence. Under
that model `cpu_pressure` had **81 raw hits and a blast radius of zero** — no
explicit key read anywhere outside tests. Its real production consumers all
register it into a *collection that some loop iterates*:

```
orion/field/pressure.py:35                              (channel tuple, reducer-iterated)
orion/field_coherence.py:9,13                           (coherence rule pairs)
services/orion-field-digester/app/digestion/decay.py:31 (NODE_DECAY_CHANNELS)
services/orion-field-digester/app/tensor/channels.py:4  (channel definition set)
```

That `decay.py:31` line is the exact mechanism behind the 2026-07-26
decayed-to-zero incident in CLAUDE.md §0A — a generic staleness loop
multiplying a channel by 0.92 per tick. Membership in that list is now
first-class lineage rather than something to rediscover by hand.

This is also the generalized form of the `transport_prediction_error`
retirement miss: a metric "retired" from one *named* consumer keeps feeding
every *generic* one. A blast radius that ignores collection membership reports
empty for exactly the metrics most likely to be silently alive.

**2. Attribute access was systematically undercounting inner-state metrics.**
Field-channel and organ metrics are string dict keys; inner-state pydantic
scalars are attributes (`frame.recon_error`). The first run reported 9
inner-state metrics with zero consumers; adding `ast.Attribute` recovered 4 of
them (+153 hits). The remaining 5 (`delta_phi`, `delta_recon_error`,
`overall_confidence`, `recon_error`, `shuffle_baseline_loss`) appear to be
genuinely unread and are reported as findings, not fixed here.

### Measured, not asserted

```
bus_channel   261    scan tokens     386
field_channel  38    files scanned  3714
inner_state    36    consumer hits  4713
organ_signal  252    runtime         ~13s
TOTAL         587 URNs
```

### Known limits, stated

- Dynamic access (`vector[name_from_config]`) is invisible to any static
  scanner. Such call sites are **undercounted, never overcounted**.
- Python matching is exact-equality on the string constant, so prose that
  merely contains a metric name produces no hit at all. The YAML/JSON scan is
  substring-based and correspondingly noisier — it is tagged `config` and
  excluded from high-confidence blast radius.
- Liveness is still **not computed** (phase 5). Every lineage card says so
  explicitly rather than leaving a blank that reads as "fine".
