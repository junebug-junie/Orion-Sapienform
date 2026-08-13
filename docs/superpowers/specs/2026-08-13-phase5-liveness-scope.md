# Phase 5 — metric liveness: scope before build

**Mode:** Design/scoping. No implementation until Juniper picks a shape.

**Date:** 2026-08-13

## Arsonist summary

Phases 1–4 answer *where does this metric come from* and *who reads it*. They
do not answer the question that actually burned us: **is the number real.**

The incident record is entirely liveness failures, not lineage failures:

- `bus_synaptic_prediction_error()` — a mathematically permanent ~0.27 floor
  (`mean(|z|)` for a calm population has expected value `sqrt(2/pi)`, not 0).
  Varied in real time, could never read calm.
- `node:substrate.route`'s `prediction_error` — decayed to subnormal because a
  generic staleness loop multiplied it by 0.92 per tick for 48+ hours.
  *Decayed-to-zero* is indistinguishable from *genuinely-calm-at-zero* without
  checking the exact geometric ratio between successive values.
- `transport_prediction_error()` — excluded from one consumer as known-dead
  while still winning budget slots in a generic one.
- **Found today, live on main:** `perception_staleness` is an edge source in
  `config/field/orion_field_topology.v1.yaml:142` but is **absent from
  `NODE_CHANNELS`** and produced by **nothing** in the digester. The edge can
  never fire, so `capability:vision` reads `pressure=0.0` and the derived
  fallbacks stamp `confidence=1.0` — a fabricated healthy vector, re-stamped
  every tick.

A naive phase 5 ("sample each metric, run the variance gate") would have
caught the third and fourth but **not the first two** — the exact two that
cost the most to find. That is the reason to scope before building.

## The decisive finding: the four surfaces are not comparable

Phases 1–4 treat all 597 URNs uniformly. For liveness they are not uniform at
all. Measured live 2026-08-13:

| surface | URNs | live history | verdict |
|---|---|---|---|
| `field_channel` | 38 | `substrate_field_state`, **95,966 rows**, `field_json` per tick | **READY** |
| `inner_state` | 37 | per-signal tables (`substrate_attention_self_model` 6,521, `substrate_attention_broadcast_log` 6,521, `substrate_self_state` **0**) | **PARTIAL** |
| `organ_signal` | 252 | in-memory window only; `substrate_organ_emissions` has 1.65M rows but **one** `organ_id` | **BLOCKED** |
| `bus_channel` | 261 | Redis streams (XLEN/XREVRANGE) | **DIFFERENT KIND** |

Three consequences, each of which changes the build:

**1. `organ_signal` is 42% of the URN space and is unsampleable.**
`services/orion-signal-gateway/app/main.py:60` returns "most recent
OrionSignalV1 per organ_id from the **in-memory window**" — no persistence.
`substrate_organ_emissions` looked like a history table and is not: 1,652,247
rows, `count(distinct organ_id) = 1`, and that one value is
`biometrics_pressure`, which **is not one of the 30 organ ids in
`ORGAN_REGISTRY`** (which has `biometrics`). So 29 of 30 organs have no
history, and the one that does is keyed by a name the registry does not use.
Any phase 5 covering this surface must first decide whether to persist organ
signals at all — a producer change, not an observability change.

**2. `bus_channel` liveness is a different question with a different answer.**
A field channel's liveness is "does this *value series* carry information".
A bus channel's is "is *traffic* flowing". Same word, different measurement,
different failure modes, and no shared classifier. Conflating them would
produce a verdict column that means two things.

**3. Only `field_channel` can ship a real verdict today** — and it already
half does, via `classify_channel_series()` behind Hub's glossary panel.

## What already exists and must not be rebuilt

- `orion/field/channel_glossary.py::classify_channel_series()` — the proven
  classifier. Verdict vocabulary already encodes the incident history:
  `never_produced` / `dead` / `ratchet_suspect` / `quiet` / `live`.
- `services/orion-hub/scripts/field_channel_glossary_routes.py` — already
  computes these live for the 38 field channels and renders them.
- The glossary YAML's own rule, which this design keeps: **verdicts are
  computed, never declared.**

So for `field_channel`, phase 5 is not "build a classifier" — it is "expose
the one that exists through the metric layer and its gate", which is a much
smaller patch than the original phase-5 line implied.

## The trap: variance is not liveness

The two most expensive incidents were **not** low-variance. Both moved
continuously and both were wrong:

- the 0.27 floor was a *rest-point* error — the metric could not reach calm
- the 0.92 decay was a *provenance* error — values changed only because a
  decay loop touched them, with no producer refreshing them

A variance gate scores both as `live`. Catching them needs two checks that are
about structure, not spread:

- **Rest-point check** — what does this metric read when the world is calm?
  Requires knowing the theoretical rest value (§0A step 3's theory anchor),
  which cannot be derived from samples alone.
- **Self-refresh check** — is the series changing because a *producer* wrote
  it, or because a *decay loop* touched it? Detectable: successive values in
  an exact geometric ratio (0.92) with no producer write between them.

`classify_channel_series`'s `ratchet_suspect` is the closest existing
analogue, and it only covers the monotone-climb direction.

## Proposed shape (for discussion, not yet a plan)

**Slice A — field channels only, computed verdict through the layer.**
Surface the existing `classify_channel_series` result on the lineage card and
in `--json`, sampled from `substrate_field_state`. 38 metrics, zero new
classifier, zero new schema. Turns "Liveness verdict: NOT COMPUTED" into a
real answer for the one surface that can honestly answer it.

**Slice B — the decay-artifact detector.**
The check that would have caught `node:substrate.route`: flag a series whose
successive ratios are constant to within epsilon and match a known decay
constant, with no producer write in between. Narrow, mechanical, and aimed at
a failure mode with two confirmed instances.

**Slice C — static "can this metric ever fire" gate.**
`perception_staleness` needs **no sampling at all** — it is a wiring gap
detectable from config: a channel named as a topology edge source that is
absent from `NODE_CHANNELS` and has no producer. This is a phase-4-style gate,
cheap, and would have caught today's fabricated-vision case before deploy.
Arguably it should jump the queue ahead of A and B.

**Deliberately deferred, with the reason:**
- `organ_signal` — blocked on a persistence decision (see finding 1). Naming
  it as "phase 5 work" would hide a producer change inside an observability
  patch.
- `bus_channel` — different measurement; deserves its own name, not a shared
  `liveness` column.
- Rest-point checking — needs a per-metric theory anchor. Cannot be automated
  from data, and a fabricated anchor is worse than none.

## Missing questions

Answered by inspection, not posed:

- *Is there organ history anywhere?* One organ, wrong key. Answered above.
- *Does a classifier exist?* Yes, proven, scoped to field channels.
- *Would sampling have caught the known incidents?* Two of four. That is the
  central scoping result.

## Open questions for Juniper

1. **Order.** Slice C (static wiring gate) is the cheapest and would have
   caught a fabricated vector that is live on main right now. A and B are the
   "real" liveness work. C first, or A first?
2. **`organ_signal` persistence.** 252 URNs stay permanently unverifiable
   until organ signals are persisted. That is a producer change to
   `orion-signal-gateway` with real cost. Worth it, or is the organ surface
   accepted as lineage-only?
3. **Scope of the word "liveness".** Do bus channels get a *separate*
   `traffic` verdict, or stay out of the liveness story entirely?

## Non-goals

- One classifier across all four surfaces. The measurement is genuinely
  different per surface; a shared column would be a keyword cathedral.
- A declared/persisted verdict column. Verdicts stay computed.
- Fixing any metric this finds. Detection only; each fix is its own patch.
- Inferring theory anchors from data.

## Acceptance checks (whichever slice is chosen)

1. For a known-live channel (`cpu_pressure`) and a known-decayed one
   (`node:substrate.route` `prediction_error`), verdicts come from real stored
   history, not fixtures.
2. Slice C fails on `perception_staleness` as it exists on main today, and
   passes once the channel is either produced or removed from the topology.
3. No verdict is written into any config file.
4. Every surface without a real verdict continues to say so explicitly rather
   than rendering a blank that reads as "fine".
