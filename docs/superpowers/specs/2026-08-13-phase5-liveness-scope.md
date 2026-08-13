# Phase 5 — signal semantics: provenance, window, commensurability

**Mode:** Design/scoping. No implementation until Juniper picks the order.

**Date:** 2026-08-13 (revised same day — see "What this revision retracts")

## What this revision retracts

The first version of this doc framed phase 5 as **liveness** and treated the
four metric surfaces being different as a *scoping obstacle*. Both were wrong.

- "Liveness" is not the problem. Every metric in the incident ledger below was
  live. They moved, they varied, they were wrong anyway.
- The surfaces differing is the **design premise**, not an obstacle. A field
  channel, an inner-state scalar, an organ signal, and a bus channel are
  different kinds of thing and get different treatment by construction. The
  first draft presented that as a blocker to be resolved. It isn't one.
- It also proposed a second answer to bus traffic. Bus traffic is already
  answered by the bus synaptic graph over `orion-bus-mirror` actuals, which
  sees multi-hop turns rather than unweighted per-tick counts. Building a
  parallel `traffic` verdict would be a duplicate mechanism.

## The one problem

**A number reaches a consumer with no record of what it is.**

Three facts are missing at the point of use, and every incident in the record
is the absence of exactly one of them:

| axis | the missing fact | failure it produces |
|---|---|---|
| **Provenance** | which producer actually wrote this, and when | a decayed or dead value reads as a calm real one |
| **Window** | what interval and what transform this summarizes | "near max but steady" is indistinguishable from "spiking" |
| **Commensurability** | is this on the same scale/semantics as what it is combined with | a low-resolution input silently dominates a merge or a ranker |

This is not a taxonomy. Each axis has a mechanical detector and a measured
instance, below.

## Incident ledger, mapped

| incident | axis | status |
|---|---|---|
| `bus_synaptic_prediction_error()` permanent ~0.27 floor (`mean(\|z\|)` rests at `sqrt(2/pi)`, not 0) | Window | fixed; live min now 0.0039 |
| `node:substrate.route` decayed by 0.92/tick for 48h, read as calm | Provenance | fixed |
| `transport_prediction_error()` excluded from one consumer, still winning budget slots in a generic one | Provenance | retired |
| `perception_staleness` wired as a topology edge source, produced by nothing → fabricated `pressure=0.0/confidence=1.0` | Provenance | fixed 2026-08-13 by the perception P4 work; edge now maps `prediction_error` |
| `thermal_pressure` (18 distinct) beating capability `pressure` (1,325 distinct) on 91.76% of ticks | Commensurability | fixed by deleting one routing entry |
| **`node:substrate.codebase` dominating the merged `prediction_error` channel** | Commensurability | **OPEN — measured below** |
| `resource_pressure` reading calm during a producer outage, crediting the in-flight action with success | Provenance | **OPEN — latent, see below** |
| "very busy at near max but steady state, so actually peaceful" is not expressible | Window | **OPEN — no mechanism exists** |

Five of eight are fixed. Each was found by hand, by a person or a review
noticing. None was found by a gate. That is the thing to change.

## Measured evidence (2026-08-13, live)

Source: `substrate_field_state`, 113,190 rows spanning 2026-08-11 → 2026-08-13.
**That is a 2.5-day window** (the corpus restarts at the disk-death of
2026-07-23), so distribution claims below are scoped to it and are not
long-run. Sample used: the most recent 20,000 ticks = 11.4h at a 2.04s median
tick.

### Commensurability: the merged `prediction_error` channel

`collect_field_channel_pressures()` merges every `PRESSURE_CHANNEL` by `max()`
across all sources. For `prediction_error` that is a max over 12 nodes:

| source | wins the merge | its own distinct values / 20,000 ticks |
|---|---|---|
| `node:substrate.codebase` | **54.2%** | **5** — effectively the constant 0.3357 |
| `node:substrate.execution` | 41.7% | 1,208 |
| `node:substrate.vision` | 4.1% | 4 (0.0 with 1.0 spikes) |
| `node:substrate.biometrics` | **0%** | 1,188 |
| `node:substrate.bus_synaptic` | **0%** | 341 |
| `atlas`, `circe`, `athena`, `prometheus`, `substrate.transport` | 0% | 1 each (permanent 0.0) |

Consequences:

- The channel has a **hard floor of 0.3357**, set by a near-constant from one
  node. It cannot read calm below that. Structurally the same defect as the
  0.27 floor, but produced by the *merge* rather than by a formula — so a
  formula-level review would never find it.
- Over the last 600 ticks (~20 min) it is **1 distinct value, exactly 1.0**.
- The two richest signals in the set (1,188 and 341 distinct) contribute
  **nothing, ever**.

`max()` across incommensurable sources does not select the most informative
one. It selects the highest-scaled one.

### Provenance: the discarded dict

`collect_field_channel_pressures()` already returns
`tuple[dict[str, float], dict[str, str]]` — values *and* a provenance dict
naming which source won each channel. `field_pressures()` discards it:

```python
channel_pressures, _provenance = collect_field_channel_pressures(field)
return map_channels_to_dimensions(channel_pressures)
```

The seam exists and is computed every tick. It is thrown away one line before
the consumer.

### Provenance: the feedback-loop trap

`orion/field/pressure.py:100-108` records, and a code review confirmed, that if
`services/orion-biometrics` goes quiet, decay drives every remaining input
toward 0, `resource_pressure` reads calm, and because
`config/feedback/feedback_policy.v1.yaml` lists `resource_pressure: decrease`
under `positive_delta_channels` (**verified present on main today**), the
in-flight action is credited with a positive outcome for a sensor outage.

Tracked to PR #1554, which **merged as docs only**. The guard was never built.

Checked for it directly: **no geometric-decay runs in the last 20,000 ticks** —
every producer stayed live across the retained window. So this is a **latent
trap, not a currently-firing bug**. It cannot be caught by monitoring, because
by the time it fires it has already written a false reward.

### Window: nothing expresses regime

Prototyped level / dispersion / drift / saturation over a declared window
against real ticks. It separates cleanly (e.g. `memory_pressure` level 0.813
with dispersion 0.001 — loaded and steady; `gpu_pressure` dispersion 0.258 and
touching both 0.0 and 1.0 — volatile and rail-saturated). No such reading
exists anywhere in the system today; consumers get one scalar.

Note the units trap this exposes: a "600-tick window" is 20 minutes at the
current 2.04s cadence, and nothing anywhere writes that down. **Windows get
declared in seconds.**

## Treatment by signal kind

Different kinds get different mechanisms. That is the point, not a compromise.

| surface | URNs | what it gets | why |
|---|---|---|---|
| `field_channel` | 38 | full treatment: provenance + regime + commensurability | real per-tick history in `substrate_field_state`; feeds the rankers |
| `inner_state` (substrate-runtime) | 37 | provenance + regime | per-signal tables exist; this is what feeds the autonomy rankers, so it is where mixing does the most damage |
| `organ_signal` | 252 | **definition-change alert only** | no persistence (in-memory window only; `substrate_organ_emissions` has 1.65M rows but one `organ_id`, and it is not in `ORGAN_REGISTRY`). Persisting it is a producer change, out of scope. Alert when someone adds to it. |
| `bus_channel` | 261 | **definition-change alert only** | bus synaptic over `orion-bus-mirror` actuals already owns traffic, including multi-hop. Alert when someone edits the defs. |

## Roadmap

Ordered. Each rung is independently shippable and independently useful.

### R1 — provenance survives the merge

Thread the already-computed provenance dict through `field_pressures()` to the
consumer. No new computation, no new schema concept, no behavior change.

*Acceptance:* for any dimension, name the source that won each contributing
channel this tick, from real stored state.

### R2 — regime readout over declared windows

Level, dispersion, drift, saturation as **separate** readings per channel,
windows declared in seconds. Surface on the lineage card and in `--json`.
Reuse `orion/bus/ewma.py::compute_ewma_update` and
`classify_channel_series()`; check `orion/substrate/prediction_error.py`,
`orion/metacog/trend_reducer.py`, and the phi autoencoder v2 running on field
signals before writing any new statistic — several already exist and this must
not add a sixth.

*Acceptance:* "near max but steady" and "volatile" produce different readouts
for two real channels. A saturated channel reports saturation rather than a
level.

### R3 — commensurability detector

Flag any merge where one source wins >50% of ticks while contributing fewer
than N distinct values, and any consumer combining channels whose declared
window semantics differ.

*Acceptance:* fires on `substrate.codebase` in the `prediction_error` merge as
it exists today, and would have fired on `thermal_pressure` before the manual
catch.

### R4 — definition-change alert

Diff-triggered notification when an agent edits bus channel defs, organ defs,
or topology channel maps. Answers "tell me when someone starts messing in
there" without a verdict column. Subsumes what the first draft called slice C,
which was a narrow static assert on one config file and is already green.

*Acceptance:* editing a channel def in a PR surfaces the change to Juniper.

### R5 — the guard (proposal mode, not this roadmap)

A staleness guard on the dimension so a decayed-to-calm reading cannot be
credited as a positive outcome. This changes a learning loop and needs its own
proposal with rollback, per CLAUDE.md §0A. **Detection first (R1-R3), so the
guard is designed against measured behavior rather than a hypothesis.**

## Non-goals

- One classifier across all four surfaces.
- Any declared or persisted verdict column. Verdicts stay computed.
- Fixing any metric this finds. Detection is its own patch; each fix is another.
- Inferring theory anchors from data. A fabricated rest-point is worse than none.
- Changing the merge, the feedback policy, or any ranker in R1-R4.

## Open decisions for Juniper

1. **Order.** R1 → R2 → R3 as written, or R3 first, given it has a measured
   open instance and R1/R2 are infrastructure for it?
2. **R5 timing.** Design the guard now in parallel, or hold until R1-R3 have
   produced real numbers on how often the trap is approached?

## Risk note — tooling, not code

`rg` output in this environment was caught on 2026-08-13 silently replacing the
searched-for identifier with a short token on large result sets, including
inside file paths (`channel_map` → `n`; `config/field/biometrics_lattice.yaml` →
`config/field/n.yaml`, a path that does not exist). Small result sets are
unaffected, so it is intermittent. Suspected source is RTK's ripgrep filter.

This matters to this roadmap specifically: every rung depends on reading
identifiers out of the repo accurately, and this corruption fabricates
plausible ones. Verify symbol and path spellings with a direct file read before
relying on grep output. A gate already exists for FCC sessions at
`~/.claude/hooks/rtk-fcc-gate.sh`; the same exclusion likely needs to cover
`rg` for this repo.
