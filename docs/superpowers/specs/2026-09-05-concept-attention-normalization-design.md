# Concept attention: giving Orion more than six things to attend to

**Status:** design only. Nothing here is built. Parked deliberately.
**Date:** 2026-09-05
**Follows:** PR #2097 (record why no override), #2101 (make override possible),
#2106 (make the absence readable).

## Arsonist summary

Orion's attention can only ever see **six things**.

The substrate graph holds 1,796 nodes. The function that decides what is worth
attending to (`_node_salience`, `orion/substrate/attention_broadcast.py`) reads
exactly one field: `metadata["dynamic_pressure"]`. Only ten `node:*` domain
nodes ever carry it, and only three carry it non-zero at any moment. The other
1,786 score exactly `0.0` and are structurally invisible.

Measured live after #2106 deployed: **80% of ticks have nothing competing at
all** (`no_open_loops`), and Orion's goals have **never once lost** a
competition (`bias_did_not_flip_winner` = 0). The machinery works. The room is
empty.

This is the third time this repo has arrived at "the gap is competition" —
2026-08-30, 2026-09-04, 2026-09-05 — and the first time with an instrument
proving it. The next real move is supply, not machinery.

## Current architecture

```
substrate graph (1,796 nodes)
  |
  +-- 10  node:substrate.*        dynamic_pressure  -> visible to attention
  +-- 1,786 sub-*                 dynamic_pressure absent or "0.0" -> invisible
        |
        +-- 816 sub-entity-topicfoundry     signals.salience == 0.0 (hardcoded,
        |                                    topic_foundry.py:506/526/550)
        +-- 786 sub-evidence-topicfoundry   signals.salience 0.027-0.457, 93 distinct
        +-- 180 sub-concept-topicfoundry    signals.salience 0.027-0.457, 79 distinct
        +--   4 sub-concept-seed            signals.salience == 1.0

_node_salience(metadata) -> reads metadata["dynamic_pressure"] ONLY.
substrate_pressure_signals() already reads node.signals.confidence two lines
below -- the signals object is in hand; the salience field on it is ignored.

gate: ORION_ATTENTION_BROADCAST_MIN_SALIENCE = 0.2
```

## The thing that looks like the answer, and is not

`signals.salience` is real and computed, not a label:

```python
# orion/substrate/adapters/topic_foundry.py:348
salience = min(1.0, max(0.0, info["count"] / total_docs))
```

That is a topic's **share of the corpus** -- how *common* it is. Attention
competes on `prediction_error` -- how *surprising* something is. Wiring share
into the same competition puts "45% of my documents are about this" on one
scale with "this domain is behaving unexpectedly," and the bigger number wins.
The most-discussed topic would permanently outrank a live anomaly.

Rejected. Same category of error as borrowing a calibrated constant across
domains: the number is fine, the quantity is wrong.

## The actual analogue

`topic_foundry_drift.topic_shares` stores `{current, baseline}` per topic, per
window (daily / weekly / monthly, 147 rows today). The per-topic delta
`|current - baseline|` is **a topic's departure from its own expectation** --
structurally the same quantity as prediction error, for concepts instead of
domains.

Measured, both families, live:

| | n | median | p90 | max |
|---|---|---|---|---|
| domain `prediction_error` (7d) | 96,650 | 0.0052 | **0.2055** | 1.0000 |
| topic drift, daily | 1,404 | 0.0307 | **0.0896** | 0.7241 |
| topic drift, weekly | 847 | 0.0259 | 0.0731 | 0.6006 |
| topic drift, monthly | 767 | 0.0167 | 0.0323 | 0.3494 |

Both are bounded in [0,1], both heavily right-skewed, both have a near-zero
median. That last property matters most: **both can return to a genuine rest
state.** Neither has the structural floor that made `bus_synaptic_prediction_error`
useless (`mean(|z|)` has expected value `sqrt(2/pi)` for a calm population, never 0).

## Normalizing the two scales

The question this doc was written to answer: they are different quantities on
different ranges. How do you compare them without fudging?

**Option A -- rank / percentile within family.** Map each node to its rank
inside its own family. Scale-free and robust. **Rejected:** a rank always has a
maximum. A completely calm concept population still fields a 1.0 candidate, so
attention could never report "nothing is going on with concepts." That is the
absence-cannot-be-detected trap this repo keeps hitting.

**Option B -- z-score against the current population.** `(x - mean) / std`.
**Rejected as the primary:** a family with tiny variance turns a trivial
absolute change into a large z. Needs a variance floor anyway, which is Option C
with extra steps.

**Option C -- normalize each family by its own rolling p90 (recommended).**

```
score = min(1.0, raw / max(p90_family, floor_family))
```

- **Why p90, not max or median.** Max is one outlier and unstable. Median is
  dominated by the quiet mass -- both families sit near zero most of the time,
  so aligning medians amplifies noise into signal. p90 is where "this is
  notable" lives, which is exactly what a gate at 0.2 is trying to catch.
  Aligning at p90 means *an event notable for a concept scores like an event
  notable for a domain* -- which is the actual claim we want to make.
- **`floor_family` is the hand-authored part**, and it is what stops a quiet
  family from inflating its own noise as its p90 collapses. It encodes
  "a change smaller than this is not meaningful for this kind of thing."
  Starting values from the table above: `0.05` for domains, `0.03` for topic
  drift. These are measurements with a date on them, not borrowed constants,
  and they must be re-measured, not inherited.
- **Rest state is preserved.** If nothing is happening, raw is near zero for
  both families and every score is near zero. No floor, no phantom winner.

**Option D -- one hand-authored gain constant.** `p90_domain / p90_drift =
0.2055 / 0.0896 = 2.29`, so multiply daily drift by ~2.3. Simplest possible
thing, and defensible *today*. Not recommended as the design, because a
hardcoded 2.3 goes stale silently and nothing would notice -- exactly the
failure mode of the `0.55` floor in `scoring.py` that made voluntary override
impossible for the life of the feature. Option C is the same idea with the
constant made observable and re-derivable.

## Missing questions

1. **Cadence mismatch.** Drift is computed daily/weekly/monthly; attention ticks
   every 30s. A concept's salience would move on a clock ~2,880x slower than a
   domain's. Does a concept that drifted this morning deserve to keep competing
   at 3am? A decay toward zero between drift computations is probably required,
   and `SubstrateActivationV1` already carries a decay mechanism -- but its
   observed values are useless as-is (all 1,786 nodes clear 0.2, 90% sit in a
   0.08-wide band around 0.45).
2. **Does adding ~966 candidates help or just add noise?** `max_open` is 5, so
   the field stays small; but the *selection* changes completely. Unknown
   whether a concept winning attention produces anything useful downstream —
   the only consumer today polls `prediction_error_confidence`.
3. **Which window?** Daily drift has the widest range (p90 0.0896) and the most
   rows; monthly is nearly flat (p90 0.0323). Probably daily, but that is a
   guess, not a measurement.
4. **The 816 hardcoded zeros.** `topic_foundry.py:506/526/550` set
   `salience=0.0` on entity/mention nodes. Deliberate or vestigial? Unfiled
   either way, and it is 45% of the graph.

## Proposed schema / API changes

None to any wire schema, bus channel, or registry entry. Everything proposed is
internal to `orion/substrate/attention*`:

- `_node_salience` gains a second source, behind a flag, returning the same
  `(float, kind)` contract it returns today.
- a small normalizer holding per-family p90 and floor, recomputed on a schedule
  and **persisted so it is inspectable** -- an invisible normalizer is the
  hardcoded constant again.
- `ORION_ATTENTION_CONCEPT_SALIENCE_ENABLED` (default false).

## Files likely to touch

- `orion/substrate/attention_broadcast.py` -- `_node_salience`, `substrate_pressure_signals`
- `orion/substrate/adapters/topic_foundry.py` -- read-only, for the drift join
- `services/orion-substrate-runtime/app/settings.py` + `.env_example` + `.env`
- `orion/substrate/tests/` -- degeneracy tests per family (see below)

## Non-goals

- **Not** wiring `signals.salience` (share) as salience. Wrong quantity; that is
  the whole point of this document.
- **Not** lowering `ORION_ATTENTION_BROADCAST_MIN_SALIENCE`. That makes the same
  six nodes compete more often; it does not enlarge what Orion can attend to.
  Worth doing separately as a cheap measured experiment, not as this.
- **Not** a taxonomy of node kinds, a plugin registry for salience sources, or a
  base class. Two sources, one function, one flag.
- **Not** fixing the 816 hardcoded zeros here. Flagged, filed, separate.

## Acceptance checks

1. `no_open_loops` drops from its measured 80% baseline. Stated before the
   change, compared after, on the same instrument.
2. Concept-sourced loops and domain-sourced loops both appear as winners in
   `broadcast_attended_node_ids` -- neither family shuts the other out.
3. Both families can read ~0 simultaneously. Pull real data during a quiet hour
   and confirm the field CAN be empty. A source that can never rest is a floor
   in disguise.
4. `top_down_bias_max` still varies, and `voluntary_override_absent_reason`
   still distributes across causes rather than collapsing to one value.
5. Degeneracy test per family, run against live data, not fixtures: distinct
   value count, median, p90, and fraction clearing the gate.
6. **Control arm:** the same measurement with the flag off, over the same
   window. Without it, a change in `no_open_loops` cannot be attributed.

## Recommended next patch

**Not this one.** Two cheaper things first, in order:

1. **Lower the gate and measure.** `MIN_SALIENCE 0.2 -> 0.05`, one env var,
   instantly reversible. Predicted from 7 days of data: ticks with >=2
   competitors go from 6.6% to 36.1%. This answers a question this design needs
   the answer to anyway -- *does more competition alone produce more deliberate
   attention?* -- for an hour of work instead of a week. If override rate does
   not move, this design's premise is weaker than it looks.
2. **File the 816 hardcoded zeros** as a real defect and decide if they are
   vestigial, before building anything that reads that field's neighbours.

Then this.
