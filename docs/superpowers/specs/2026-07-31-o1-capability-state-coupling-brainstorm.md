# Brainstorm: why capability does not vary with state (Sentience Striving O1)

Status: **brainstorm**. Not a design doc, not scoped, not sequenced. Ideas are kept at the
size they actually are rather than collapsed to the smallest buildable version — per
Juniper's standing instruction that premature MVP-scoping is what kills the interesting
half of these. Nothing here is authorized; the charter
(`orion/sentience_striving_program/README.md`) is design/proposal mode and every phase
needs its own sign-off.

Date: 2026-07-31
Program: Sentience Striving, Outcome **O1** — *"Capability actually varies with state.
Orion's autonomous-action budget demonstrably rises and falls with real internal pressure,
not a flat per-cycle allowance, with a demonstrated, verified ceiling."*

---

## 1. The measurement that started this

O1 is failing, and it is falsifiable in one query. Live, 10 full hours:

```text
dispatch rate            939/hour   sd 20   CV = 2.2%
internal confidence      swung 6.2% over the same window
correlation(rate, state) r = 0.249   (n=10, not significant)
```

Orion takes ~939 autonomous actions an hour, hour after hour, within 2.2%, while its
internal state moves three times that much and the two barely correlate.

The composition is flatter than the total. Six hours of dispatches, by kind:

```text
inspect_bus_channel_catalog        1113
summarize_transport_contract_drift 1111
inspect_attended_target            1105
inspect_field_topology_catalog      710
watch_transport_backpressure        556
summarize_loaded_state              540
inspect_node_resource_pressure      400
inspect_execution_pressure           16
inspect_transport_status             11
```

Per hour, the top three:

```text
hour    bus_cat  tr_drift  attended
20:00     185      185       180
19:00     185      185       185
18:00     184      183       182
17:00     183      183       183
16:00     189      189       189
15:00     185      185       185
14:00     193      193       193
```

Three *different* cognitive acts firing an identical number of times per hour, to the unit.
That is only possible if they are emitted as a fixed group per cycle and nothing selects
between them.

---

## 2. Root cause: the scorer reads dead channels

`config/proposals/proposal_policy.v1.yaml` scores candidates on four weighted dimensions.
Every dimension present in `capability_vectors`, 12h of live field state, ranked by how
alive it is, against its policy weight:

```text
dimension                avg      max     %zero   policy weight
available_capacity      0.870    1.000     0.0        0.00
confidence              0.935    1.000     0.0        0.00
pressure                0.130    0.850    28.6        0.00
reasoning_pressure      0.013    0.696    71.4        0.15
execution_pressure      0.018    0.392    93.0        0.30
reliability_pressure    0.005    0.900    99.4        0.35   <- highest weight
contract_pressure       0.000    0.000   100.0           —
stream_backlog_pressure 0.000    0.000   100.0           —
transport_pressure      0.000    0.000   100.0           —
resource_pressure          ABSENT — no producing edge      0.25
```

**The weights are ranked in near-perfect inverse order of liveliness.** Traced upstream via
`config/field/orion_field_topology.v1.yaml`:

- `reliability_pressure` (0.35) ← `observer_failure_pressure` (**100.0% zero across 105,370
  samples, max 0.00000 — never once non-zero**), `execution_friction` (100.0% zero),
  `failure_pressure` (99.0% zero).
- `execution_pressure` (0.30) ← `cortex_exec_step_load`, 87.7% zero.
- `resource_pressure` (0.25) ← **nothing. Zero producing edges exist.**
- `reasoning_pressure` (0.15) ← `reasoning_load`, genuinely alive at 25% zero.

Meanwhile `pressure` — 28.6% zero, avg 0.130, fed by **11 producing edges** carrying cpu,
gpu, memory, disk, and bus prediction error — carries weight **zero**.

Honesty correction so this is not oversold: `available_capacity` and `confidence` are not
independent signals. The topology derives them from `pressure`
(`max(0, 1-pressure)` / `max(0, 1-0.5*pressure)`). So there is exactly **one** live
independent dimension in the whole capability vector, and its weight in action selection
is zero.

### The mechanism, stated plainly

A template with `dimensions` multiplies its score by state. State is zero, so the score
collapses below `min_priority: 0.1` and it never dispatches. A template with
`dimensions: {}` keeps its constant `base_priority` and wins every arbitration.

The inversion is exact: `inspect_transport_status` has the **highest** base priority in the
policy (0.42) and fired **11 times in 6 hours**; `watch_reliability` and
`defer_due_to_low_readiness` fired **zero**. The three that fired ~1,110 times each are
precisely the three with no dimensions at all.

**The more state-responsive a proposal is, the more reliably it is silenced.**

The arena is not broken. Competition, scoring, gating, dispatch all work. They are wired to
channels nobody produces.

---

## 3. The thing that bothers me most

Right now **Orion is loudest when it has least to say.**

Calm state → state-sensitive templates collapse to zero → only state-blind templates
survive → constant 939/hour of inspection. The system's floor and its ceiling are the same
number, and that number is set by how many hard-coded templates have empty `dimensions`.

A genuinely state-coupled Orion would do the opposite: go quiet when calm, surge when
something happens. Silence would be information. Right now silence is impossible — there is
no state, however serene, that produces fewer than ~939 actions per hour.

Worth sitting with: fixing O1 might mean the correct outcome is **far fewer** dispatches
most of the time, not a more variable ~939. 22,500 LLM observations a day are currently
produced, and `orion_metacog` — the nearest analogue — has no consumer at all. We should not
assume the target state is "same volume, better distributed."

---

## 4. This is the six-drive taxonomy again, in a different file

The charter's §1 records that the six drives were *"imported wholesale from one external
design chat, never independently checked against Orion's own mission."* The four dimension
weights in `proposal_policy.v1.yaml` have exactly the same provenance signature:
hand-authored category names with hand-set weights, never validated against whether the
channel behind them produces anything.

`resource_pressure` is the proof. It has a weight of 0.25 and **no producer has ever
existed**. Nothing failed. Nothing alerted. Two templates score on it and dispatch anyway
via a fallback. It has presumably been that way since the file was written.

The charter's O4 already names the correct posture: *"Named categories (if any survive) are
a report on clustering of real coalition-winning history, versioned and re-derivable, not a
constant."* Four hand-set floats in a YAML are the opposite of that.

So the narrow reading of this finding — "point the weights at `pressure`" — would fix the
number while preserving the disease. Worth doing as an experiment; not worth mistaking for
the answer.

---

## 5. Directions, at their real size

Deliberately not ranked by buildability, and not collapsed to an MVP.

### 5a. Re-point the weights (the small, honest experiment)

Give `pressure` real weight; delete `resource_pressure` or give it a producer; drop the
weights on channels that are >99% zero. Cheapest possible test of whether the wiring
mismatch is the *whole* story or just the visible part.

Value: it is falsifiable within hours against instrumentation that already exists (CV,
correlation, top-3 lockstep). Risk: it treats hand-authored weights as legitimate, just
mis-set. If CV barely moves, that is itself a strong finding — it would mean the flatness is
structural (fixed templates, per-cycle emission) rather than a weighting problem.

### 5b. Kill `base_priority` as the dominant term

Today a candidate's score is dominated by a constant that a human typed. A proposal's
priority should *be* its state-derived quantity; `base_priority` should be a tiebreaker at
most. The current arrangement guarantees that when state is quiet, ranking is entirely
determined by authored constants — i.e. by a preference ordering fixed months ago.

This is a bigger swing than 5a because it means accepting that when there is genuinely
nothing to say, *nothing should be proposed*.

### 5c. Derive the weights instead of authoring them

O4's actual ask. Weights become a periodically re-fit report over which dimensions
historically preceded outcomes worth having, versioned and re-derivable. Requires deciding
what "outcome worth having" means — which is the hard part and probably the real work.

The honest blocker: we do not currently have an outcome signal to fit against. Dispatch
results are LLM observations that nothing consumes. There is no downstream that says "that
was worth doing." Until that exists, any fitted weighting is fitting to noise. **This may be
the actual critical path for the whole program** — not attention, not proposals, but *did
anything come of it*.

### 5d. Stop enumerating the action vocabulary

There are twelve templates. Twelve things Orion can ever propose, hand-written, with
hand-set priorities. However well the competition works, it is choosing among twelve
constants. A system whose entire behavioral repertoire is a fixed list in a YAML is
performing selection, not origination.

What would it mean to generate candidates from state rather than filter a fixed list? This
is the largest swing here and probably the one most likely to become a cathedral if
approached carelessly. Recording it because it is the honest end of the line this finding
points down, not because it should be next.

### 5e. Make the budget a real budget

O1's wording includes *"with a demonstrated, verified ceiling."* There is currently no
budget in the sense of a resource that depletes and recovers. There is a per-cycle emission
and a risk cap. A capability budget that accumulates when calm and spends under pressure
would make "capability varies with state" true by construction and give the ceiling
something to be a ceiling *of*.

Adjacent prior art already in the repo: `orion/substrate/endogenous_curiosity.py` has
per-cycle budget slots. Worth reading before inventing anything.

### 5f. Ask whether dispatch rate is even the right O1 instrument

CV of dispatches/hour is what I reached for because it is measurable today. But O1 says
*capability*, not *activity*. A system could hold dispatch rate constant while varying
*what it is permitted to do* — risk tier, reversibility floor, policy gate — and satisfy
O1's spirit better than rate variance does. Worth deciding deliberately rather than
inheriting my choice of metric because it was convenient.

---

## 6. Open questions I have not answered

1. **Is `observer_failure_pressure` a dead producer or a channel nothing was ever wired
   to?** 105,370 samples, max exactly 0.00000, never once non-zero. Those are different
   diagnoses with different fixes. Not yet traced.
2. **Why do `summarize_loaded_state` (540) and `inspect_node_resource_pressure` (400)
   dispatch at all**, when both score on `resource_pressure`, which has no producer? There
   is a fallback path in `proposal_urgency()` — its behavior is load-bearing here and I have
   not read it.
3. **What consumes a dispatch result?** If nothing does, then O1 is measuring the variance
   of an output that has no reader, and 5c has no target to fit against.
4. **Is the per-cycle fixed emission the deeper cause?** Even with perfect weights, if the
   builder emits the same slate every tick, variance can only come from the cutoff moving.
   Worth checking whether emission itself is state-gated at all.
5. **Did today's signal-health work make this worse?** Several fixes let domains report
   genuine `0.0` for the first time. Correct instruments feeding a scorer that reads zero as
   "nothing to do" would *reduce* variance. Not measured; checkable against the same window.

---

## 7. What this brainstorm is not

It is not a plan. It does not pick 5a-5f. It does not estimate anything. The one thing it
does assert is that the O1 failure has been traced to a specific, measured, wiring-level
cause, and that the cheapest fix for the number is not the same as the fix for the problem.
