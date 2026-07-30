# proposal_priority() theory-anchor design — findings, not a patch

2026-07-30. Follow-up to the proposals scoping session that produced
PR #1493 (dead-dimension template fix). That patch closed the honest-cleanup
half of the redesign; this doc is the "live data as follow-up" investigation
into the harder half — what should replace `proposal_priority()`'s
hand-weighted `0.4 * match_score + 0.2 * urgency + 0.1 * confidence` blend
(`orion/proposals/scoring.py:235-244`, untouched by PR #1493).

## Arsonist summary

Went looking for real dispatched-proposal outcome data to calibrate
`proposal_priority()`'s weights against. Found two things that matter more
than the weights:

1. **The only outcome signal that exists cannot be used for calibration.**
   `action_outcomes.success` (written by
   `services/orion-execution-dispatch-runtime/app/worker.py:588`) is
   `raw_len > 0` — "did the LLM produce non-empty text," not "was this the
   right proposal to have surfaced." Live query, 2026-07-30: success rate is
   >99% across every template that has ever dispatched, and two templates
   (`inspect_transport_status`, `watch_reliability`) have **never once
   failed** in the entire window. There is no variance in this label to
   explain — it cannot serve as a supervised signal for whether priority
   correctly ranked proposals. `action_outcomes.surprise` is borrowed from
   `bus_synaptic_prediction_error()`, a general system-wide bus metric, not
   anything specific to the dispatched proposal.

2. **The current formula produces real, measured, near-total single-template
   capture — a bigger problem than the weight constants.** Real dispatch
   data (Postgres `conjourney.action_outcomes`, `action_id LIKE
   'dispatch:proposal:%'`, pulled live via psql, 7-day window 2026-07-23
   through 2026-07-30):

   | template_key | n dispatched | success | first seen | last seen |
   |---|---|---|---|---|
   | `inspect_node_resource_pressure` | 34,864 | 34,832 | 2026-07-26 | 2026-07-30 |
   | `inspect_execution_pressure` | 630 | 621 | 2026-07-23 | 2026-07-30 |
   | `inspect_transport_status` | 88 | 86 | 2026-07-23 | 2026-07-29 |
   | `watch_reliability` | 27 | 27 | 2026-07-29 | 2026-07-30 |
   | (8 other live templates) | 0 | — | never | never |

   `inspect_node_resource_pressure` captured **98% of every real dispatch
   slot** in the window. `max_dispatches_per_tick: 1`
   (`config/execution_dispatch/execution_dispatch_policy.v1.yaml:53`) means
   `proposal_priority()` is not a soft weighting — it is a hard winner-take-
   all ranking function across 12 heterogeneous templates for one slot per
   tick, and one template has been winning almost every tick since a sharp
   regime change on 2026-07-28 (daily count jumped 139 → 15,072). 8 of 12
   live templates — including every template PR #1493 just fixed — have
   never been dispatched at all in this window.

Net: before touching the 0.4/0.2/0.1 weights, there's a real question of
whether reweighting an additive blend is even the right lever, given the
observed problem is single-template near-starvation under a hard top-1
dispatch policy, not "the ranking is close but slightly mis-weighted."

## Current architecture

- `orion/proposals/scoring.py::proposal_priority()` — `clamp01(base_priority
  + 0.4 * match_score + 0.2 * urgency + 0.1 * confidence)`. All three
  weights hand-picked at original implementation, no theory anchor, no
  citation — unlike `dimension_confidence()` (precision-weighted, Feldman &
  Friston, real EWMA calibration, PR from 2026-07-28/29) or Candidate A's
  attention salience (same theory anchor, PR #1484), this function has never
  been through the metric-quality-gate.
- Consumer: `orion/execution_dispatch/builder.py` sorts/selects candidates
  by `priority_score`; `execution_dispatch_policy.v1.yaml`'s
  `max_dispatches_per_tick: 1` means only the single highest-priority
  *approved* candidate (`allowed_policy_decisions: [approved_read_only]`)
  actually dispatches each tick.
- Outcome tracking: `services/orion-execution-dispatch-runtime/app/worker.py`
  writes one `action_outcomes` row per real dispatch. `success = raw_len >
  0` (line 588). `surprise` = `self._store.latest_bus_synaptic_prediction_error()`
  (line 628), falling back to `0.0` — a system-wide bus metric, not a
  per-proposal quality signal (see that line's own comment: "surprise was a
  hardcoded 0.0 placeholder... honest").
- `action_id` embeds the template key
  (`dispatch:proposal:<template_key>:<field_tick_id>:...`), which is how
  the table above was built — `split_part(action_id, ':', 3)`.
- No Postgres table stores `FieldStateV1`/`field_pressures()` history —
  field state is bus/Falkor-native, not SQL-persisted — so the *cause* of
  the 07-28 regime change (real resource-pressure elevation vs. a scoring
  artifact that makes `resource_pressure` structurally easier to saturate
  than `execution_pressure`) could not be traced from Postgres alone in
  this pass.

## Missing questions

1. **Is `success = raw_len > 0` ever going to be a real decision-quality
   signal, or does calibrating `proposal_priority()` against real outcomes
   require genuinely new instrumentation** (e.g., did an operator act on,
   dismiss, or correct the generated inspect output; did the named target
   turn out to matter later)? If the latter, this is new-metric work under
   the full metric-quality-gate, not a scoring-formula edit — and it has to
   happen before any data-driven reweighting is defensible.
2. **Was the 07-28 regime change real signal or a scoring artifact?**
   `resource_pressure` going from 139/day to 15,072/day of captured dispatch
   slots is either (a) node:atlas's resource pressure was genuinely elevated
   for ~2.5 straight days relative to every other template's pressure
   dimension, or (b) `dimension_score`/`dimension_confidence` saturate near
   1.0 more easily for `resource_pressure` than for `execution_pressure`
   given their different `DIMENSION_PRECISION_MIN_VARIANCE` floors
   (`resource_pressure: 5e-5` vs `execution_pressure: 1e-4` —
   `resource_pressure`'s floor is half the size, meaning smaller real
   deviations register as larger z-scores / higher confidence for it).
   Not traced here — would need real `field_pressures()` time series.
3. **Is `max_dispatches_per_tick: 1` even the right dispatch shape** for a
   priority function meant to rank 12 heterogeneous proposal kinds? A softer
   selection policy (round-robin among templates above `min_priority`, or an
   explicit per-template dispatch floor/cooldown) might matter more to the
   observed problem than reweighting the 0.4/0.2/0.1 blend, since the
   blend's output only ever matters insofar as it changes who wins the one
   slot.
4. Still open from the first scoping doc, untouched by PR #1493 or this doc:
   the `operator_review` dead-end gate, `proposal_risk()`'s own hand-picked
   bumps, and the three-way "proposal" naming collision
   (`orion/proposals/` vs. context-exec's `ProposalLedgerRecordV1` ledger vs.
   Hub's `routing_threshold_patch`).

## Proposed schema / API changes

None recommended yet — the finding here is that the data needed to justify
a *specific* replacement formula doesn't exist. Two candidate directions,
neither implemented, offered for discussion:

- **(a) Precision-weighted, not additive.** Replace the independent
  `0.1 * confidence` term with confidence as a precision/trust gate on
  `match_score` (`base_priority + match_score * confidence`, `urgency` as a
  separate floor/gate rather than a third additive weight) — the same
  Feldman & Friston precision-weighting theory already shipped and reviewed
  for Candidate A and for `dimension_confidence()` itself, not a new
  invention. Would need real testing against the same live-data discipline
  used for those two prior patches before shipping.
- **(b) Diversity-aware dispatch, independent of the priority formula.**
  Address question 3 directly — e.g. a per-template dispatch floor or
  cooldown in `execution_dispatch_policy.v1.yaml` — since the observed
  98%-capture problem may be substantially a dispatch-selection question,
  separable from whatever `priority()` becomes.

These are not mutually exclusive and neither is scoped as a patch yet.

## Files likely to touch (once a direction is chosen — not started)

- `orion/proposals/scoring.py` (`proposal_priority`)
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml` (if
  direction (b))
- `tests/test_proposal_scoring.py`
- `tests/test_execution_dispatch_*.py`
- This doc, updated with whichever direction is chosen and why.

## Non-goals

- Not touching `proposal_risk()`'s hand-picked bumps here (separate,
  smaller, already-named question).
- Not building new outcome instrumentation in this doc.
- Not resolving the `operator_review` dead-end or the three-way naming
  collision here.
- Not picking direction (a) vs (b) here — that's the next real decision,
  and it's a fork with genuine trade-offs, not an obvious call.

## Acceptance checks

N/A — no patch proposed yet. This doc exists to change what gets proposed
next, per this repo's design-mode contract (a concrete design artifact with
non-goals and acceptance checks, even when the acceptance check is "the
open question is now traceable to real code and real data instead of
assumed").

## Recommended next patch

Not a `proposal_priority()` edit yet. Recommend resolving question 2 first
(real signal vs. scoring artifact) since it's cheap relative to a full
reweighting effort and determines whether direction (a) alone would even
address the observed 98%-capture problem, or whether (b) is required
regardless of what `priority()` becomes.
