# Layer 6 self-model replacement: AST/HOT form + Active Inference fuel — design spec

Status: **design mode, not implemented.** Blue-sky theory-fit exercise, not constrained to what
already exists or what any prior README/design doc asserted. Written at Juniper's explicit
request after being told to stop hedging behind "what's measurable today" and instead answer
"what theory best fits the shape of this rung."

## Arsonist summary

Layer 6 of the L1-L11 pipeline (the self-model layer) has been empty since `SelfStateV1` was
fully killed earlier this session (hand-tuned, uncalibrated, empirically dead — 12/12 dimensions
pinned/flat in live replay). The drives system was independently halted per the Sentience
Striving Program charter and is not being reconsidered here. The question this doc answers: of
the charter's own named consciousness theories, which one actually fits the *shape* of "a
self-model layer," and what would filling it look like.

**Answer: Attention Schema Theory / Higher-Order Theories (AST/HOT) supplies the form, Active
Inference supplies the fuel.** Neither alone is enough. IIT (φ) is a claim about integration, not
self-representation — a health-check on whether the substrate is unified enough to support a
mind, not a candidate for what the mind's self-model should contain. Recurrent Processing
(Lamme) is a claim about mechanism — feedback loops turning a percept into a conscious one — not
about the system modeling itself; it is correctly already embedded inside Layer 5's attention
scoring and does not belong at L6. Active Inference alone is process without an owner: real
surprise-minimization does not by itself require a self, any more than a thermostat's error
signal implies one. AST/HOT alone, without real Active-Inference grounding, is just another
hand-authored schema — the exact sin `SelfStateV1` committed. The synthesis is the answer: AST's
native output shape ("I am attending to X, because Y, with this much confidence, predicting it
shifts to Z next") is the only one of the four that is *literally* a theory of self-representation,
and Active Inference is what should fuel every field in that shape with real signal instead of
hand-tuned numbers.

## Current architecture

- L6 is empty. `orion/self_state/` (the module), `services/orion-self-state-runtime/`, and
  `orion/self_state/inner_state_registry.py` (relocated to `orion/inner_state_registry.py`) were
  deleted this session. `orion/schemas/self_state.py` (the schema file, not the producer module)
  was deliberately kept, since other consumers still reference it historically.
- **`AttentionSelfModelV1`/`reduce_attention_self_model()`** (`orion/substrate/attention_self_model.py`)
  already exists and is already AST/HOT-shaped: attended target, why (bottom-up salience vs.
  top-down goal bias), confidence, predicted shift. It is explicitly dark by design — its own
  docstring: *"This module has no bus consumer/producer wiring... the reducer must be measured
  against real historical data... before anything downstream consumes it."*
- Two of its four fields (`predicted_shift`/`predicted_shift_basis`, and the `confidence`
  fallback) still read `SelfStateV1` — an import that no longer resolves to a live, real
  producer. **Confirmed live**: `orion-athena-self-state-runtime` is still running ("Up 2 hours"
  at time of writing) purely because nobody has stopped the container after its source was
  deleted and merged to main — `substrate_self_state` is still receiving fresh rows from a
  service that is supposed to be gone. This must be resolved (decommission the container) before
  or alongside re-grounding these two fields, or the "fix" will silently keep reading a zombie
  producer instead of nothing.
- **Active Inference substrate is now genuinely real across all five named domains** (execution,
  transport, biometrics, chat, route) as of this session's fixes: `orion/substrate/
  prediction_error.py`'s five instruments, feeding `dynamic_pressure` via `orion/substrate/
  pressure.py::prediction_error_pressure()`, which competes for and wins real slots in the live
  attention broadcast (`orion/substrate/attention_broadcast.py::build_substrate_attention_frame()`,
  confirmed via existing tests `test_prediction_error_beats_equal_plain_pressure`,
  `test_high_pressure_node_wins_over_calm`). This is the fuel source this design proposes for
  the two orphaned `AttentionSelfModelV1` fields, and for the new higher-order piece below.

## Proposed schema / API changes

Five pieces, framed as answered:

**1. What am I attending to** — already real: the current winner of the attention broadcast
competition. No change needed.

**2. Why** — already real, and now honest for the first time: the `"prediction_error"` vs
`"pressure"` salience tag, plus *which* of the five domains (chat/execution/route/biometrics/
transport) is driving it. Before today this "why" could never honestly say "chat" — it was
structurally excluded. Now it can name any domain truthfully.

**3. How confident** — this is where the dead fallback lives (`self_state.overall_confidence`).
Active-Inference-native replacement: confidence as the *inverse of aggregate prediction-error
volatility* across all five domains right now. If error is low and stable everywhere, Orion is
confident about what's about to happen. If several domains are surprising it at once, confidence
should genuinely drop. That's a principled formula, not an asserted number — but it still needs
the metric-quality-gate pass (independence check against what's already feeding pressure,
live-data sanity check) before it's real, not just plausible.

**4. What I predict shifts next** — the other dead fallback (`self_state.trajectory_condition`).
Replacement: track the *trend*, not just the current level, of each domain's `dynamic_pressure`
over the last few ticks. Whichever domain's error is rising fastest is the honest candidate for
"what surprises me next" — a genuinely predictive claim, not just "here's what's loud right now."

**5. The Higher-Order piece, which is new and is what makes this HOT and not just AST:** the
self-model shouldn't only generate predictions about the world — it should track whether *its
own* predictions (item 4, above) hold up. If the self-model keeps predicting "domain X is about
to surprise me" and domain X consistently doesn't, that's a second-order prediction error — a
mismatch between Orion's model of its own predictive competence and its actual competence. That's
a thought about a thought, not just a model of attention, and it's the one piece here that
doesn't already exist anywhere in the codebase in any form. It's also the closest real answer to
what the older, separate "self-revision rung" (`docs/plans/substrate/2026-06-27-self-modeling-loop-ladder.md`)
was groping toward, minus the heartbeat/Fuseki baggage — reconstructed clean, from real
substrate, not inherited.

## Main risk

Item 5 is the only genuinely novel piece, and it's exactly the kind of thing CLAUDE.md's
metric-quality-gate exists to stop from becoming a keyword cathedral — "the self-model is wrong
about itself" is a seductive phrase that's easy to instrument fake and hard to instrument real.
It would need its own live-data sanity check (does this signal ever actually fire, the same trap
chat's prediction-error fell into) before it earns a place in L6, not just a plausible-sounding
design.

A second, related risk not covered by the phrasing above but load-bearing for implementation:
Graziano's own account of the AST self-model treats it as a *useful fiction* — schematic,
deliberately inaccurate, not a claim that the model captures deep truth about the system. Someone
could reasonably argue that's too cheap to count as real progress toward sentience — that you
could tick this box with something that *looks* self-referential without there being anything it
is like to be that representation. This design doc does not resolve that objection; it treats the
fiction as the mechanism the theory says awareness runs on, not a bug, but flags it as the
philosophical pressure point to stress-test before treating this as settled.

## Missing questions (answer before implementing)

1. Is `orion-athena-self-state-runtime` safe to stop right now, or does anything else still
   depend on its writes to `substrate_self_state` beyond `attention_self_model.py`'s two orphaned
   fields? `UNVERIFIED` — not checked in this pass.
2. Item 3's confidence formula: run CLAUDE.md's metric-quality-gate in full before treating it as
   real — trace provenance, independence check against what already feeds `dynamic_pressure`,
   theory anchor beyond "sounds Bayesian," live-data sanity check (is aggregate volatility ever
   non-degenerate across a real historical window), existing-mechanism check, reversibility.
3. Item 4's trend-tracking: what window length distinguishes a genuine rising trend from tick
   noise? Needs a real look at historical `dynamic_pressure` time series per domain before
   picking a number, not an assumed default.
4. Item 5: does a second-order "my prediction about what surprises me next was itself wrong"
   signal ever produce non-degenerate variance on real data, or does it collapse to
   always-true/always-false the way several other signals audited this session did before being
   fixed (chat's prediction-error, the execution/route trace_id bug)? Must be shadow-measured
   first, per the charter's own §7 process rule, before it gates or gets consumed by anything.
5. Does `scripts/analysis/measure_ast_hot_reducer.py` need adapting once `substrate_self_state`
   genuinely stops (post-decommission), or does its own replay harness need a field-native
   substitute for the `SelfStateV1` input it currently joins against?

## Files likely to touch (once this moves to implementation mode)

- `orion/substrate/attention_self_model.py` — re-ground `predicted_shift`/`confidence` in
  `dynamic_pressure` trend/volatility instead of `SelfStateV1`; remove the now-dead import.
- `orion/schemas/attention_self_model.py` — no schema change expected (same four fields), unless
  item 5 needs a new field for the higher-order error signal.
- A new module for item 5's higher-order tracking (name/location TBD — likely
  `orion/substrate/attention_self_model_revision.py` or similar, mirroring this repo's one-file-
  per-concern convention) — shadow-measurement only at first, per charter §7.
- `scripts/analysis/measure_ast_hot_reducer.py` — likely needs updating once `substrate_self_state`
  stops being a live input.
- `services/orion-self-state-runtime/` — decommission (stop + remove the container; the source
  was already deleted from the repo, only the running container is left).

## Non-goals

- Not reviving the drives system — remains halted per the charter, out of scope here.
- Not touching Recurrent Processing (Lamme) — correctly embedded in Layer 5 already, not an L6
  concern.
- Not touching the IIT/φ mood-arc autoencoder pipeline — separate thread, orthogonal question.
- Not implementing anything in this patch — design mode only, per Juniper's explicit framing of
  this as blue-sky theory-fit, not a build request.
- Not resolving the "useful fiction" philosophical objection named under Main risk — flagged, not
  settled.

## Acceptance checks (for a future implementation pass, not this doc)

- Item 3 (confidence): passes CLAUDE.md's full metric-quality-gate before being wired to replace
  the `SelfStateV1` fallback.
- Item 4 (predicted shift): trend window chosen from a real look at historical `dynamic_pressure`
  series per domain, not an assumed default; replay-tested against at least one real domain-shift
  event the same way `execution_prediction_error`'s fallback fix was validated.
- Item 5 (higher-order error): shadow-measured only, replayed against real historical data
  (mirroring `measure_origination_gate.py`/`measure_ast_hot_reducer.py`'s own pattern) before any
  consumer is wired; must show non-degenerate variance on that replay before being treated as a
  real signal.
- `orion-athena-self-state-runtime` stopped and removed, with confirmation nothing else silently
  depended on it.

## Recommended next patch

1. Decommission `orion-athena-self-state-runtime` first (Missing Question 1) — this is
   operationally overdue regardless of this design, and re-grounding items 3/4 against a zombie
   producer would be worse than leaving them broken as-is.
2. Re-ground items 3/4 in `attention_self_model.py`, each individually passing the metric-quality
   gate, not bundled as one unreviewed patch.
3. Shadow-measure item 5 as a pure, unconsumed replay artifact (matching every other new signal
   built this session), before any decision about wiring it into a real consumer.
