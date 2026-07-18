# Concept reinforcement-on-recall

**Date:** 2026-07-18
**Status:** Implemented in this patch
**Scope:** `concept_region.py`'s live turn-time collector only. Not the crystallization
system (`orion/memory/crystallization/`), which already has its own, separately-shipped
`recall_boost()`/`decay()` wiring — this document is about giving `ConceptNodeV1`
(the concept-atlas substrate graph) the same kind of signal.

## Arsonist summary

PR #1166 fixed concept-node decay from being 100%-structurally-inert (every concept born
at `activation=0.0`, no half-life, so Hub's live decay scheduler had nothing to act on) to
actually decaying every concept from a real seeded value. That patch explicitly deferred
one piece: decay only ever moves activation *down*. Nothing moves it back up when a
concept is still genuinely relevant — a concept that's discussed constantly decays on
exactly the same curve as one nobody has mentioned in months. Golden concepts
(Orion/Juniper/the relationship) got a real starting salience (1.0) in the same follow-up,
but even they just decay monotonically from there with no way to reflect "this is still
alive and load-bearing."

The natural, already-existing signal for "this concept is still relevant": it just got
surfaced in a real chat turn. `services/orion-recall/app/collectors/concept_region.py`'s
`fetch_concept_region_fragment()` already runs on the live chat-turn hot path
(`app/worker.py:1606-1625`) and already knows exactly which concept nodes matched the
turn's text via cheap label-substring matching. That match event is the reinforcement
signal — it just wasn't wired to write anything back.

## Design questions resolved in conversation, not assumed

**Sync inline write vs. async via a bus event.** Considered and rejected async. Falkor
writes measured sub-millisecond live (`GRAPH.QUERY` internal execution time ~0.66ms for a
single-node lookup during PR #1166's live verification); this codebase's bus channels are
ephemeral pub/sub with no replay guarantee, so routing reinforcement through a bus event
consumed by a separate worker doesn't buy real safety over a direct write — it just adds a
dependency that can silently be down, for no reduction in worst-case data loss. A direct
write, wrapped in the same never-raise/degrade convention this collector already uses on
its read side, has an equivalent failure mode (this turn's reinforcement doesn't happen)
without the extra moving part.

**Where the write happens.** Not inside `fetch_concept_region_fragment()` itself — that
function's own docstring already promises "never persists anything," used by existing
callers/tests as a hard contract. Breaking that quietly would be exactly the kind of
scope-creep this repo's conventions warn against. Instead: a new, separately-testable
function `reinforce_matched_concepts()`, and a thin composing wrapper
`fetch_concept_region_fragment_and_reinforce()` that calls the existing read function
unchanged, then reinforces whatever it matched. The live call site in `worker.py` swaps to
the wrapper; nothing about the original function's contract changes for any other caller.

**Boost math.** Mirrors `orion/memory/crystallization/dynamics.py::recall_boost()` exactly:
`activation = current + (1 - current) * boost`, `boost = 0.08`. Asymptotic toward 1.0,
diminishing returns as a concept approaches ceiling — reusing an already-validated
constant and shape rather than inventing a new one for a structurally identical problem
(a weaker "this was retrieved" signal, distinct from a stronger "this recurred
independently" signal that a real Reinforce-on-formation event would represent — no such
formation-level reinforcement event exists for concept nodes today, so only the recall-side
boost is being wired here).

**What does NOT move.** Only `signals.activation.activation`. Not `recency_score`
(deliberately left to be temporally derived at read/tick time, matching the same
restraint `orion/substrate/adapters/_common.py::make_activation()` already applies — see
its docstring). Not `temporal.observed_at` (carries specific "last touched" semantics read
by decay math and pressure propagation elsewhere in `orion/substrate/dynamics.py`; PR #1131
already established that mutating it inside a decay-adjacent write path is a correctness
hazard, and this patch treats that constraint as a "no" for reinforcement too, conservatively,
rather than opening a second design question in the same patch). Not `confidence` or
`salience` — those describe "how true/prevalent is this," not "how often has this come up
in conversation," and conflating them would reproduce the exact conformity-bias failure
mode the crystallization system's own reinforcement spec
(`docs/superpowers/specs/2026-07-13-memory-recall-reinforcement-decay-wiring-spec.md`)
explicitly named as a hard invariant for its own `recall_boost()`.

## Non-goals

- Reinforcement on concept *formation*/recurrence (the stronger signal `reinforce()` maps to
  in the crystallization system) — no equivalent formation-dedup event exists for concept
  nodes today; out of scope here.
- Retirement/dormancy surfacing from decay — already separately handled by
  `SubstrateDynamicsEngine`'s dormancy transitions (currently gated off,
  `SUBSTRATE_DYNAMICS_TICK_ENABLED=false`), not touched by this patch.
- Any change to `fetch_concept_region_fragment()`'s existing signature, return shape, or
  "never persists" contract.
- Drive/goal-state-conditioned reinforcement weighting — a first-time coupling between the
  motivational subsystem and this signal would need its own proposal-mode pass per
  AGENTS.md, same reasoning the crystallization spec used to defer the equivalent idea
  there.

## Correction found in review: don't call `store.snapshot()` on this path

The first draft resolved each matched node's current activation and identity_key via
`store.snapshot()`, mirroring the exact pattern Hub's `decay_concept_activations()` already
uses. Code review caught that this undermines the sync-write latency rationale above:
`FalkorSubstrateStore.snapshot()` re-hydrates from Falkor with a currently-unbounded query
whenever the write generation has moved since the last call, and *every reinforcement write
this function makes bumps that generation* -- so on a busy conversation, each reinforcing
turn would force the next one to pay for a full-graph rehydrate, not the sub-millisecond
single-node cost the design above assumes. Fine for Hub's 120s decay cadence; not fine on a
live per-turn hot path.

Fixed by adding `get_identity_key_by_node_id()` to the `SubstrateGraphStore` protocol (a
new method mirroring the existing `get_node_id_by_identity()`, implemented across
`InMemorySubstrateGraphStore`, `FalkorSubstrateStore`, `GraphDBSubstrateStore`, and
`RoutedSubstrateGraphStore`) and switching `reinforce_matched_concepts()` to
`store.get_node_by_id()` + `store.get_identity_key_by_node_id()` -- both read straight from
the in-process cache with no refresh-triggering cost, confirmed by reading each backend's
actual implementation, not assumed. A node with no known identity_key is skipped rather
than reinforced with a falsy identity_key, since Falkor's codec writes `identity_key or ""`
unconditionally on every upsert -- passing a missing one would durably clobber the node's
real identity, not leave it alone.

## Acceptance checks

1. A concept matched in a live turn has a measurably higher `activation` immediately after,
   asymptotic toward 1.0 — not a flat overwrite, not unbounded growth past 1.0.
2. Two concepts, one recalled repeatedly across distinct turns and one never recalled,
   diverge over time — recall visibly counteracts decay instead of just being immediately
   erased by the next decay tick.
3. `confidence` and `salience` are provably unchanged by any number of reinforcement calls.
4. A failed reinforcement write (store error, missing node) never raises out of the
   turn-assembly path — matches `fetch_concept_region_fragment()`'s own existing
   degrade-never-raise convention.
5. `fetch_concept_region_fragment()` itself (the original function) is untouched: same
   signature, same "never persists" behavior, existing tests pass with zero modification.
