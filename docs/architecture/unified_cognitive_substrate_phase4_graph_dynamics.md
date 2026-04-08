# Unified Cognitive Substrate — Phase 4: Graph Dynamics and Pressure Propagation

## Scope

Phase 4 adds a deterministic, bounded dynamics layer over the already materialized substrate graph. Structural ontology and materialization remain canonical; this phase only introduces dynamic state updates.

## Dynamics model

### Activation (dynamic)

`SubstrateDynamicsEngine` computes per-node activation each tick from deterministic seed factors:

- recency score from `temporal.observed_at`
- structural salience (`signals.salience`)
- dynamic pressure (`metadata.dynamic_pressure`)
- unresolved contradiction boost
- node-type specific boost for tensions and state snapshots

Activation propagation is bounded by:

- allowed predicates (`supports`, `associated_with`, `activates`, `seeks`, etc.)
- attenuation per hop
- max hop depth
- minimum propagation threshold

This avoids runaway spread and keeps updates inspectable.

### Salience relationship

Structural salience (`signals.salience`) remains canonical/static for Phase 4. Dynamic activation is computed each tick and can rise/fall independently.

### Decay, dormancy, revival

After propagation, activation is decayed using node half-life settings (`signals.activation.decay_half_life_seconds`) and floor.

Dormancy rules:

- become dormant when activation and recency are both under threshold
- revive when activation crosses revival threshold
- transitions are stored in node metadata (`dormant`, `dormancy_updated_at`) and returned as typed transition records

### Contradiction amplification

Unresolved contradiction nodes amplify pressure as a function of:

- contradiction severity (`metadata.severity`)
- persistence (time unresolved)

Resolved contradictions (`metadata.resolved=true`) contribute no amplification. Amplified pressure propagates to involved nodes with bounded attenuation.

### Drive/goal pressure propagation

Active drives seed pressure from salience + optional explicit metadata pressure. Pressure propagates along graph edges with bounded hop depth and attenuation.

Goal state affects pressure:

- blocked goals amplify pressure
- satisfied goals damp pressure

Pressure is bounded to `[0, 1]` and recorded on nodes as `metadata.dynamic_pressure`.

## Typed result contracts

Phase 4 introduces minimal typed dynamics contracts:

- `ActivationUpdateV1`
- `PressureUpdateV1`
- `DormancyTransitionV1`
- `SubstrateDynamicsResultV1`

These are inspectable outputs of each deterministic tick.

## Non-goals (explicit)

- no learned/GNN dynamics
- no frontier/teacher graph mutation expansion
- no broad runtime workflow refactor
- no replacement of canonical substrate ontology/materialization
