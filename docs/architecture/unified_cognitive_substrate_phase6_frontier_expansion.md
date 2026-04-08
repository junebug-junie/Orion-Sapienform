# Unified Cognitive Substrate — Phase 6: Frontier Expansion as Typed Graph-Delta Generation

## Why this phase exists

Phases 1–5 established canonical substrate structure, materialization, deterministic dynamics, and deterministic perception. Phase 6 adds a controlled frontier lane for **typed graph expansion proposals** rather than freeform critique text.

## Frontier expansion vs frontier critique

- **Critique lane**: can remain advisory/narrative.
- **Expansion lane (this phase)**: must emit typed graph-delta items that map into substrate-native node/edge candidates plus contradiction/evidence-gap candidates.

This keeps frontier contributions inspectable and machine-actionable.

## Core contracts

Phase 6 introduces typed contracts for:

- bounded expansion requests (`FrontierExpansionRequestV1`)
- structured frontier responses (`FrontierExpansionResponseV1`)
- per-item typed deltas (`FrontierDeltaItemV1`)
- mapped substrate-native candidate bundles (`FrontierGraphDeltaBundleV1`)

## Bounded context philosophy

`FrontierContextPackBuilder` builds bounded context packs from materialized substrate state:

- bounded focal nodes/edges
- activation/pressure hotspots
- contradiction references
- evidence refs
- explicit truncation marker

No full-graph dump and no unbounded memory dump.

## Zone-aware landing posture

Expansion bundles carry target-zone + suggested landing posture:

- `world_ontology` → `fast_track_proposal`
- `concept_graph` → `moderate_proposal`
- `autonomy_graph` → `conservative_proposal`
- `self_relationship_graph` → `strict_proposal_only`

This enables policy-aware promotion and review in later phases.

## Identity/autonomy protection

For autonomy/self-relationship zones, mapper enforces proposal-only posture and rejects canonical node writes from frontier output. This prevents direct canonical overwrite of protected self/relationship/autonomy state.

## Provider-agnostic seam

`FrontierExpansionService` uses a provider protocol and deterministic mapper/context builder:

1. build bounded context pack
2. call provider seam
3. validate request/task/zone consistency
4. map to substrate-native delta bundle

No hard dependency on live provider integration is required.

## What later phases can add

- promotion/HITL integration for landing posture execution
- runtime-triggered frontier invocation policies
- curiosity loops over contradiction/evidence-gap clusters
- richer taxonomy/refinement item handling
