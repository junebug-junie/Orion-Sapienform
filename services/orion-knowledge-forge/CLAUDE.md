# Orion Knowledge Forge Service Instructions

This service exists to compile messy research/design material into reviewed, source-grounded execution context.

## V1 boundaries

Allowed:
- read raw sources, claims, specs, decisions, reviews, context packs
- search lexical knowledge
- summarize status
- compile context packs
- propose review artifacts

Forbidden unless explicitly requested:
- silent canonical mutation of accepted claims/specs/decisions
- GraphDB/RDF integration
- vector search
- autonomous background rewriting
- chat auto-ingest
- full engineering harness

## Core primitives

- Source: raw artifact
- Claim: atomic source-backed assertion
- Decision: reviewed choice and rationale
- Spec: current design or plan intent
- Context pack: task-specific execution bundle
- Review: proposed mutation awaiting human approval

## Ideation outputs

Claude should write proposals as review artifacts, not canonical truth.

Preferred paths:
- reviews/pending/ideation-*.md
- reviews/pending/spec-delta-*.md
- context_packs/generated/*.md
