# Social Relational Memory

## Purpose

This phase adds a compact, inspectable relational-memory layer on top of the existing `social_room` flow. The goal is to help Orion carry safe continuity with recurring peers and rooms without widening private recall or turning the transport layer into an agent runtime.

## What gets synthesized

After `social.turn.stored.v1` is emitted, `orion-social-memory` performs lightweight post-turn synthesis and updates:

- **peer continuity** keyed by `platform + room_id + participant_id`
- **room continuity** keyed by `platform + room_id`
- **social stance snapshot** keyed to Orion's social-room mode

Each artifact is compact, evidence-backed, and built only from stored social-turn material plus safe external-room metadata already attached to the turn.

Shared peer-local and room-local cues are now tracked with typed consent states (`accepted`, `declined`, `deferred`, `unknown`) so the memory layer can keep mutual legibility modest, scoped, and explainable.

## Recall boundary

The live bridge path remains thin:

1. room message enters `orion-social-room-bridge`
2. bridge optionally fetches compact social-memory summaries
3. Hub invokes `chat_profile=social_room`
4. sql-writer persists the social turn
5. `orion-social-memory` synthesizes updated summaries post-turn

This does **not** widen blocked/private recall. The relational summaries are derived only from safe social-turn evidence, not from sealed internal traces or unrestricted Juniper material.
If a turn explicitly declines or defers a shared cue, that turn does not expand the scoped shared artifact for the affected peer/room surface.

## Prompt grounding

`chat_social_room` can now receive:

- peer continuity summary
- room continuity summary
- Orion social stance snapshot
- existing safe memory digest / concept evidence

These are intentionally compact and prompt-safe, so the live response stays fast and inspectable.

## Intentional non-goals

This layer does **not**:

- enable tools or actions
- perform autonomous relationship planning
- expose broad private introspection
- simulate deep psychology
- recreate Cortex inside social memory
