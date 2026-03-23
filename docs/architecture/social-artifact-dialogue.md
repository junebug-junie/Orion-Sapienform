# Social Artifact Dialogue for `social_room`

## Purpose

This phase adds a compact dialogue layer for shared social artifacts inside the existing `social_room` path.

The goal is to let Oríon and a peer briefly negotiate **what** would be carried forward, **how** it would be worded, and **which scope** it would live in, without turning the flow into a protocol engine or planner lane.

## Typed states

The flow uses three typed records:

- `SocialArtifactProposalV1`
- `SocialArtifactRevisionV1`
- `SocialArtifactConfirmationV1`

These records are prompt-safe and inspectable. They stay attached to the existing Hub → social turn → social-memory seam.

## Conservative scope handling

Supported dialogue scopes are:

- `session_only`
- `room_local`
- `peer_local`
- `no_persistence`

When scope is ambiguous, Oríon defaults to `session_only` or asks one concise clarifying question.

No broadening from `session_only` to `room_local` / `peer_local` happens without explicit agreement.

## Proposal / revision / confirmation flow

1. Hub detects a narrow shared-artifact dialogue cue.
2. Hub renders a compact proposal or revision into prompt metadata and, when useful, into the narrow `social_artifact_dialogue` skill result.
3. The turn is stored through the normal `social.turn.v1` / `social.turn.stored.v1` path.
4. `orion-social-memory` persists the latest proposal / revision / confirmation records for the relevant peer-local or room-local surface.
5. Only an **accepted** confirmation with a durable scope becomes an active continuity input.

## Persistence boundary

- **Proposed** artifacts are inspectable but non-expanding.
- **Revised** artifacts are inspectable but non-expanding.
- **Declined / deferred** artifacts are inspectable but non-expanding.
- Only **accepted** `room_local` / `peer_local` confirmations can expand active continuity.
- `session_only` stays narrow and does not become durable continuity.

## Safety / privacy rules

- blocked / private / sealed material never enters a proposal, revision, or confirmation
- unresolved proposals do not silently become durable memory
- decline / defer ends the exchange without pressure or repetition
- the dialogue stays compact, socially natural, and non-agentic

## Non-goals

- no general tools or actions
- no planner-react / agent-chain behavior
- no open-ended negotiation loop
- no generic protocol framework
- no hidden durable persistence
