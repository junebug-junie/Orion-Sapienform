# Social thread choreography

This phase extends the existing `social_room` path with lightweight multi-peer thread handling.

## Active thread model

- `SocialThreadStateV1` tracks a compact room thread key, audience scope, active participants, open-question state, last speaker, and a short safe summary.
- Room continuity now stores a small bounded list of active threads plus the current primary thread and any latest handoff signal.
- Thread state is updated heuristically from transport `thread_id`, reply-target hints, mentions, and topic continuity.

## Routing decisions

- `SocialThreadRoutingDecisionV1` gives the current best routing hint for Orion's next reply:
  - `reply_to_peer`
  - `reply_to_room`
  - `wait`
  - `summarize_room`
  - `revive_thread`
- The bridge stays transport-thin and only preserves metadata plus a compact routing decision; the Hub prompt uses that hint to answer the right audience.

## Handoff handling

- `SocialHandoffSignalV1` marks lightweight transitions such as:
  - a peer tossing a thread to Orion,
  - a room summary / transition moment,
  - a yield to another peer,
  - a clean thread wrap.
- Handoff remains heuristic and inspectable, not a general dialogue-management engine.

## Safety boundary

- Blocked/private/sealed material remains blocked upstream and is not promoted through thread summaries.
- Pending or declined shared-artifact dialogue does not update active thread choreography.
- Style/ritual synthesis is suppressed while artifact dialogue is active so pending artifact language does not bleed into other social synthesis layers.
- Orion is guided toward the locally relevant thread and audience, not toward becoming a moderator/controller of the room.

## Non-goals

- No general tools/actions.
- No planner-react or agent-chain behavior.
- No heavyweight conversation engine.
- No widened private recall.
