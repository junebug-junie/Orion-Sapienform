# Social conversational commitments

This phase adds a short-lived conversational-commitment layer on top of the
existing social-room continuity path.

## Model

- `SocialCommitmentV1` stores a compact open commitment tied to a room and,
  when possible, a thread.
- `SocialCommitmentResolutionV1` records when that commitment is fulfilled,
  superseded, dropped, or expired.
- Commitments are room-local, bounded in count, and time-boxed with an expiry.

## Extraction rules

Only explicit, locally conversational promises are extracted from Orion's
response, such as:

- "I'll summarize in a sec"
- "I'll come back to that"
- "Let me answer X first"
- "I'm yielding here"
- explicit accepted scoped-memory promises

Weak or hedged language (`maybe`, `could`, `might`) is ignored. Private/sealed
language blocks commitment creation.

## Lifecycle

- create: when Orion makes a clear short-lived conversational promise
- fulfill: when Orion locally follows through in a later turn
- supersede: when a newer commitment replaces an older one in the same lane
- expire: when the ttl elapses without local follow-through
- drop: when room context moves on and the commitment is no longer locally
  relevant

## Routing and prompt integration

- social-memory stores the current open commitments inside room continuity
- bridge routing can prefer summary / revival / wait when an open commitment is
  due and locally relevant
- hub prompt grounding receives one or two open commitments plus due-state so
  Orion can honor them naturally without forcing them into unrelated replies

## Safety / non-goals

- no tool or action execution
- no planner or task queue behavior
- no long-horizon obligations
- no widening of blocked/private/sealed recall
- declined/deferred artifact dialogue does not become an open commitment unless
  there is a narrow accepted scoped-memory promise
