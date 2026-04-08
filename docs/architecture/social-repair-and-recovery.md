# Social repair and misunderstanding recovery

This phase adds a compact, room-local repair layer for common conversational
misalignment inside shared rooms.

## Repair signals

- direct peer correction (`that was for Cadence, not you`)
- thread / audience mismatch (`wrong thread`, `wrong person`)
- contradiction against Orion's recent local commitment or reply
- scope correction (`room-local, not peer-local`, `session-only`, `private`)
- redirect / yield cue (`let Archivist take this one`)
- clarification request after an apparent mismatch

Signals stay heuristic, conservative, and inspectable. Weak cues can be logged
but ignored.

## Repair decisions

- `repair`: brief correction, then move on
- `clarify`: ask one narrow clarifying question
- `yield`: stop pressing and let the target peer take the exchange
- `reset_thread`: suppress the wrong-thread reply and retarget cleanly
- `ignore`: low-confidence or non-actionable signal

## Routing and prompt integration

- bridge policy detects repair signals from the inbound room turn plus current
  social-memory context
- routing can suppress wrong-thread replies, retarget the audience, or yield
- prompt grounding receives both the active repair signal and the recommended
  repair decision so Orion can respond naturally

## Interaction with commitments and consent

- repair can suppress open commitments when yielding or resetting is safer
- scope corrections only narrow behavior; they never broaden carry-forward
- pending / declined artifact dialogue stays non-expanding
- private / sealed language remains blocked and should not be surfaced

## Safety / non-goals

- no tools, task execution, or planner escalation
- no heavy self-critique loop
- no widening of blocked/private recall
- no treating every disagreement as a repair event
- no verbose or defensive apology behavior
