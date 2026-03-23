# Social-room realism tuning

This pass tightens the existing social-room heuristics so Oríon feels less like a moderator and more like a natural peer in live multi-peer rooms.

## Main adjustments

- **Bridge summaries are more selective.**
  They now require genuine room-level usefulness instead of firing whenever disagreement merely exists.

- **Clarifying questions are rarer.**
  They now win mainly when ambiguity is explicit or clarification pressure is genuinely high, rather than as a default response to mild disagreement.

- **Live local state beats decorative background more consistently.**
  Active thread, disagreement, repair, and commitments now outrank ritual/style hints, resumptive snapshots, and stale consensus more aggressively.

- **Floor management is more conservative.**
  `leave_open` is favored more often, while `yield_to_peer`, `invite_room`, and closure are used more sparingly.

- **Repair and epistemic guidance are shorter.**
  The wording hints now aim for plain conversational language instead of bureaucratic meta-language.

## Why

The new inspection/debug surfaces made it easier to see where the system felt over-structured:

- too eager to summarize the room
- too quick to ask clarification questions
- too willing to manage handoff/closure
- too verbose about uncertainty

These tuning changes keep the same safety and consent boundaries while making the behavior feel more human and locally grounded.
