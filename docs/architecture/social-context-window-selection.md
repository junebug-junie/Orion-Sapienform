# Social context window selection

## Context candidate model

The social-room stack now assembles a compact context window for each social-memory summary using:

- `SocialContextCandidateV1` for each possible grounding candidate
- `SocialContextSelectionDecisionV1` for the final selection rationale and budget accounting
- `SocialContextWindowV1` for the selected prompt-safe set

Candidates can represent thread state, claims, consensus, divergence, calibration, commitments, ritual/style hints, freshness warnings, handoff, deliberation, and participant/room continuity.

## Selection rules

Selection is conservative and local-first:

- addressed-peer continuity outranks generic room background
- active thread state outranks room-global summaries
- open commitments outrank older ritual/style hints
- fresher contested/divergent claim state outranks stale consensus
- fresh calibration can stay active, while stale calibration softens or drops
- `refresh_needed` freshness hints are kept when they stop stale assumptions from governing the turn

## Budget and window semantics

The selector ranks candidates by inclusion decision, priority band, freshness, and relevance score, then keeps a compact budgeted set for prompt grounding.

- `include`: active governing context
- `soften`: useful background, not governing
- `exclude`: considered, but intentionally left out of the active window

The prompt consumes the selected window first and treats excluded/softened stale state as non-governing unless the turn explicitly reopens it.

## Safety and non-goals

- blocked/private/sealed material stays out
- pending/declined artifact dialogue does not become active context unless explicitly represented as pending dialogue state
- the selector does not enable tools or hidden actions
- this is not long-context maximization; it is compact, situation-appropriate grounding
