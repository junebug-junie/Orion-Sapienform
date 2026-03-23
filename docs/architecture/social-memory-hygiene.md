# Social memory hygiene and re-grounding

## Freshness model

The social-room stack now carries compact freshness surfaces alongside continuity, calibration, and consensus state:

- `SocialDecaySignalV1` records why a remembered social artifact is losing support.
- `SocialRegroundingDecisionV1` records the bounded action to take (`keep`, `soften`, `reopen`, `expire`, `refresh_needed`).
- `SocialMemoryFreshnessV1` gives prompt-safe freshness hints for participant- and room-local grounding.

These surfaces are scoped locally by platform / room / participant / thread when available and track:

- artifact kind
- freshness state
- decay level
- confidence and evidence count
- last supporting update timestamp
- rationale / reasons / metadata

## Decay, softening, and reopening

Freshness checks are conservative and only use already-available social-room evidence. Signals can come from:

- time since the last supporting evidence
- low evidence count
- contradiction / correction against older consensus
- thread or context shift
- style / ritual hints that stop repeating
- calibration hints that are no longer refreshed
- commitments whose TTL elapsed

When support weakens, Orion does **not** treat old state as silently true. Instead it applies bounded re-grounding:

- stale consensus softens toward `emerging` / contested instead of staying strong
- stale peer calibration weakens toward `unknown` / cautious
- expired commitments are removed cleanly
- old style / ritual hints fade toward lighter adaptation
- old bridge / closure cues are reopened or expired on thread shift

## Re-grounding semantics

Re-grounding is intentionally narrow:

- `soften`: keep as low-strength orientation only
- `reopen`: assume the room may need clarification or fresh evidence
- `expire`: stop carrying the state forward as active
- `refresh_needed`: keep it visible only as a prompt to check again before relying on it

Prompt grounding surfaces these as compact hints so Orion prefers refresh / clarification over acting as if older assumptions are settled current truth.

## Safety and non-goals

- Blocked / private / sealed material stays blocked.
- Pending / declined artifact dialogue does not count as freshness support.
- Freshness changes caution and grounding weight, not truth or authority.
- No tool execution or action execution is introduced by this layer.
- This is not a hidden ranking system; it is a local, reversible memory hygiene pass.
