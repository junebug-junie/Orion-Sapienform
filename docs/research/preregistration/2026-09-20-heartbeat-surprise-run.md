# Heartbeat surprise

Pre-reg: `docs/research/preregistration/2026-09-20-heartbeat-surprise.md`

Generated: 2026-09-20T22:29:54.818596+00:00

- sessions scored=12 median_slope=-0.0026 frac_neg=0.67 median_drop=0.0488 settles=False
- controls scored=12 median_slope=-0.0009 frac_neg=0.67 median_drop=-0.0492 settles=False

## Decision

**mixed: chat start is louder, but no session-specific decline**

holds=False thermometer=False mixed=True

## Notes

The first control batch landed before grammar retention (empty windows). Controls were re-picked inside the live grammar range without changing the fail line, then merged with the already-scored sessions.

There is no forecast model. Surprise is change in the 9-cut profile versus the last snapshot.

## Plain reading

When a real chat stretch starts, heartbeat’s picture jumps more than it does in a random stretch of the same length. It does **not** then reliably quiet down for the rest of that chat. Eight of twelve sessions sloped down; we needed ten. Random stretches did the same eight-of-twelve. So the “settles during a conversation” half of the old charter question did not hold. The “jumps when something starts” half did.

Nothing was wired. Keep publishing the heartbeat verdict.
