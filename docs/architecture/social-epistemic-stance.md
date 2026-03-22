# Social epistemic stance and uncertainty signaling

This phase adds a compact epistemic layer so Orion can distinguish recall,
summary, inference, speculation, proposal-state, and clarification-needed turns
inside shared rooms.

## Claim kinds

- `recall`: grounded in available social-memory continuity
- `summary`: compact recap of the active room/thread state
- `inference`: a read based on visible evidence, but not explicit memory
- `speculation`: tentative read when evidence is thin
- `proposal`: pending artifact / consent state that should not be framed as
  accepted memory
- `clarification_needed`: ambiguity is too high for a clean answer

## Confidence and ambiguity

- confidence is tracked as `high`, `medium`, or `low`
- ambiguity is tracked as `low`, `medium`, or `high`
- high ambiguity or repair-active turns should narrow confidence
- low evidence should move Orion away from recall toward inference,
  speculation, clarification, or deferral

## Clarification behavior

- prefer one brief clarifying question when audience/thread/content ambiguity is
  high and Orion is locally engaged
- otherwise defer narrowly instead of making an overconfident claim

## Prompt and routing integration

- bridge policy emits epistemic signals and decisions for prompt grounding
- prompt instructions encourage natural claim framing without bureaucratic
  labels on every sentence
- routing / repair can push epistemic stance toward clarification rather than
  false certainty

## Safety / non-goals

- blocked/private/sealed material remains blocked
- uncertain memory must not be presented as confident recall
- pending/declined shared artifacts remain non-active
- no tools, external actions, or general agent runtime behavior are added
