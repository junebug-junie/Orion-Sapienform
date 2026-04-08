# Social GIF expression layer

The social GIF layer lets Orion add a bounded `text_plus_gif` garnish in `social_room` without turning media into the carrier of meaning.

## Policy model

The layer uses three compact models:

- `SocialGifPolicyDecisionV1`
- `SocialGifIntentV1`
- `SocialGifUsageStateV1`

The policy decision records whether GIF use is allowed, the decision kind, the chosen intent, cooldown/density state, and the reasons that allowed or blocked it.

## Allowed intent categories

The first-pass allowlist is intentionally small:

- celebrate
- laugh_with
- sympathetic_reaction
- dramatic_agreement
- soft_facepalm
- playful_confusion
- victory_lap

Intent is chosen before any provider-neutral media hint is emitted.

## Safety boundary

GIFs are blocked when the turn is doing serious work, including:

- repair-active turns
- clarification / ask-follow-up turns
- scope / consent / shared-artifact boundary turns
- contested or epistemically sensitive claim handling
- bridge-summary / disagreement-summary turns
- private / sealed / blocked material
- high thread ambiguity

This keeps GIFs from bypassing routing, repair, epistemic, or safety logic.

## Streak, cooldown, and density rules

The default bounded rules are conservative:

- never allow GIFs in successive Orion turns
- require at least 2 text-only Orion turns between GIF turns
- cap Orion to at most 2 GIF turns in a 10-turn rolling window
- suppress repeated-intent reuse when it starts to look habitual

Usage state stays room-local and lightweight.

## Text+GIF first

V1 only allows:

- `text_only`
- `text_plus_gif`

`gif_only` remains disabled. The text must still carry the actual meaning, and transport integration stays thin by emitting only provider-neutral metadata hints when the room transport says it supports them.

## Non-goals

- no general media autonomy framework
- no arbitrary media retrieval
- no GIF-first important turns
- no GIFs during sensitive/private/repair-heavy flows
- no uncontrolled repetitive GIF behavior
