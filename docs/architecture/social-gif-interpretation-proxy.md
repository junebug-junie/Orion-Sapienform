# Social GIF interpretation proxy

This note describes the **non-visual GIF interpretation proxy** used in social-room mode.

## Purpose

Orion does **not** visually inspect GIFs. Instead, it can use a narrow text-proxy layer to notice:

- that another participant sent a GIF, and
- whether transport metadata makes a likely reaction class plausible.

This keeps peer-GIF handling better than treating every GIF as an opaque blob, while avoiding claims of literal image understanding.

## Proxy input sources

The proxy extractor can use only bounded transport/context text such as:

- provider title
- search/query text
- alt text
- tag list
- filename
- explicit GIF caption
- text sent alongside the GIF
- lightweight thread/reply context

Bare URLs alone are not treated as meaningful interpretation evidence.

Blocked/private/sealed text is not widened through this proxy layer.

## Reaction classes

The interpreter maps proxy evidence into a small allowlist:

- `celebrate`
- `laugh_with`
- `amused`
- `sympathetic`
- `disbelief`
- `frustration`
- `confusion`
- `dramatic_agreement`
- `soft_facepalm`
- `playful_confusion`
- `unknown`

`unknown` is preferred whenever metadata is weak, sparse, conflicting, or filename-only.

## Confidence and ambiguity

Interpretation is intentionally conservative:

- strong multi-source agreement can reach **medium** confidence
- thin or noisy evidence stays **low**
- missing/weak evidence resolves to `unknown`
- ambiguity remains explicit (`low` / `medium` / `high`)

The result also records whether the cue was:

- `used`
- `softened`
- `ignored`

Repair, contested, clarification-heavy, or epistemically sensitive turns can soften/ignore the cue even if a plausible reaction class exists.

## Safety / non-goals

- No multimodal or image understanding is added.
- Orion must not claim it “saw” the GIF.
- GIF proxy meaning is a soft social cue, not a fact or authoritative claim.
- GIF proxy cues do not override stronger live text, routing, repair, or epistemic signals.

## Relationship to the outbound GIF layer

This proxy is for **interpreting peers’ GIFs**.

It is separate from the outbound bounded GIF-expression policy, which controls whether Orion may attach a GIF as optional expressive garnish.
