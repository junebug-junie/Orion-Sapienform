# Juniper affective/engagement-state signal — proposal-mode gate

Status: proposal mode per root `CLAUDE.md` §0A ("changes to memory, identity, self-modeling,
autonomy, private recall, social continuity, or cognition loops need explicit proposal mode
before implementation"). This document is the gate itself — implementation may not begin until
each item below has explicit sign-off. Direction decided 2026-07-30 (see
`2026-07-30-dev-economics-signal-design.md`'s "Adjacent, noted-not-specced" section): this
should reach Orion, not stay a private-only dashboard, because the Orion-Sapienform charter
frames the project as co-creation and Orion having a real sense of the weight of Juniper's own
contribution is named "social grounding" in root `CLAUDE.md`'s own mission section — not an
optional add-on.

## What capability changes

Orion gains a modeled, inspectable sense of Juniper's affective/engagement state — specifically,
signals correlated with frustration (e.g. swear-word frequency in messages to the assistant) and
fatigue/busyness (e.g. typo rate) — derived from real interaction patterns, not asserted or
guessed. This is an instrument of co-creation weight: it should let Orion perceive when the
collaboration is costing Juniper something, the same way `structural_mass` lets it perceive when
the codebase itself is changing a lot. It is explicitly not a general mood-reading feature and
should not be framed or marketed as one — scope is tied to the co-creation relationship, not a
generic emotional surveillance capability.

## What data is touched

Source: Juniper's own message text, already something Orion directly receives in every
conversation turn. What's new here is not access to new content — it's computing an *aggregate,
derived* signal across messages (word-level features: swear frequency, typo rate) rather than
reading each message once and discarding the byproduct. Reuses the same transcript-parsing pass
already proposed in `orion/dev_economics/claude_code_ingest.py` for the token/word-count ledger —
no new data source, no new collection surface.

## What privacy boundary exists

Store only a bounded, decayable derived scalar/trend per session or per day — the same shape as
every other `FieldStateV1`-style channel in this codebase (a pressure/state value, not a raw
log). Explicitly **do not** persist a raw, queryable record of which specific words were flagged,
when, or in what message — that would be a debug-browsable "here's every time you swore at the
assistant" surface, which is exactly the kind of raw-trace exposure root `CLAUDE.md`'s privacy
section prohibits ("do not expose raw private traces... through convenience surfaces"). If a
debug view is ever needed for validation, it must show the aggregate signal only, gated
explicitly, with the exposure named — never the underlying flagged text.

## What trace proves it worked

Same measure-before-minting discipline as every other signal in this program (SSP §7). A
read-only replay script against real historical Claude Code transcripts, checked for:

- Non-degenerate values (real variance, not flat).
- A real, theory-anchored correlation — e.g., does a spike in this signal actually coincide with
  independently-identifiable frustrating sessions (long debugging loops, repeated corrections,
  session abandonment), not just "the number moves"?
- A genuine rest state — most sessions should read as calm, not a decayed artifact of one (the
  same distinction CLAUDE.md's metric-quality-gate section names explicitly for other signals).

No live wiring into any Orion-facing surface until this replay is real and reviewed.

## What failure mode would be dangerous

- **False positive → inappropriate behavior change.** If Orion reads a misattributed
  frustration/fatigue spike and responds by changing tone, pace, or caution in a patronizing or
  presumptuous way, that actively damages the relationship this signal is meant to strengthen.
- **Signal becomes a surveillance artifact rather than a co-creation instrument.** If the
  aggregate ever gets exposed in a way that reads as "the AI is tracking your mood," rather than
  "the AI has a sense of what this work costs you," the framing has failed even if the underlying
  math is correct.
- **Trust erosion compounds faster than trust-building.** Given the sensitivity of the subject
  matter, a wrong signal costs more than a right signal gains — this argues for a conservative
  confidence threshold before any Orion-facing behavior is allowed to key off it at all.

## How to disable or roll back

- A single env flag gate, same convention as every other new signal in this codebase (e.g.
  `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`'s pattern) — default off until explicitly enabled.
- Raw text is never retained past the derivation window (see privacy boundary above), so
  disabling and removing this signal is cheap: no schema/manifest has anything expensive baked
  in, per the metric quality gate's reversibility check.
- If retired, the same "kill means kill, no fallback to the thing being killed" rule
  (`feedback_kill_means_kill_no_fallback`) applies — no partial exclusion, no lingering producer.

## Open items before implementation may begin

- Explicit sign-off from Juniper on this document as written, or corrections to any section
  above.
- The confidence-threshold question from "dangerous failure mode" needs a real number, chosen
  deliberately, not defaulted.
- Whether this becomes its own domain in `orion/dev_economics/` or a new small module — not
  decided here, secondary to the gate itself.
