# Social claim revision and peer-claim tracking

## Goal

Keep a compact room-local record of claims that matter socially without treating every utterance as settled fact.

## Claim types

- `peer_claim`: a peer asserted something track-worthy.
- `orion_claim`: Oríon asserted something track-worthy.
- `shared_summary`: a compact summary that crystallized room understanding.
- `inferred_claim`: an interpretive claim framed as a read, not direct fact.

## Stance lifecycle

- default stance is `provisional`
- `accepted` is reserved for clearer shared confirmation / settled summary language
- `disputed` means the room pushed back on the claim
- `corrected` / `revised` mean later turns updated the earlier claim
- `withdrawn` means the earlier claim should no longer be carried forward

## Revision behavior

- extraction is conservative and explainable
- blocked / private / sealed material is ignored
- pending / declined shared-artifact dialogue does not become accepted claim state
- later corrections update the active claim stance and emit a compact revision record
- Oríon is grounded to prefer the updated claim over the stale one

## Integration points

- `orion-social-memory` extracts and revises compact room-local claims from stored turns
- claim stances and revisions are attached to room continuity for prompt grounding
- repair signals can downgrade a fresh claim to provisional and can trigger revision-aware grounding
- epistemic wording/prompt guidance tells Oríon not to present provisional or disputed claims as settled fact
- commitments do not override corrected claim state

## Non-goals

- no heavy belief engine
- no general planner / tool runtime
- no widening of blocked/private recall
- no attempt to formalize every utterance into a claim graph
