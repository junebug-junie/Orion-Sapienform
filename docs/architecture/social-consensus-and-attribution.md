# Social consensus, divergence, and attribution

## Goal

Keep room-local claim tracking speaker-aware so Oríon can tell the difference between a peer-held view, Oríon’s own stance, partial alignment, and active disagreement.

## Attribution model

- claim attribution stays attached to a compact normalized claim key
- peer participant stances are tracked conservatively as `support`, `question`, `dispute`, `correct`, `withdraw`, or `unknown`
- Oríon’s stance is tracked separately so the room does not collapse into a single flat belief

## Consensus states

- `none`: one speaker asserted something, but the room has not converged
- `partial`: more than one holder exists, but support is still narrow
- `emerging`: support is growing, often including Oríon, but still not broad enough to call settled
- `contested`: dispute / withdrawal is present
- `consensus`: broader support is visible
- `corrected`: the room revised a previously held version

## Prompt integration

- social-room grounding includes attributed claims plus compact consensus/divergence hints
- prompt instructions tell Oríon to attribute views clearly and not flatten contested claims into fake agreement
- epistemic wording stays narrower when claim consensus is partial, contested, or corrected

## Safety / non-goals

- blocked / private / sealed material remains out of attribution and consensus state
- pending / declined artifact dialogue does not become consensus evidence
- no heavy argumentation engine
- no planner / tool / orchestration surface
