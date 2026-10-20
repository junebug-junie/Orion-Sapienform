## Summary

- The personality cue, which reads Orion's strongest themes back to them, now shows every active theme. Before, it showed only the top one.
- The cue length limit goes from 600 to 900 characters (`PERSONALITY_CUE_MAX_CHARS`).
- Each theme label shows up to 64 characters, cut at a whole word (`LABEL_SHOWN_CHARS`, `_label`). It was 120.
- The rule that drops the weakest theme when the cue is too long is unchanged.

## Outcome moved

Step 1 of the personality handoff (PR #2766). At 600 characters only the top node fit. Because of that, the self-inquiry theme ("I am the seventy-fourth answer…") never reached Orion. It is the only active theme that is not prediction-error narration. Rendering the live frame of 2026-10-20 03:38 UTC:

- before: 1 of 4 nodes shown, `disposition_ids=[disp-a6a805961f7be5d9]`
- after: 4 of 4 nodes shown, 732 characters, plus the choice line

## Current architecture

One pure renderer, `orion/personality/cue.py`, serves every lane. Each service builds `PersonalityCueV1` locally from the durable-runs frame: orion-thought (stance, reverie) and orion-hub (curiosity kickoff, outreach, `/api/personality/cue`).

## Architecture touched

Only the shared renderer and the schema constant. No bus, DB or env changes.

## Files changed

- `orion/personality/cue.py`: shorter labels cut at a word; guards against "……", trailing punctuation, and a label that collapses to one word.
- `orion/schemas/personality.py`: `PERSONALITY_CUE_MAX_CHARS` 600 → 900.
- `orion/personality/tests/test_cue.py`: a regression test for the live shape (4 live labels and lane mixes, a fifth node, and the choice line), plus label edge cases.

## Schema / bus / API changes

- Behavior changed: `PersonalityCueV1.text` `max_length` goes from 600 to 900. The limit is looser, so it is backward compatible. Every service builds the cue locally, no reader parses it off the wire, and no DB column stores the text. Outreach `grounding_summary` stores only the IDs.

## Env/config changes

None.

## Tests run

```text
orion/personality/tests: 114 passed
services/orion-thought -k personality: 19 passed
services/orion-durable-runs -k personality: 33 passed, 3 skipped
services/orion-hub -k personality: 57 passed, 4 skipped
```

## Evals run

No eval harness covers the cue. The live-frame render above is the evidence: 4 of 4 nodes, 732 characters. Whether lanes act differently with more themes is the later measurement step. Juniper's standing direction is graph and read-back first.

## Docker/build/smoke checks

Not deployed from this branch. Live deploys run from main after merge.

## Review findings fixed

- Finding: a double "……" when the cut kept a word that ends in "…".
  - Fix: the strip set now includes "…", ".", "—" and "-".
  - Evidence: `test_label_shortening_edges`.
- Finding: a label whose only space is near the start collapsed to "a…".
  - Fix: cut back to a whole word only if at least half the width remains. Otherwise cut hard.
  - Evidence: `test_label_shortening_edges`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build && scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Run this from the primary checkout on main after merge. The order does not matter, because the limit is looser.

## Risks / concerns

- Severity: low.
  - Concern: about 300 more prompt characters per lane, at most.
  - Mitigation: the drop-weakest rule still enforces the limit.
- Severity: note.
  - Concern: 3 of the 4 themes Orion now sees are prediction-error narration. That is handoff step 2 (correlated lanes). This PR does not fix it.

## PR link

See PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
