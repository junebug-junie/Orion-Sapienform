# PR report: drive-history reflection synthesis — fix second real grounding-guardrail rejection

## Summary

- Follow-up to the merged `feat/drive-history-identity-reflection` (PR #1031), which already fixed one real live rejection (percentage paraphrased as "all"/"every" instead of the literal `%` token).
- Live-reproduced a **second, distinct** real rejection while attempting to re-verify that first fix: the LLM correctly wrote `100%` this time, but dropped fact 1's other required token (`500`, the raw tick count) entirely — treating the percentage as making the count redundant.
- Added an explicit prompt instruction: facts bundling more than one number require ALL of those numbers verbatim, not just whichever reads most naturally in prose.
- Could not re-verify this specific fix end-to-end live — see "Docker/build/smoke checks" for why, and for a real, separate infra finding surfaced while investigating.

## Outcome moved

Two real, independently-observed LLM compression patterns that were defeating the grounding guardrail are now both explicitly addressed in the prompt. The guardrail itself has not been weakened at any point — both fixes tighten the model's compliance with it rather than loosening what it checks.

## Current architecture (before this patch)

`_build_narrative_prompt()` (merged in #1031, then adjusted once already for the percentage-paraphrase case) still let the model drop a fact's second number when one number seemed to imply the other.

## Architecture touched

`scripts/drive_history_reflection_synthesis.py` only — prompt text, no logic change.

## Files changed

- `scripts/drive_history_reflection_synthesis.py`: one additional paragraph in `_build_narrative_prompt()`'s instructions, explicit about multi-number facts.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
$ python -m pytest tests/test_drive_history_reflection_synthesis.py -q
45 passed
```
No test asserts exact prompt wording; unaffected by this change.

## Evals run

Not applicable — see live verification notes below for the real evidence this patch is based on.

## Docker/build/smoke checks

**Real live rejection this fix addresses (captured before the fix):**
```text
RAW_LLM_CONTENT_DEBUG: '{"narrative": "Over the period from 2026-06-14 to 2026-06-19,
autonomy was the dominant drive in 100% of audited ticks, demonstrating a consistent
focus on autonomy. On 2026-06-18, autonomy was the dominant drive with 81 ticks, and
on 2026-06-19, it was dominant with 419 ticks, reflecting a strong and sustained
emphasis on autonomy as noted in audit drive-audit-4431285570f10004 at 2026-06-19
05:24 UTC.", "cited_fact_numbers": [1, 2, 3, 6]}'

LLM output rejected as ungrounded: cited fact 1 but not all of its real tokens
('autonomy', '100%', '500') appear verbatim in the narrative
```
Note: `100%` IS present (the first fix worked); `500` is not present anywhere in the narrative — the model treated it as implied by "100%" and dropped it.

**Could not re-verify the fix live.** Three follow-up attempts (varying `--max-events`, up to a 4-minute budget) all timed out with no output. Diagnosed why rather than assuming it was this fix:

```text
$ docker stats orion-athena-fuseki --no-stream
CPU %: 241.57%   MEM: 37.95GiB / 40GiB (94.88%)

$ time curl -s -m 15 -X POST http://localhost:3030/orion/query \
    --data "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } LIMIT 1"
real 0m0.031s   # trivial query still fast in isolation

$ docker logs orion-athena-fuseki --since 15m | tail -40
# repeating, ~3-second-interval query pattern against a completely different
# graph (http://conjourney.net/graph/substrate, "harness_closure"/
# "sub-identity-*" node/edge queries with ~70-item VALUES clauses) -- not
# this script, not the drives graph. Individual queries in this loop complete
# in 10-53ms each, but the loop itself is sustained and heavy.
```

Fuseki is under real, sustained resource pressure (94.88% of its 40GB memory limit, 241% CPU) from an unrelated, high-frequency query loop. This is the most plausible explanation for this script's intermittent timeouts, not a regression from a pure prompt-text change (the code path to the point of hanging is otherwise identical to the two prior successful runs that did complete). Not investigated further — identifying what's issuing that loop is a separate, unrelated task outside this patch's scope — but flagging plainly since it's a real, currently-live resource-pressure condition on a shared container, not a one-off blip.

## Review findings fixed

N/A — this patch is itself a fix, evidence-based from a real live run, not a response to a separate review pass.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: Medium (carried over from the parent PR, now compounded) — still no fully-observed successful end-to-end run (crystallization actually created) for this script. Two real, distinct rejection patterns have now been found and fixed; whether a third exists is unknown until a clean run is observed. Recommend re-attempting once Fuseki's load has settled, ideally with visibility into what's driving the "harness_closure" query loop.
- Severity: Low — the Fuseki resource-pressure finding above is real and current at time of writing; if it persists or worsens, it may be worth its own investigation (separate from this script or its consumers).

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/drive-history-prompt-count-omission?expand=1`
