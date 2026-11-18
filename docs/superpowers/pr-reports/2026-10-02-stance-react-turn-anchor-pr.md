## Summary

- The stance step (stance_react) no longer asks the model to copy this turn's 36-character ID. It cites the bare token `hub:turn`, and code expands it to `hub:turn:<correlation_id>`.
- The prompt no longer shows the turn ID anywhere. In the attended-node lists the anchor appears as `hub:turn`, and `correlation_id` is dropped from the prompt-facing association and repair-bundle summaries.
- `canonicalize_turn_refs` turns `hub:turn`, the bare ID, or any garbled `hub:turn:*` into the real anchor, in both `evidence_refs` and `strain_refs`.
- Fixes a hole: model-written `strain_refs` widen the set of allowed evidence, so a garbled anchor there used to let the same garbled evidence ref through.
- Thinking is unchanged (no reasoning budget), per Juniper.

## Outcome moved

A Hub chat turn on 2026-10-02 (corr `39adc920`) deferred with "stance_react exec result missing thought payload". The model (Qwen3.6-35B, chat lane) produced a good stance in about 2k characters of reasoning. It then spent about 14k characters re-checking its copy of the UUID ("Wait, the prompt says…"), hit max_tokens=8000 (`finish_reason=length`) and returned empty content. The gateway call took 114s and the Hub waited 144s. Counted from `orion_metacognitive_trace` (stance reasoning that mentions `correlation_id` 10 or more times): 12–34 of these spirals a day on 09-25..09-28, 1–4 a day since 09-29.

## Current architecture

Hub → orion-thought `run_stance_react` → cortex-exec `stance_react` verb → gateway chat lane. Model JSON → `parse_stance_react_payload` → `apply_stance_react_pipeline` → `align_evidence_refs_to_coalition` → quality and disposition checks. The alignment step already fell back to the anchor when citations were empty or invalid, so the model never needed to copy the ID.

## Architecture touched

orion-thought prompt context only. No bus, schema or env change.

## Files changed

- `orion/thought/coalition.py`: adds `HUB_TURN_REF_TOKEN`, `canonicalize_turn_refs` and `prompt_turn_refs`. Alignment now canonicalizes evidence and strain refs.
- `orion/thought/stance_react.py`: the prompt-facing association summary shows `hub:turn` and drops `correlation_id`. The repair-bundle summary drops `correlation_id`.
- `services/orion-thought/app/bus_listener.py`: `coalition_projection` shows `hub:turn`.
- `orion/cognition/prompts/stance_react.j2`: asks for the literal `hub:turn`, never the ID. The subset rule names `hub:turn` as the anchor.
- `orion/thought/evals/stance_task_boundary.py`: canonicalizes refs the same way the runtime does.
- Tests: `orion/thought/tests/test_stance_react_pipeline.py`, `orion/thought/tests/test_stance_task_boundary_eval.py`, new `services/orion-thought/tests/test_stance_react_prompt_hides_turn_id.py`.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: duplicate refs are now deduplicated (nothing downstream relies on duplicates). Any `hub:turn:*` ref maps to the current turn. That is safe because only the current turn's anchor is ever in the coalition (`orion/hub/association.py:76`).
- Compatibility notes: the full anchor is still accepted.

## Env/config changes

None.

## Tests run

```text
pytest orion/thought/tests orion/harness/tests/test_stance_scope_hierarchy.py services/orion-thought/tests -q
1 failed, 579 passed, 28 skipped
```

The one failure, `test_settings_mind_enrichment.py::test_mind_enrichment_defaults_off`, also fails on main without this change. It reads the local service `.env` (`http://mind:6611`).

The new rendered-prompt test fails against the old `stance_react.py` and passes with the fix.

## Evals run

`stance_task_boundary` is updated to accept `hub:turn`, and its unit tests pass. The live gateway eval was not run: UNVERIFIED against the live model.

## Docker/build/smoke checks

Not run. This is a Python and template change in orion-thought. A live check after deploy: watch for `stance_react error` in the orion-thought logs, and for stance `pre_answer` traces with 10 or more `correlation_id` mentions.

## Review findings fixed

- Finding: the prompt still rendered the full `hub:turn:<uuid>` in `attended_node_ids` (association and coalition_projection), plus `correlation_id`, while line 113 required evidence to be a subset of those IDs. That contradicted the bare-token instruction and could restart the spiral.
  - Fix: the prompt-facing summaries show `hub:turn` and drop `correlation_id`. Line 113 names `hub:turn` as the anchor.
  - Evidence: `test_rendered_stance_prompt_never_contains_the_turn_id` renders the real template and asserts the UUID is absent.
- Noted (latent, not fixed): if earlier turns' anchors are ever put into the coalition, canonicalization would relabel them as the current turn.

## Restart required

```bash
scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: low
- Concern: the model can still spiral on other long IDs, such as open-loop IDs. Thinking is not budgeted.
- Mitigation: the spiral count query above detects recurrence.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2478

🤖 Generated with [Claude Code](https://claude.com/claude-code)
