## Summary

- Every cortex verb prompt now prints dicts and lists as JSON instead of Python repr (`{'key': 'value'}`).
- The fix lives in one shared Jinja environment, `orion/cognition/prompt_render.py`, used by cortex-exec `_render_prompt`.
- The stance, reverie and reading evals and the model replay render through the same environment, so they test the prompt production sends.

## Outcome moved

After the gateway JSON fix (#3355) deployed, 11 `stance_react` calls still failed (agent lane, 2026-11-14). All were "Expecting property name enclosed in double quotes".

- **Cause:** `stance_react.j2` printed `association`, `repair_bundle` and `coalition_projection` raw, so the model saw Python-literal dicts and answered in kind.
- **Why the gateway couldn't save them:** the replies mix quotes (`'Juniper's'`), which no safe repair can parse.
- **Effect of this fix:** the model now sees only JSON in its prompt.

## Current architecture

`_render_prompt` used a plain `Environment(autoescape=False)`, so `{{ dict }}` printed Python repr.

## Architecture touched

- New: `orion/cognition/prompt_render.py`.
- `services/orion-cortex-exec/app/executor.py`.
- Evals and replay: `orion/thought/evals/stance_task_boundary.py`, `services/orion-hub/evals/test_reading_stance_prompt_eval.py`, `services/orion-thought/evals/test_reverie_chain_continuity_eval.py`, `services/orion-thought/evals/run_temporal_self_cue_latency_eval.py`, `orion/evals/model_replay/extract.py`.

## Files changed

- `orion/cognition/prompt_render.py`: `json_finalize`, `prompt_environment()`.
- `services/orion-cortex-exec/app/executor.py`: `_render_prompt` uses the shared environment.
- `services/orion-cortex-exec/tests/test_prompt_dicts_render_as_json.py`: new tests.
- The five eval and replay files above: render through the shared environment.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: prompt text. Dicts and lists print as JSON. Inside them, `None/True/False` become `null/true/false` and `'` becomes `"`.
- Compatibility notes: `finalize` changes only `{{ }}` output. `{% if %}`, filters, indexing and `~` still see the Python object, and `| tojson` output passes through. Prompt length is unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: not needed
- skipped keys requiring operator action: none

## Tests run

```text
test_prompt_dicts_render_as_json.py              3 passed
  (the stance test fails on the old renderer: association, repair_bundle, coalition_projection
   printed as {'...'})
test_metacog_trend_cue_prompt_render.py          3 passed
test_dream_cycle_prompt_render.py                3 passed
test_substrate_probe_prompt_grounding.py         8 passed
orion/evals/model_replay                         99 passed
test_chat_relational_stance.py                   5 failed, 32 passed (identical on main: cwd-relative
                                                 template paths, pre-existing)
test_chat_general_stance_plumbing.py             1 failure, identical on main ('metadata' is undefined)
```

## Evals run

```text
None run live. The change is the prompt text. Live check after deploy: stance_react JSON 502s on the
agent lane, against the 11/day baseline (2026-11-14).
```

## Docker/build/smoke checks

```text
No dependency or compose change. orion-cortex-exec rebuilds on the post-pull run.
```

## Review findings fixed

- Finding: the evals and the model replay rendered stance with a plain `Environment()`, so they no longer tested the production prompt.
  - Fix: shared `prompt_environment()`, used by all of them.
  - Evidence: `test_evals_and_replays_render_like_production`.
- Finding: a `RecursionError` from very deep nesting was not caught.
  - Fix: caught; falls back to repr.
- Reviewer found no correctness bug in the change itself. It checked all 71 templates for `| safe`, `| string` and quoted dict interpolation, and found none.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

The post-pull rebuild covers it.

## Risks / concerns

- Severity: low
  - Concern: Python repr built in Python before rendering is not covered. Example: the reading Stage-2 `handoff=` text inside `user_message`.
  - Mitigation: none here. That text is passed with `| tojson`, as a quoted string.
- Severity: low
  - Concern: `| string`, `| safe` and sets still print Python repr.
  - Mitigation: no template uses them today.
- Issues: refs #3289. Closes when stance_react JSON 502s are seen to stop live.

## PR link

(added after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
