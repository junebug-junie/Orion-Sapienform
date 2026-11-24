## Summary

- Reverie questions now turn the model's thinking off and raise the answer budget to 1024 tokens. This copies the fix stance already uses.

## Outcome moved

In the first day live, 7 of 8 reveries were stored as `unparseable`. Every one of the 28 `reverie_question` calls in `llm_call_usage` ended at exactly 512 completion tokens, the gateway default. The agent lane thinks by default, so the reasoning used up the budget before any JSON was written. The one success used 474 tokens.

Truncated answers also used up the 8-a-day cap, and some were retried as lane failures, which is why there were 28 calls for 8 rows.

## Files changed

- `services/orion-thought/app/reverie.py`: the question call's context now sends `chat_template_kwargs.enable_thinking=False` and `max_tokens=1024`. cortex-exec already forwards both keys to the gateway (`executor.py` around line 4461).
- `services/orion-thought/tests/test_reverie_questions.py`: a regression test, and the context-key set is updated.

## Tests run

`pytest services/orion-thought/tests/test_reverie_questions.py services/orion-thought/tests/test_reverie_thin_import_boundary.py`: 49 passed.

## Review

This is a 6-line change that copies an existing, reviewed pattern (stance_react). I did not run a separate review subagent.

## Restart required

The post-pull rebuild rebuilds orion-thought.

## Risks / concerns

- **UNVERIFIED live** until the next quiet stretch after deploy. The check is that `llm_call_usage.completion_tokens` for verb `reverie_question` stays under 1024 and `reverie_question.refused_reason='unparseable'` drops to about 0.
- **Today's cap is already spent** by the 7 failed rows. The first real test comes after midnight UTC.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
