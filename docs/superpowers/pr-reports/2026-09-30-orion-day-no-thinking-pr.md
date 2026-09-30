## Summary

- The Orion's Day letter's two model calls (note, carry-forward) now send `chat_template_kwargs.enable_thinking=False`, the repo's standard switch (same as curiosity supervisor, pre-turn appraisal, concept relation).
- One-line change in `runner._call_verb_text`, whose only caller is the `orion_day.letter` durable graph. Test pins it.

## Outcome moved

First live letter (run `orion-day-2026-09-28-1`, 2026-09-30) got its GPU hold, then failed `verb_truncated_at_max`: gateway corr `a50f4fd6-1d2e-4b1c-a363-55fa2ca4b898` shows `completion_tokens=12000 finish_reason=length`, `reasoning_len=32070`, content empty. Every attempt would repeat this, so the letter could never be written.

## Current architecture

`_call_verb_text` built a `CortexClientRequest` with `policy_dispatch_only`, route, and `gpu_lease`; cortex-orch passes `options` through and cortex-exec forwards `chat_template_kwargs` to the gateway (`executor.py` `_fwd_key` loop). Qwen3.8-27B thinks by default and reasoning counts against `max_tokens`.

## Architecture touched

- `services/orion-durable-runs/app/runner.py`: `chat_template_kwargs={"enable_thinking": False}` in `_call_verb_text` options.
- `services/orion-durable-runs/tests/test_verb_text.py`: asserts it.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
pytest services/orion-durable-runs/tests -q  -> 246 passed, 71 skipped (Postgres; CI runs them)
mutation: removing the option -> test_verb_text fails (1 failed, 7 passed)
```

## Evals run

No new eval; `orion/orion_day/evals` checks prompt separation/grounding, unaffected.

## Docker/build/smoke checks

Live evidence of the failure above. Fix UNVERIFIED live until durable-runs is rebuilt; the waiting run resumes on the new code.

## Review findings fixed

See PR comments.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build   # from main after merge
```

## Risks / concerns

- Severity: low. Concern: without thinking, note quality may be less planned. Mitigation: read the first letters; revert is one line, or raise `LLM_ORION_DAY_NOTE_MAX_TOKENS` instead.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
