## Summary

- Orion's town speech now reads only the model's answer. On an empty answer it no longer falls back to the model's reasoning.
- `is_injectable` (`orion/embodiment/speech.py`) now requires at least one letter. Live 2026-10-29 (`c:609091`), Orion posted `[]`, an empty structured answer that passed the whitespace-only check.
- Both speech paths in `services/orion-embodiment/app/worker.py` move from `extract_cortex_payload_text` (reasoning fallback on by default) to `extract_cortex_answer_text`. That module's own docstring says text shown as Orion's words must use the latter.
- Live 2026-10-29 01:38 UTC (`c:608777`, Orion↔Sofia): Orion opened a chat, so there was no partner line yet, and the prompt allows an empty reply. The model reasoned its way to empty. The gateway separated `reasoning_content` (1,890 chars) correctly. Embodiment's extractor then posted that reasoning, which quotes the full speech prompt, as Orion's line.

## Outcome moved

Failure mode: model reasoning posted into AI Town as Orion speaking. It also went to social memory through the turn ingest. After this, an empty answer takes the existing `embodiment_speech_empty_reply_skipped` path.

## Current architecture

`_request_utterance` and `_request_utterance_quick` decode the cortex result and call `extract_cortex_payload_text(result)` with `include_reasoning=True` (the default).

## Architecture touched

`services/orion-embodiment/app/worker.py` and `orion/embodiment/speech.py` (`is_injectable`).

## Files changed

- `services/orion-embodiment/app/worker.py`: both call sites use `extract_cortex_answer_text`.
- `orion/embodiment/speech.py`: `is_injectable` requires a letter.
- `orion/embodiment/tests/test_speech.py`: `test_letterless_replies_are_not_speech`.
- `services/orion-embodiment/tests/test_worker_conversation_memory.py`: `test_request_utterance_quick_never_speaks_the_reasoning` (empty content plus reasoning gives `""`).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: reasoning is never spoken.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

Operator note: the live `services/orion-embodiment/.env` was set back to `EMBODIMENT_SPEECH_QUICK_LLM_ROUTE=quick_background` on 2026-10-29 to stop the leak. Set it back to `quick_embodied` after this deploys.

## Tests run

```text
pytest services/orion-embodiment/tests -> 102 passed
pytest orion/embodiment -> 101 passed
mutation: old extractor restored -> test_request_utterance_quick_never_speaks_the_reasoning fails
```

## Evals run

None. Live check after deploy: Orion's town lines contain no prompt text, and an empty answer logs `embodiment_speech_empty_reply_skipped`.

## Docker/build/smoke checks

Not deployed.

## Review findings fixed

(filled after review)

## Restart required

After merge, from the clean primary main checkout on athena:

1. `git pull --ff-only` (rebuilds orion-embodiment).
2. Set `EMBODIMENT_SPEECH_QUICK_LLM_ROUTE=quick_embodied` in `services/orion-embodiment/.env`.
3. `scripts/safe_docker_build.sh orion-embodiment up -d --force-recreate`

## Risks / concerns

- Severity: low. Concern: the leaked line (speech prompt and reasoning, no memory or private data) remains in the town transcript of `c:608777` and in `social_room_turns`. Mitigation: removal is a production write and awaits Juniper.
- Severity: low. Concern: an opener with no partner line asks the model whether to speak at all, which invites long reasoning. Mitigation: follow-up; the empty-reply path now handles it.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
