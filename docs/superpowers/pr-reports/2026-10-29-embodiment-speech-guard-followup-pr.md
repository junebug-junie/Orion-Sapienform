## Summary

- Follow-up to #2955. These review fixes were pushed after that PR merged.
- `is_injectable` accepts letters or digits (a reply of `42` is speech) and rejects plumbing error text through the canonical `looks_like_error_text`.
- A junk grounded reply (`[]`, error text) now falls back to the quick path instead of skipping the turn.

## Outcome moved

Two failure modes. Orion would post an error report (`[Error: llamacpp timed out…]`) as speech, and would drop a numeric answer.

## Current architecture

On main, `is_injectable` requires a letter (#2955). Error text with letters passes, and digit-only replies are dropped. `_request_utterance` returns any non-blank grounded reply.

## Architecture touched

`orion/embodiment/speech.py`, `services/orion-embodiment/app/worker.py`.

## Files changed

- `orion/embodiment/speech.py`: `isalnum`, plus `looks_like_error_text`.
- `services/orion-embodiment/app/worker.py`: grounded reply gated by `is_injectable`.
- `orion/embodiment/tests/test_speech.py`: digits and error-text cases.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: speech guard only.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: town speech is OFF in the live `services/orion-embodiment/.env` (`EMBODIMENT_SPEECH_ENABLED=false`, since 2026-10-29) because of the reasoning/`[]` leak. Turn it back on after this deploys.

## Tests run

```text
pytest orion/embodiment -> 101 passed
pytest services/orion-embodiment/tests -> 102 passed
```

## Evals run

None.

## Docker/build/smoke checks

Not deployed.

## Review findings fixed

These are the #2955 review's findings 2–4 (see the #2955 report).

## Restart required

After merge, from the clean primary main checkout on athena:

1. `git pull --ff-only`
2. In `services/orion-embodiment/.env`: set `EMBODIMENT_SPEECH_ENABLED=true` and `EMBODIMENT_SPEECH_QUICK_LLM_ROUTE=quick_embodied`.
3. `scripts/safe_docker_build.sh orion-embodiment up -d --force-recreate`

## Risks / concerns

- Severity: low. Concern: junk with letters (`null`, `*stays silent*`) still passes. Mitigation: follow-up on the speech prompt's "reply with an empty string".

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2961

🤖 Generated with [Claude Code](https://claude.com/claude-code)
