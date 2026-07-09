## Summary

- Added normalized town-speech repeat detection for AI Town Orion utterances.
- Taught the embodiment worker to compare a candidate reply against both the in-memory last injected line and Orion's latest transcript line.
- Blocked punctuation/case/spacing-only repeats before AI Town message injection.
- Added regression tests for same-process repeats, post-restart transcript repeats, and stale-memory transcript precedence.
- Seeded the embodiment arbitration eval's `__new__` worker with the fields its process-intent path uses.

## Outcome moved

Orion no longer injects the same canned AI Town line over and over when the model returns an unchanged reply to later partner turns.

## Current architecture

`orion-embodiment` polls AI Town, shapes `WorldPerceptionV1`, gates speech in `_speak_once`, requests an utterance from cortex, and injects it via `startTyping` plus `messages:writeMessage`. Before this patch, speech avoided empty replies and obvious self-echo when Orion authored the last town message, but it did not reject a repeated generated line after a newer partner response.

## Architecture touched

Service: `services/orion-embodiment`

Runtime seam changed: town speech injection gate only. No bus, schema, API, dependency, Docker, or env contract changes.

## Files changed

- `orion/embodiment/speech.py`: added normalized utterance comparison helpers.
- `orion/embodiment/tests/test_speech.py`: covered whitespace/case/punctuation repeat normalization.
- `services/orion-embodiment/app/worker.py`: tracked last injected line per conversation and compared candidate replies against memory plus transcript before injection.
- `services/orion-embodiment/tests/test_worker_speech.py`: added repeated-reply regression coverage.
- `services/orion-embodiment/evals/test_embodiment_arbitration_eval.py`: completed `__new__` worker field seeding used by the eval path.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: repeated AI Town speech candidates are skipped before injection.
- Compatibility notes: no payload shape or channel changes.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not required; no `.env_example` changed
- skipped keys requiring operator action: none

## Tests run

```text
venv/bin/python -m pytest orion/embodiment/tests -q
70 passed

venv/bin/python -m pytest services/orion-embodiment/tests -q
62 passed

git diff --check
passed
```

## Evals run

```text
venv/bin/python -m pytest services/orion-embodiment/evals -q
1 passed
```

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-embodiment/.env -f services/orion-embodiment/docker-compose.yml config
passed

docker compose --env-file .env --env-file services/orion-embodiment/.env -f services/orion-embodiment/docker-compose.yml build
passed
```

## Review findings fixed

- Finding: Repeat detection preferred stale in-memory state over a newer transcript line.
  - Fix: compare candidate replies against both in-memory and transcript last-spoken text.
  - Evidence: `test_repeated_reply_checks_transcript_even_with_stale_memory`; `services/orion-embodiment/tests` passed.
- Finding: punctuation-only variants could repeat.
  - Fix: normalize punctuation out for repeat comparison.
  - Evidence: `test_repeated_utterance_normalizes_case_and_spacing`; `orion/embodiment/tests` passed.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-embodiment/.env -f services/orion-embodiment/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low
- Concern: This blocks exact normalized repeats only, not paraphrased canned variants.
- Mitigation: The guard is deterministic and narrow; paraphrase-level suppression can be added later with a small similarity threshold if live traces show that pattern.

## PR link

TBD after push/PR creation.
