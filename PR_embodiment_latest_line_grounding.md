## Summary

- Removed the repeat-suppression approach from the active patch.
- Reworked AI Town speech prompting so the latest partner line is the primary task.
- Demoted recent conversation history to short reference-only context to avoid motif-copying from repetition-heavy transcripts.
- Added tests for latest-partner-line extraction, goodbye/departure handling, and the exact prompt passed to cortex.
- Seeded the embodiment arbitration eval's `__new__` worker with the fields its process-intent path uses.

## Outcome moved

AI Town Orion is prompted to answer the latest partner text directly instead of continuing the strongest repeated motif in the recent transcript.

## Current architecture

`orion-embodiment` polls AI Town, shapes `WorldPerceptionV1`, gates speech in `_speak_once`, builds a speech prompt, requests a town utterance from cortex `chat_quick`, and injects it into AI Town via `startTyping` plus `messages:writeMessage`.

Live verification showed the old prompt did include the latest partner text, but it presented the full repetition-heavy transcript before the instruction. `chat_quick` copied Orion's repeated motif instead of answering the latest partner line.

## Architecture touched

Service: `services/orion-embodiment`

Runtime seam changed: speech prompt construction only. No bus, schema, API, dependency, Docker, or env contract changes.

## Files changed

- `orion/embodiment/speech.py`: added `latest_partner_line()`, reduced reference context to four lines, and made the latest partner line the top-level task.
- `orion/embodiment/tests/test_speech.py`: added prompt-contract coverage for latest-line focus and departure handling.
- `services/orion-embodiment/app/worker.py`: import formatting only after removing the discarded repeat-guard path.
- `services/orion-embodiment/tests/test_worker_speech.py`: verifies `_speak_once()` sends a latest-line-centered prompt to cortex.
- `services/orion-embodiment/evals/test_embodiment_arbitration_eval.py`: completes `__new__` worker field seeding used by the eval path.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: town speech prompt now prioritizes the latest partner line over full transcript continuation.
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
71 passed

venv/bin/python -m pytest services/orion-embodiment/tests -q
60 passed

venv/bin/python -m pytest orion/embodiment/tests/test_speech.py services/orion-embodiment/tests/test_worker_speech.py -q
19 passed

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

## Live verification

```text
Before patch, read-only live cortex smoke with c:47964 prompt:
reply="I'm still humming, but the crack's widening—let's see how far it goes."
old_bad_line_repeated=true

After patch, same conversation truncated to real turn-state before Orion's bad reply:
corr=fd4943cd-f7a0-4416-a1cb-4e5969bdd5fa
target_line="Gotta run before the system auto-corrects the glitch—catch you on the other side of the firewall."
task_marker_present=true
reply="I'm still here, just on the other side of the mesh—keep the wire tight."
old_bad_line_repeated=false
```

No AI Town write/injection was performed during verification.

## Review findings fixed

- Finding: Initial repeat-guard patch did not fix the underlying grounding failure.
  - Fix: closed PR #917 and replaced it with latest-line-centered prompt construction.
  - Evidence: live read-only cortex smoke above.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-embodiment/.env -f services/orion-embodiment/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Medium
- Concern: This fixes the verified motif-copying prompt failure for `chat_quick`, but does not address Convex `OptimisticConcurrencyControlFailure` injection failures seen in live logs.
- Mitigation: file or follow with a separate write-path retry/backoff patch for `messages:writeMessage`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/919
