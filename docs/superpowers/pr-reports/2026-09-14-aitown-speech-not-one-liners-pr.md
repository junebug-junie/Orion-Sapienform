## Summary

- Stop forcing AI Town speech into “exactly one short spoken line.”
- Tell `chat_quick` (when `surface=aitown`) to match reply length to what the partner said, and drop the hub-style Compact / “stay compact” squeeze on that path.
- Lock both contracts with prompt/speech tests.

## Outcome moved

Town Orion can answer a substantive Juniper turn with a few spoken sentences instead of a dismissive one-liner, without turning on the heavier grounded speech lane yet.

## Current architecture

`orion-embodiment` builds a speech user prompt via `build_speech_prompt`, then dispatches `chat_quick` with `metadata.surface=aitown`. Live default keeps `EMBODIMENT_SPEECH_UNIFIED_ENABLED=false`, so the quick template is the system prompt that shapes length.

## Architecture touched

- Shared embodiment speech prompt (`orion/embodiment/speech.py`)
- Quick chat template (`orion/cognition/prompts/chat_quick.j2`)
- Embodiment README note + focused tests

## Files changed

- `orion/embodiment/speech.py`: length/substance contract
- `orion/cognition/prompts/chat_quick.j2`: aitown length + drop Compact/stay-compact
- `orion/embodiment/tests/test_speech.py`: assert old one-liner wording is gone
- `services/orion-cortex-exec/tests/test_chat_prompt_context_guardrails.py`: aitown vs hub style
- `services/orion-cortex-exec/tests/test_situation_prompt_integration.py`: pass `metadata={}` so chat_quick can render
- `services/orion-embodiment/README.md`: document speech length contract

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: prompt-only speech length guidance for aitown
- Compatibility notes: hub `chat_quick` (non-aitown) still uses Compact / stay compact

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: n/a
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=services/orion-embodiment:. pytest \
  orion/embodiment/tests/test_speech.py \
  services/orion-cortex-exec/tests/test_chat_prompt_context_guardrails.py \
  services/orion-cortex-exec/tests/test_situation_prompt_integration.py \
  -q
# 26 passed
```

## Evals run

```text
No eval harness for this prompt-length seam; structural prompt/speech tests cover the contract.
```

## Docker/build/smoke checks

```text
Not run. Prompt/template change; needs embodiment + cortex-exec-chat restart to take effect live. Live reply quality UNVERIFIED until a town turn after deploy.
```

## Review findings fixed

- Finding: IDENTITY RULES still said “stay compact” for aitown, fighting the new length contract
  - Fix: gate that line on `surface != aitown`; aitown gets substance-matching wording instead
  - Evidence: guardrail tests assert `but stay compact` absent on aitown and present otherwise

## Restart required

```bash
# From an embodiment worktree (or after merge on deploy host), rebuild/restart both:
scripts/safe_docker_build.sh orion-embodiment up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build cortex-exec-chat
# Exact compose service names may vary by host PROJECT; use the chat-lane cortex-exec that serves
# orion:cortex:exec:request:chat.
```

## Risks / concerns

- Severity: low
- Concern: prompt-only; model may still short-answer under timeout pressure (`EMBODIMENT_SPEECH_TIMEOUT_SEC`)
- Mitigation: next step remains optional grounded lane / timeout work; this PR only removes the forced one-liner contract

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2225
