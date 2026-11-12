## Summary

- Make thinking explicitly off for each `fast`-class llama.cpp request when the caller omits a choice.
- Preserve explicit per-call `enable_thinking` values and existing structured-output policy.
- Cover gateway bus, native completion, and OpenAI HTTP paths; leave metacog and other backends alone.

## Outcome moved

The gateway now constructs a request-level thinking choice for fast llama.cpp calls. This is verified in local dispatch tests and a Docker build. Live gateway behavior is `UNVERIFIED` until merge and deployment. Circe's worker already defaults to thinking off; the earlier 512-token empty-final result was from a deliberately thinking-on probe, not an ordinary quick call.

## Current architecture

Circe's Qwen3.5 fast worker serves four 4,096-token slots on port 8013 with launch-default thinking off. The separate metacog worker remains Qwen3 8B Q5 on port 8012. The gateway routes bus and OpenAI HTTP calls through GPU pool grants. Several fast callers omit `enable_thinking`, so their request payloads currently rely on the worker launch default.

## Architecture touched

The gateway adds a small shared payload helper at its two dispatch boundaries. It applies only after a `fast` class llama.cpp grant. No worker profile, pool route, or caller process changes.

## Files changed

- `services/orion-llm-gateway/app/fast_thinking.py`: copy the payload and fill an omitted `enable_thinking` with `false`.
- `services/orion-llm-gateway/app/llm_backend.py`: apply default to bus calls, including native completion probes.
- `services/orion-llm-gateway/app/passthrough_proxy.py`: apply default to OpenAI HTTP calls, streaming and nonstreaming.
- `services/orion-llm-gateway/tests/`: verify omitted and explicit choices, metacog isolation, and native probe forwarding.
- `services/orion-llm-gateway/README.md`, `docs/operations/qwen35-fast-lane-trial.md`: document behavior and the forced-thinking probe.

## Schema / bus / API changes

- Added, removed, or renamed fields: none.
- Behavior changed: omitted `chat_template_kwargs.enable_thinking` becomes `false` on fast llama.cpp grants. Explicit `true` or `false` is preserved by the gateway; the existing structured-output artifact policy can still disable thinking.
- Other pool classes and backends retain their current behavior.

## Env/config changes

- No env template or config changes. Ignored local `.env` files were used for build and parity checks only.
- `python3 scripts/check_env_template_parity.py orion-llm-gateway`: PASS.

## Tests run

```text
PYTHONPATH=services/orion-llm-gateway:. /tmp/orion-qwen35-test-venv/bin/python -m pytest -q services/orion-llm-gateway/tests/test_llm_backend.py services/orion-llm-gateway/tests/test_openai_passthrough.py
49 passed, 2 preexisting warnings
git diff --check: PASS
```

## Evals run

The already-deployed Qwen3.5 worker returned a six-token final answer with no reasoning for a default quick call. A deliberately thinking-on 512-token call exhausted its output budget; the same prompt completed at 1,024 tokens after 658 completion tokens. Gateway request-level behavior remains `UNVERIFIED` live until this patch deploys.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-llm-gateway build: PASS
```

## Review findings fixed

- Independent review found no functional issue. It flagged that explicit thinking-on is not unconditional for structured artifact requests. The README and runbook now state the existing policy exception.

## Restart required

After merge, from a clean primary `main` synchronized with `origin/main`, rebuild and restart `orion-llm-gateway` using `scripts/safe_docker_build.sh orion-llm-gateway up -d --build`. Verify a live omitted-choice quick call returns final content without reasoning, explicit thinking-on remains selectable, and metacog routing remains unchanged. This production action requires Juniper's confirmation under `AGENTS.md` section 13.

## Risks / concerns

- The 4,096-token fast context and short caller output caps can still be exhausted when a caller explicitly turns thinking on. Callers choosing thinking must allocate an adequate generation budget.
- A fast-class spill to a non-llama.cpp backend retains that backend's policy; the new default is scoped to llama.cpp grants.
- Live gateway request-level behavior is `UNVERIFIED` until deployment and smoke.

## PR link

Pending.
