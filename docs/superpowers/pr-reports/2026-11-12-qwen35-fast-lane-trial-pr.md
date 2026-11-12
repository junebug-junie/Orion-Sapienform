## Summary

- Added a Circe GPU3 fast-worker profile for Unsloth Qwen3.5 9B UD-Q5_K_XL.
- Preserved fast's four 4,096-token slots and 8,192-token generation cap; left metacog's profile and route unchanged.
- Made the fast worker default to thinking off while allowing `chat_template_kwargs.enable_thinking` per call.
- Added a direct worker eval and a caller census/runbook for a narrow rollout.

## Outcome moved

The exact GGUF booted in an isolated GPU3 shadow server and passed default/off/on thinking, four simultaneous requests, JSON, logprobs, and tool-call checks. Production fast remains on the old model until Circe's local `.env` selects this profile and `atlas-fast` is redeployed.

## Current architecture

Circe's `atlas-fast` llama.cpp worker serves pool role `fast` at port 8013; `atlas-metacog` separately serves port 8012. Both use GPU3. The gateway already forwards per-call `chat_template_kwargs` to llama.cpp. The fast worker currently runs Qwen3 8B Q4 with four 4k slots; metacog runs Qwen3 8B Q5 with four 4k slots.

## Architecture touched

Only the llamacpp-host profile/launch policy, its local selection template, tests, eval, and operations docs changed. No bus/schema or gateway contract changed.

## Files changed

- `config/llm_profiles.yaml`: exact Unsloth source and fast profile with four 4k slots and request-overridable thinking.
- `services/orion-llamacpp-host/.env_example`: new fast profile selection.
- `services/orion-llamacpp-host/app/thinking_policy.py`, `app/main.py`: avoid a launch-wide zero budget and a misleading warning for explicit `reasoning=auto`.
- `services/orion-llamacpp-host/tests/`: profile/launch checks and correction of a stale metacog expectation; no metacog runtime change.
- `services/orion-llamacpp-host/evals/run_qwen35_fast_eval.py`: direct compatibility eval.
- `services/orion-llamacpp-host/README.md`, `docs/operations/qwen35-fast-lane-trial.md`: usage, observed callers, rollout watch, and rollback.

## Schema / bus / API changes

- Added: one model profile.
- Removed: none.
- Renamed: none.
- Behavior changed: the new profile accepts request-level thinking on/off without a global zero budget.
- Compatibility notes: gateway request options already forward `chat_template_kwargs`; metacog-class grants can spill to fast under the existing pool policy.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: `ATLAS_FAST_PROFILE_NAME` selects the new profile.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: run; this helper excludes llamacpp-host from its default list, so local ignored host `.env` files were also synced manually. Circe's deployed `.env` still selects the old profile and needs an explicit one-key change at deployment.
- Skipped keys requiring operator action: none for llamacpp-host.

## Tests run

```text
PYTHONPATH=services/orion-llamacpp-host:. /tmp/orion-qwen35-test-venv/bin/python -m pytest -q services/orion-llamacpp-host/tests/test_thinking_policy.py services/orion-llamacpp-host/tests/test_qwen35_fast_profile.py services/orion-llamacpp-host/tests/test_profile_forwarding.py
30 passed
python3 scripts/check_gpu_pool_config.py: PASS
python3 scripts/check_env_template_parity.py orion-llamacpp-host: PASS
git diff --check: PASS
```

## Evals run

```text
ssh circe python3 - --base-url http://127.0.0.1:18013 < services/orion-llamacpp-host/evals/run_qwen35_fast_eval.py
PASS: 4 x 4096 slots, parallel isolation, thinking on/off, strict JSON, topic labels, logprobs, tools
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-llamacpp-host --compose-file docker-compose.atlas-workers.yml build atlas-fast: PASS
Circe llama.cpp b10398 shadow /health: ok, /slots: 4 x 4096
Circe GPU3 with shadow + existing fast + metacog + Kev: 26,421 MiB used, 6,074 MiB free
Shadow stopped; GPU3 returned to 19,491 MiB used, 13,004 MiB free
scripts/safe_graphify_update.sh: unavailable on this host; /mnt/storage-warm/orion-graphify is not mounted/writable
```

## Review findings fixed

- Finding: structured-output thinking-off claim was too broad.
  - Fix: document the exact schema plus policy condition.
  - Evidence: `structured_output.py` condition and corrected runbook.
- Finding: eval sent explicit thinking kwargs for every call, so it missed the launch default.
  - Fix: add a no-kwargs request that asserts no reasoning content.
  - Evidence: revised shadow eval passed.
- Finding: `.env_example` alone does not change Circe's local profile selection; README examples conflicted.
  - Fix: state the exact Circe `.env` edit and correct the examples.
  - Evidence: runbook and README now agree.
- Finding: new profile initially lowered `n_predict` relative to old fast.
  - Fix: retain 8,192 to isolate the model swap.
  - Evidence: profile and launch test assert the value.

## Restart required

After merge, on Circe's clean primary `main` synchronized with `origin/main`, set only `ATLAS_FAST_PROFILE_NAME=qwen35-9b-udq5kxl-v100-32gb-circe-fast-4k` in `services/orion-llamacpp-host/.env`, then run:

```bash
scripts/safe_docker_build.sh orion-llamacpp-host --compose-file docker-compose.atlas-workers.yml up -d --build atlas-fast
```

Verify `/props` reports `Qwen3.5-9B-UD-Q5_K_XL.gguf`, `/slots` remains four 4096-token slots, and the caller rails in the runbook move.

## Risks / concerns

- Severity: medium. Concern: a weak `json_object` prompt sometimes returned fenced JSON despite `finish_reason=stop`. Mitigation: strict prompts passed; monitor cortex-orch annotation parse/fallback logs and topic-foundry artifacts.
- Severity: medium. Concern: a thinking-on call with only 128 output tokens produced reasoning but empty final content. Mitigation: callers opting into thinking must allocate enough generation budget and room inside 4k.
- Severity: medium. Concern: 24-hour pre-turn probes reached 4,082 prompt tokens, near the 4,096 slot limit. Mitigation: track context overflows and pre-turn probe failures after rollout.
- Severity: low. Concern: existing metacog-class fallback can route some grants to fast. Mitigation: include those grants in fast monitoring; metacog worker is unchanged.
- Severity: medium. Concern: production fast model is UNVERIFIED until deployment and caller smoke.

## PR link

Pending PR creation.
