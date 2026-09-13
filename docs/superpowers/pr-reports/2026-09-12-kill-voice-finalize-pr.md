## Summary

- Kill mandatory `orion_voice_finalize` (5c). Aligned drafts with strain resolved return the exact motor text.
- Conditional `orion_response_repair` runs only when 5b is `misaligned`, `uncertain`, or `strain_unresolved`.
- Motor receives `response_policy_summary` so speech policy is not deferred to a second writer.
- `surprise_resolved` follows reflection (aligned ∧ ¬strain ∧ ¬finalize_failed), not rewrite/epsilon.
- Hub still recognizes legacy `orion_voice_finalize` status strings as finalize-phase failures.
- Env: `VOICE_FINALIZE_TIMEOUT_SEC` → `RESPONSE_REPAIR_TIMEOUT_SEC` (local `.env` synced).

## Outcome moved

Ordinary “yo / hey I’m here” turns stop getting rewritten into assumption / next-concrete-move scaffold by a mandatory voice pass. Rejected drafts still fail closed if repair fails.

## Current architecture

FCC motor → 5a → 5b → always `orion_voice_finalize` (except structured/reading_only) → outcome/closure.

## Architecture touched

- Harness finalize chain gate + repair rename (`orion/harness/finalize.py`)
- Cognition verb/prompt: delete voice, add `orion_response_repair`
- Cortex-exec agent-lane routes
- Harness-governor schema plumbing + timeout env
- Hub `_finalize_phase_error` historical reader
- Field-digester / attention comments for new lane name

## Files changed

See `git log origin/main..HEAD --stat`. Primary seams: `orion/harness/*`, `orion/cognition/verbs|prompts`, `services/orion-cortex-exec`, `services/orion-harness-governor`, `orion/hub/turn_orchestrator.py`.

## Schema / bus / API changes

- Added: `HarnessRunV1.response_repair_ran`, `response_repair_reason` (and outcome molecule mirrors)
- Removed: `finalize_overlay` as a second-pass producer concept; verb `orion_voice_finalize`
- Renamed: callable repair helpers; cortex lane string → `orion_response_repair`
- Behavior changed: `finalize_ran` means 5a/5b completed; repair is separate
- Compatibility notes: Hub maps both `orion_response_repair` and legacy `orion_voice_finalize` in `grounding_status`

## Env/config changes

- Added keys: `RESPONSE_REPAIR_TIMEOUT_SEC`
- Removed keys: `VOICE_FINALIZE_TIMEOUT_SEC`
- Renamed keys: timeout alias as above
- `.env_example` updated: yes (`services/orion-harness-governor/.env_example`)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (old key removed from live `.env`)
- skipped keys requiring operator action: none for this change

## Tests run

```text
pytest orion/harness/tests -q
# 328 passed

pytest services/orion-cortex-exec/tests/test_default_llm_route_for_step.py \
  services/orion-cortex-exec/tests/test_harness_finalize_route.py \
  services/orion-hub/tests/test_turn_orchestrator_ws_frames.py -k finalize -q
# pass

python -c "from orion.cognition.plan_loader import load_verb_yaml; ..."
# orion_response_repair loads; orion_voice_finalize FileNotFoundError
```

## Evals run

```text
pytest orion/harness/evals -q
# introspection fixture updated for run_orion_response_repair
# layer attribution: repair text change under misaligned reflection
```

## Docker/build/smoke checks

```text
Not rebuilt in this session (code + unit/gate tests only). Restart commands below after merge.
```

## Review findings fixed

- Finding: Task tool quota blocked formal per-task subagent review mid-plan
  - Fix: controller continued inline with the same task boundaries and commits
  - Evidence: `.superpowers/sdd/progress.md` + commit history on `fix/kill-voice-finalize`
- Finding: introspection fixture still pointed at `run_orion_voice_finalize`
  - Fix: updated to `run_orion_response_repair`
  - Evidence: `orion/harness/evals/fixtures/unified_turn_introspection.json`

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: Formal Task-tool subagent reviews were quota-blocked mid-plan; controller finished Tasks 4–8 inline.
- Mitigation: Focused pytest green on harness (328) + cortex-exec/hub finalize suites; live smoke after restart still needed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2208
