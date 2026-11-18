## Summary

- Orion's autonomy knobs in Hub and the GPU pool had one value written in code and a different one in the config template (`.env_example`) and in production. If a key ever went missing from a live `.env`, Orion's behaviour would silently change (for example the curiosity daily cap would fall from 7 to 3, the cooldown would jump from 30 minutes to 4 hours, and the GPU shed lever would quietly disarm).
- Every behavioural code default in `services/orion-hub/app/settings.py` and `services/orion-gpu-pool/app/settings.py` now equals `.env_example`, which equals the live `.env` and the running container. Same for docker-compose `${KEY:-x}` fallbacks, which are the *real* default for any key compose lists explicitly.
- `scripts/check_settings_defaults.py` gains `--example-drift`; CI (`orion-static-gates.yml`) now fails when a Hub or GPU-pool default drifts from its template again.
- Three keys were deliberately not decided (see Risks): `HUB_PROPOSAL_REVIEW_ENABLED`, `GPU_POOL_ORION_SHED_ENABLED`, and `CHAT_HISTORY_LOG_CHANNEL` (equivalent by construction).
- No live behaviour changes: every value written into code is the value production already runs.

## Outcome moved

Failure mode closed: "a missing `.env` key silently changes an autonomy knob". Before this patch, `check_settings_defaults.py orion-hub --example-drift` against main reports 54 drifted behavioural defaults/fallbacks in Hub and 4 in the GPU pool; after, 0 (with 40 host-specific keys and 3 reasoned keys exempt, each exemption self-expiring).

## Current architecture

- `scripts/check_settings_defaults.py` only checked that orion-actions' Settings fields had *a* default, not that it matched the template.
- `scripts/check_env_template_parity.py` compares the live `.env` to `.env_example` at deploy time; it never looks at code defaults.
- Hub's compose file carries ~187 `${KEY:-x}` fallbacks; for those keys the compose fallback, not the Settings default, is what a missing key produces.

## Architecture touched

- Services: orion-hub (settings defaults, compose fallbacks, tests), orion-gpu-pool (settings default, compose fallback, `.env_example` comment).
- Gate: `scripts/check_settings_defaults.py --example-drift`, CI step in `.github/workflows/orion-static-gates.yml`.
- No bus, schema, or API changes.

## Files changed

- `scripts/check_settings_defaults.py`: new `--example-drift` mode. Parses each Settings field's `.env_example` value through the field's own pydantic type and compares it to the code default; also compares compose `${KEY:-x}` literals (nested `${A:-${B}}` skipped). `HOST_SPECIFIC_EXAMPLE_KEYS` (URLs, paths, DSNs, location) and `REASONED_DRIFT` (key -> reason) are exempt, and an exemption that is no longer drifted fails the check.
- `tests/test_check_settings_defaults.py`: drifted fixture fails, aligned fixture passes, compose-fallback drift fails, stale exemption fails, unparseable value fails, JSON/report-only, plus a regression test that real orion-hub and orion-gpu-pool stay aligned.
- `.github/workflows/orion-static-gates.yml`: runs the tests and the check for both services.
- `services/orion-hub/app/settings.py`: 36 defaults aligned (list below); stale "off by default" comments updated.
- `services/orion-hub/docker-compose.yml`: 18 fallbacks aligned.
- `services/orion-gpu-pool/app/settings.py`, `docker-compose.yml`, `.env_example`: `GPU_POOL_SHED_ENABLED` default true; comment no longer claims "code default false".
- `services/orion-hub/README.md`: `ENABLE_PRE_TURN_APPRAISAL` default `true`.
- `services/orion-hub/tests/conftest.py` + 5 test files: new `pre_turn_appraisal_off` fixture for tests that drive the legacy post-turn substrate path with fakes that have no appraisal RPC (they silently depended on the old `false` default).
- `services/orion-hub/tests/test_curiosity_contractor_peer_flag.py`: asserts the new `True` default.

### Keys aligned (old code default -> now; `.env_example`, live `.env`, and container all already equal the new value)

orion-hub Settings:
- `HUB_CURIOSITY_INVESTIGATION_DAILY_CAP` 3 -> 7
- `HUB_CURIOSITY_INVESTIGATION_MIN_COOLDOWN_SEC` 14400 -> 1800
- `HUB_CURIOSITY_SELF_INQUIRY_DAILY_CAP` 3 -> 7
- `HUB_CURIOSITY_SELF_INQUIRY_ENABLED` false -> true
- `HUB_CURIOSITY_INVESTIGATION_ENABLED` false -> true
- `HUB_CURIOSITY_INVESTIGATION_WINDOW_START_HOUR` / `_END_HOUR` 8 / 22 -> -1 / -1 (no window)
- `HUB_CURIOSITY_OUTREACH_ENABLED`, `HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED`, `HUB_CURIOSITY_DREAM_HYPOTHESES_ENABLED`, `HUB_CURIOSITY_DURABLE_ADMISSION_ENABLED` false -> true
- `HUB_ENDOGENOUS_OUTREACH_ENABLED` false -> true; `HUB_ENDOGENOUS_OUTREACH_TZ` UTC -> America/Denver
- `SUBSTRATE_AUTONOMY_ENABLED`, `_APPLY_ENABLED`, `_COGNITIVE_PROPOSALS_ENABLED`, `_ROUTING_APPLY_ENABLED`, `SUBSTRATE_REVIEW_SCHEDULER_ENABLED`, `SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE` false -> true
- `SUBSTRATE_STORE_BACKEND` sparql -> falkor
- `ORION_UNIFIED_TURN_ENABLED`, `ORION_HARNESS_GOVERNOR_ENABLED`, `ENABLE_PRE_TURN_APPRAISAL` false -> true
- `HUB_AGENT_CLAUDE_ENABLED`, `HUB_AGENT_CLAUDE_MCP_ENABLED`, `HUB_AGENT_CURIOSITY_HINT_ENABLED`, `HUB_AITOWN_ENABLED`, `GRAPHITI_ENABLED`, `WORLD_PULSE_UI_FIXTURE_RUN_ENABLED`, `ORION_BUS_ENFORCE_CATALOG` false -> true
- `HUB_RECALL_SHADOW_EVAL_TIMEOUT_SEC` 20 -> 45
- `ORION_SITUATION_WEATHER_PROVIDER` stub -> openmeteo
- `WHISPER_MODEL_SIZE` distil-medium.en -> distil-small.en; `WHISPER_COMPUTE_TYPE` float16 -> float32
- `SERVICE_VERSION` 0.3.0 -> 0.4.0

orion-hub compose fallbacks: the curiosity contractor/dream/durable-admission flags, `HUB_AGENT_CLAUDE(_MCP)_ENABLED`, `HUB_AITOWN_ENABLED`, `GRAPHITI_ENABLED`, `AUTONOMY_GOAL_EXECUTION_ENABLED`, the four `SUBSTRATE_AUTONOMY_*` flags, `SUBSTRATE_REVIEW_SCHEDULER_ENABLED`, `WORLD_PULSE_UI_FIXTURE_RUN_ENABLED` false -> true; `HUB_AGENT_CONTEXT_EXEC_ENABLED` true -> false (orion-context-exec is not deployed; settings, template, and live were already false); `ORION_SITUATION_WEATHER_PROVIDER` stub -> openmeteo; `CORTEX_GATEWAY_REQUEST_CHANNEL` / `CORTEX_GATEWAY_RESULT_PREFIX` `orion-cortex-gateway:*` -> `orion:cortex:gateway:*`.

orion-gpu-pool: `GPU_POOL_SHED_ENABLED` false -> true (settings and compose).

Live-value evidence: for every key above, a script compared the primary checkout's `services/<svc>/.env` and `docker exec orion-athena-{hub,gpu-pool} env` against `.env_example` and printed only equality booleans (no values, so no secrets); all were equal.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: only what happens when a key is *missing* from `.env`; nothing changes for the current production `.env`.
- Compatibility notes: a fresh host or bare test run with no `.env` now gets production behaviour (flags on), per the standing "feature flags ship ON" rule.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: `services/orion-gpu-pool/.env_example` (comment only)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, run from the primary checkout root; no keys added for hub or gpu-pool; no keys skipped. (Its pre-existing "diverged" list for other services is unrelated and untouched.)
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_check_settings_defaults.py -q                         -> 24 passed
python scripts/check_settings_defaults.py orion-hub --example-drift      -> OK (exit 0)
python scripts/check_settings_defaults.py orion-gpu-pool --example-drift -> OK (exit 0)
same check pointed at unmodified main                                    -> exit 1 for both, naming
    HUB_CURIOSITY_INVESTIGATION_DAILY_CAP default=3 .env_example=7, the cooldown, the self-inquiry cap,
    and GPU_POOL_SHED_ENABLED (settings + compose) -- the gate catches the original incident
pytest services/orion-gpu-pool/tests -q                                   -> 155 passed, 15 skipped
pytest services/orion-hub/tests -q (branch)                               -> 39 failed, 3374 passed, 86 skipped
pytest services/orion-hub/tests -q (main, detached worktree, same commit base)
                                                                          -> 39 failed, 3374 passed, 86 skipped
    failure sets are identical (diffed by test id); all 39 are pre-existing on main
```

## Evals run

```text
None. This is a config-default alignment plus a deterministic static gate; the gate's own tests are the
evidence. No eval harness applies.
```

## Docker/build/smoke checks

```text
Not run: no image or runtime change for production. Compose fallbacks only apply when a key is missing
from services/<svc>/.env, and every changed key is present there with the same value.
```

## Review findings fixed

Review ran in a subagent against `origin/main...HEAD`. It found no blocking issues and no test regressions (it also confirmed the 39 hub failures are already on main).

- Finding: compose fallbacks were collected one value per key (last one wins), and comment lines were scanned too. A drifted first occurrence could hide behind a matching later one.
  - Fix: comment lines are skipped and every occurrence is compared.
  - Evidence: `test_example_drift_compose_comment_and_duplicate_occurrences`.
- Finding: fields could silently drop out of the comparison. `AliasChoices` aliases, `env_prefix` and un-aliased lowercase fields under `case_sensitive=False` were all missed.
  - Fix: names are resolved the way pydantic-settings resolves them, and the lookup is upper-cased when the settings are case-insensitive.
  - Evidence: `test_example_drift_lowercase_unaliased_field_is_compared`, `test_example_drift_env_prefix_and_alias_choices`.
- Finding: `_same` crashed on unhashable values. `_unquote` left the quotes on a value written as `"x" # note`.
  - Fix: an explicit check for None or empty string; the quoted token is taken first.
  - Evidence: `test_example_drift_false_is_not_unset`, `test_example_drift_default_factory_and_none_vs_empty`.
- Finding: three hub README lines still stated the old defaults.
  - Fix: updated `SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED`, `SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED` and `HUB_AGENT_CURIOSITY_HINT_ENABLED`.
  - Evidence: README diff.
- Finding: the conftest settings sweep called `getattr` on every module in `sys.modules`.
  - Fix: it now only looks at the `scripts`, `app` and `orion` packages and uses `vars()`.
  - Evidence: the 30 focused hub tests behave the same as before.
- Finding (not fixed, scope narrowed): a bare compose `${KEY}` with no fallback overrides the code default with `""` when the key is missing. About 50 hub keys are like this. Code that calls `os.getenv` directly is also invisible to the check; `SUBSTRATE_STORE_BACKEND` is read that way, so changing its Settings default has no runtime effect.
  - Fix: both gaps are written down in the check's scope comment as known gaps. Follow-up needed.
  - Evidence: `scripts/check_settings_defaults.py` scope comment.

## Restart required

```text
No restart required. Production already runs every value now written into code.
```

## Risks / concerns

- Severity: medium
- Concern: `HUB_PROPOSAL_REVIEW_ENABLED` is on in `.env_example` and live, but its API (orion-context-exec, :8096) is not deployed and nothing listens on 8096. That looks like a mistake, so its code default stays `false` and it is listed in `REASONED_DRIFT` for Juniper to decide (turn the live flag off, or redeploy context-exec).
- Mitigation: the gate prints it on every run; once decided, the stale-exemption rule forces the entry out.

- Severity: medium
- Concern: `GPU_POOL_ORION_SHED_ENABLED` (Orion's own self-shed) is on in `.env_example` and live, but `.env_example` records it as "Juniper's call; code default stays off". A recorded decision, so not auto-flipped.
- Mitigation: listed in `REASONED_DRIFT` with that reason.

- Severity: low
- Concern: with no `.env` at all, Hub now defaults to production autonomy (substrate autonomy apply, curiosity outreach, unified turn). That is the intended "flags ship ON" posture, but anyone booting Hub without a `.env` gets live behaviour, not a quiet one.
- Mitigation: kill switches are unchanged; set the key to `false` in `.env`. `GRAPHITI_ENABLED` now defaults on while `GRAPHITI_ADAPTER_URL` defaults empty. Whether the crystallization path does nothing safely on an empty URL is UNVERIFIED. AI Town's empty Convex URL returns a 400, so that case is handled.

- Severity: low
- Concern: two gaps the check cannot see: bare compose `${KEY}` without a fallback (about 50 hub keys), and readers that call `os.getenv` directly.
- Mitigation: written down in the check's scope comment; follow-up.

- Severity: low
- Concern: `tests/test_substrate_effect_pipeline.py::test_pipeline_handles_internal_failure_without_raising` fails on main and on this branch (pre-existing). With appraisal defaulting on it briefly passed for the wrong reason; the new fixture pins appraisal off so it fails honestly again.
- Mitigation: follow-up, out of scope.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2592

🤖 Generated with [Claude Code](https://claude.com/claude-code)
