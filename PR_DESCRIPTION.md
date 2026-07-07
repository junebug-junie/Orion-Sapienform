# feat(orion-thought): advisory Mind stance enrichment for the unified turn

Open PR: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/mind-unified-stance-enrichment

## Summary

- Optionally run `orion-mind` before the `stance_react` verb and inject a strict allow-listed, mode-agnostic self/attention `mind_coloring` block into the verb context — so the unified turn feels more alive without contradicting grounding or mis-framing technical/agent/coding turns.
- New `services/orion-thought/app/mind_enrichment.py`: four seams — allow-list coloring selector, light Mind snapshot builder (no cognitive-projection cold rebuild), fail-open `httpx` client, `MindRunArtifactV1` publisher (`mode="orion"`).
- Wiring in `run_stance_react` behind `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED` (default **false**); `stance_react` stays the sole author of `ThoughtEventV1` and reconciles the advisory coloring.
- Everything is flag-gated and **fails open**: Mind unconfigured/unreachable/slow/low-quality/erroring → byte-identical to prior stance behavior.
- Unified-turn Mind runs publish to the existing `orion:mind:artifact` rail tagged `mode="orion"` so the state-journaler → `mind_runs` → hub EKG lights up.

## Outcome moved

Unified-turn stance gains an advisory self/attention prior (attention frontier, reflective themes, curiosity threads, self/juniper relevance) computed by Orion's own Mind — without ceding stance authorship or leaking task-control into non-relational turns. When disabled (default) behavior is provably unchanged.

## Current architecture

Unified turn: Hub `turn_orchestrator` → `orion-thought` (RPC on `orion:thought:request`) → `orion-harness-governor`. `orion-thought` computed stance cold via `stance_react` with no Mind involvement. The legacy Brain path (cortex-orch) already runs Mind — this PR does NOT touch that path; it only borrows the HTTP-call + artifact-publish pattern.

## Architecture touched

- `services/orion-thought` only (service-bounded), plus the shared prompt `orion/cognition/prompts/stance_react.j2`, the shared bus catalog `orion/bus/channels.yaml` (producer registration), and `scripts/sync_local_env_from_example.py` (env allow-list).
- Reuses the already-registered `MindRunArtifactV1` schema and `orion:mind:artifact` channel — no new event shape.

## Files changed

- `services/orion-thought/app/mind_enrichment.py`: new module (selector, snapshot builder, fail-open HTTP client, artifact publisher, local UUID-coercion helper).
- `services/orion-thought/app/bus_listener.py`: thread `mind_coloring` through context/plan-request; `_maybe_build_mind_coloring` (flag-gated, fully fail-open); wire into `run_stance_react`.
- `orion/cognition/prompts/stance_react.j2`: advisory `PRIOR SELF-SIGNAL` block ("reconcile, do not obey"; existing inputs win; adds no output keys).
- `services/orion-thought/app/settings.py`: 9 default-off Mind settings.
- `services/orion-thought/.env_example`, `docker-compose.yml`, `requirements.txt` (`httpx>=0.27`): config/deps/compose.
- `orion/bus/channels.yaml`: add `orion-thought` to `producer_services` for `orion:mind:artifact`.
- `scripts/sync_local_env_from_example.py`: add `ORION_THOUGHT_MIND_`/`ORION_MIND_` prefixes + `CHANNEL_MIND_ARTIFACT` to the sync allow-list.
- `services/orion-thought/README.md`: feature docs + preconditions.
- Tests: `test_mind_coloring_selector.py`, `test_mind_light_snapshot.py`, `test_settings_mind_enrichment.py`, `test_mind_http_client.py`, `test_stance_context_mind_coloring.py`, `test_stance_prompt_renders_coloring.py`, `test_mind_enrichment_fail_open.py`, `test_mind_artifact_mode_tag.py`; eval `evals/test_mind_enrichment_eval.py`.

## Schema / bus / API changes

- Added: none (reuses `MindRunArtifactV1` / `orion:mind:artifact`).
- Removed: none.
- Renamed: none.
- Behavior changed: `orion-thought` becomes a second producer on `orion:mind:artifact` (registered in `channels.yaml`); artifacts tagged `request_summary_jsonb.mode = "orion"`.
- Compatibility notes: default-off; no `StanceReactRequestV1` / channel-semantics changes.

## Env/config changes

- Added keys (9): `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED`, `ORION_MIND_BASE_URL`, `ORION_THOUGHT_MIND_TIMEOUT_SEC`, `ORION_THOUGHT_MIND_WALL_MS`, `ORION_THOUGHT_MIND_ROUTER_PROFILE`, `ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES`, `ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED`, `ORION_THOUGHT_MIND_COLORING_MAX_ITEMS`, `CHANNEL_MIND_ARTIFACT`.
- Removed/renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: yes — the sync allow-list was missing these prefixes (they would never have synced), so `scripts/sync_local_env_from_example.py` was extended and the operational `services/orion-thought/.env` was synced (9 keys added, all default-safe). Verified via `should_sync_key` and an end-to-end sync run.
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/tests/ services/orion-thought/evals/ -q
# 106 passed, 4 failed
# The 4 failures (test_reverie_spontaneous_thought.py x3, test_settings_salience_flags.py::test_salience_flags_default_off)
# are PRE-EXISTING — they fail identically on the base commit (3cffe1cd) and are unrelated to this branch.
# All 36 new tests/eval-tests pass, including the fail-open + no-nested-leak regressions added in review.
```

## Evals run

```text
PYTHONPATH=services/orion-thought:. python -m pytest services/orion-thought/evals/ -q
# 4 passed (3 new anti-contradiction/aliveness + pre-existing reverie hollow-guard)
```

## Docker/build/smoke checks

```text
docker compose --env-file <root>/.env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml config
# resolves all 9 new ORION_THOUGHT_MIND_* / ORION_MIND_BASE_URL / CHANNEL_MIND_ARTIFACT env values
```

## Review findings fixed

- Finding: fail-open hole — `build_light_mind_request`/`select_mind_coloring` were called unguarded, so with the flag ON an enrichment bug would emit a failure thought.
  - Fix: wrapped the `_maybe_build_mind_coloring` body in `try/except -> return None`.
  - Evidence: new `test_enrichment_selector_raises_fails_open` (green).
- Finding: `juniper_relevance` counted toward the substance gate but was never rendered → possible near-empty-shell block.
  - Fix: render `juniper_relevance` in `stance_react.j2`.
  - Evidence: `test_block_present_with_coloring` asserts it renders.
- Finding: `orion-thought` published to `orion:mind:artifact` without being a registered producer.
  - Fix: added `orion-thought` to `producer_services` in `channels.yaml`.
  - Evidence: YAML parses; producers = `["orion-cortex-orch", "orion-thought"]`.
- Finding (minor): allow-listed string fields not coerced; redundant slice.
  - Fix: explicit `str()` coercion for `self_relevance`/`identity_salience`/`juniper_relevance`; dropped redundant `[:max_items]`.
  - Evidence: selector tests green.

**Second review round (approve — full-branch pre-merge review):**

- Finding (minor): `identity_salience` was not length-clipped like its siblings; a non-scalar value could inject an unbounded repr / nested keys.
  - Fix: added `_clip_str_or_none` used for all three scalar coloring fields — it length-bounds scalars and **drops non-scalar values (dict/list) entirely** rather than stringifying them, closing the last theoretical nested-leak vector. Also normalizes whitespace-only → `None`.
  - Evidence: new `test_scalar_fields_clipped_and_no_nested_leak` (dict `identity_salience` → `None`, no `task_mode` in blob) and `test_whitespace_only_scalar_normalizes_to_none` (both green).

## Restart required

Operator (not agent) must rebuild for `httpx` + new env, then run the rollout pre-flight:

```bash
docker compose --env-file .env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build
```

Rollout pre-flight (design §Rollout): (1) confirm `orion-mind` has `MIND_LLM_SYNTHESIS_ENABLED=true` (separate service — the only path that yields `meaningful_synthesis`); (2) confirm `orion-thought` can reach `ORION_MIND_BASE_URL`; (3) flip `ORION_THOUGHT_MIND_ENRICHMENT_ENABLED=true`; (4) manual acceptance gate — confirm at least one real unified turn produces a non-empty `meaningful_synthesis` coloring and that technical/agent turns keep their `task_mode`/`conversation_frame` (latent-LLM check, measured at rollout).

## Risks / concerns

- Severity: low — Manual acceptance gate (end-to-end `meaningful_synthesis` coloring + no chat-framing on technical turns) is latent-LLM behavior verified at rollout, not automatable here. The structural anti-contradiction guarantee is covered by the eval.
- Severity: low — Oversized-response cap is applied post-download (httpx buffers non-streaming bodies); it prevents parse/validation of oversized bodies but not full download. Acceptable for an internal trust boundary; streaming enforcement is a possible follow-up. (Only remaining Minor from the two review rounds — both reviews returned **Approve**.)
- Severity: low — `_envelope_correlation_id` is now a third identical copy (`bus_listener.py`, `reverie.py`, `mind_enrichment.py`). Consolidation into a shared helper is a recommended follow-up (deferred to avoid restructuring untouched modules in this changeset).
- Note: repo lacks the AGENTS.md-referenced `check_env_template_parity.py` / `check_schema_registry.py` / `check_bus_channels.py` gate scripts; env/schema/channel parity was verified via `sync_local_env_from_example.py`, direct registry/catalog inspection, and the full pytest suite instead.

## PR link

Open via: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/mind-unified-stance-enrichment
