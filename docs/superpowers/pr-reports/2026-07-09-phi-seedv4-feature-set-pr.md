# feat(phi): add seed-v4 trainable feature set (spec 2/3)

Branch: `feat/phi-seedv4-feature-set`
Compare: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/phi-seedv4-feature-set

## Summary

- Re-version the phi cognitive-encoder trainable feature set from `seed-v3` to `seed-v4` (additive, opt-in via `INNER_FEATURES_VERSION=seed-v4`; shipped default stays `seed-v3`).
- Excise the proven-frozen SelfStateV1 theater trio (`coherence`, `continuity_pressure`, `social_pressure`) from the trainable vector — still recorded in `infra` for provenance, never scaled/trained on.
- Drop the structurally-sparse `exec_step_fail_rate`/`execution_friction` cognitive slots (0.3% signal in production).
- Add real `execution_load`, `reasoning_load`, `reasoning_present` cognitive slots sourced from orion-thought's `reasoning_activity` projection (spec 1 of this initiative), each independently `.none`-tagged with a truthful zero when its source is dark — never a fake floor.
- `execution_load` falls back to `log1p(summed execution_trajectory step_count)` when the reasoning_activity projection itself is dark.

## Outcome moved

`encoder_trainable_feature_names("seed-v4")` now resolves to `agency_readiness, execution_pressure, reasoning_pressure, overall_intensity, recall_gate_fired, reasoning_present, execution_load, reasoning_load` — 8 live dims, replacing 2 dead cognitive dims and 3 frozen felt dims with real, currently-live signal.

## Current architecture

`services/orion-spark-introspector/app/inner_state.py` computed a seed-v3 trainable vector = 6 felt dims (incl. 3 now-proven-frozen) + `overall_intensity` + 4 cognitive slots sourced from the `execution_trajectory` projection, 2 of which (`exec_step_fail_rate`, `execution_friction`) carry near-zero live signal.

## Architecture touched

- `services/orion-spark-introspector/app/inner_state.py` — `SEEDV4_THEATER_FELT`, `SEEDV4_COGNITIVE_FEATURE_NAMES`, `encoder_trainable_feature_names()` seed-v4 branch, `_active_trajectory_runs`/`_recall_gate_fired` helpers factored out of `cognitive_features_from_trajectory` (seed-v3 behavior byte-identical, regression-tested), new `cognitive_features_seed_v4()` builder, `build_inner_state_features()` widened infra-routing + `include_cognitive` conditions and new `reasoning_activity_projection` param.
- `services/orion-spark-introspector/app/substrate_reads.py` — `ReasoningActivitySnapshot`, `fetch_reasoning_activity()` (mirrors `fetch_execution_trajectory`'s fail-closed pattern), `SubstrateReadCache.put_reasoning_activity`/`get_reasoning_activity`.
- `services/orion-spark-introspector/app/settings.py` + `.env_example` — new `ORION_THOUGHT_BASE_URL` setting (default `http://orion-athena-thought:7155`).
- `services/orion-spark-introspector/app/worker.py` — cached `_fetch_orion_thought_reasoning_activity()` fetcher, threaded into `handle_self_state`.

## Files changed

- `services/orion-spark-introspector/app/inner_state.py`: seed-v4 feature set, cognitive slot builder, shared helper factoring
- `services/orion-spark-introspector/app/substrate_reads.py`: `fetch_reasoning_activity` + cache slots
- `services/orion-spark-introspector/app/settings.py`: `orion_thought_base_url` setting
- `services/orion-spark-introspector/.env_example`: `ORION_THOUGHT_BASE_URL` key
- `services/orion-spark-introspector/app/worker.py`: cached reasoning-activity fetch wired into `handle_self_state`
- `services/orion-spark-introspector/tests/test_inner_state_seed_v4.py`: new — seed-v4 feature set, dark/live/fallback cases, refactor regression
- `services/orion-spark-introspector/tests/test_substrate_reads.py`: `fetch_reasoning_activity` + cache tests
- `services/orion-spark-introspector/tests/test_inner_state_emit.py`: `handle_self_state` seed-v4 integration test + reasoning-activity mocks added to existing tests

## Schema / bus / API changes

- Added: none (no schema/bus changes — `InnerFeatureV1`/`InnerStateFeaturesV1` untouched).
- Removed: none.
- Renamed: none.
- Behavior changed: `features_version="seed-v4"` rows carry a different trainable vector shape than `seed-v3`; `seed-v3`/legacy rows unchanged.
- Compatibility notes: additive re-version. `scripts/fit_phi_encoder.py`'s `input_features_for_version` needed no code change (generic passthrough), verified by test.

## Env/config changes

- Added keys: `ORION_THOUGHT_BASE_URL=http://orion-athena-thought:7155` (spark-introspector)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced: `python scripts/sync_local_env_from_example.py` did not pick up this key (it's not in the sync script's prefix/exact allowlist, and that script is out of this patch's scope) — added `ORION_THOUGHT_BASE_URL=http://orion-athena-thought:7155` to the local `services/orion-spark-introspector/.env` by hand, mirroring the existing `SUBSTRATE_RUNTIME_URL` line
- skipped keys requiring operator action: none

## Tests run

```
venv/bin/python -m pytest services/orion-spark-introspector/tests -q
# 97 passed, 1 pre-existing failure (test_phi_reward_emitted_when_encoder_ok,
# confirmed present on main before this branch via git stash — unrelated), 1 skipped

venv/bin/python -m pytest tests/test_corpus_gate.py tests/test_inner_state_trajectory_features.py tests/test_inner_state_features.py tests/test_inner_state_schema.py -q
# 28 passed
```

## Orchestrator review pass (post-subagent, before merge)

Read every touched file against the spec, ran the full suite, then closed one
gap the subagent had correctly flagged but left out of scope:

- **Fixed** (commit `b8feebbe`): `handle_self_state`'s corpus-health gate
  hardcoded the seed-v3 `COGNITIVE_FEATURE_NAMES` tuple regardless of
  `features_version`. Now branches on `inner.features_version` and uses
  `SEEDV4_COGNITIVE_FEATURE_NAMES` for seed-v4 rows. Verified: under today's
  sourcing logic this was structurally inert (execution_load/reasoning_load's
  liveness is causally coupled to recall_gate_fired/reasoning_present's, so
  the two name sets always produced the same accept/reject verdict — proved
  by reverting the fix and confirming the new regression test
  `test_handle_self_state_corpus_gate_uses_seedv4_cognitive_names` fails
  without it, passes with it) — but it's a real latent gap that would
  silently weaken corpus hygiene the moment either signal's sourcing logic
  changes independently, so closed now rather than left as a landmine.
- Also fixed the `scripts/sync_local_env_from_example.py` allowlist gap for
  `ORION_THOUGHT_BASE_URL` (same recurring failure mode as the two prior
  pushes on this initiative) and a minor import-order cleanup in `worker.py`.

## Evals run

No dedicated eval harness exists for spark-introspector's phi feature set beyond the corpus/promote gates in `scripts/fit_phi_encoder.py`, which are out of scope for this patch (spec explicitly defers the fit/backfill run to a follow-up once `PUBLISH_REASONING_TELEMETRY` is live and accruing real data).

## Docker/build/smoke checks

Not run — no runtime config/dependency/port changes; pure Python feature-computation logic + one new outbound HTTP read (fail-closed, cached, matches existing `fetch_execution_trajectory` pattern).

## Review findings fixed

- Finding: `cognitive_features_seed_v4`'s `reasoning_load` branch used `isinstance(x, (int, float)) and x` (truthy) to validate `thinking_tokens_sum`, which (a) let a negative value reach `math.log1p`, raising `ValueError: math domain error` and violating the function's own "never raises" contract, and (b) treated Python `bool` (a subclass of `int`) as a valid numeric count, so a JSON `true` would fabricate a nonzero `reasoning_load` instead of the mandated truthful zero.
  - Fix: replaced with an explicit `isinstance(..., (int, float)) and not isinstance(..., bool) and thinking_tokens_sum > 0` guard.
  - Evidence: `test_seed_v4_reasoning_load_negative_thinking_tokens_never_raises`, `test_seed_v4_reasoning_load_bool_thinking_tokens_not_treated_as_number` in `test_inner_state_seed_v4.py`, both passing.
- Finding (fixed by orchestrator, commit `b8feebbe`, after subagent correctly flagged it out of its authorized scope): `handle_self_state`'s `is_corpus_row_healthy(inner, cognitive_feature_names=COGNITIVE_FEATURE_NAMES)` call was hardcoded to the seed-v3 4-name tuple regardless of `features_version`. Now selects `SEEDV4_COGNITIVE_FEATURE_NAMES` when `inner.features_version == "seed-v4"`. See "Orchestrator review pass" above for the inertness analysis and regression test.
- Finding (reviewed, not changed — deliberate, disclosed): the refactored `_active_trajectory_runs` helper wraps `datetime.fromisoformat` in a `try/except Exception: continue`, where the original inline seed-v3 loop had no such guard (a malformed `last_updated_at` would previously raise and propagate out of `handle_self_state`). This is a genuine behavior difference from strict byte-identical, but it converts a crash into the same truthful-zero degrade-gracefully pattern already used everywhere else in this file (matches CLAUDE.md's "never raise on absent input — degrade gracefully" mandate and the existing fail-closed convention in `substrate_reads.py`). Kept intentionally; noted here rather than silently changed.
- Finding (reviewed, not changed — matches task's explicit instructions verbatim): `handle_self_state` now calls `_fetch_orion_thought_reasoning_activity()` unconditionally every tick, even for `seed-v3`/`seed-v2` rows that never consume the result. This was specified verbatim in the task brief (no version gate requested) and is low-severity: fail-closed, 2s cache, 2s timeout, matches the existing `fetch_execution_trajectory` pattern.

## Restart required

```text
No restart required for this PR by itself (seed-v4 is opt-in; default INNER_FEATURES_VERSION stays seed-v3). To exercise seed-v4 in a running deployment, set INNER_FEATURES_VERSION=seed-v4 and ORION_THOUGHT_BASE_URL for the orion-spark-introspector service, then:

docker compose \
  --env-file .env \
  --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml \
  up -d --build orion-spark-introspector
```

## Risks / concerns

- Severity: low
  - Concern: unconditional reasoning_activity HTTP fetch on every self-state tick, even when unused (seed-v3 default).
  - Mitigation: fail-closed, cached (2s TTL), 2s timeout — matches existing pattern; explicit task requirement.

## PR link

Not opened — `gh` CLI is unauthenticated in this environment. Branch pushed to `origin/feat/phi-seedv4-feature-set`; open at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/phi-seedv4-feature-set
