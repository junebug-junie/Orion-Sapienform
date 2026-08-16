# PR #1699 -- tension-driven mutating dispatch, no operator gate

https://github.com/junebug-junie/Orion-Sapienform/pull/1699

## Summary

- Wires the merged, read-only `orion.attention.tension` deviation gate into a live, mutating execution-dispatch path -- the piece the sensing-layer spec (2026-08-14) deliberately deferred ("wiring tension to anything that acts is a separate patch requiring proposal sign-off").
- New live producer (`services/orion-field-digester/app/digestion/tension.py`) runs the gate once per digestion tick, persisting baselines on `FieldStateV1` across restarts.
- `deviation_pressure` becomes a 5th `PRESSURE_DIMENSIONS` entry, cleared CLAUDE.md 0A's metric gate on 41,973 real ticks.
- 3 new proposal templates score on it and route to already-shipped-but-never-wired verbs: `observe_tension_via_camera` (read-only, the perception-ladder "consumer" wiring), `prune_dangling_images`, `prune_stopped_containers` (real mutation).
- **No `operator_review` gate anywhere in this path**, per explicit instruction. Safety comes from the existing three independent gates (`mode.allow_mutating_dispatch`, `maintenance_bounded` scope, each skill's own runtime gate).
- No perception-ladder file touched -- `look_at_camera` is referenced by verb name only.

## Outcome moved

Orion's internal deviation state can now become a real, routed, mutating proposal without an operator-approval step anywhere in the path -- closing the gap the tension arc was chartered to close. Verified end-to-end by hand: a synthetic tension spike drives `action_warrant: warranted`, a real proposal, and a real route resolution to `skills.runtime.image_prune.v1` under `maintenance_bounded` scope.

## Current architecture

- `orion.attention.tension`: pure, read-only package, no live producer.
- `PRESSURE_DIMENSIONS`: closed 4-element set, each backed by a per-tick precision EWMA baseline.
- `execution_dispatch_policy.v1.yaml`'s `template_to_cortex` override (shipped 2026-08-13): inert, `{}`.
- `skills.runtime.image_prune.v1` / `skills.runtime.docker_prune_stopped_containers.v1`: shipped mutating verbs, no live template routed to either.
- `skills.perception.look_at_camera.v1`: shipped read-only verb, unreachable (no `allowed_scope` declared anywhere).

## Architecture touched

- `orion/attention/tension/{deviation_gate,competition,__init__}.py` -- persistence contract (`export_baselines`/`import_baselines`) and `deviation_pressure()` derivation, added to the existing pure package.
- `orion/schemas/field_state.py` -- 4 new fields (`tension_baseline_mu/_var/_n`, `tension_deviation_pressure`), same pattern as `dimension_precision_ewma*`.
- `services/orion-field-digester/app/digestion/tension.py` (new) + `app/tensor/update_rules.py` -- the live producer, wired into the real digestion tick.
- `orion/field/pressure.py`, `orion/proposals/scoring.py`, `orion/proposals/templates.py` -- consumer wiring.
- `config/proposals/proposal_policy.v1.yaml`, `config/execution_dispatch/execution_dispatch_policy.v1.yaml` -- 3 templates, 3 routes.

## Files changed

- `orion/attention/tension/deviation_gate.py`: `export_baselines()`/`import_baselines()` -- the persistence contract a live producer needs, that the offline measurement scripts never did.
- `orion/attention/tension/competition.py`: `DEVIATION_PRESSURE_SATURATION`, `deviation_pressure()`, delegated persistence on `FieldTensionCompetition`.
- `orion/schemas/field_state.py`: 4 new fields, bounded cardinality (4 nodes x 33 channels), not unbounded per-event growth.
- `services/orion-field-digester/app/digestion/tension.py` (new): the live producer, `lru_cache`'d direction-map load.
- `services/orion-field-digester/app/tensor/update_rules.py`: wired before the existing precision-baseline update.
- `orion/field/pressure.py`: `deviation_pressure` injected into `field_pressures_with_provenance()`, always present (0.0 = honest "nothing admitted").
- `orion/proposals/scoring.py`: `PRESSURE_DIMENSIONS` + derived variance floor + full metric-gate evidence inline; `_LEGACY_EMPTY_DIMENSIONS_FALLBACK` decoupling (code-review fix, see below).
- `orion/proposals/templates.py`: `"maintain"` added to the `ProposalKind` type hint (hygiene) + copy for the 3 new templates.
- `config/proposals/proposal_policy.v1.yaml`: 3 templates, `dimension_weights.deviation_pressure`, `limits.max_suppressed` 10->20.
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: 3 `template_to_cortex` entries, disclosed skill-gate asymmetry.
- 4 new test files, 6 pre-existing test files updated where this patch's real behavior change needed a new, correct assertion.
- `docs/superpowers/specs/2026-08-16-tension-driven-mutating-dispatch-design.md`: full design, metric gate, corrections, acceptance checks.

## Schema / bus / API changes

- Added: `FieldStateV1.tension_baseline_mu/_var/_n`, `tension_deviation_pressure`. Backward-compatible (Pydantic `default_factory`, degrades to cold start on an upgraded persisted row -- tested).
- Added: `PRESSURE_DIMENSIONS` gains `deviation_pressure`; `DIMENSION_PRECISION_MIN_VARIANCE` gains its derived floor (assertion keeps them in sync, enforced at import time).
- No bus channel changes. No removed or renamed fields.
- Compatibility notes: `field_pressures()` output now always includes `deviation_pressure` (unlike the other 4 dimensions, which are genuinely absent some ticks) -- disclosed, tested, and 4 pre-existing tests updated to assert the new correct behavior.

## Env/config changes

- No env keys added, removed, or renamed. `.env`/`.env_example` untouched -- confirmed via `git status --short` and `git diff --check`.

## Tests run

```
/tmp/r4venv/bin/python -m pytest <28 relevant test files/dirs across orion/attention/tension,
  orion/proposals, orion/execution_dispatch, orion/field, orion/schemas, and the field-digester
  producer> -q
389 passed, 1 warning (unrelated pydantic deprecation warning) in 98s
```

Broader `tests/` sweep run to check for collateral damage; the ~125 collection errors encountered are pre-existing environment gaps in this venv (`ModuleNotFoundError: numpy/sqlalchemy/psycopg2/requests/app`), confirmed unrelated by tracing each error's import chain -- none touch a file this patch modified.

## Evals run

No dedicated eval harness for this seam. The design doc's "Acceptance checks" section (5 checks, including a by-hand end-to-end smoke test) serves as the eval for this patch; not automated into a committed eval script in this pass.

## Docker/build/smoke checks

Not run -- no Docker/runtime config touched, no new dependency, no port/health-check change. The live producer wiring was verified via direct Python import + call against the real shipped YAML configs (see design doc's "Acceptance checks" #5), which exercises the same code path `run_digestion_tick()` calls in production without needing the full field-digester container (an unrelated pre-existing `requests` import gap in this dev venv blocks importing `run_digestion_tick` directly, traced and confirmed unrelated to this patch).

## Review findings fixed

- Finding: `PRESSURE_DIMENSIONS` gaining `deviation_pressure` silently widened `_pressure_dimension_ids()`'s empty-`dimensions` fallback, so 5 already-shipped templates that never declared `deviation_pressure` would have started scoring urgency/confidence off it anyway.
  - Fix: decoupled the fallback into `_LEGACY_EMPTY_DIMENSIONS_FALLBACK`, frozen to the original 4 dimensions.
  - Evidence: `tests/test_proposal_scoring.py::test_empty_dimension_template_fallback_excludes_deviation_pressure` (new), asserts all 5 templates' resolved dimension set excludes `deviation_pressure`.
- Finding: the live producer reconstructed `FieldTensionCompetition()` with its default `directions` every tick, re-parsing `config/attention/channel_direction_map.yaml` from disk roughly every ~2s, forever.
  - Fix: `@lru_cache(maxsize=1)` around `load_direction_map()` in the producer module; gate baselines still rehydrate fresh from `state` every tick (correctness-load-bearing), only the static config is cached.
  - Evidence: `services/orion-field-digester/tests/test_tension_pressure_baseline.py::test_direction_map_is_cached_not_reloaded_every_tick`.
- Finding: this PR's own comments/design doc overstated `docker_prune_stopped_containers`' safety gate as equivalent to `image_prune`'s real measured disk/count threshold; it has no such floor, only an env flag + label/age filters.
  - Fix: corrected in both yaml config files and the design doc -- disclosed as a real, larger blast radius for that specific route, not softened.
  - Evidence: `config/proposals/proposal_policy.v1.yaml` (prune_dangling_images comment), `config/execution_dispatch/execution_dispatch_policy.v1.yaml` (template_to_cortex comment), design doc's §4 and "Corrections found by code review" section.

## Restart required

```
No restart required for this PR alone -- the new producer step only takes effect once
services/orion-field-digester is rebuilt/restarted from this branch:

docker compose \
  --env-file .env \
  --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml \
  up -d --build orion-field-digester
```

Juniper: please run this restart command yourself per repo policy (no `sudo`/deploy commands run by the agent).

## Risks / concerns

- Severity: **high** (by design, per explicit instruction). Concern: real, unattended Docker image/container deletion can now fire without an operator-approval step. Mitigation: three independent gates unchanged from the existing `prune_build_cache` precedent (`mode.allow_mutating_dispatch`, `maintenance_bounded` scope allowlist, each skill's own runtime gate) -- this patch adds routes behind an already-open gate, it does not open a new one or widen `hard_blocks`.
- Severity: medium. Concern: `prune_stopped_containers` has no resource-pressure or count floor of its own (disclosed above) -- it will prune every matching exited container once triggered and dispatched, not only when disk pressure is genuinely high. Mitigation: none built in this patch; flagged as a real, larger blast radius than `prune_dangling_images` for Juniper's awareness before this merges.
- Severity: low. Concern: `deviation_pressure`'s scoring weights (`0.30` dimension weight, `0.55`/`0.35`/`0.25` per-template weights), `DEVIATION_PRESSURE_SATURATION = 10.0`, and `limits.max_suppressed = 20` are all disclosed, uncalibrated starting values. Mitigation: same discipline as every other constant in this arc -- revisit against real post-deploy dispatch data.
- Severity: low. Concern: node-level target binding (proposal target = the tension Borda winner) was named as a real capability but deliberately not built this patch (fixed targets used instead). Mitigation: explicitly logged as a non-goal in the design doc, not silently dropped.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/tension-driven-dispatch
