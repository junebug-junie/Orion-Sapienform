# PR: Self-state/mesh substrate redesign — Phase 0 (hygiene) + Phase 1 (evidence integrity)

## Summary

- Full archaeology of the overlapping self-state/drive/phi/mesh-weighting metrics substrate (`docs/notes/2026-07-12-metrics-swamp-arsonist-review.md`), a phased redesign spec (`docs/superpowers/specs/2026-07-12-self-state-mesh-substrate-redesign-design.md`), and implementation plan (`docs/superpowers/plans/2026-07-12-self-state-mesh-substrate-redesign.md`), plus the L6-L11 flow now documented in `services/orion-substrate-runtime/README.md`.
- **Phase 0 (hygiene):** removed the dead `policy_pressure` dimension (hardcoded `0.0` forever, zero producer); replaced `SelfStateV1`'s templated per-dimension `reasons` string with real evidence-derived text; committed a previously-uncommitted NO-GO measurement-gate verdict for `endogenous_origination.py`; disambiguated it from the unrelated live `endogenous_runtime.py`; applied an already-decided-but-never-applied flag flip disabling autonomous concept clustering.
- **Phase 5 research (parallel, read-only):** confirmed the L7-L11 substrate ladder (proposal→policy→execution-dispatch→feedback→consolidation) terminates in Hub debug tiles with no live consumer; confirmed the phi encoder never reads `SelfStateV1.confidence`, only `.score` — informing Phase 1's risk assessment.
- **Phase 1 (evidence integrity):** fixed a real bug where 11 raw channels double-counted against their own topology-edge-weighted diffusion (making per-node mesh weights functionally inert); real per-dimension confidence from channel count + agreement, replacing a uniform proxy; promoted `transport_integrity` from display-only to an actual contributor to `overall_intensity`.
- **Review round (high-effort, 5 of 8 angles dispatched, cross-confirmed findings):** caught and fixed a real regression the Phase 1 fix introduced in `orion-spark-introspector`'s tissue-viz hardware bypass, a second regression where two dimensions would report `confidence=0.0` forever, a naming collision, and several smaller cleanups.

## Outcome moved

`SelfStateV1` (Layer 6) — Orion's literal self-model substrate — goes from a shape that overstated its own evidentiary grounding (uniform confidence, templated reasons, a dead dimension, topology edge weights silently defeated by double-counting) toward one that's honest about what it does and doesn't know, without breaking any of the ~15 known downstream consumers or the live phi encoder's training distribution in ways that weren't explicitly flagged.

## Current architecture

Before this PR: `orion-self-state-runtime` recomputed `SelfStateV1` every 2s from `substrate_field_state` + attention frames, with a metric shape that had accumulated three independent problems (documented in the arsonist review): a dead dimension, uniform per-dimension metadata, and per-node topology weights that were declared in `config/field/orion_field_topology.v1.yaml` but never actually took effect at Layer 6 because raw and diffused channel values competed via `max()` for the same dimension.

## Architecture touched

- `orion/self_state/` (schema, builder, scoring, policy) — the core L6 synthesis
- `orion/proposals/scoring.py` — a live consumer of the dimension set
- `orion/autonomy/endogenous_origination.py`, `services/orion-cortex-exec/app/endogenous_runtime.py` — docstring-only disambiguation, no logic change
- `services/orion-spark-concept-induction/.env_example` — one flag flip
- `services/orion-spark-introspector/app/worker.py` — the phi/tissue-viz consumer whose hardware bypass this PR both broke and then fixed within the same session
- `services/orion-substrate-runtime/README.md` — new L6-L11 architecture documentation
- `config/self_state/self_state_policy.v1.yaml` — the actual tuning surface for most of this change

## Files changed

- `docs/notes/2026-07-12-metrics-swamp-arsonist-review.md`: new — full evidence-grounded inventory of overlapping metric systems (self-state, DriveEngine, AutonomyStateV2, the L7-L11 ladder, three disconnected mesh-weighting schemes)
- `docs/notes/2026-07-12-phase5-research-findings.md`: new — L7-L11 real-behavior trace + phi encoder input-surface trace + an orchestrator addendum re-verifying the phi-impact caveat against the actual landed Phase 1 diff
- `docs/superpowers/specs/2026-07-12-self-state-mesh-substrate-redesign-design.md`: new — north star, design invariants, target shape, non-goals
- `docs/superpowers/plans/2026-07-12-self-state-mesh-substrate-redesign.md`: new — 6-phase implementation plan (this PR ships phases 0-1)
- `docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md`: appended the 2026-07-08 measurement-gate NO-GO verdict (previously only in `/tmp/autonomy-gate/report.md`)
- `services/orion-substrate-runtime/README.md`: new "Downstream of this service: Layers 6-11" section — flow diagram, per-dimension formula table, evolution mechanics, known-issues list
- `orion/schemas/self_state.py`: removed `policy_pressure` from the `dimension_id` enum
- `orion/self_state/builder.py`: removed `policy_pressure` from `DimensionId`/`ALL_DIMENSION_IDS`/hardcoded score; real per-dimension `reasons`; real per-dimension `confidence` via `channel_dimension_confidence()` with a `COMPOSITE_DIMENSION_IDS` fallback to `overall_confidence` for `field_intensity`/`agency_readiness`/`transport_integrity`; `transport_integrity` computed before `overall_intensity` so its policy weight actually applies; `evidence_for_dimension()` now unions `channel_dimension_map` and the new `evidence_channel_map`
- `orion/self_state/scoring.py`: `weighted_overall_intensity()` skips (not zero-defaults) a weighted dimension absent from `dimension_scores`; new `channels_mapped_to_dimension()` shared filter; new `channel_dimension_confidence()` (renamed from an initial `dimension_confidence()` to avoid colliding with a pre-existing `orion.proposals.scoring.dimension_confidence`); new `COMPOSITE_DIMENSION_IDS`
- `orion/self_state/policy.py`: new `evidence_channel_map: dict[str, str]` field
- `config/self_state/self_state_policy.v1.yaml`: removed 11 double-counted channels from `channel_dimension_map`/`pressure_channels`; added those same 11 back under a new `evidence_channel_map` (evidence-only, does not feed score); added `transport_integrity: 0.05` to `dimension_weights` (funded by trimming `reliability_pressure`/`social_pressure` slightly)
- `orion/proposals/scoring.py`: removed `policy_pressure` from `PRESSURE_DIMENSIONS`
- `services/orion-spark-introspector/app/worker.py`: fixed the `policy_ease` dead reference to a removed schema key (was already behaviorally a no-op, now an explicit constant with a comment instead of a phantom `.get()`)
- `services/orion-cortex-exec/app/endogenous_runtime.py`: new module docstring disambiguating from the unrelated `endogenous_origination.py`
- `orion/autonomy/endogenous_origination.py`: extended module docstring with NO-GO status, why it's kept (not deleted — `bus_worker.py` still eagerly imports it behind an always-false flag), and the disambiguation note
- `services/orion-spark-concept-induction/.env_example` (+ local `.env`, not tracked): `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED` `true` → `false`
- Test files: `tests/test_self_state_builder.py`, `tests/test_self_state_builder_hardening.py`, `tests/test_self_state_policy_loader.py`, `tests/test_self_state_reliability_decontamination.py`, `tests/test_self_state_scoring.py`, `tests/test_self_state_transport_dimension.py`, `tests/test_proposal_transport_readonly_candidates.py`, `services/orion-spark-introspector/tests/test_tissue_viz_arousal.py`, `test_tissue_viz_novelty.py`, `test_inner_state_emit.py` — updated fixtures for the removed dimension/channels, plus new regression tests for every fix listed above

## Schema / bus / API changes

- Removed: `SelfStateDimensionV1.dimension_id` no longer accepts `"policy_pressure"`
- Added: `SelfStatePolicyV1.evidence_channel_map` (config schema field, not a bus/API contract)
- Behavior changed: `SelfStateDimensionV1.confidence` is now genuinely per-dimension (previously uniform); `.reasons` is now evidence-derived (previously a fixed template); `resource_pressure`/`execution_pressure`/`reliability_pressure` scores change value on any tick where the previously-double-counted channels were active (see Risks below)
- No bus channel or schema-registry changes — this is entirely internal to `SelfStateV1`'s existing shape plus one new policy-config field
- Compatibility: additive-first per the redesign's own design invariant — no consumer needs to change for `SelfStateV1` to remain valid; `dominant_evidence`/`reasons`/`confidence` change in *content*, not shape

## Env/config changes

- Added keys: none (policy YAML is not an env template)
- Removed keys: none
- Changed: `services/orion-spark-concept-induction/.env_example`: `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false`
- `.env_example` updated: yes
- local `.env` synced: yes, done directly (this key isn't in `sync_local_env_from_example.py`'s default whitelist, flagged and handled manually)
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_self_state_builder.py tests/test_self_state_builder_hardening.py \
  tests/test_self_state_schemas.py tests/test_self_state_scoring.py \
  tests/test_self_state_policy_loader.py tests/test_self_state_prediction.py \
  tests/test_self_state_reliability_decontamination.py tests/test_self_state_runtime_store.py \
  tests/test_self_state_transport_dimension.py tests/test_proposal_transport_readonly_candidates.py \
  services/orion-spark-introspector/tests/test_tissue_viz_arousal.py \
  services/orion-spark-introspector/tests/test_tissue_viz_novelty.py \
  services/orion-spark-introspector/tests/test_inner_state_emit.py tests/test_inner_state_features.py \
  orion/autonomy/tests/test_endogenous_origination.py \
  orion/spark/concept_induction/tests/test_endogenous_origination_wiring.py -q
→ 111 passed, 0 failed
```

New regression tests added this session: dead-dimension removal + real-reasons generation (Phase 0); double-counting fix + per-dimension confidence variance + `weighted_overall_intensity` skip-vs-zero-default (Phase 1); composite-dimension confidence fallback + evidence-channel-map restoration + an end-to-end test running the real `build_self_state()` pipeline to prove the tissue-viz hardware bypass survives (review round).

`git diff --check`: clean. `.env` confirmed ungitignored-but-unstaged throughout.

## Evals run

None. This work touches `orion-self-state-runtime`'s scoring substrate but no eval harness exists for it beyond the unit/regression tests above — flagged as a known gap, not addressed in this PR.

## Docker/build/smoke checks

Not run — no live Docker environment available in this session. Restart commands below are for the operator to run.

## Review findings fixed

High-effort code review (5 of 8 planned angles dispatched — line-by-line scan, removed-behavior audit, cross-file tracer, reuse, simplification; efficiency/altitude/conventions angles and formal verification were not run, session moved straight to fixing after two independently-cross-confirmed regressions surfaced).

- Finding: Phase 1's `channel_dimension_map` fix silently removed 11 raw channels from `dominant_evidence`, breaking `orion-spark-introspector`'s `_hardware_resource_pressure()`/`_execution_load_pressure()` bypass (built for a real 2026-07-10 incident) — both would always return `None` and fall back to the raw, still-saturatable score. Found independently by two review angles via different methods (static reasoning vs. empirical trace).
  - Fix: new `evidence_channel_map` policy section restores the 11 raw channels to `dominant_evidence`/`reasons` only, explicitly not read by score computation.
  - Evidence: new end-to-end test `test_hardware_bypass_survives_real_build_self_state_pipeline` running the real `build_self_state()` pipeline, plus `test_evidence_channel_map_restores_raw_channel_without_double_counting_score`.
- Finding: `field_intensity`/`agency_readiness` (synthesized dimensions, never channel-mapped) would report `confidence=0.0` forever under the new formula — a permanent structural falsehood, not an honest "no evidence this tick" — feeding a live `proposal_confidence()` calculation for two proposal templates.
  - Fix: new `COMPOSITE_DIMENSION_IDS` set; these dimensions (plus `transport_integrity`, structurally the same case) fall back to `overall_confidence`.
  - Evidence: new test `test_composite_dimension_confidence_falls_back_to_overall_confidence`.
- Finding: the new `orion.self_state.scoring.dimension_confidence` collided by name with a pre-existing, differently-shaped `orion.proposals.scoring.dimension_confidence` in a sibling package.
  - Fix: renamed to `channel_dimension_confidence`.
  - Evidence: grep confirmed no test or caller referenced the old name directly.
- Finding: `evidence_for_dimension` and the new confidence function independently re-implemented the same `channel_dimension_map` filtering loop (cross-confirmed by two angles).
  - Fix: extracted shared `channels_mapped_to_dimension()`.
  - Evidence: both call sites now delegate to it; full suite re-run green.
- Finding: `services/orion-spark-introspector/app/worker.py:278` read `dimensions.get("policy_pressure")`, a key removed in Phase 0 — currently benign only because the `_s()` default happened to reproduce the prior always-zero value.
  - Fix: replaced with an explicit constant and a comment explaining why, removing the phantom schema dependency.
  - Evidence: `test_tissue_viz_arousal.py` suite unchanged and passing.
- 2 findings not fixed (documented, non-blocking, both single-angle, lower severity): `dimension_confidence()`'s `n==1` branch simplified per one suggestion but the deeper "reimplements `DeviationGate`'s variance-based confidence" observation (a different, more principled statistical approach already exists in `orion/autonomy/deviation_gate.py`) was not adopted — scoped as a possible Phase 2 improvement, not a Phase 1 blocker, since it would change the confidence formula's actual output distribution and needs its own consideration rather than a same-session swap. `_emit_summary_labels`'s post-hoc `transport_summary_labels` set-union (minor, could be folded into the single-pass label builder) left as-is — cosmetic, no behavior change either way.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-self-state-runtime/.env \
  -f services/orion-self-state-runtime/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```

`orion-proposal-runtime` imports `orion/proposals/scoring.py` — restart it too if running (same pattern, `services/orion-proposal-runtime/docker-compose.yml`).

## Risks / concerns

- Severity: **high — do not restart `orion-self-state-runtime` in production until this is resolved.**
  Concern: the double-counting fix genuinely changes the `.score` distribution of `agency_readiness`, `execution_pressure`, `reasoning_pressure`, and `overall_intensity` — exactly the four `SelfStateV1`-sourced features the live phi encoder (seedv4 weights) trains on. The fix is correct (it repairs topology edge weights that were functionally inert), but the encoder was trained on the pre-fix, double-counted distribution.
  Mitigation: three options recorded in `docs/notes/2026-07-12-phase5-research-findings.md`'s addendum (accept-and-document / version-pin a drift gate / retrain the corpus) — deliberately left as an open decision for Juniper, not defaulted by this PR. Code is correct and fully tested; deploy-readiness is a separate gate.
- Severity: medium
  Concern: no eval harness exists for `orion-self-state-runtime`'s scoring substrate; this PR is covered by unit/regression tests only.
  Mitigation: flagged as a known gap; building one is out of scope for this PR.
- Severity: low
  Concern: `channel_dimension_confidence()`'s agreement formula (max-min spread) is cruder than `orion/autonomy/deviation_gate.py`'s existing variance/EWMA-based approach to the same "confidence from count+spread" problem — two different formulas for a conceptually similar signal now exist in the codebase.
  Mitigation: not adopted this session (would change output distribution, needs its own consideration); named as a candidate Phase 2 improvement.
- Severity: low
  Concern: this PR was developed directly on `main` for several commits before being moved to a branch (an process gap on my part) — `origin/main` had also moved in the meantime (2 unrelated file-deletion commits). No content conflict; branch created cleanly from the tip of that local work. Local `main` currently still sits 6 commits ahead of a stale point and would need a `git reset --hard origin/main` to realign once this PR merges — not done in this session pending your confirmation, since that's a destructive-shaped operation on `main` specifically.
  Mitigation: after this PR merges, run `git fetch && git checkout main && git reset --hard origin/main` to realign local `main` — flagging rather than doing it unprompted.

## PR link

<!-- filled in after `gh pr create` -->
