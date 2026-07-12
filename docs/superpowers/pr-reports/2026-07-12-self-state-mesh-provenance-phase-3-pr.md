# PR: Self-state Phase 3 — node-attributed mesh provenance + Phase 4 research

## Summary

- **Phase 3 (mesh embodiment) code portion**: threads which node/capability contributed each diffused field channel through to `SelfStateV1`'s `reasons` field — the concrete mechanism for "my reasoning capacity is strained because Circe is down" instead of an anonymous pressure number.
- **Phase 4 (drive unification) research**: traced `CLUSTER_ROLE_WEIGHTS`/`orion-state-service` node-weighting live on the running mesh (docker logs/exec, not README claims) — found a *third* independent, previously-unflagged reimplementation of the same node-weighting question, and confirmed the arsonist doc's original consumer was itself already dark.
- Medium-effort code review (2 angles) found and fixed one real, empirically-reproduced bug in the provenance-tracking logic before it shipped.
- Two Phase 3 sub-items from the plan remain out of scope for this PR: enabling biometrics on Atlas/Circe is a remote-deployment action on physical hosts this session has no access to; revisiting the capacity-pressure-vs-continuity-threat distinction is explicitly gated on real multi-node data existing first.

## Outcome moved

Layer 6's evidence trail gains real mesh identity. Previously, a capability-level pressure number (e.g. `resource_pressure`) was anonymous even after Phase 1's evidence-channel-map fix restored raw-channel visibility — nothing said *which node* was behind it. Now, once Atlas/Circe biometrics come online (a separate operator action), self-state's `reasons` will name them by name.

## Current architecture

Before this PR: `apply_diffusion` (orion-field-digester) already applied real per-node topology edge weights (fixed in Phase 1's double-counting repair), but discarded which node fed which value the moment it accumulated into a capability channel. `collect_field_channel_pressures` further flattened node/capability vectors into one anonymous channel-name-keyed dict.

## Architecture touched

- `orion/schemas/field_state.py` — new schema field
- `services/orion-field-digester/app/digestion/diffusion.py` — provenance tracking during diffusion
- `orion/self_state/scoring.py`, `orion/self_state/builder.py` — provenance threaded into evidence
- `docs/notes/` — Phase 4 research findings

## Files changed

- `orion/schemas/field_state.py`: new `capability_provenance: dict[str, dict[str, str]]` field on `FieldStateV1` — per-tick "primary contributor" proxy, not a historical ledger (documented limitation)
- `services/orion-field-digester/app/digestion/diffusion.py`: `apply_diffusion()` tracks and records the largest single weighted contribution per `(target_id, channel)` this tick; fixed a real bug (see Review findings) where a zero-contribution edge could still claim/overwrite provenance
- `orion/self_state/scoring.py`: `collect_field_channel_pressures()` now returns `(pressures, provenance)` instead of just `pressures` — a breaking signature change, contained to its one production caller
- `orion/self_state/builder.py`: new `_provenance_label()`/`_reason_for_evidence()` helpers; `reasons` now names the contributing node when known — `dominant_evidence`'s exact `channel_name=value` format is deliberately untouched (protects `orion-spark-introspector`'s hardware-bypass parsing)
- `services/orion-field-digester/tests/test_diffusion_provenance.py`: new — 5 tests covering single-edge provenance, larger-contribution-wins, zero-contribution-doesn't-clobber (the bug fix), capability-capability edges
- `tests/test_self_state_builder.py`, `tests/test_self_state_scoring.py`: updated for the new tuple return + new acceptance test (`test_resource_pressure_reasons_name_the_contributing_node`) proving the Phase 3 goal end-to-end
- `docs/notes/2026-07-12-phase4-cluster-weighting-research.md`: new — Phase 4 research findings (see below)

## Schema / bus / API changes

- Added: `FieldStateV1.capability_provenance` (additive, `default_factory=dict`, safe for pre-existing rows/messages missing the field — verified against `extra="forbid"` semantics, which reject unexpected extras, not omitted-with-default fields)
- Behavior changed: `collect_field_channel_pressures()`'s return type (internal to `orion/self_state/`, one production caller, updated in this diff); `SelfStateV1.reasons`' text content gains an optional `(node: X)` suffix when provenance is known — `dominant_evidence`'s format is unchanged
- No bus channel or schema-registry changes — `FieldStateV1` is registered by class reference, not a hand-maintained field list

## Env/config changes

None.

## Tests run

```text
pytest tests/test_self_state_builder.py tests/test_self_state_builder_hardening.py \
  tests/test_self_state_schemas.py tests/test_self_state_scoring.py \
  tests/test_self_state_policy_loader.py tests/test_self_state_prediction.py \
  tests/test_self_state_reliability_decontamination.py tests/test_self_state_runtime_store.py \
  tests/test_self_state_transport_dimension.py tests/test_self_state_deviation.py \
  tests/test_proposal_transport_readonly_candidates.py \
  services/orion-spark-introspector/tests/test_tissue_viz_arousal.py \
  services/orion-spark-introspector/tests/test_tissue_viz_novelty.py \
  services/orion-spark-introspector/tests/test_inner_state_emit.py -q
→ 87 passed, 0 failed

pytest services/orion-field-digester/tests/ -q
→ 16 passed, 0 failed
```

`git diff --check`: clean.

## Evals run

None — no eval harness exists for the field-digester/self-state substrate beyond the unit/regression tests above (known, pre-existing gap, not introduced by this PR).

## Docker/build/smoke checks

Not run this session — a live Docker mesh **is** running on this host (confirmed via `docker ps`: `orion-athena-self-state-runtime`, `orion-athena-field-digester`, etc., all up), and research for the Phase 4 doc used it read-only (`docker logs`/`docker exec`), but no image was rebuilt/restarted with this PR's code. Restart commands below are ready whenever this merges.

## Review findings fixed

Medium-effort code review (2 of 8 angles: line-by-line scan, cross-file tracer — scoped to this diff's size).

- Finding: `apply_diffusion`'s "largest contribution wins provenance" check (`contribution >= _max_contribution.get(key, 0.0)`) used `0.0` as both the "nothing recorded yet this call" sentinel and a legitimate zero-contribution value. A single edge contributing nothing this tick (its source has no value for the mapped channel) could satisfy the comparison and silently overwrite provenance genuinely recorded on a *prior* tick — even though the accumulated value itself doesn't change (adding `0.0` is a no-op). Found and empirically reproduced by the review agent.
  - Fix: added an explicit `contribution > 0.0` guard before the comparison.
  - Evidence: new regression test `test_zero_contribution_edge_does_not_clobber_prior_tick_provenance`.
- Cross-file tracer angle: checked every caller of the changed `collect_field_channel_pressures` signature and every `FieldStateV1` constructor/deserialization path in the repo — found nothing broken. Confirmed `capability_provenance`'s `default_factory=dict` is safe under `extra="forbid"` for old data missing the field.

## Phase 4 research summary (separate from the code change above)

`docs/notes/2026-07-12-phase4-cluster-weighting-research.md` traces whether `CLUSTER_ROLE_WEIGHTS`/`orion-state-service` are a legitimate separate "ops health" concern or a duplicate of the field-topology node-weighting question. Findings:

- **A third, previously-unflagged reimplementation exists**: `orion-hub`'s own `BIOMETRICS_ROLE_WEIGHTS_JSON` fallback (`{"atlas":0.6,"athena":0.4}` — a third, different number set from `CLUSTER_ROLE_WEIGHTS`' `{"atlas":0.7,"athena":0.3,"other":0.5}`), which is the one **actually active today** as a fallback.
- `CLUSTER_ROLE_WEIGHTS`' own code path (`BiometricsHub.publish_cluster`) never runs live — `BIOMETRICS_MODE=agent`, confirmed zero `biometrics.cluster.v1` messages ingested over a 6-hour log window.
- Unlike the L7-L11 ladder, `CLUSTER_ROLE_WEIGHTS` does have one real non-display consumer when it fires: `orion-cortex-exec`'s metacognition LLM prompt templates via `metacog_biometrics_cue`. Its `spark.signal.v1`/`DriveEngine` path is a dead end (the `level` dimension is unmapped in `signal_drive_map.yaml`).
- `orion-state-service`'s aggregation answers a genuinely different question (single-writer/most-recent snapshot selection) and is **not** a duplicate — the arsonist doc's framing was imprecise on this point.
- Recommendation: before Atlas/Circe come online, derive Hub/metacog node-weighting from the field-topology file's already-declared per-node weights instead of a third bespoke scalar, and retire both `CLUSTER_ROLE_WEIGHTS` and `BIOMETRICS_ROLE_WEIGHTS_JSON`.
- Incidental bug found, not fixed here: `orion-state-service`'s `GET /state/latest` HTTP route throws a live `TypeError` (missing required kwarg) — only the bus RPC path is actually exercised.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-field-digester/.env \
  -f services/orion-field-digester/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-self-state-runtime/.env \
  -f services/orion-self-state-runtime/docker-compose.yml up -d --build
```

This PR's changes do **not** touch any dimension's `.score` formula (only `reasons` text and a new schema field) — unlike the still-open Phase 1 phi-corpus decision, there is no equivalent deployment gate here. Safe to restart once merged.

## Risks / concerns

- Severity: low
  Concern: `capability_provenance` is a "largest contribution this tick" proxy, not a historical ledger — a capability channel's accumulated value can reflect contributions from multiple nodes over many ticks, but provenance only ever names the single biggest contributor in the *current* tick's diffusion pass.
  Mitigation: documented explicitly in the schema field's docstring; acceptable scope for this phase per the plan's own "don't build a bigger feature than needed" guidance. A fuller historical ledger is a candidate follow-up if this proves insufficient once Atlas/Circe are live.
- Severity: low
  Concern: no test exists for "old `FieldStateV1` payload missing `capability_provenance` entirely" as an explicit regression (the review agent confirmed this is safe by Pydantic semantics, but didn't add a dedicated test).
  Mitigation: flagged as a minor test-coverage gap, not a live bug; cheap to add later if desired.
- Severity: informational
  The Phase 4 research surfaced a third weighting duplicate and a live HTTP bug in `orion-state-service`, neither fixed in this PR (out of scope — this PR is Phase 3 code + Phase 4 *research*, not Phase 4 *implementation*).

## PR link

<!-- filled in after push -->
