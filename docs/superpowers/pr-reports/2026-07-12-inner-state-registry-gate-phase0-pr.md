## Summary

- Phase 0 of the inner-state unification plan: a registry (`orion/self_state/inner_state_registry.py`) enumerating every "what does Orion currently feel/perceive" signal in the repo, plus a deterministic gate (`scripts/check_inner_state_registry.py`) that fails if the registry goes stale or a new inner-state-shaped schema/channel appears unregistered.
- This is the foundational fix for a failure mode independently rediscovered five separate times in one session: a real, computed signal silently duplicating another one, or never reaching cognition (DriveEngine/AutonomyStateV2, `CLUSTER_ROLE_WEIGHTS` x3, the phi/heuristic split, `FieldAttentionFrameV1`'s discarded per-node scores).
- Running the gate for the first time against the real repo surfaced 16 real findings, each individually triaged (not blanket-suppressed) — see "Review findings fixed" below.
- Also found, documented, and explicitly NOT fixed as scope creep: `make agent-check` and 2 of its 3 referenced scripts don't exist despite CLAUDE.md describing them as real.
- README updates: new `orion/self_state/README.md` (the package had none); a new section in `services/orion-self-state-runtime/README.md`.

## Outcome moved

The next inner-state-shaped signal added anywhere in the repo either gets a registry entry or fails a deterministic gate — not a sixth manual grep-archaeology discovery weeks later.

## Current architecture

Nine signals across six services (`orion-field-digester`, `orion-attention-runtime`, `orion-self-state-runtime`, `orion.spark.concept_induction`, `orion.autonomy`, `orion-spark-introspector`, `orion-biometrics`, plus the 5-service L7-L11 ladder) each independently answer some version of "what does Orion currently feel." No prior mechanism declared which was canonical, which reached cognition, or which duplicated another.

## Architecture touched

`orion/self_state/` (new registry module + README), `scripts/` (new gate script), `tests/` (new gate tests), `Makefile` (new target), `services/orion-self-state-runtime/README.md`. No runtime behavior changed — this is bookkeeping infrastructure, not a code path any service executes.

## Files changed

- `orion/self_state/inner_state_registry.py` (new): `Cadence`/`CompositionStatus` enums, `InnerStateSignal` frozen dataclass with validation (`DUPLICATE` requires `duplicate_of`, `SHADOW` requires `shadow_reason`), and all 9 current signals populated with real producer/cadence/consumer data traced from the design spec.
- `scripts/check_inner_state_registry.py` (new): rot check (every registered schema still imports as a real `BaseModel`) + new-duplicate heuristic (keyword scan of `orion/bus/channels.yaml` and `orion/schemas/registry.py`'s flat name dict, cross-checked against the registry plus a documented `_EXTRA_COVERED_SCHEMA_NAMES` exclusion list).
- `tests/test_inner_state_registry_gate.py` (new): 8 tests — rot-check pass/fail, heuristic pass/fail against both the real repo and synthetic fixtures, `InnerStateSignal` validation, full gate exit code.
- `Makefile`: new `check-inner-state-registry` target.
- `orion/self_state/README.md` (new), `services/orion-self-state-runtime/README.md`: documentation.
- `docs/superpowers/specs/2026-07-12-inner-state-unification-design.md`, `docs/superpowers/plans/2026-07-12-inner-state-unification-plan.md`: the design spec and phased plan this implements (written earlier this session, committed here for the first time).

## Schema / bus / API changes

None. No existing schema, channel, or API changed shape.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. pytest tests/test_inner_state_registry_gate.py -q
8 passed

PYTHONPATH=. pytest tests/test_self_state_runtime_store.py tests/test_self_state_deviation.py \
  tests/test_self_state_transport_dimension.py tests/test_self_state_builder.py \
  tests/test_self_state_reliability_decontamination.py tests/test_self_state_prediction.py \
  tests/test_self_state_policy_loader.py tests/test_self_state_builder_hardening.py \
  tests/test_self_state_scoring.py tests/test_self_state_schemas.py \
  tests/test_inner_state_registry_gate.py -q
62 passed

python scripts/check_inner_state_registry.py
inner_state_registry gate OK (9 entries checked)
```

## Evals run

None applicable — this is deterministic bookkeeping infrastructure, not a behavior/quality question.

## Docker/build/smoke checks

Not applicable — no service, no runtime behavior change, nothing to deploy.

## Review findings fixed

The gate's first real run against the repo surfaced 16 matches with no registry entry. Each was individually traced and resolved, not blanket-suppressed:

- Finding: `SelfStateDimensionV1`, `FieldAttentionTargetV1`, `DriveAuditV1`, `DriveNodeV1` matched keywords with no registry entry.
  - Resolution: confirmed each is a sub-object/companion of an existing registry entry (per-dimension row, per-target row, audit-trail companion, graph-node projection respectively) — added to `_EXTRA_COVERED_SCHEMA_NAMES` with the specific relationship documented, not just suppressed.
- Finding: `ArticleClusterV1` matched the `cluster` keyword.
  - Resolution: traced to `orion/schemas/world_pulse.py` — news-article clustering for topic-foundry, a genuine keyword collision unrelated to node-health clustering. Documented as a false positive, not silently ignored.
- Finding: `AttentionFrameV1`, `AttentionSignalV1`, `AttentionSalienceTraceV1`, `AttentionLoopOutcomeV1`, `PendingAttentionCardV1`, `ChatAttentionRequest`, `ChatAttentionAck`, `ChatAttentionState` all matched the `attention` keyword.
  - Resolution: traced these to two distinct, real, already-existing subsystems (`orion/schemas/attention_frame.py` + `orion/schemas/attention_salience.py` — conversational curiosity/open-loop attention allocation; `orion/schemas/notify.py` — chat-notification attention) structurally different from `FieldAttentionFrameV1`'s per-node/capability salience scoring. Explicitly excluded as out of scope for this registry's felt-state focus, with a note flagging them as worth their own future audit — not silently dismissed as noise.
- Finding: `PhiEncoderManifestV1`, and the five L7-L11 ladder frame schemas (`ProposalFrameV1` etc.) matched keywords.
  - Resolution: confirmed as direct lineage companions of `phi_intrinsic_reward.v1` and `l7_l11_ladder` respectively, added to the exclusion list with that reasoning.

Separately, code review confirmed: `orion/schemas/registry.py:660`'s `_REGISTRY` really is `Dict[str, Type[BaseModel]]` as the gate script assumes; the `scripts/platform/`-shadows-stdlib-`platform` import ordering issue (same class of bug documented in `scripts/fit_phi_encoder.py`) was pre-emptively worked around in the new script; all `InnerStateSignal.__post_init__` validation invariants hold across all 9 entries; Makefile recipe uses a real tab, not spaces.

## Restart required

```text
No restart required.
```

No service reads this registry at runtime — it's a static bookkeeping module and a CI-style gate script.

## Risks / concerns

- Severity: low
- Concern: the new-duplicate heuristic is a maintained keyword list, not a formal proof — documented explicitly in the script's own docstring and in `orion/self_state/README.md`. A cleverly-named 10th duplicate could evade it.
- Mitigation: a shared marker base class was considered and rejected (real migration cost across 6+ existing schemas for marginal precision gain); the keyword list is easy to extend when the next miss is found, same discipline as `config/autonomy/signal_drive_map.yaml`'s own closed-list approach.
- Severity: low
- Concern: this patch surfaces (but does not resolve) that `make agent-check`, `check_schema_registry.py`, `check_bus_channels.py`, and `check_env_template_parity.py` — all referenced in CLAUDE.md §11/§17 as if they exist — do not.
- Mitigation: named explicitly here rather than silently fixed as scope creep or silently ignored. Juniper's call whether this is a follow-up patch.

## PR link

Branch pushed: `feat/inner-state-registry-gate`
