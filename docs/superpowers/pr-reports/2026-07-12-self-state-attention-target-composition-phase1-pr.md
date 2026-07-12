## Summary

- Phase 1 of the inner-state unification plan: `orion/self_state/builder.py` read the full per-node/per-capability `FieldAttentionTargetV1` list every tick — real, non-theater scoring from `orion-attention-runtime` (`weighted_pressure`/`urgency_score`/`confidence_from_vector`) — and kept only the top-5 bare `target_id` strings on `SelfStateV1.dominant_attention_targets`, discarding `pressure_score`/`dominant_channels`/`reasons` one hop downstream.
- New additive field `SelfStateV1.dominant_attention_target_details: list[AttentionTargetSummaryV1]` carries that structured data through, alongside (not replacing) the existing bare-string list — same target_ids, same order.
- `inner_state_registry.py`'s `field_attention_frame.v1` entry flipped from `SHADOW` to `COMPOSED` in the same PR — this is the actual proof the registry contract (Phase 0) works, not a separate afterthought.
- Code review caught a real gap in my own first test draft (checked presence, not correctness of the "top-1" selection) — fixed with an order-distinguishable fixture test, verified it actually fails against an injected regression before confirming it passes against the real code.
- README updates: `orion/self_state/README.md`, `services/orion-self-state-runtime/README.md`.

## Outcome moved

Node-attributed attention data that was computed for real and thrown away now survives into `SelfStateV1` — the schema-level prerequisite for Phase 2 (node-attributed phi) and Phase 3 (embodiment narrative), both still gated on this phase's real-traffic variation check post-deploy.

## Current architecture

`orion/self_state/builder.py:302` (`dominant_targets = [t.target_id for t in attention.dominant_targets[:5]]`) was the exact discard point. `FieldAttentionTargetV1` (produced by `orion/attention/field_attention/selectors.py`) already carries `salience_score`, `pressure_score`, `novelty_score`, `urgency_score`, `confidence_score`, `dominant_channels: dict[str, float]` (pre-sorted descending by `weighted_pressure()`), and `reasons: list[str]` (pre-sorted the same way by `_reasons_from_dominant()`).

## Architecture touched

`orion/schemas/self_state.py`, `orion/self_state/builder.py`, `orion/self_state/inner_state_registry.py`. No service code changed — `orion-self-state-runtime` picks this up on its next rebuild, no new dependency or migration.

## Files changed

- `orion/schemas/self_state.py`: new `AttentionTargetSummaryV1` (`target_id`, `target_kind`, `pressure_score`, `dominant_channel: str | None`, `reason: str | None`) + `SelfStateV1.dominant_attention_target_details`.
- `orion/self_state/builder.py`: new `_attention_target_summary()` pure helper (top-1 extraction via `next(iter(...))` / `reasons[0]`, trusting the upstream pre-sort rather than re-sorting); wired into `build_self_state()` alongside the existing `dominant_targets` extraction, same slice, same order.
- `orion/self_state/inner_state_registry.py`: `field_attention_frame.v1` `SHADOW` → `COMPOSED`, with the composition documented in `notes`.
- `tests/test_self_state_builder.py`: 3 new tests — end-to-end (real `build_attention_frame()` + `build_self_state()` pipeline), and 2 direct unit tests on `_attention_target_summary()` (correct top-1 selection with an order-distinguishable fixture; graceful `None`/`None` when a target has no channels/reasons).
- `orion/self_state/README.md`, `services/orion-self-state-runtime/README.md`: documentation.
- `docs/superpowers/plans/2026-07-12-inner-state-unification-plan.md`: Phase 1 checkboxes updated — 3 of 4 items done, the traffic-window question and the "verified against Postgres" acceptance check explicitly marked as blocked on deployment, not silently checked off.

## Schema / bus / API changes

- Added: `SelfStateV1.dominant_attention_target_details: list[AttentionTargetSummaryV1]` (additive, defaults to empty list — no existing consumer breaks; `extra="forbid"` on `SelfStateV1` means old rows without this key still validate fine via the field's default).
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing consumers — `dominant_attention_targets` (read by `orion/proposals/builder.py`'s `motivating_targets`) is untouched.
- Compatibility notes: fully backward compatible in both directions (old code reading new rows: unknown-but-unused field; new code reading old rows: field defaults to `[]`).

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. pytest tests/test_self_state_builder.py tests/test_self_state_schemas.py \
  tests/test_self_state_builder_hardening.py tests/test_self_state_runtime_store.py \
  tests/test_self_state_deviation.py tests/test_self_state_transport_dimension.py \
  tests/test_self_state_reliability_decontamination.py tests/test_self_state_prediction.py \
  tests/test_self_state_policy_loader.py tests/test_self_state_scoring.py \
  tests/test_inner_state_registry_gate.py tests/test_proposal_frame_builder.py \
  tests/test_proposal_frame_schemas.py -q
79 passed

python scripts/check_inner_state_registry.py
inner_state_registry gate OK (9 entries checked)
```

Regression-confirmed: temporarily broke `_attention_target_summary()`'s channel-selection logic (picked last instead of first) — the new unit test failed correctly (`assert 'thermal_pressure' == 'gpu_pressure'`), then reverted and re-confirmed green.

## Evals run

None applicable.

## Docker/build/smoke checks

Not run — not deployed. This is a schema/builder change with no runtime behavior for existing fields; `orion-self-state-runtime` needs a rebuild to pick it up, but nothing breaks if it doesn't (the new field just won't populate until redeployed).

## Review findings fixed

- Finding: the first draft of `test_dominant_attention_target_details_carries_structured_data` only asserted `dominant_channel is not None` / `reason is not None` — a bug that picked the wrong channel/reason (e.g. lowest instead of highest, or a stale/cached one) would have passed silently.
  - Fix: added `test_attention_target_summary_picks_first_channel_and_reason`, a direct unit test on the helper with a hand-built `FieldAttentionTargetV1` fixture whose channels/reasons are deliberately order-distinguishable.
  - Evidence: verified the new test actually fails when the selection logic is broken (temporarily changed `next(iter(...))` to pick the last entry instead — test failed with `AssertionError: assert 'thermal_pressure' == 'gpu_pressure'`), then reverted and confirmed the suite is green again.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-self-state-runtime/.env \
  -f services/orion-self-state-runtime/docker-compose.yml \
  up -d --build self-state-runtime
```

Not run — deliberately left for Juniper (per "merged but didn't pull/redeploy" — this PR isn't merged yet either, listed here for when it is).

## Risks / concerns

- Severity: low
- Concern: the "does the dominant target actually vary tick-to-tick, or get dominated by one or two nodes constantly" question from the plan's Phase 1 acceptance criteria is genuinely unanswered — requires a real post-deploy traffic window, not something a unit test or this PR can verify.
- Mitigation: explicitly left unchecked in the plan doc rather than assumed; Phase 2 (node-attributed phi) is already gated on this being answered first, per the plan.

## PR link

Branch pushed: `feat/self-state-attention-target-composition`
