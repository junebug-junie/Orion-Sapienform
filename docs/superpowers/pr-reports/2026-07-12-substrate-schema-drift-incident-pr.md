# PR: Live incident — schema-drift crash loop + stuck FIFO queues across the substrate ladder

## Summary

- Live production incident, found while verifying the Phase 3 mesh-provenance deploy: Phase 0 removed `policy_pressure` from `SelfStateV1`'s valid `dimension_id` values. Every service that loads a historical `substrate_self_state` row via `SelfStateV1.model_validate()` can hit a row saved *before* that change and crash.
- `orion-self-state-runtime` had been crash-looping on every single tick since redeploy (confirmed via `docker logs`, ~2 hours behind live `field_state`).
- Three more services were also crash-looping on the identical error: `orion-policy-runtime`, `orion-proposal-runtime`, `orion-execution-dispatch-runtime`.
- Beyond the crash itself, `policy-runtime` and `execution-dispatch-runtime` were **permanently stuck on one broken FIFO-queue item each** — a naive "skip on missing self-state" would have retried the same oldest broken proposal/policy-frame forever, blocking every real item queued behind it (confirmed: policy-runtime stuck on 1 broken proposal with 35 legitimate newer ones piled up behind it).
- All fixes deployed live and verified: every service now processes continuously, the stuck queues drained.

## Outcome moved

The entire L6-L11 substrate ladder (self-state → proposal → policy → execution-dispatch → feedback → consolidation) is genuinely live and processing continuously for the first time this session, including real Atlas/Circe mesh data flowing all the way through with node-attributed evidence.

## Current architecture

Before this fix: any schema change that narrows an already-`extra="forbid"` model's accepted values (removing an enum literal) had no migration story for already-persisted rows. Every consumer that does a strict `model.model_validate()` on a stored JSON blob was silently assuming forward compatibility that didn't exist.

## Architecture touched

- `services/orion-self-state-runtime/app/store.py` (prior commit on this branch)
- `services/orion-policy-runtime/`, `services/orion-proposal-runtime/`, `services/orion-execution-dispatch-runtime/`, `services/orion-feedback-runtime/`, `services/orion-consolidation-runtime/` — store-level loader fixes
- `orion/policy/builder.py`, `orion/execution_dispatch/builder.py` — new "unevaluable frame" builders for the two services with FIFO-queue-stuck risk

## Files changed

- `services/orion-policy-runtime/app/store.py`: `load_self_state()` catches `ValidationError`, degrades to `None`
- `services/orion-policy-runtime/app/worker.py`: on missing/incompatible self-state, builds and saves an honest "unevaluable" `PolicyDecisionFrameV1` instead of returning silently — the FIFO queue advances
- `orion/policy/builder.py`: new `build_unevaluable_policy_decision_frame()` — empty decisions, `operator_review_required=True`, warning naming the missing dependency, reuses the existing `stable_policy_frame_id` scheme (naturally superseded, not duplicated, if the dependency later loads)
- `services/orion-execution-dispatch-runtime/app/store.py`: same `load_self_state()` fix (plus a leftover duplicate line cleaned up)
- `services/orion-execution-dispatch-runtime/app/worker.py`: same "unevaluable frame" pattern, covering both the missing-proposal and missing-self-state branches
- `orion/execution_dispatch/builder.py`: new `build_unevaluable_execution_dispatch_frame()` mirroring the policy-runtime builder
- `services/orion-proposal-runtime/app/store.py`: `load_latest_self_state()` degrades to `None` — no queue-stuck risk here (keyed on "latest," not a fixed id)
- `services/orion-feedback-runtime/app/store.py`: both self-state loaders degrade to `None` — no additional worker fix needed, `build_feedback_frame` already treats these inputs as optional
- `services/orion-consolidation-runtime/app/store.py`: generic `_parse_json`/`_load_rows` skip a schema-incompatible row instead of failing the whole windowed batch (used across multiple model types, not just self-state)
- `tests/test_policy_runtime_worker.py`, `tests/test_execution_dispatch_runtime_worker.py`: updated 2 existing tests that asserted the old silent-skip behavior; both now assert the recorded unevaluable frame's shape
- `tests/test_policy_runtime_store.py`, `tests/test_execution_dispatch_runtime_store.py`: new regression tests reproducing the exact legacy payload shape

## Schema / bus / API changes

None — no schema fields added or removed. This is purely error-handling and queue-advancement logic.

## Env/config changes

None.

## Tests run

```text
# Each service run in isolation (module-name collisions occur if batched --
# expected, every service's app.worker/app.store share module names by convention)
pytest tests/test_policy_runtime_store.py tests/test_policy_runtime_worker.py -q
→ 7 passed

pytest tests/test_execution_dispatch_runtime_store.py tests/test_execution_dispatch_runtime_worker.py \
  tests/test_execution_dispatch_builder.py -q
→ 14 passed

pytest tests/test_proposal_runtime_store.py tests/test_proposal_runtime_worker.py -q
→ 5 passed

pytest tests/test_feedback_runtime_store.py tests/test_feedback_builder.py \
  tests/test_feedback_transport_outcomes.py -q
→ 16 passed

pytest tests/test_consolidation_runtime_store.py tests/test_consolidation_runtime_worker.py -q
→ 7 passed
```

`git diff --check`: clean.

## Evals run

None applicable.

## Docker/build/smoke checks — done live, this is the actual verification

This fix was deployed to the live mesh as part of incident response, not deferred to post-merge:

```bash
# All five rebuilt and restarted:
docker compose --env-file .env -f services/orion-policy-runtime/docker-compose.yml up -d --build
docker compose --env-file .env -f services/orion-proposal-runtime/docker-compose.yml up -d --build
docker compose --env-file .env -f services/orion-execution-dispatch-runtime/docker-compose.yml up -d --build
docker compose --env-file .env -f services/orion-feedback-runtime/docker-compose.yml up -d --build
docker compose --env-file .env -f services/orion-consolidation-runtime/docker-compose.yml up -d --build
```

Verified via `docker logs` post-restart: each service hit the one broken legacy row exactly once, logged a graceful warning + `*_saved_unevaluable` line, then resumed continuous normal processing (`policy_decision_frame_saved ... decisions=10`, `execution_dispatch_frame_saved ... candidates=5 blocked=5`, `feedback_frame_saved ... outcome_status=dry_run_only`, repeating for every real item). Confirmed via direct SQL query against the live `substrate_proposal_frames`/`substrate_policy_decision_frames` join that the backlog (127 proposals, 7 policy frames — 2+ hours of accumulated real work, not stuck items) is draining normally, not stuck on a single id.

## Review findings fixed

No separate review round was run for this PR — it was incident response with continuous live verification substituting for a static review pass (each fix was confirmed against real production behavior immediately after deploying it, which is a stronger signal than a code-review agent would provide for this specific class of bug).

## Restart required

Already done live, as part of this incident response (see Docker/build/smoke checks above). No further action needed once this merges — the running containers already have this code.

## Risks / concerns

- Severity: low
  Concern: the "unevaluable frame" pattern (empty decisions/candidates, a warning) is new, minimal, and only exercised live against exactly one real broken item per service so far — it hasn't been tested against a broader variety of failure shapes (e.g., a policy frame with a missing proposal *and* a missing self-state simultaneously).
  Mitigation: each failure branch is handled independently and idempotently (stable frame IDs mean re-processing is always safe); acceptable given the narrow, well-understood failure mode this addresses.
- Severity: informational
  This incident is a general lesson for future schema changes to `SelfStateV1` (or any `extra="forbid"` model with historical persisted rows): removing an enum value needs either a migration/backfill step, or — as done here — every consumer needs to tolerate `ValidationError` on old rows. Worth calling out in the redesign spec's design invariants for future phases.

## PR link

<!-- filled in after push -->
