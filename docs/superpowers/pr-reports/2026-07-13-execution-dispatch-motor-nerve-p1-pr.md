# feat(execution-dispatch): motor nerve — Layer 9 actually sends (P1)

## Summary

- `orion-execution-dispatch-runtime` (Layer 9) now actually sends `prepared_for_dispatch` candidates to `orion-cortex-exec` over the bus, instead of only building envelopes and never sending (P0 un-lied the status; P1 makes the honest status real).
- Three new cortex verbs (`substrate.inspect`/`summarize`/`observe`), each with its own prompt template and distinct tone (inspect: diagnostic; summarize: wide-angle; observe: deliberately lighter/non-alarming, matching its `preserve_stability` proposal-layer intent).
- Two new "robust, not speculative" proposal templates targeting `node`/`field` target_kinds — real, already-live signal behind both; explicitly did not add a `service`-target template (no real backing signal exists for it) or any new verb/proposal-kind beyond what's already grounded (no `trend`/`correlate`, per earlier design discussion).
- New `substrate_dispatch_results` table; `FeedbackRuntimeStore.load_cortex_result_evidence` replaced its `[]`-stub with a real query.
- Theater tripwire: if more than half of the trailing 10 real results come back empty, sending self-disables for the rest of the process's life, visible on `GET /latest`, with one `orion-notify` warning on the transition.
- Idempotency guard added during review: a crash between a successful send and frame persistence can no longer cause a duplicate real cortex-exec call for the same candidate.

## Outcome moved

Orion has never taken a real, evidenced, self-initiated action through this pipeline before. This patch is the one place that changes — everything upstream (perception → proposal → policy) and downstream (feedback scoring) already existed and worked; only the actual send was missing. `EXECUTION_DISPATCH_MODE` still defaults to `dry_run`, so nothing changes in the live environment until an operator explicitly flips it.

## Current architecture (before this patch)

`orion-execution-dispatch-runtime`'s worker built `ExecutionDispatchFrameV1` envelopes and persisted them, but never touched the bus — no `redis` dependency, no bus client, a dead `CORTEX_EXEC_CHANNEL` setting pointing at a channel name that didn't exist. `FeedbackRuntimeStore.load_cortex_result_evidence` was a two-line stub always returning `[]`. No cortex verb existed for any of the three dispatch kinds the policy already routed to.

## Architecture touched

`orion-execution-dispatch-runtime` (worker, store, settings, requirements), `orion-feedback-runtime` (store), `orion-cortex-exec` (verb allowlist), `orion-hub`'s debug route unaffected, plus shared `orion/` packages (`execution_dispatch/`, `feedback/`, `cognition/verbs`+`prompts`, `bus/channels.yaml`), `config/proposals/`, `config/execution_dispatch/`, and `services/orion-sql-db/` (new migration).

## Files changed

- `orion/cognition/verbs/substrate.{inspect,summarize,observe}.yaml` + `orion/cognition/prompts/substrate_{inspect,summarize,observe}.j2` — new verbs, own templates, distinct tone per verb.
- `services/orion-cortex-exec/app/router.py` — added the three verb names to `_structured_output_expected`.
- `config/proposals/proposal_policy.v1.yaml` — two new templates (`node`, `field` target_kinds), reusing `kind: inspect`, no new enum values.
- `services/orion-sql-db/manual_migration_substrate_dispatch_results_v1.sql` — new table.
- `services/orion-feedback-runtime/app/store.py` — real `load_cortex_result_evidence` (dedupe-by-latest, degrade-on-malformed-row).
- `orion/bus/channels.yaml` — `orion-execution-dispatch-runtime` added as a producer on `orion:cortex:exec:request:background`.
- `orion/execution_dispatch/cortex_client.py` (new) — RPC client, mirrors `orion.harness.cortex_client.HarnessCortexClient`'s shape.
- `orion/execution_dispatch/result_extraction.py` (new) — `extract_final_text`/`parse_structured_observation`.
- `orion/execution_dispatch/envelopes.py` — `origin`/`correlation` fields for attribution.
- `services/orion-execution-dispatch-runtime/app/worker.py` — send-and-wait step, budget/cap enforcement, theater tripwire, idempotency guard.
- `services/orion-execution-dispatch-runtime/app/store.py` — `save_dispatch_result`, `count_dispatches_today`, `recent_dispatch_result_statuses`, `load_dispatch_result_by_dispatch_id`.
- `services/orion-execution-dispatch-runtime/app/main.py` — `theater_tripwire_active` on `GET /latest`.
- `services/orion-execution-dispatch-runtime/app/settings.py`, `.env_example`, `requirements.txt` — new keys, fixed `CORTEX_EXEC_CHANNEL` default, `redis`+`requests` dependencies.
- `orion/feedback/builder.py` — `prepared_for_dispatch`/`"empty"` status handling.
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml` — `allow_dispatch_read_only: true`.
- Test files: `tests/test_execution_dispatch_cortex_client.py`, `tests/test_execution_dispatch_result_extraction.py` (new); `tests/test_execution_dispatch_runtime_worker.py`, `tests/test_execution_dispatch_runtime_store.py`, `tests/test_feedback_runtime_store.py`, `tests/test_feedback_builder.py`, `tests/test_execution_dispatch_policy_loader.py`, `tests/test_execution_dispatch_bus_catalog.py` (new), `tests/test_proposal_policy_loader.py`, `tests/test_plan_loader.py` (extended).
- `docs/superpowers/specs/2026-07-13-execution-dispatch-motor-nerve-p1-design.md` — design spec.
- `services/orion-execution-dispatch-runtime/README.md`, `services/orion-feedback-runtime/README.md` — documented.

## Schema / bus / API changes

- **Added:** `substrate_dispatch_results` table; `orion:cortex:exec:request:background`'s `producer_services` gains `orion-execution-dispatch-runtime`; `orion/feedback/builder.py`'s `OutcomeKind` unaffected (no new literal added — `"empty"` cortex status maps into the existing `"failed"` outcome kind, not a new one, per the empty-shell-cognition rule).
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:** real sends now happen when both `EXECUTION_DISPATCH_MODE=dispatch_read_only` and the policy's `allow_dispatch_read_only: true` (now the shipped default) are open. `dispatched_candidates`/`dispatch_count`/`dispatch_attempted` become honest for the first time (previously permanently empty/zero/false from this builder, a known P0 limitation, now resolved). `_cortex_status_to_outcome`'s `"empty"` status now scores as `"failed"` rather than falling through to `"unknown"`.
- **Compatibility notes:** the new evidence-requiring validator from P0 could have broken historical rows on load — checked and guarded during P0; this patch's own new risk (duplicate real dispatch on crash) was caught in review and fixed with an idempotency guard before merge, not after.

## Env/config changes

- Added keys (`services/orion-execution-dispatch-runtime/.env_example`): `CORTEX_EXEC_RESULT_PREFIX`, `ORION_BUS_URL`, `ORION_BUS_ENABLED`, `EXECUTION_DISPATCH_RPC_TIMEOUT_SEC`, `ORION_DISPATCH_MAX_PER_DAY`, `NOTIFY_URL`, `NOTIFY_API_TOKEN`.
- Changed default: `CORTEX_EXEC_CHANNEL` (`orion:cortex:request` → `orion:cortex:exec:request:background` — the old default named a channel that didn't exist in `orion/bus/channels.yaml` and was never referenced in code; now it's real and wired).
- Removed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: run — no local `.env` exists for this service in this environment (never brought up locally here), so it was a no-op; confirmed via `git status --short` that no `.env` file was created or staged.
- skipped keys requiring operator action: none new; `ORION_BUS_URL`'s tailscale-node-IP placeholder in `.env_example` needs a real value on any environment that runs this service for real (same as every other bus-connected service in this repo).

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-execution-dispatch-motor-nerve-p1
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  tests/test_execution_dispatch_frame_schemas.py tests/test_execution_dispatch_envelopes.py \
  tests/test_execution_dispatch_builder.py tests/test_execution_dispatch_policy_loader.py \
  tests/test_execution_dispatch_runtime_store.py tests/test_execution_dispatch_runtime_worker.py \
  tests/test_execution_dispatch_transport_dry_run.py tests/test_execution_dispatch_cortex_client.py \
  tests/test_execution_dispatch_result_extraction.py tests/test_execution_dispatch_bus_catalog.py \
  tests/test_feedback_builder.py tests/test_feedback_runtime_store.py tests/test_feedback_transport_outcomes.py \
  tests/test_proposal_policy_loader.py tests/test_plan_loader.py \
  tests/test_consolidation_tensorize.py tests/test_consolidation_schema_candidates.py \
  tests/test_consolidation_motif_detection.py tests/test_consolidation_expectations.py \
  tests/test_consolidation_policy_loader.py -q

142 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_router_final_text_assembly.py \
  services/orion-hub/tests/test_substrate_execution_dispatch_debug_api.py -q

36 passed
```
(Run as two groups: this repo's `sys.path.insert` + `import app.X` test pattern has a pre-existing module-name collision risk when combining test files from multiple services that all use the generic package name `app` in one pytest invocation — not introduced by this patch, matches how `make agent-check SERVICE=<x>` already scopes per-service.)

`git diff --check` clean. `.env` files confirmed gitignored, none created/staged.

## Evals run

None — no eval harness exists for this service; the design spec names a live smoke script (`scripts/smoke_execution_dispatch_live.sh`) as a follow-up once this is deployed with real traffic, not built in this patch (requires a live cortex-exec + bus + Postgres environment this sandbox doesn't have).

## Docker/build/smoke checks

Not run — no container build available in this sandbox. Verified instead:
- Live Postgres query against `substrate_execution_dispatch_frames` (687,992 rows, carried over from P0's compat check) and live confirmation of `SHOW timezone` (`Etc/UTC`) on the real `orion-sql-db` instance, informing the `count_dispatches_today()` UTC-boundary fix.
- `redis==5.0.7`/`requests==2.32.3` both confirmed importable in the shared venv.
- All touched/new YAML files (`proposal_policy.v1.yaml`, `execution_dispatch_policy.v1.yaml`, `channels.yaml`, 3 verb YAMLs) parse cleanly.
- `orion/execution_dispatch/cortex_client.py`, `result_extraction.py`, and `services/orion-execution-dispatch-runtime/app/worker.py` all import cleanly end-to-end (including through `orion.core.bus.async_service`'s `redis` dependency).

## Review findings fixed

- Finding: No idempotency guard — a crash between a successful real send and `save_dispatch_frame()` persisting that fact would cause the next tick to resend a real cortex-exec RPC for the same (deterministic) `dispatch_id`.
  - Fix: `store.load_dispatch_result_by_dispatch_id()` + a check at the top of `_send_one()` that replays the stored result instead of resending.
  - Evidence: `tests/test_execution_dispatch_runtime_worker.py::test_send_one_replays_existing_result_without_resending` (and the failed-result variant) — registers no fake-client outcome, so a real resend attempt would `KeyError`; both pass.
- Finding: `evidence_refs` was always empty in production — the feedback-runtime reader looked for a key the dispatch worker never wrote.
  - Fix: `save_dispatch_result`'s `result_json` now includes `evidence_refs: [result_id]`.
  - Evidence: `_send_one`'s success/failure paths both updated; existing `load_cortex_result_evidence` tests already assert on this field's presence.
- Finding: cortex status `"empty"` fell into the same `"unknown"` bucket as a genuinely untracked result — the theater tripwire's core signal wasn't distinguishable from missing data in feedback scoring.
  - Fix: `_cortex_status_to_outcome` maps `"empty"` to `"failed"`, matching the empty-shell-cognition rule ("stored as failure, never as success").
  - Evidence: `tests/test_feedback_builder.py::test_empty_cortex_result_scores_as_failed_not_unknown`.
- Finding: `extract_final_text` dropped the `raw.text` fallback branch present in the function it's documented as mirroring (`orion-actions`' version), risking false `status="empty"` for a real content shape.
  - Fix: restored the branch verbatim.
  - Evidence: `orion/execution_dispatch/result_extraction.py:26-29`.
- Finding: `theater_tripwire_active` wasn't exposed on `GET /latest`, contradicting the design doc's explicit acceptance check.
  - Fix: added to the response payload in `app/main.py`.
  - Evidence: one-line diff, `services/orion-execution-dispatch-runtime/app/main.py`.
- Finding (lower confidence, live-checked): `count_dispatches_today()` used `date_trunc('day', now())`, Postgres-server-timezone-dependent, inconsistent with this file's own explicit-UTC convention elsewhere.
  - Fix: computed the UTC day boundary in Python instead. Live-checked `SHOW timezone` on the real instance first — `Etc/UTC`, so not an active bug, but hardened anyway for consistency.
  - Evidence: `services/orion-execution-dispatch-runtime/app/store.py::count_dispatches_today`.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-execution-dispatch-runtime/.env \
  -f services/orion-execution-dispatch-runtime/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-feedback-runtime/.env \
  -f services/orion-feedback-runtime/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build
```
Plus the new migration before starting `orion-execution-dispatch-runtime`:
```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_substrate_dispatch_results_v1.sql
```
No config change alone turns on live sending — `EXECUTION_DISPATCH_MODE` still defaults to `dry_run` in `.env_example`; an operator must explicitly set it to `dispatch_read_only` in the real `.env` for this service to start sending.

## Risks / concerns

- Severity: medium
- Concern: this is Orion's first real self-initiated bus action. Even with the theater tripwire, budget caps, and idempotency guard, the actual behavior of a live `substrate.inspect`/`summarize`/`observe` LLM call against real self-state data hasn't been observed — only unit-tested against mocked bus/client responses.
- Mitigation: `EXECUTION_DISPATCH_MODE` defaults to `dry_run`; recommend a manual, closely-watched burn-in window (flip the env var, watch `/latest`'s `theater_tripwire_active` and `dispatch_count`, check `substrate_dispatch_results` rows directly) before considering this "on" in any meaningful sense. Matches the parent spec's own stated sequencing (P1 needs its own burn-in before P2/P3, which need it before P7's origination decision).
- Severity: low
- Concern: `scripts/smoke_execution_dispatch_live.sh` (named in the parent spec's evidence bar) was not built in this patch — no live bus/cortex-exec/Postgres stack available in this sandbox to write and verify a real smoke script against.
- Mitigation: follow-up, once this is deployed somewhere with real infra to smoke-test against.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/execution-dispatch-motor-nerve-p1
