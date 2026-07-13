# feat(autonomy): experience loop -- Layer 9 actions become memory + chat evidence (P2)

## Summary

- Layer 9 (`orion-execution-dispatch-runtime`, P0+P1, merged) reached nothing outside its own table until now — zero references to episode journaling or action outcomes existed anywhere in that service. Every real dispatch result (success, empty observation, or RPC failure) now publishes an `ActionOutcomeEmitV1` event onto the existing `orion:autonomy:action:outcome` bus channel — the same always-on route `orion-spark-concept-induction` already produces onto for curiosity-fetch outcomes. No new pipe.
- `services/orion-cortex-exec/app/chat_stance.py` gained `_project_recent_dispatch_actions`, querying `load_action_outcomes(subject="orion")` directly and projecting to a bounded, privacy-safe `{kind, summary, success, observed_at}` shape (never `action_id`/`query`/`articles`/`salience`), rendered into `chat_general.j2`'s existing EVIDENCE-GATED CLAIMS section.
- Re-grounded against live code before writing anything: the original brainstorm-level P2 plan assumed a new `felt_state_reader.py` lane and flipping `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED`. Both were wrong — the data already reaches chat-turn state via a different, already-wired path, and that flag gates a *retired* duplicate-email feature, not anything Layer-9-related. Neither is touched by this patch.

## Outcome moved

A real autonomous action, once it happens (still gated behind `EXECUTION_DISPATCH_MODE=dispatch_read_only`, still not the default anywhere), now becomes something Orion can truthfully reference in conversation — "I looked into X" backed by a real evidence row, not invented. Before this patch, a completed Layer 9 dispatch produced a database row nothing else in the system ever read.

## Current architecture

Two real, already-durable mechanisms existed and were both orphaned from Layer 9: the `ActionOutcomeEmitV1` → sql-writer → `action_outcomes` bus route (fired only from one curiosity-fetch call site), and `AutonomyStateV2.last_action_outcomes` (already wired into `chat_stance.py` at the reducer level, but never rendered into any template). This patch connects Layer 9 to the producer side of the first, and connects the second's already-flowing data to a template — two thin patches, not the three-part design the pre-grounding brainstorm assumed.

## Architecture touched

`orion-execution-dispatch-runtime` (worker, settings, `.env_example`), `orion/bus/channels.yaml` (producer registration), `orion-cortex-exec` (`chat_stance.py`, `chat_general.j2`). No new schemas, no new tables, no new services.

## Files changed

- `services/orion-execution-dispatch-runtime/app/worker.py` — `_emit_action_outcome`, wired into all three `_send_one` outcome paths including idempotent replay.
- `services/orion-execution-dispatch-runtime/app/settings.py`, `.env_example` — `BUS_ACTION_OUTCOME_OUT` / `action_outcome_channel`.
- `orion/bus/channels.yaml` — `orion-execution-dispatch-runtime` added as a second producer on `orion:autonomy:action:outcome`.
- `services/orion-cortex-exec/app/chat_stance.py` — `_project_recent_dispatch_actions`.
- `orion/cognition/prompts/chat_general.j2` — renders `chat_recent_dispatch_actions` in the evidence-gated section.
- `tests/test_execution_dispatch_runtime_worker.py`, `tests/test_execution_dispatch_bus_catalog.py` — new/extended.
- `services/orion-cortex-exec/tests/test_chat_stance_recent_dispatch_actions_projection.py` (new).
- `services/orion-execution-dispatch-runtime/README.md`, `services/orion-cortex-exec/README.md` — documented.
- `docs/superpowers/specs/2026-07-13-autonomy-experience-loop-p2-design.md` — design spec.

## Schema / bus / API changes

- **Added:** none — reuses `ActionOutcomeEmitV1` and the `orion:autonomy:action:outcome` channel exactly as they already exist.
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:** `orion:autonomy:action:outcome` now has two producers instead of one. `chat_general.j2` renders an additional evidence block when `chat_recent_dispatch_actions` is non-empty (harmless/absent otherwise).
- **Compatibility notes:** `sql-writer`'s route for this channel upserts by `action_id` (SQL primary key) via `merge()` — confirmed during review, this is why re-emitting on replay is safe rather than a duplication risk.

## Env/config changes

- Added keys: `BUS_ACTION_OUTCOME_OUT` (`services/orion-execution-dispatch-runtime/.env_example`, default `orion:autonomy:action:outcome`).
- Removed/renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: no local `.env` exists for this service in this sandbox (never run locally here) — confirmed via `git status --short`, nothing created.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-autonomy-experience-loop-p2
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest tests/test_execution_dispatch_runtime_worker.py tests/test_execution_dispatch_bus_catalog.py -q
22 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-cortex-exec/tests/test_chat_stance_recent_dispatch_actions_projection.py -q
10 passed
```
(Run as two invocations — combining `orion-execution-dispatch-runtime` and `orion-cortex-exec` test files in one pytest process hits a pre-existing cross-service `app.*` module-name collision, same as encountered in P1; not introduced by this patch.)

`git diff --check` clean. Broader execution-dispatch/feedback suite (100 tests) also re-run and green after the review-fix commit.

## Evals run

None — no eval harness exists for either touched service; this patch doesn't add a new quality dimension an eval would measure (it wires existing, already-tested mechanisms together).

## Docker/build/smoke checks

Not run — no container build available in this sandbox. The data-flow chain (emit → sql-writer route → SQL table → `load_action_outcomes` → chat_stance projection → template) was traced file-by-file and confirmed intact during code review, but not live-exercised end-to-end (would require a running bus + sql-writer + real dispatch, none available here).

## Review findings fixed

- Finding (HIGH): the idempotent-replay path in `worker.py` skipped the `ActionOutcomeEmitV1` emit entirely, justified by a "would duplicate the record" comment that was factually wrong — `action_outcomes.action_id` is the SQL primary key and sql-writer's route upserts via `merge()`, so a repeat emit for the same `dispatch_id` overwrites rather than duplicates. Skipping it meant a crash between `save_dispatch_result` and the emit (or a transient bus failure inside the emit's own swallowed try/except) permanently lost that outcome, since every later tick also hits the replay branch.
  - Fix: replay path now re-emits, reconstructing summary/success from the stored result.
  - Evidence: `tests/test_execution_dispatch_runtime_worker.py::test_send_one_re_emits_action_outcome_on_idempotent_replay`.
- Finding (MEDIUM): `_project_recent_dispatch_actions` read `ctx["chat_autonomy_state_v2"]["last_action_outcomes"]`, which silently returns `[]` whenever `_run_autonomy_reducer` resolved a different subject for that turn (e.g. `"relationship"` during autonomy contextual fallback, a real production path) even though real Layer 9 outcomes exist under `subject="orion"`.
  - Fix: queries `load_action_outcomes(subject="orion")` directly, decoupling the feature from the ambient reducer's per-turn subject.
  - Evidence: `services/orion-cortex-exec/tests/test_chat_stance_recent_dispatch_actions_projection.py::test_queries_the_orion_subject_directly_not_ctx`.
- Checked and confirmed correct (no fix needed): the full data-flow chain from emit to template render (traced line-by-line by the reviewer); the `chat_autonomy_state_v2` vs `chat_autonomy_state` key choice (the design doc's originally-specified key doesn't have the needed field at all — caught by the implementing agent mid-build via runtime verification, not the reviewer); sort/newest-first ordering; a claimed test-pollution issue was independently reproduced against unmodified `origin/main` and confirmed pre-existing, unrelated to this patch.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-execution-dispatch-runtime/.env \
  -f services/orion-execution-dispatch-runtime/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build
```
No migration needed — reuses the existing `action_outcomes` table and channel. No config flip required for this patch's own code to be inert-safe; it only produces visible behavior once real dispatches occur, which itself still requires `EXECUTION_DISPATCH_MODE=dispatch_read_only` to be set (not the default).

## Risks / concerns

- Severity: low
- Concern: the full chain (bus emit → sql-writer → SQL → chat projection → template) is unit-tested at each link but not live-exercised end-to-end in this sandbox.
- Mitigation: recommend confirming this during the same P1 burn-in window already planned (watch for `action_outcomes` rows appearing with `subject='orion'` when a real dispatch fires, then confirm a subsequent chat turn's stance context contains `chat_recent_dispatch_actions`).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/autonomy-experience-loop-p2
