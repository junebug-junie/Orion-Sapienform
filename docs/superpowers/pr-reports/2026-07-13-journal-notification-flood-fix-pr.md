# PR report — journal/notification flood fix

## Summary

- Fixed the root cause of the "drowning in journals" bug: `orion-actions`' scheduler cursor/workflow-schedule state lived in ephemeral `/tmp` with zero volume mount, so every container recreate wiped it and re-fired all 3 daily jobs (`daily_journal`, `daily_pulse_v1`, `world_pulse`) same-day. Fire counts had climbed from 1/day (March–June) to 6–17/day (July) as redeploys became more frequent.
- Fixed a scheduler-daily journal double-email bug (two independent codepaths, two un-shared dedupe-key namespaces).
- Replaced the ad-hoc per-trigger_kind email/in-app dispatch logic with a declarative, fail-closed registry (`orion/journaler/dispatch_registry.py`), backed by a completeness gate script.
- Turned in-app journal notifications off by default across all trigger kinds — data-backed: 1796 sent in 14 days, 0 ever opened.
- Fixed RDF recall candidate scoring: chat-turn/claim fragments were ordered by a lexical UUID sort (a permanent, arbitrary "most recent" winner) instead of the real timestamp already written to the graph, and hardcoded `ts=0.0` so they never decayed by age. This is why journals kept reflecting on the same stale topic across many days.
- Added a report-only schedule-collision lint documenting that Daily Pulse and Daily Journal are configured to fire at the exact same local minute, by construction.

## Outcome moved

- Daily-journal fire count should return to ≤1/day regardless of container recreate frequency (previously scaled with restart frequency, up to 17/day observed).
- Scheduler-daily journals now produce exactly one email, not two.
- Recall candidate scoring now decays by real age instead of treating a fixed historical turn as permanently "fresh."
- ~128/day of unread in-app notification noise removed (0% open rate over the full 14-day sample).

## Current architecture (before this patch)

- `services/orion-actions/docker-compose.yml` had no `volumes:` entries at all.
- Two bespoke, independently-dedupe-keyed codepaths in `main.py` each emailed persisted `daily_summary`/`scheduler` journal entries.
- `journal_entry_index.trigger_kind` (and the underlying `JournalEntryWriteV1`/`journal.entry.created.v1` event) never carried `trigger_kind` at all, so no clean dispatch policy could be keyed on it.
- `services/orion-recall/app/storage/rdf_adapter.py`'s two RDF fragment-fetch functions ordered candidates by `ORDER BY DESC(STR(?turn))` (a UUID string sort) and hardcoded `ts=0.0`.

## Architecture touched

- `services/orion-actions`: durable state volume, dispatch registry consolidation, settings/env-parity cleanup.
- `orion/journaler`: schema (`JournalEntryWriteV1.trigger_kind`), `build_write_payload`, `build_journal_entry_index_payload`, new `dispatch_registry.py`.
- `orion/bus/channels.yaml`: documented the new optional field on `journal.entry.write.v1` / `journal.entry.created.v1`.
- `services/orion-recall`: RDF fragment recency scoring.
- `scripts/`: two new gate/lint scripts (`check_journal_dispatch_registry.py`, `check_daily_schedule_collisions.py`), wired into `Makefile` as standalone targets (there is no `make agent-check` in this repo — confirmed absent, not invented).

## Files changed

- `services/orion-actions/docker-compose.yml` — mount `${ORION_DATA_ROOT:-/mnt/graphdb}/orion-actions/state:/data/orion-actions`.
- `services/orion-actions/.env_example`, `.env` (local, not committed) — repoint `ACTIONS_SCHEDULER_CURSOR_STORE_PATH`/`ACTIONS_WORKFLOW_SCHEDULE_STORE_PATH` off `/tmp`; retire `ACTIONS_SCHEDULER_DAILY_JOURNAL_MESSAGES_ENABLED`, `ACTIONS_SCHEDULER_DAILY_JOURNAL_EMAIL_ENABLED`, `ACTIONS_JOURNAL_POST_PERSIST_EMAIL_EXCLUDE_SOURCE_KINDS`.
- `services/orion-actions/app/settings.py` — drop the 3 retired settings fields and the now-dead `post_persist_email_excluded_source_kinds()` method.
- `services/orion-actions/app/main.py` — removed `_is_scheduler_daily_journal`, `_build_scheduler_daily_journal_message_payload`, `_build_scheduler_daily_journal_email_request`, `_should_email_persisted_journal`, `_build_post_persist_journal_email_request`; added single `_dispatch_journal_notifications`, called once from the post-persist consumer.
- `services/orion-actions/README.md` — documents both the state-volume fix and the dispatch registry.
- `orion/journaler/schemas.py` — `JournalEntryWriteV1.trigger_kind: JournalTriggerKind | None = None`.
- `orion/journaler/worker.py` — `build_write_payload` populates it.
- `orion/journaler/indexing.py` — `build_journal_entry_index_payload` prefers `write.trigger_kind` as fallback.
- `orion/journaler/dispatch_registry.py` (new) — `JOURNAL_DISPATCH_REGISTRY` + fail-closed `resolve_policy()`.
- `orion/journaler/README.md` — documents the new field + registry.
- `orion/bus/channels.yaml` — `journal.entry.write.v1` / `journal.entry.created.v1` description updates.
- `services/orion-recall/app/storage/rdf_adapter.py` — `fetch_rdf_chatturn_fragments` / `fetch_rdf_graphtri_fragments` now select+order by the real `ORION.timestamp` predicate; `_parse_rdf_timestamp` helper handles both epoch and ISO literal shapes.
- `services/orion-recall/README.md` — recency-fix note.
- `scripts/check_journal_dispatch_registry.py` (new) — registry completeness gate.
- `scripts/check_daily_schedule_collisions.py` (new) — report-only cadence-collision lint.
- `scripts/README.md`, `Makefile` — new sections/targets for both scripts.
- Tests: `services/orion-actions/tests/test_scheduler_cursor_state_path.py` (new), `test_journal_actions.py` (extended), `test_async_notify_producers.py` (removed tests for deleted functions); `orion/journaler/tests/test_trigger_kind_roundtrip.py` (new); `services/orion-sql-writer/tests/test_journal_entry_trigger_kind_filtering.py` (new); `services/orion-recall/tests/test_rdf_recency_scoring.py` (new); `tests/test_check_journal_dispatch_registry.py`, `tests/test_check_daily_schedule_collisions.py` (new, repo root).

## Schema / bus / API changes

- Added: `JournalEntryWriteV1.trigger_kind: JournalTriggerKind | None = None` — optional, backward-compatible.
- Behavior changed: `journal.entry.write.v1` and `journal.entry.created.v1` payloads now carry `trigger_kind`. `orion/bus/channels.yaml` updated in the same changeset.
- Behavior changed: journal notification emails now all use `event_kind="orion.journal.notify"` (previously `orion.journal.daily.scheduler` and `orion.journal.persisted` were two separate event kinds). **Compatibility note:** any external dashboard/alert keyed on the old `event_kind` values needs updating — flagging this explicitly since it wasn't in the original scope but is a direct consequence of consolidating to one dispatch path.
- Removed: no fields removed. `journal_entries` (raw append-only SQL table) does not get a `trigger_kind` column — proved via a real SQLite round-trip test that `_write_row`'s mapper-based field filtering silently and safely drops it (`test_journal_entry_trigger_kind_filtering.py`).

## Env/config changes

- Added keys: none (paths already existed; values changed).
- Removed keys: `ACTIONS_SCHEDULER_DAILY_JOURNAL_MESSAGES_ENABLED`, `ACTIONS_SCHEDULER_DAILY_JOURNAL_EMAIL_ENABLED`, `ACTIONS_JOURNAL_POST_PERSIST_EMAIL_EXCLUDE_SOURCE_KINDS`.
- Changed values: `ACTIONS_SCHEDULER_CURSOR_STORE_PATH`, `ACTIONS_WORKFLOW_SCHEDULE_STORE_PATH` — `/tmp/orion-actions/...` → `/data/orion-actions/...`.
- `.env_example` updated: yes.
- Local `.env` synced: done by hand for `services/orion-actions/.env` in the main checkout (`scripts/sync_local_env_from_example.py` only covers a curated feature-key subset that doesn't include these two — confirmed via `--dry-run`, so a manual edit was correct here, not a workaround).
- Skipped keys requiring operator action: **the host directory `${ORION_DATA_ROOT:-/mnt/graphdb}/orion-actions/state` must be created before the container starts** — `/mnt/graphdb` is root-owned and this session's user cannot `mkdir` under it. See "Restart required" below.

## Tests run

```text
services/orion-actions/tests                                    92 passed
orion/journaler/tests                                             4 passed
services/orion-recall/tests/test_rdf_recency_scoring.py
  + test_rdf_adapter_graph_iri.py                                 8 passed
services/orion-sql-writer/tests/test_journal_entry_indexing.py
  + test_journal_entry_trigger_kind_filtering.py                  9 passed
tests/test_check_journal_dispatch_registry.py
  + tests/test_check_daily_schedule_collisions.py                20 passed
services/orion-sql-writer/tests/test_journal_entry_payload_boundary.py
  1 failed (test_journal_entry_write_uses_nested_payload_for_validation_and_write)
  -- pre-existing on main (e7ccca8b), confirmed untouched by this patch's diff, not caused by it.
```

## Evals run

No dedicated eval harness exists for `orion-actions`, `orion-journaler`, or the recall scoring path touched here — this patch is infra/dispatch/scoring-logic, verified via the gate tests above plus live script runs (registry completeness, schedule-collision detection against the real `.env_example`).

## Docker/build/smoke checks

```text
python scripts/check_service_env_compose_parity.py orion-actions
  -> declares env_file: -- all 95 .env_example keys reach the container. N/A (no drift).

python scripts/check_service_env_compose_parity.py orion-recall
  -> OK -- all 128 .env_example keys are exposed via environment:.

python scripts/check_journal_dispatch_registry.py
  -> OK -- all 8 trigger_kind(s) in _TRIGGER_TO_MODE have a JOURNAL_DISPATCH_REGISTRY row.

python scripts/check_daily_schedule_collisions.py
  -> COLLISION: 1 pair(s) within 30 minutes of each other: Daily Journal <-> Daily Pulse: 0m apart
  -> (report-only, known/expected, not a regression from this patch)
```

Could not run `docker compose up -d --build` for `orion-actions` in this session: the host directory backing the new volume mount (`/mnt/graphdb/orion-actions/state`) requires root to create (`/mnt/graphdb` is `root:root`), and this session does not run `sudo`. See "Restart required."

## Review findings fixed

- Finding: `services/orion-actions/app/settings.py` had a missing blank line (PEP8) between `subscribe_patterns()` and the module-level `@lru_cache` `get_settings()` after `post_persist_email_excluded_source_kinds()` was removed.
  - Fix: restored the blank line.
  - Evidence: `python3 -c "import ast; ast.parse(...)"` confirmed syntax was always valid; fixed for style only.
- Finding (caught by the implementing agent's own subagent review before this orchestrator review): `_dispatch_journal_notifications` left `in_app_ok`/`email_ok` as `None` in the policy-disabled branches, causing a test assertion failure (`assert None is True`).
  - Fix: both are set to `True` explicitly when the policy disables that channel (not required → doesn't block dedupe completion).
  - Evidence: `test_missing_trigger_kind_sends_nothing` and the full `test_journal_actions.py` suite pass.
- Finding: after removing the two scheduler-daily-specific settings fields, `ACTIONS_SCHEDULER_DAILY_JOURNAL_MESSAGES_ENABLED`/`ACTIONS_SCHEDULER_DAILY_JOURNAL_EMAIL_ENABLED` were left live in `.env_example`/`docker-compose.yml` with no consumer — silent config drift.
  - Fix: removed from both files (and the live `.env` in the main checkout), plus the now-dead `post_persist_email_excluded_source_kinds()` method.
  - Evidence: `check_service_env_compose_parity.py orion-actions` clean; `grep` for the retired keys across the service returns no consumers.
- Finding: does `journal_entries` (raw SQL table) need a `trigger_kind` column now that `JournalEntryWriteV1` has one?
  - Fix: verified — not needed. `_write_row`'s mapper-attribute filtering silently drops unmapped fields.
  - Evidence: `test_journal_entry_trigger_kind_filtering.py`'s real SQLite round-trip test (not just mapper inspection).

## Restart required

**Before restart**, an operator with root must create the new state directory (this session's user cannot):
```bash
sudo mkdir -p /mnt/graphdb/orion-actions/state
sudo chown -R 1000:1000 /mnt/graphdb/orion-actions   # match the orion-actions container's runtime uid, verify against the Dockerfile if unsure
```

Then, from repo root on the machine actually running these services:
```bash
docker compose \
  --env-file .env \
  --env-file services/orion-actions/.env \
  -f services/orion-actions/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-recall/.env \
  -f services/orion-recall/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml \
  up -d --build
```

`services/orion-actions/.env` in the main checkout has already been updated with the new `/data/orion-actions/...` paths and the 3 retired keys removed (done as part of this session, not committed — `.env` is gitignored).

## Risks / concerns

- Severity: low. Concern: `journal.entry.write.v1`/`journal.entry.created.v1` payloads now include an additional optional field; any external consumer doing strict schema validation without tolerance for new optional fields could be affected. Mitigation: field defaults to `None`, and `orion/bus/channels.yaml`'s registered consumer is `orion-sql-writer` only (plus wildcard `*` on the created event) — no other in-repo consumer found.
- Severity: low. Concern: all journal notification emails now share one `event_kind` (`orion.journal.notify`) instead of two. Mitigation: called out explicitly above; no in-repo consumer of the old event_kind strings was found, but this wasn't exhaustively checked against anything outside the repo (e.g. an external Grafana/alerting rule).
- Severity: medium (operational, not code). Concern: the host directory for the new volume mount doesn't exist yet and requires root to create. Mitigation: exact commands given above; until created, `docker compose up` for `orion-actions` will either fail or Docker will auto-create the directory as root-owned, which may cause a permission error inside the container depending on its runtime uid — create it explicitly first.
- Severity: low. Concern: `services/orion-actions/app/settings.py`, `workflow_schedule_store.py`, and `scheduler_cursor_store.py` still hardcode `/tmp/orion-actions/...` as in-code fallback defaults if the env var is ever unset (found by the implementing agent, left untouched as out-of-scope). Mitigation: `.env_example`/`.env` now set both paths explicitly, so the fallback shouldn't be hit in practice; worth a follow-up if someone ever unsets these vars.

## PR link

<opened manually — see PR body below, `gh` is not authenticated in this environment>
