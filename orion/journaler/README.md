# Orion Journaler

`orion.journaler` is a shared domain/business worker, not a deployable service.

Boundaries:
- `orion-actions` owns journaling trigger policy.
- Cortex owns prose composition through `journal.compose`.
- `orion-sql-writer` owns persistence.

Semantics:
- Journal entries are append-only; there is no update or replace-latest path.
- Journaling is distinct from Collapse Mirrors.
- Collapse-response journaling should consume the semantic stored event (`collapse.mirror.stored.v1`).
- Metacog journaling is currently provisional and uses the existing metacog trigger event until a dedicated digest/cycle-complete event exists.
- Journal metadata stays intentionally minimal: trigger/source refs plus correlation linkage.

## `trigger_kind` on `JournalEntryWriteV1`

`JournalEntryWriteV1.trigger_kind` (`orion/journaler/schemas.py`) is an optional
`JournalTriggerKind | None` field (default `None`), added 2026-07-13. It carries the
originating `JournalTriggerV1.trigger_kind` forward from the ephemeral, in-process-only
trigger object onto the persisted/bus-facing write payload:

- `build_write_payload` (`worker.py`) populates it from `trigger.trigger_kind`.
- `build_created_event_payload` (`worker.py`) needs no extra code to propagate it —
  it already does `write.model_dump(mode="json")`, and `trigger_kind` is now a real
  declared field on that model, so it survives automatically.
- `build_journal_entry_index_payload` (`indexing.py`) prefers the separately-passed
  `trigger` param's `trigger_kind` when present, falling back to `write.trigger_kind`.
- The raw append-only `journal_entries` SQL table
  (`services/orion-sql-writer/app/models/journal_entry.py::JournalEntrySQL`) does
  **not** have a `trigger_kind` column, and does not need one — `_write_row`
  (`services/orion-sql-writer/app/worker.py`) filters any schema field without a
  matching column rather than erroring. `journal_entry_index` already had this column
  (it was simply always `NULL` before this patch, since nothing populated it).

This is why a dispatch-policy registry can now key on `trigger_kind`: before this
change, no bus-facing journal event ever carried it.

## Journal dispatch registry

`orion/journaler/dispatch_registry.py` declares `JOURNAL_DISPATCH_REGISTRY`, a
`trigger_kind -> JournalDispatchPolicy` table (`email_enabled`, `in_app_enabled`,
`dedupe_scope`, `recall_profile_setting`), plus `resolve_policy(trigger_kind)`, a
fail-closed lookup (an unregistered `trigger_kind` resolves to an all-disabled
policy). `services/orion-actions` is the sole consumer today — see that service's
README, "Journal notification dispatch registry" section, for what this replaced and
why in-app notifications are off by default. `scripts/check_journal_dispatch_registry.py`
gates the registry against `orion.journaler.worker._TRIGGER_TO_MODE` so a new
trigger_kind can't silently ship without a deliberate dispatch policy.

## Chat discussion window (time-bounded SQL)

The cognitive workflow `journal_discussion_window_pass` journals a **recent discussion window** derived from persisted chat turns in `chat_history_log`, not from `session_id` or vector recall.

- **Trigger builder:** `build_discussion_window_journal_trigger` wraps `build_manual_trigger` with a `source_ref` of the form `chat_history_log:window:<start_iso>:<end_iso>:turns=N` and puts the compact transcript in `prompt_seed` (consumed by `journal_compose_prompt.j2`).
- **Compose / write:** Same as other manual paths: `build_compose_request(..., journal.compose)` then `draft_from_cortex_result` / `build_write_payload` / publish `journal.entry.write.v1` on `orion:journal:write`.

### Runtime checks (why `journal_entries` might stay empty)

- **Orch bus:** `services/orion-cortex-orch` uses the same Redis-style bus as `orion-sql-writer`. If **`ORION_BUS_ENABLED=false`** (or the bus is not connected) on Orch, `publish` is a **no-op** and no row is written; workflows now raise **`journal_write_bus_disabled`** instead of claiming success.
- **Catalog validation:** With a strict channel catalog, an invalid `JournalEntryWriteV1` payload raises **`journal_write_bus_catalog_rejected`** at publish time (see `OrionBusAsync._validate_payload`).
- **Downstream:** `orion-sql-writer` must be running, subscribed to **`orion:journal:write`**, and using a **database URL that points at the same Postgres** you query for `journal_entries`.
