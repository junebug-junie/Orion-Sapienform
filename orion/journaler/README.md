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

## Chat discussion window (time-bounded SQL)

The cognitive workflow `journal_discussion_window_pass` journals a **recent discussion window** derived from persisted chat turns in `chat_history_log`, not from `session_id` or vector recall.

- **Trigger builder:** `build_discussion_window_journal_trigger` wraps `build_manual_trigger` with a `source_ref` of the form `chat_history_log:window:<start_iso>:<end_iso>:turns=N` and puts the compact transcript in `prompt_seed` (consumed by `journal_compose_prompt.j2`).
- **Compose / write:** Same as other manual paths: `build_compose_request(..., journal.compose)` then `draft_from_cortex_result` / `build_write_payload` / publish `journal.entry.write.v1` on `orion:journal:write`.

### Runtime checks (why `journal_entries` might stay empty)

- **Orch bus:** `services/orion-cortex-orch` uses the same Redis-style bus as `orion-sql-writer`. If **`ORION_BUS_ENABLED=false`** (or the bus is not connected) on Orch, `publish` is a **no-op** and no row is written; workflows now raise **`journal_write_bus_disabled`** instead of claiming success.
- **Catalog validation:** With a strict channel catalog, an invalid `JournalEntryWriteV1` payload raises **`journal_write_bus_catalog_rejected`** at publish time (see `OrionBusAsync._validate_payload`).
- **Downstream:** `orion-sql-writer` must be running, subscribed to **`orion:journal:write`**, and using a **database URL that points at the same Postgres** you query for `journal_entries`.
