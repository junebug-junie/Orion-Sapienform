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
