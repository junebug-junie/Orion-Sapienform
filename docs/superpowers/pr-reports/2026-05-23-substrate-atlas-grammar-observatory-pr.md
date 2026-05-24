# PR: Substrate Atlas / Grammar Observatory MVP

**Branch:** `feat/substrate-atlas-grammar-observatory`  
**Base:** `main` (rebased on `e55bdabd`, May 2026)  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/feat-substrate-atlas-grammar-observatory`

## Summary

Ships a **read-only visual instrument** over canonical grammar events: typed atoms, edges, layers, dimensions, temporal hops, compactions, and projections. Events flow through `orion:grammar:event` → sql-writer append-only ledger → Hub `/api/substrate/atlas/*` → Cytoscape UI at `/substrate-atlas`.

## Architecture

```text
Organs emit GrammarEventV1
  → orion:grammar:event (bus)
  → orion-sql-writer: apply_grammar_event (dedupe by event_id)
  → Postgres grammar_* tables
  → orion/grammar/query.py (materializer)
  → Hub grammar_atlas_routes + substrate-atlas.js (Cytoscape)
```

**Not** the same as `schema_kernel.ConceptAtomV1` or `SubstrateMoleculeV1` — episodic trace grammar is a separate layer.

## Changes by area

| Area | Key paths |
|------|-----------|
| Schemas | `orion/schemas/grammar.py`, registry |
| Ledger | `orion/grammar/ledger.py`, `grammar_ledger_handler.py` |
| Query | `orion/grammar/query.py`, `graph_view.py`, `constants.py` |
| SQL | `manual_migration_grammar_atlas.sql`, `models/grammar_trace.py` |
| Bus | `orion/bus/channels.yaml`, sql-writer subscribe + route map |
| Hub API | `scripts/grammar_atlas_routes.py` |
| Hub UI | `templates/substrate_atlas.html`, `static/js/substrate-atlas.js` |
| Seed | `scripts/seed_substrate_atlas_demo.py` → `trace:vision:demo` |

## Code review fixes (post-review)

- Rebased onto current `origin/main` (includes llm-uncertainty + vision-retina-canonical)
- Cached SQLAlchemy engine in Hub atlas routes (no per-request engine)
- Temporal-path BFS: `visited_atoms` separate from `visited_hops`
- `IntegrityError` on ledger commit → log + treat as skip (duplicate child PK)
- `GRAMMAR_ATLAS_POLL_INTERVAL_MS` in settings + template
- Migration file header with apply command

## Operator checklist

1. **Apply DDL** (required):
   ```bash
   psql "$DATABASE_URL" -f services/orion-sql-db/manual_migration_grammar_atlas.sql
   ```
2. **Env** — sync `orion-hub` and `orion-sql-writer` `.env` from `.env_example` (`orion:grammar:event`, `GRAMMAR_ATLAS_*`).
3. **Seed demo**:
   ```bash
   cd services/orion-sql-writer
   DATABASE_URL=... POSTGRES_URI=... PYTHONPATH=../.. python ../../scripts/seed_substrate_atlas_demo.py
   ```
4. Open **`/substrate-atlas`** → select **`trace:vision:demo`**.

## Tests

```bash
PYTHONPATH=. pytest orion/grammar/tests -q          # 13 passed
PYTHONPATH=services/orion-sql-writer:. pytest services/orion-sql-writer/tests/test_grammar_event_routing.py -q  # 2 passed
PYTHONPATH=services/orion-hub:. pytest services/orion-hub/tests/test_grammar_atlas_api.py -q  # 5 passed
```

## Deferred / follow-ups

- Live vision-retina grammar emitter (Task 10)
- Hub WebSocket live stream (MVP uses 3s poll)
- Integration tests against real Postgres
- Hub auth gate on atlas API for production exposure
- `annotation_emitted` child table (MVP: event row only)

## Assessment

**Ready to merge:** Yes (lab/staging) after DDL applied on target DB.
