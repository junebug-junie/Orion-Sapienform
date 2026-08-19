# AI Town table split, Phase 2 migration design — `orion-recall`

Status: DESIGN, not implemented. Answers the open question left in
`docs/superpowers/specs/2026-08-19-aitown-table-split-phase2-consumer-audit.md`:
*"does `orion-recall`'s shared join point query both tables and merge, or
does the caller need to know which table a turn lives in?"* Ground-truthed
2026-08-19 by reading every bucket-(b) query site directly, not inferred
from the earlier audit summary.

## Arsonist summary

`orion-recall`'s 7 bucket-(b) files aren't one shared join point — they're
**four distinct query shapes across two different Postgres drivers**, each
needing a different fix. There is no single mechanical change that covers
all of them. The good news: within each shape, the fix is the same pattern
repeated, so this is 4 small changes, not 7 independent ones.

## Current architecture

| shape | driver | files | pattern |
|---|---|---|---|
| **1. Id-batch lookup** | `asyncpg` (async) | `sql_chat.py::fetch_chat_turns_by_id`, `fetch_chat_turn_timestamps` | `WHERE id = ANY($1::text[])` against a batch of known ids, returns a `Dict[id, ...]`. 3 callers (`falkor_chat_adapter.py`, `falkor_neighborhood_adapter.py`, `worker.py`) just consume the dict — zero changes needed in the callers themselves. |
| **2. Recency/content search** | `psycopg2`-style sync cursor, wrapped in `asyncio.to_thread` | `sql_timeline.py`'s 3 functions (`fetch_recent_fragments`, `fetch_related_by_entities`, `fetch_exact_fragments`) | `WHERE created_at >= ...` / `ILIKE` search, `ORDER BY created_at DESC LIMIT N`. Column names are **dynamically introspected** per call (`_pick_id_col`/`_pick_session_col`/etc. query `information_schema` against whatever `RECALL_SQL_TIMELINE_TABLE` is configured — not always `chat_history_log`, e.g. `collapse_mirror`). |
| **3. Bounded time-window scan** | same sync cursor | `storage/sql_adapter.py::fetch_sql_fragments`'s `include_chat` branch | `WHERE created_at >= %s ORDER BY created_at DESC LIMIT 300`. Static columns, no dynamic introspection — simpler than shape 2. |
| **4. Correlation-id platform resolution** | psycopg2, one-off scripts | `bulk_reject_aitown_proposals.py`, `smoke_aitown_crystallization_gate.py` | `LEFT JOIN chat_history_log h ON h.correlation_id = w.correlation_id`, reads only `h.client_meta`/`h.prompt`/`h.response` for classification, not a batch/search query. |

Plus a 5th, lower-stakes shape: **`backfill_recall_falkor_chat_tags_snapshot.py`**
does a plain unfiltered `SELECT ... FROM chat_history_log ORDER BY created_at ASC`
for a one-off manual tool — simplest fix, lowest urgency (not a live service).

`chat_source_tagging.py` itself needs no change — it's a pure library all
of the above import; it goes back to being useful the moment its callers
see AI-Town rows again.

## The real design question: what happens during the dual-write transition window

Phase 1 (shipped, off by default) writes an AI-Town row to **both** tables.
Phase 3 (cutover, not started) will write it to **only** the mirror table.
Between Phase 2 shipping and Phase 3 cutover, an AI-Town row's id can exist
in **both** tables simultaneously. That window is temporary and
self-resolving (it ends at cutover), but Phase 2's migrated code will be
live and running *during* it, not just after — so the fix has to handle it
correctly, not just the eventual post-cutover state.

This plays out differently per shape:

- **Shape 1 (id-batch lookup)**: the caller already knows the exact ids it
  wants. If an id exists in both tables, a duplicate-key collision doesn't
  even come up naturally — the caller just needs each id's true content,
  once.
- **Shapes 2/3 (recency/search)**: these don't know ids in advance; they
  search by time/content. If a query naively unions both tables, the *same*
  turn could appear twice in the result set during the transition window
  (found once in each table).
- **Shape 4 (correlation-id join)**: a `LEFT JOIN`, not a search — at most
  one row per side, no fan-out risk.

### Recommendation

**Shape 1 — two queries, Python-merge, mirror wins on conflict.**
Query `chat_history_log` and `aitown_chat_history_log` for the same id
batch (two `asyncpg` calls, same query shape, different table name — trivial
given the table name is already a settings-driven f-string param, same
pattern `RECALL_SQL_CHAT_TABLE` already uses). Merge the two dicts with the
mirror table's entries applied last (so they win on any id present in
both). This is correct across all three lifecycle states with the *same*
code, no phase-awareness needed:
- Pre-Phase-1 / dual-write off: mirror query returns nothing extra, no-op.
- Dual-write window: id present in both, mirror wins — reasonable, since the
  mirror table is the row's eventual canonical home.
- Post-cutover: id present in exactly one table, whichever query finds it
  wins — already correct, nothing to change when Phase 3 ships.

One extra network round-trip per call. These are already batch lookups
(`id = ANY($1)`), not per-row queries, so the added cost is one extra query
per *call*, not per row — cheap, and only paid when `turn_ids` is non-empty.

**Shapes 2/3 — `UNION ALL` at the SQL level, accept a bounded cosmetic risk.**
Postgres already handles `ORDER BY ... LIMIT N` efficiently over a UNION,
so wrapping both tables in one query and letting the DB do the ordering is
simpler and cheaper than merging two Python-side result sets — *if* the
column shapes are guaranteed identical (`aitown_chat_history_log` was built
as an exact mirror of `chat_history_log`, so `_pick_id_col`/etc. resolve
identically for both — verified, not assumed).

The tradeoff: `UNION ALL` does not dedupe. During the transition window, a
turn matching the search predicate could appear **twice** in results — a
duplicate memory fragment shown once from each table. This is a real but
low-severity, self-resolving-at-cutover cosmetic bug, not data loss or a
crash. The alternative (`DISTINCT ON` with a table-priority tiebreaker, or
a two-query Python merge matching shape 1's approach) fully avoids it at
real added complexity — an extra subquery layer for every call site in
`sql_timeline.py`'s dynamic-column code, which is already the most complex
file in this set.

**My recommendation is `UNION ALL`, accepting the bounded duplicate risk**,
with the exact tradeoff documented in a code comment at each call site —
not because the alternative is wrong, but because occasionally seeing one
memory fragment twice for a few weeks is a genuinely minor cost against a
real complexity increase in code that's already hard to read (dynamic
per-call `information_schema` introspection). If Juniper would rather pay
the complexity to close the duplicate risk entirely, shape 1's two-query
merge pattern generalizes to shapes 2/3 too — flagging this as a real
decision point, not a foregone conclusion.

**Shape 4 — extend the `LEFT JOIN`, `COALESCE`.** No new design question:
`LEFT JOIN aitown_chat_history_log a ON a.correlation_id = w.correlation_id`
alongside the existing join, `COALESCE(h.client_meta, a.client_meta)` (and
same for `prompt`/`response`). No fan-out risk (correlation_id is unique per
table by construction), no meaningful duplicate risk (COALESCE just picks
one side; both sides carry equivalent content by dual-write's own
contract).

**Shape 5 (backfill snapshot script)** — lowest priority, not a live
service. Run the same query against both tables, concatenate, re-sort by
`created_at` before writing `snapshot.json`.

### A caveat worth naming, not glossing over

Shapes 1/4's "both sides carry equivalent content" assumption depends on
Phase 1's dual-write actually succeeding for every write. It's designed to
degrade safely when it doesn't (`_maybe_dual_write_aitown_chat_history`'s
SAVEPOINT-contained failure, live-verified in PR #1734) — but a contained
failure means the mirror row can genuinely be **missing or stale** relative
to the primary table, not just delayed. Shape 1's "mirror wins on conflict"
rule is still correct in that case (if the mirror row doesn't exist, the
primary-table query finds it and there's nothing to conflict with); Shape
4's `COALESCE(h.client_meta, a.client_meta)` also degrades safely (falls
back to the primary's client_meta, which is the pre-dual-write behavior
today — a straight improvement, not a regression). Named here so this
isn't discovered as a surprise mid-implementation.

## Proposed schema / API changes

No schema changes — `aitown_chat_history_log` already exists (PR #1734).

- `services/orion-recall/app/settings.py`: new `RECALL_SQL_AITOWN_CHAT_TABLE`
  (default `"aitown_chat_history_log"`), mirroring how `RECALL_SQL_CHAT_TABLE`
  already works, so the second table name is configurable/overridable the
  same way the first one is.
- `sql_chat.py::fetch_chat_turns_by_id`/`fetch_chat_turn_timestamps`: query
  both tables, merge dicts (mirror wins).
- `sql_timeline.py`'s 3 functions' chat-branch queries (the
  `timeline_table == RECALL_SQL_CHAT_TABLE` branch only — the
  `collapse_mirror`/generic branch is untouched, it was never AI-Town-aware
  and stays that way): wrap in `UNION ALL` against both tables.
- `storage/sql_adapter.py::fetch_sql_fragments`'s `include_chat` branch:
  same `UNION ALL` treatment.
- `bulk_reject_aitown_proposals.py`, `smoke_aitown_crystallization_gate.py`:
  extend the existing `LEFT JOIN` with a second one + `COALESCE`.
- `backfill_recall_falkor_chat_tags_snapshot.py`: second query + concat +
  re-sort.

## Files likely to touch

- `services/orion-recall/app/settings.py`
- `services/orion-recall/app/sql_chat.py`
- `services/orion-recall/app/sql_timeline.py`
- `services/orion-recall/app/storage/sql_adapter.py`
- `services/orion-recall/.env_example` (new `RECALL_SQL_AITOWN_CHAT_TABLE` key)
- `scripts/bulk_reject_aitown_proposals.py`
- `scripts/smoke_aitown_crystallization_gate.py`
- `scripts/backfill_recall_falkor_chat_tags_snapshot.py`
- Tests: `services/orion-recall/tests/` (whatever covers `sql_chat.py`/
  `sql_timeline.py`/`sql_adapter.py` today — needs a fixture with rows in
  both tables to actually exercise the merge/union logic, not just one).

No changes needed in `falkor_chat_adapter.py`, `falkor_neighborhood_adapter.py`,
`worker.py`, or `chat_source_tagging.py` themselves — they consume shape 1's
output and it's already correct once `sql_chat.py` is fixed.

## Non-goals

- Not fixing the pre-existing lost-update race in the *primary*-table
  `_apply_spark_meta_patch` (flagged, out of scope, PR #1734's own report).
- Not building AI Town's own separate concept graph (deferred, tracked in
  the original design doc's "Recommended next patch" step 3).
- Not deciding Phase 3's cutover timing here — this only prepares Phase 2's
  reads; cutover is still gated on every bucket-(b) consumer (this PR's
  scope) actually shipping first.
- Not attempting to eliminate shapes 2/3's transition-window duplicate risk
  by default — flagged as an open decision for Juniper, not resolved here.

## Acceptance checks

- Live test: seed one row with the same id/correlation_id in both
  `chat_history_log` and `aitown_chat_history_log` (simulating the dual-write
  window), confirm `fetch_chat_turns_by_id` returns exactly one entry per
  id (no duplicate-key crash, no silently-wrong dict overwrite direction).
- Live test: seed an AI-Town-only row in `aitown_chat_history_log` with no
  matching row in `chat_history_log` (simulating post-cutover), confirm all
  four shapes surface it correctly (a search predicate finds it, an entity
  join resolves it, the crystallization gate correctly classifies its
  platform as `aitown` rather than `NULL`).
- `test_ensure_dataset_and_model_creates_when_missing`-style regression
  coverage isn't applicable here, but the existing `sql_chat.py`/
  `sql_timeline.py` test suite must pass unmodified for the single-table,
  dual-write-off case — same "don't change existing behavior" bar Phase 1
  held itself to.

## Recommended next patch

One PR: `services/orion-recall`'s 5 files (shapes 1-3 + the settings key),
live-verified against real Postgres with rows seeded in both tables. The
2 crystallization-gate scripts (shape 4) and the backfill snapshot script
(shape 5) are small and mechanical enough to ride in the same PR or split
into a second one — Juniper's call, not a real dependency either way.

Not started until Juniper confirms the shape-2/3 tradeoff recommendation
(`UNION ALL`, accept the bounded duplicate risk) above.
