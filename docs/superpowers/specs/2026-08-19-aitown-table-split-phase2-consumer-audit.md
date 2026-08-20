# AI Town chat-history table split, Phase 2 — consumer audit

Status: AUDIT COMPLETE, migration NOT started. Track B Phase 2 of
`docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md`.

Every one of the ~50 real `chat_history_log` readers in the repo (Postgres
SQL against the table or the `ChatHistoryLogSQL` ORM model, excluding
`orion-sql-writer`'s own producer-side write path shipped in Phase 1 — PR
#1734) was read directly and bucketed as:

- **(a) doesn't need AI Town data** — naturally and correctly stops seeing
  AI Town rows once Phase 3 cuts AI Town writes over to
  `aitown_chat_history_log`. No code change needed, ever, for this file.
- **(b) needs AI Town data** — depends on seeing AI Town rows to function
  correctly. Needs an explicit read from `aitown_chat_history_log` added
  alongside its existing `chat_history_log` read, in a later migration PR.

50 files audited, dispatched as 5 parallel read-only investigations (not
guessed from file names — each file was actually opened and its
`chat_history_log`/`ChatHistoryLogSQL` usage traced).

## Headline finding

**`orion-recall` is almost entirely bucket (b) — 7 of its 9 audited files —
and every one of them fails *silently open*, not loudly.** None of these
paths raise an error, log at error level, or fail a health check when an
AI-Town row disappears from `chat_history_log` post-cutover. They just quietly
return fewer results:

- `sql_chat.py::fetch_chat_turns_by_id` — id not found → simply absent from
  the returned map, callers drop it.
- `storage/falkor_chat_adapter.py` / `falkor_neighborhood_adapter.py` — Falkor
  `:ChatTurn`/`:Entity` nodes for AI Town turns keep existing (they're not
  platform-scoped), but the Postgres join for their text silently misses,
  `continue`, fragment dropped.
- `sql_timeline.py` — three functions do live `ILIKE` searches over
  prompt/response; any recall query about an AI-Town-only conversation
  returns **zero hits**, not an error.
- `storage/sql_adapter.py::fetch_sql_fragments` — `kind='chat'` fragment
  volume **shrinks by roughly 90%** (AI Town is ~90% of current row volume).
- `worker.py` — entity-relatedness injection silently drops AI-Town turns on
  join-miss; `_window_rdf_chatturn_candidates`'s time-windowing step treats an
  unresolvable row as "outside the window," fully excluding it from
  time-windowed recall, not just missing a timestamp.
- `chat_source_tagging.py` — not itself a query, but the shared library all
  six of the above import to label/tag AI-Town rows. Its whole reason to
  exist goes dark for AI Town content the moment its callers stop seeing
  AI-Town rows, without anyone needing to touch this file itself.

This is the real, concrete cost of NOT doing Phase 2/3 correctly: recall
about AI Town conversations degrades invisibly. Nothing crashes, nothing
alerts, the numbers just quietly get smaller. Worth reading twice before
scheduling Phase 3's cutover.

## Full results by batch

### Batch 1 — bucket (a), no action needed (10/10)

| file | reasoning |
|---|---|
| `orion/cognition/chat_history_compactor/constants.py` | Pure constants, no query. |
| `orion/cognition/chat_history_compactor/digest.py` | Operates on an already-fetched window; never queries the DB. |
| `orion/cognition/chat_history_compactor/window.py` | Explicitly exists to reflect "organic conversation" — losing AI Town rows only improves its output. |
| `orion/cognition/compactor/index.py` | Cache-key builder only, no DB access. |
| `orion/cognition/workflows/registry.py` | Table name appears only in human-readable workflow metadata. |
| `orion/discussion_window/sql_fetch.py` | The real `SELECT` reader for journaling/compacting real conversation; currently reads everything (a preexisting gap, arguably), but post-cutover naturally narrows to real Orion/Juniper turns — the *desired* behavior. |
| `orion/hub/turn_orchestrator.py` | Publisher, not a reader — writes envelopes `orion-sql-writer` persists. |
| `orion/journaler/worker.py` | Uses the table name only as a label string on an already-fetched transcript. |
| `orion/memory/crystallization/formation_policy.py` | Discards AI-Town-tagged crystallizations by policy already (`DEFAULT_DISCARD_PLATFORMS`) — Phase 3 converges with its own intent. |
| `orion/memory/crystallization/intake_pipeline.py` | Same shape as `formation_policy.py`; consumes an already-tagged object, no direct SQL read. |

### Batch 2 — 7 bucket (a), 3 bucket (b)

| file | bucket | reasoning / consequence |
|---|---|---|
| `orion/memory_graph/consolidation_draft_hydrate.py` | (a) | Only reachable via a dormant `graph_draft` mode, dead in the live default config (`MEMORY_CONSOLIDATION_OUTPUT=crystallization_propose`). Would become (b) if that mode were ever enabled — worth remembering, not urgent. |
| `orion/schemas/chat_history.py` | (a) | Pure Pydantic schema, no query. |
| `orion/schemas/discussion_window.py` | (a) | Pure Pydantic schema, no query. |
| `orion/schemas/memory_consolidation.py` | (a) | Pure schema; carries a `source_platform` field but doesn't extract it itself. |
| `orion/substrate/relational/adapters/recall.py` | (a) | Passthrough mapper of upstream-fetched fragments; nothing to migrate here itself. |
| `scripts/analysis/measure_metacog_trend_baseline.py` | (a) | Diagnostic probe; gets *more* accurate without AI-Town noise diluting the signal. |
| `scripts/backfill_chat_history_from_bus_fallback.py` | (a) | Already explicitly scoped to `_HUB_SOURCES` only — AI Town's `orion-embodiment` source never matches, by design. |
| `scripts/backfill_recall_entity_graph_cleanup_reconcile.py` | **(b)** | Diffs Falkor `:ChatTurn` nodes against a `chat_history_log` snapshot with no platform filter. A future re-run's snapshot (post-cutover) would make every historical AI-Town `:ChatTurn` node **permanently invisible** to the reconcile loop — the exact stale-edge bugs this script exists to fix never get corrected for that ~90% of nodes, and `coverage_gap` would report a large, permanent, misleading gap. |
| `scripts/backfill_recall_falkor_chat_tags_extract_and_write.py` | **(b)** | Consumes the snapshot below; a re-run would silently skip untagged AI-Town turns forever, with no indication in its own output counts. |
| `scripts/backfill_recall_falkor_chat_tags_snapshot.py` | **(b)** | The actual live-table reader for the pair above — `SELECT ... FROM chat_history_log ORDER BY created_at ASC`, no platform filter, snapshots *every* turn by design. This is the natural single place to add the `aitown_chat_history_log` read for all three. |

### Batch 3 — 9 bucket (a), 2 bucket (b), 2 not-real-readers (producer side)

| file | bucket | reasoning / consequence |
|---|---|---|
| `scripts/bulk_reject_aitown_proposals.py` | **(b)** — confirmed | `LEFT JOIN chat_history_log` to resolve each window's real platform before bulk-rejecting AI-Town-unanimous windows. Post-cutover the join returns `NULL` for every AI-Town turn → every AI-Town-only window misclassifies as `keep` instead of `external` → **the purge tool silently stops rejecting the AI-Town backlog, reporting "nothing to do."** |
| `scripts/print_recent_turn_effects.py` | (a) | Generic debug dump, no platform logic. |
| `scripts/smoke_aitown_crystallization_gate.py` | **(b)** — confirmed | Same `LEFT JOIN` pattern as above, used to replay/prove the live formation gate discards AI-Town windows. Post-cutover: `resolved_platform` becomes `None` for AI-Town windows, `would_discard` collapses toward 0 — **the smoke starts reporting a false negative on the exact thing it verifies**, even though the deployed gate (which reads `provenance->>'source_platform'`, set independently at crystallization-build time) keeps working correctly. |
| `scripts/smoke_memory_consolidation_pipeline.py` | (a) | Tests its own synthetic self-issued turn, not real AI-Town traffic. |
| `scripts/sql/turn_effect_timeseries_postgres.sql` | (a) | Generic dashboard query; narrows to real signal post-cutover. |
| `services/orion-actions/app/workflow_schedule_bootstrap.py` | (a) | String literal only, not a query. |
| `services/orion-cortex-exec/app/verb_adapters.py` | (a) | Drives the same real-conversation-scoped `discussion_window` path as batch 1's `sql_fetch.py`. |
| `services/orion-cortex-orch/app/workflow_runtime.py` | (a) | Same discussion-window path, journal-drafting use case. |
| `services/orion-dream/app/aggregators_sql.py` | (a) | Random 24h sample folded into dream/reverie material; losing NPC-dialogue noise improves fidelity. |
| `services/orion-embodiment/app/settings.py` | not a reader | Comment only; producer-side config. |
| `services/orion-embodiment/app/worker.py` | not a reader | **This is the producer** — publishes the AI-Town `chat.history` envelopes in the first place. Needs its own separate producer-side repointing in Phase 3 (already handled structurally by Phase 1's dual-write; Phase 3 flips it to single-write), not a Track B Phase 2 "add a read" fix. |

### Batch 4 — 8 bucket (a) (some not real readers), 1 deferred-(b) caveat

| file | bucket | reasoning / consequence |
|---|---|---|
| `services/orion-hub/app/settings.py` | (a) | Config/channel names only. |
| `services/orion-hub/scripts/api_routes.py` | (a) | Writer, not a reader. |
| `services/orion-hub/scripts/chat_history_rehydrate.py` | (a) | Session-scoped (`WHERE session_id = :sid`); AI Town rows use `session_id="aitown:{convo_id}"`, structurally never matches a Hub websocket session. |
| `services/orion-hub/scripts/chat_turn_trace_routes.py` | (a) | Its own docstring says it deliberately does not read `chat_history_log` today. |
| `services/orion-hub/scripts/concept_atlas_routes.py` | (a) today; **deferred (b)** | Already filters AI Town OUT of Orion's own corpus (PR #1721/#1726, shipped) — unaffected by the split. BUT the design doc names this file as the future home of "AI Town's own concept graph" (deferred, not yet built) — when that ships, it needs to read `aitown_chat_history_log`, or it would ingest zero documents. Nothing to do here today. |
| `services/orion-hub/scripts/endogenous_outreach.py` | (a) | Same session-scoping exclusion as `chat_history_rehydrate.py`. |
| `services/orion-hub/scripts/skill_runner_catalogue.py` | (a) | Pure string dispatch table, no query. |
| `services/orion-memory-consolidation/app/retry_degraded_classifies.py` | (a) | Reads unfiltered today, but provably inconsequential for AI-Town rows — `formation_policy.py`'s unconditional `DISCARD` makes their classification quality downstream-irrelevant regardless. |
| `services/orion-memory-consolidation/app/settings.py` | (a) | Config only (the `MEMORY_FORMATION_DISCARD_PLATFORMS` gate itself lives in `formation_policy.py`, not here). |
| `services/orion-meta-tags/app/settings.py` | (a) | Config only (`RECALL_FALKOR_DISCARD_PLATFORMS`, consumed elsewhere from the bus, not a direct SQL read here). |

### Batch 5 — 2 bucket (a), 7 bucket (b) — the headline finding

| file | bucket | reasoning / consequence |
|---|---|---|
| `services/orion-recall/app/chat_source_tagging.py` | **(b)** | The canonical tagging library itself — no query of its own, but all 6 files below import it. Goes functionally dark for AI Town content the moment its callers stop seeing AI-Town rows. |
| `services/orion-recall/app/settings.py` | (a) | Config constants only. |
| `services/orion-recall/app/sql_chat.py` | **(b)** | `fetch_chat_turns_by_id` — shared join used by 3 independent Falkor-backed consumers below. Post-cutover: silent drop on join-miss for every AI-Town turn discovered via Falkor. |
| `services/orion-recall/app/sql_timeline.py` | **(b)** | 3 functions, live `ILIKE` search over prompt/response. Any recall query about an AI-Town-only conversation returns **zero hits**. |
| `services/orion-recall/app/storage/falkor_chat_adapter.py` | **(b)** | Fragment-build loop `continue`s silently on join-miss — recall fragment coverage degrades with no visible error. |
| `services/orion-recall/app/storage/falkor_neighborhood_adapter.py` | **(b)** | Same join-miss failure mode for entity-based recall specifically. |
| `services/orion-recall/app/storage/sql_adapter.py` | **(b)** | `kind='chat'` fragment output **shrinks ~90%** post-cutover (matches AI Town's current share of table volume). |
| `services/orion-recall/app/worker.py` | **(b)** | Two dependencies: entity-relatedness injection (silent join-miss drop) and `_window_rdf_chatturn_candidates` time-windowing (unresolvable row treated as "outside the window," **fully excluded** from time-windowed recall, not just missing a timestamp). |
| `services/orion-topic-foundry/app/pipelines/chat_corpus_builder/repository.py` | (a) | Unfiltered read, but the pipeline's own content detectors (code/log/traceback regexes) show it's built for engineering conversation, not NPC roleplay — no AI-Town-awareness anywhere downstream to preserve. |

## Bucket (b) tally — real migration work, by service

| service | files | shared fix shape |
|---|---|---|
| `orion-recall` | 7 (`chat_source_tagging.py`, `sql_chat.py`, `sql_timeline.py`, `storage/falkor_chat_adapter.py`, `storage/falkor_neighborhood_adapter.py`, `storage/sql_adapter.py`, `worker.py`) | The real bulk of Phase 2. `sql_chat.py::fetch_chat_turns_by_id` is the shared join point most of these route through — the natural place to add a second query against `aitown_chat_history_log` and merge results, rather than touching all 7 call sites independently. |
| `scripts/` (crystallization gate tooling) | 2 (`bulk_reject_aitown_proposals.py`, `smoke_aitown_crystallization_gate.py`) | Both use the identical `LEFT JOIN chat_history_log h ON h.correlation_id = w.correlation_id` pattern — needs a second `LEFT JOIN aitown_chat_history_log` (or a `UNION`) so `client_meta`/platform still resolves post-cutover. |
| `scripts/` (Falkor entity-graph backfill trio) | 3 (`backfill_recall_falkor_chat_tags_snapshot.py`, `_extract_and_write.py`, `_cleanup_reconcile.py`) | One-off/occasional tooling, not live services. Only the snapshot script (`_snapshot.py`) actually queries Postgres — fix there covers all 3. |

**12 files total need a real code change** (not counting `concept_atlas_routes.py`'s deferred future need, or `orion-embodiment`'s producer-side repointing, both already tracked elsewhere).

## What Phase 2 migration actually looks like (not started)

This audit is Phase 2's first deliverable per the design doc
("`Phase 2 (consumer audit + migration, reviewed per-consumer, not
batch-decided here)`"). The migration itself — actually adding the
`aitown_chat_history_log` reads to the 12 bucket-(b) files — is real,
separate work, likely more than one PR:

1. **`orion-recall`** (7 files, one shared join point): probably its own PR.
   Real design question not answered here: does `fetch_chat_turns_by_id`
   query both tables and merge (`UNION ALL` or two queries + dict merge), or
   does the caller need to know which table a given `turn_id` lives in
   up front? Needs its own investigation, not decided in this audit.
2. **Crystallization gate tooling** (2 files, identical pattern): small,
   mechanical, likely one PR alongside orion-recall or standalone.
3. **Falkor entity-graph backfill trio** (3 files via 1 real query site):
   smallest, lowest-frequency-run of the three groups.

None of this should start without confirming Phase 1's dual-write has
actually been running long enough to have real AI-Town rows in
`aitown_chat_history_log` to test against — as of this audit,
`SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED` is still `false` (Phase 1 shipped off
by default, per its own PR report).

## Non-goals

- No code changes in this PR — audit only, per Track B's own phased
  sequencing (each phase reviewed on its own).
- No pre-decision on `orion-recall`'s exact fix shape (union vs. two-query
  merge vs. something else) — that's the next PR's design question.
- No Phase 3 (cutover) planning here — Phase 3 is explicitly gated on every
  bucket-(b) consumer identified here actually shipping first.

## Acceptance checks (for this audit)

- All ~50 real readers traced with direct file reads, not inferred from
  names or grep alone — done, 5/5 batches.
- Every bucket-(b) classification names a concrete, specific failure mode,
  not a vague "might be an issue" — done, see per-file tables above.
- Cross-checked one subagent's speculative "gap" claim
  (`orion/memory/crystallization/intake_consolidation_window.py`) directly
  against the file before including it — confirmed it has zero DB access
  and is not actually a `chat_history_log` reader; excluded from the final
  count rather than propagated as a false lead.
