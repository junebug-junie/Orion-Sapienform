-- grammar_traces retention + Atlas listing indexes (2026-08-20).
--
-- WHY. grammar_traces was the one grammar table with no retention at all, while its
-- children (grammar_events/grammar_atoms/grammar_edges) are pruned at 3 days. Measured
-- live before this patch: 487,970 trace rows, oldest 2026-07-23, and 205,465 of them
-- (42%) already had ZERO atoms -- the Grammar Atlas was already listing hollow traces
-- whose contents had been pruned out from under them. That is CLAUDE.md's "empty-shell
-- cognition": a UI panel rendered with no real backing artifact.
--
-- Two separate indexes, for two separate queries:
--
--   (created_at, trace_id)  retention's batched delete, which does
--                           WHERE created_at < :cutoff ORDER BY created_at, trace_id
--                           LIMIT :batch_size. Same shape as idx_grammar_atoms_created_at
--                           and its siblings, added for the same reason.
--
--   (started_at DESC, trace_id)  orion/grammar/query.py::list_traces, which the Atlas
--                           calls on every page load:
--                           ORDER BY started_at DESC LIMIT 50. Measured live with
--                           EXPLAIN (ANALYZE, BUFFERS) against the real 143 MB table:
--                           Parallel Seq Scan over all 487,970 rows, 11,523 blocks read,
--                           58.8 ms, every single load. The table has exactly one index
--                           today (the trace_id primary key), which this query cannot use.
--
-- CONCURRENTLY: cannot run inside a transaction block. Run each statement on its own.
-- An interrupted build leaves an INVALID index -- drop it and re-run rather than
-- assuming the index is present.

create index concurrently if not exists idx_grammar_traces_created_at
    on grammar_traces (created_at, trace_id);

create index concurrently if not exists idx_grammar_traces_started_at
    on grammar_traces (started_at desc, trace_id);
