# Investigation: is `grammar_traces`' `trace_id`-only conflict key safe to fix now?

Date: 2026-07-16
Branch: `chore/grammar-traces-conflict-key-investigation` (docs-only, no code changes)
Trigger: PR #1068 (`fix/trace-id-collision`, merged) accepted-not-fixed finding —
`orion/grammar/ledger.py::_upsert_trace()`'s conflict key is `trace_id` alone,
so any future producer/lane pairing that reproduces the same `trace_id` would
reproduce the same class of ledger clobber that #1068 fixed by convention
(giving harness its own `"harness_motor"` lane) rather than structurally.

## Conclusion

**Not safe to implement as a thin patch right now.** No code changed on this
branch. Changing `_upsert_trace()`'s conflict key to `(trace_id,
source_service)` alone would be an *incomplete and actively misleading* fix —
it would stop the `grammar_traces` summary row (`started_at`/`ended_at`/
`status`) from being clobbered, but every other table that stores the
substantive grammar content (`grammar_events`, `grammar_atoms`,
`grammar_edges`, `grammar_temporal_hops`, `grammar_compactions`,
`grammar_projections`) is *also* keyed and queried by `trace_id` alone, with
no `source_service` (or equivalent) discriminator anywhere in their schema or
in any read-path query. A real fix to "two producers can't safely share a
`trace_id`" requires coordinated changes across at least 4 files/services,
not one. Doing only the `grammar_traces` piece would produce a schema-valid
change that doesn't fix the actual risk — the exact "empty-shell" anti-pattern
this repo's contract (`AGENTS.md` §0A) warns against.

The right general fix for this collision *class* is the one #1068 already
used: keep `trace_id` itself collision-resistant per producer (lane
suffixing), and treat this doc's finding as confirmation that the DB-level
compound-key approach is a trap, not a to-do.

## (a) Schema safety — does anything assume one `grammar_traces` row per `trace_id`, and would other tables need to become compound too?

### Schema itself

`services/orion-sql-writer/app/models/grammar_trace.py`:

- `GrammarTraceSQL.trace_id` (L10): `Column(String, primary_key=True)` — single-column PK. No `ForeignKey` declarations anywhere in this file.
- `GrammarEventSQL.trace_id` (L26): plain `String`, `nullable=False`, indexed (`idx_grammar_events_trace_id`, L45) — **not** a real DB foreign key to `grammar_traces.trace_id`, just a convention-only join column. Has its own `source_service` column (L37) but it is not part of any index or constraint tying it to `grammar_traces`.
- `GrammarAtomSQL.trace_id` (L55), `GrammarEdgeSQL.trace_id` (L84), `GrammarTemporalHopSQL.trace_id` (L108), `GrammarCompactionSQL.trace_id` (L129), `GrammarProjectionSQL.trace_id` (L146): all plain `String`, all indexed by `trace_id` only (or not indexed at all, for compactions/projections). **None of these tables have a `source_service` column at all.**

So today, nothing in the schema enforces "one `grammar_traces` row per
`trace_id`" via a DB constraint beyond the PK itself — but every child table's
only way to associate itself with "its" trace is the bare `trace_id` string.
If two producers emit under the same `trace_id`, their atoms/edges/hops/
compactions/projections **already land in the same rows of the same child
tables today**, indistinguishable from each other, regardless of anything
done to `grammar_traces`.

### `orion/grammar/ledger.py::_upsert_trace()` (L46–94)

Confirmed exactly as PR #1068 described: `on_conflict_do_update`/
`on_conflict_do_nothing(index_elements=["trace_id"])`, no `source_service` in
the key (L75, L87, L92). `apply_grammar_trace_batch()` (L292–312) calls
`_upsert_trace()` once per `trace_started`/`trace_ended` event in a batch,
then bulk-inserts events (`on_conflict_do_nothing(index_elements=["event_id"])`,
L230) and derived atoms/edges/hops/compactions/projections
(`on_conflict_do_nothing` on each table's own PK, L259/265/271/277/283) —
**all independent of trace ownership**. Two producers sharing a `trace_id`
today: their `trace_started` events race for the single `grammar_traces` row
(the bug #1068 fixed for one specific pairing); their `grammar_events`/
`grammar_atoms`/`grammar_edges` rows do **not** collide on insert (different
`event_id`/`atom_id`/`edge_id`) but **do** end up filed under the same
`trace_id`, merged in every read.

### Read path — `orion/grammar/query.py`

Every trace-scoped read function filters children by `trace_id` alone, with
no `source_service` awareness:

- `list_traces()` (L184–206): `session.query(GrammarTraceSQL)` with no
  ordering tiebreaker beyond `started_at.desc()` — if two rows for the same
  `trace_id` existed (post compound-key change), both would show up as two
  separate "traces" in this list, each showing the same downstream
  atom/edge/layer counts (`_count_atoms`, `_count_edges`, etc. at
  L146–181, all filtered by `trace_id` alone) — i.e. **both list rows would
  report identical, merged atom/edge counts**, which is actively confusing
  UI output, not a fix.
- `get_trace()` (L209–258): `session.query(GrammarTraceSQL).filter(trace_id
  == trace_id).first()` (L210–214) — **`.first()` with no deterministic
  order**. If a compound key let two rows exist, this silently returns
  whichever one Postgres happens to return first (no `ORDER BY`), while the
  `atoms`/`edges`/`hops`/`compactions`/`projections` queried right after
  (L218–247) are **still pulled by `trace_id` alone** — i.e. they'd include
  content from *both* producers' events even though only one arbitrary
  `grammar_traces` metadata row is shown. This is the single clearest piece
  of evidence that the compound-key fix is incomplete: the "fixed" table's
  read function still returns non-deterministic single-row metadata glued to
  a merged, two-producer content graph.
- `get_trace_graph()` (L319–361): same pattern — `.first()` on
  `GrammarTraceSQL` for existence-check only, then atoms/edges pulled by
  `trace_id` alone (L333–342) and fed through `_trace_graph_atom_ids()`
  (BFS/graph-view construction) with zero producer awareness. The rendered
  graph for a colliding `trace_id` would silently interleave two unrelated
  producers' atoms/edges as if they were one causal graph.
- `get_atom_neighborhood()` (L364–408) and `get_temporal_path()`
  (L513–582): both start from a single `atom_id` (already globally unique
  by construction, not trace-scoped), then pull *all* edges/hops for that
  atom's `trace_id` (L379–383, L528–532) with no `source_service` filter —
  same merge risk.
- `get_atom_provenance()` (L411–510): pulls edges/compactions/projections
  by `trace_id` alone (L431–496) — same merge risk.

### API / consumers

- `services/orion-hub/scripts/grammar_atlas_routes.py` — the Grammar Atlas
  read API (`GET /api/substrate/atlas/traces`, `/traces/{trace_id}`,
  `/traces/{trace_id}/graph`, `/atoms/{atom_id}/...`) is a thin wrapper
  around every `orion/grammar/query.py` function above (L152–228) and
  inherits every merge risk described there. No additional trace_id
  assumptions of its own.
- `orion/substrate/execution_loop/reducer.py` — `ExecutionTrajectoryProjectionV1.runs`
  is an **in-memory dict keyed by `trace_id`** (L52–66, L171–174), separate
  from the SQL ledger entirely. This was already flagged in #1068's report
  as incidentally protected by the lane fix. A DB-level compound key on
  `grammar_traces` does **nothing** for this in-memory reducer — it would
  need its own `(trace_id, source_service)` (or lane-in-id) disambiguation
  independently. Confirms the fix genuinely has to happen at the trace_id
  *value* level (as #1068 already did), not just in one SQL table.
- `orion/substrate/execution_loop/grammar_extract.py` — parses
  `correlation_id`/`node_id` back out of `trace_id` via
  `parse_execution_trace_id()` (`cortex.exec:{node}:{correlation_id}[:{lane}]`
  format, `orion/substrate/execution_loop/ids.py` L4–27). This is the exact
  mechanism #1068 used to fix the specific collision (giving harness a
  `:harness_motor` lane). Confirms `trace_id` generation is **deterministic**,
  not random-UUID-based, for the `cortex.exec:` namespace — i.e. real
  collisions between two producers computing the same `(node, correlation_id,
  lane-or-none)` tuple are the actual, structural risk (not a theoretical
  edge case), which is exactly what happened in the incident #1068 fixed.
- `orion/substrate/chat_loop/grammar_extract.py` — separate `trace_id`
  namespace (`CHAT_TRACE_PREFIX`, parsed via `_parse_trace_id()`, L24–28),
  own producer, not audited further here (out of scope — different
  namespace, no evidence of collision with `cortex.exec:`).
- `services/orion-bus/app/grammar_emit.py` — `BusTransportGrammarCollector`
  has its own `bus_transport_trace_id(node_id, sample_window_id)` namespace
  (L17), explicitly called out in #1068's report as "not audited... out of
  scope for this specific cortex.exec: namespace collision." Still true here.
- `orion/spark/concept_induction/dossier.py` — reads `trace_id` off bus
  envelopes (`env.trace.get("trace_id")`, L15-19) for provenance tagging
  only; does not join against `grammar_traces` or assume single-row
  ownership.

### Would other tables need to become compound too?

Yes, structurally — if the actual goal is "grammar data from two producers
sharing a `trace_id` never gets treated as one causal trace," then
`grammar_events`, `grammar_atoms`, `grammar_edges`, `grammar_temporal_hops`,
`grammar_compactions`, and `grammar_projections` would all need a
`source_service` (or equivalent) column, and **every** `orion/grammar/query.py`
read function (7 of them) would need to accept/filter on it, and the Grammar
Atlas API routes would need a way to pass it through (a `trace_id` alone is
no longer sufficient to unambiguously identify "a trace" for `GET
/traces/{trace_id}` once collisions are DB-legitimate rather than DB-rejected).
That is real, multi-file, multi-service fan-out — not a thin seam.

## (b) Worth doing now vs. leaving as documented accepted risk

**Leave as documented accepted risk — do not implement the compound-key
change.** Reasoning:

1. **The compound key on `grammar_traces` alone is worse than useless as a
   "collision fix."** It would silence the specific symptom #1068 found
   (clobbered `started_at`/`status`) while leaving the substantively more
   important data (atoms, edges — the actual grammar/cognition content) still
   merged across producers in every read path. A future engineer seeing "we
   fixed the conflict key" could reasonably assume the collision class is
   closed, when the real risk (two unrelated causal graphs rendered as one)
   would still be live. That's a regression in trustworthiness of the fix,
   not an improvement.
2. **A complete fix has real cross-service fan-out**, matching the task's
   "not safe to implement quickly" criterion exactly: 6 SQL tables' schemas
   (migration + SQLAlchemy models in `services/orion-sql-writer/app/models/grammar_trace.py`),
   7 read functions in `orion/grammar/query.py`, the Grammar Atlas API
   surface (`services/orion-hub/scripts/grammar_atlas_routes.py`) and
   presumably its UI consumer, plus the independent in-memory
   `ExecutionTrajectoryProjectionV1.runs` reducer
   (`orion/substrate/execution_loop/reducer.py`) which a DB-level fix
   wouldn't even touch.
3. **The already-proven fix is cheaper and more correct: keep `trace_id`
   collision-resistant at generation time**, as #1068 did. The `cortex.exec:`
   namespace's `lane` mechanism
   (`orion/substrate/execution_loop/ids.py::cortex_exec_trace_id()`) already
   exists precisely for this. The residual risk isn't "the DB doesn't enforce
   uniqueness correctly" — it's "a future producer might forget to pick a
   distinct lane." That's a convention/discipline gap, better addressed by:
   - a lint/test-level guard (e.g. a repo-wide test enumerating every known
     `GrammarCollector`-shaped class's `trace_id` property and asserting no
     two produce the same string for the same `(node, correlation_id)` input
     — cheap, deterministic, catches the *actual* failure mode), or
   - a comment/registry of reserved lane names (the mitigation #1068 already
     applied to `CORTEX_EXEC_ISOLATED_TRACE_LANES`).
4. Both known additional producer namespaces (`BusTransportGrammarCollector`
   in `services/orion-bus/app/grammar_emit.py`, chat_loop's
   `CHAT_TRACE_PREFIX`) were flagged as unaudited in #1068 and remain
   unaudited here — another reason a DB-level "fix" would be premature: we
   don't yet have full inventory of every trace_id-producing surface to know
   whether a compound key's second column should even be `source_service`
   (there may be other legitimate reasons two producers share a session
   without being a "collision" — e.g. a well-known co-processing pattern —
   that a blanket compound key would need to distinguish from bugs).

## What a real fix would require, if picked up later

1. Inventory every `trace_id`-producing collector (`HarnessGrammarCollector`,
   `CortexExecGrammarCollector`, `BusTransportGrammarCollector`, chat_loop's
   collector, and any others) and confirm each has a namespace/lane scheme
   that makes accidental cross-producer collisions structurally impossible
   (not just conventionally avoided).
2. Add a deterministic test (not an eval) that constructs two of these
   collectors with intentionally-colliding inputs (same node/correlation_id
   where applicable) and asserts their `trace_id` properties differ — a
   standing regression guard against the entire collision class, not just
   the one instance #1068's tests cover.
3. Only if (1)+(2) still leave a real gap (e.g. a producer that cannot be
   given a distinct namespace/lane for architectural reasons) would a
   DB-level compound key be worth it — and at that point it must be done
   across all 6 grammar child tables and all 7 `query.py` read functions in
   one coordinated changeset, with the Grammar Atlas API updated to accept
   an optional `source_service` disambiguator on trace-scoped routes.

## Recommendation

Do not touch `orion/grammar/ledger.py`'s conflict key right now. If this
keeps surfacing, the next concrete patch to pick up is item 2 above (a
cross-collector `trace_id`-uniqueness regression test) — it's genuinely thin,
catches the real failure mode (a forgotten/misused lane), and doesn't require
the multi-table, multi-service migration a compound DB key would.
