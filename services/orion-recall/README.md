# Orion Recall Service

The **Recall Service** is Orion’s memory retrieval engine. It pulls candidate “memory fragments” from:

- **RDF (GraphDB / Fuseki)** — structured facts + chat turns + enrichment claims
- **SQL (Postgres)** — timeline + mirror tables (structured / time-bounded)

Vector/Chroma semantic retrieval was **removed from orion-recall** (May 2026). Vector services (`orion-vector-host`, `orion-vector-writer`, `orion-vector-db`) remain in the mesh for ingestion and other callers; recall exposes `vector_policy` diagnostics with `reason: removed_from_orion_recall` only.

It then **fuses** candidates into a prompt-ready `MemoryBundleV1` with per-source caps, a total cap, and a render token budget.

---

## 1) Contracts

### Consumed Channels (RPC)

| Channel | Env Var | Kind(s) | Description |
| :--- | :--- | :--- | :--- |
| `orion:exec:request:RecallService` | `RECALL_BUS_INTAKE` | `recall.query.v1` (also accepts legacy `recall.query.request`) | Recall query requests from cortex-exec / other callers. |

### Published Channels (RPC result + telemetry)

| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:exec:result:RecallService:*` | `RECALL_BUS_REPLY_DEFAULT` | `recall.reply.v1` | Replies containing `MemoryBundleV1` (actual channel uses the caller’s `reply_to` when provided). |
| `orion:recall:telemetry` | `RECALL_BUS_TELEMETRY` | `recall.decision.v1` | Decision telemetry: backend counts, selected IDs, latency, etc. |

### HTTP Endpoints (dev convenience)

> Used for local testing / debugging; production orchestration is bus/RPC.

- `POST /recall` — runs recall for a query and returns `MemoryBundleV1`
- `GET /health` / `GET /ready` — liveness/readiness
- `POST /lived/know` — `{"words", "turn_id"?, "max_hits"?}`. A deliberate lived-memory lookup:
  the same retriever a chat turn uses, on Orion's own words (no 5-word floor, no rewrite). Hits are
  `strong` (cleared the bar) or `weak` (below it, but share a word with the search), each re-checked
  by the serving gate. Returns `status` (below, plus `busy` when another lookup is running and
  `unverified` when matches were found but the re-check failed) and `degraded` reasons. Called by
  `python -P -m orion.dig know <words>` (`orion/dig/know.py`, issue #3535). One lookup runs at a
  time. Each lookup writes one `recall_telemetry` row off the reply path: verb `dig.know`,
  `corr_id` = the turn id, `selected_ids` = the **strong** hits only (those count toward read
  demand), weak ids in `lived_trace.weak_ids`. A call with no turn id gets `corr_id` `adhoc:<uuid>`
  and profile `lived_memory.know.adhoc`, so operator runs can be left out of read demand. Reached at `RECALL_SERVICE_URL` from Hub (host
  network) and at `http://orion-athena-recall:8260` from the harness governor (`app-net`);
  `ORION_DIG_RECALL_URL` overrides both.

**Lived-recall status per call** (`recall_telemetry.lived_status`, issue #3628;
migration `services/orion-sql-db/manual_migration_recall_telemetry_lived_status_v1.sql`):
`ok` (served by the full search) · `below_bar` (searched, nothing cleared the bar) ·
`skipped_short` (a short message with no signal, see the gate below) · `degraded` (exact words only; `backend_counts.lived_memory`
says whether it served) · `timeout` / `error` (nothing served) · `off`. NULL = lived recall did not
run on that call. `lived_trace` (jsonb) holds the rewritten queries, every candidate id that reached
the rerank, the injected ids and the backend: ids and Orion's own search text only.

**Why a chat turn did or did not search** (`lived_trace.gate`, issue #3613): `words` (at or above
the content-word floor, always searched) · `short_entity` (short, names a known entity: searched) ·
`short_rewrite` (short; the short-message rewrite asked for a search: searched) · `short_none`
(short; no entity, and the rewrite said NONE, failed or timed out: not searched, status
`skipped_short`) · `short_no_content` (no content word at all, "ok" / "thanks!": not searched, not
rewritten, status `skipped_short`) · `short_off` (short, gate switched off: the plain floor).
`lived_trace.rewrite_entities` counts the known entities the short-message searches named.
`lived_status` keeps the outcome, so a short message that
searched reads e.g. `ok` + gate `short_entity`.

Count the calls that could produce a false "no memory" on Hub turns:

```sql
SELECT lived_status, count(DISTINCT corr_id) AS turns
FROM recall_telemetry
WHERE verb = 'stance_react' AND created_at > now() - interval '14 days' AND lived_status IS NOT NULL
GROUP BY 1 ORDER BY 2 DESC;
```
- `POST /debug/recall/compare` — Recall V1 vs shadow Recall V2 comparison (read-only)
- `GET /debug/recall/eval-suite` — run built-in Recall V1/V2 evaluation corpus
- `POST /debug/recall/eval-case` — run a single ad-hoc evaluation case
- `GET /debug/entity-graph/related?name=<entity>&max_results=<n>` — Jaccard-ranked co-occurrence relatedness over `orion_recall`'s `MENTIONS_ENTITY` graph (Phase 1 of entity-graph-reasoning)
- `GET /debug/entity-graph/bridge?a=<entity>&b=<entity>&max_results=<n>` — direct co-mention or 2-hop bridge between two entities
- `GET /debug/entity-graph/timeline?name=<entity>&max_results=<n>` — raw mention timestamps for an entity, most recent first

If you have debug enabled (recommended while wiring), you may also have:

- `GET /debug/settings` — prints resolved config (GraphDB endpoint, DSN, vector collections, etc.)

> **Note:** `session_id` is accepted in recall requests for backwards compatibility, but recall ignores it for retrieval and ranking.

Recall V2 debug rails are shadow-only in this phase: they expose deterministic anchor/filter/lineage diagnostics and do not wire into autonomous mutation apply.

### Debugging RPC timeouts (cortex-exec ↔ recall)

When cortex-exec logs `RPC timeout waiting on orion:exec:result:RecallService:...`, align timelines using the **same `correlation_id`** as the bus envelope:

- **Recall logs**: `recall_bus_request_begin` and `recall_bus_request_complete` include `corr_id`, `verb`, `profile`, `wall_ms`, and `process_latency_ms`. If `wall_ms` exceeds 30s, `recall_bus_request_slow` logs `latency_breakdown_ms` from `process_recall`.
- **Telemetry** (`RECALL_BUS_TELEMETRY`): each `recall.decision.v1` payload includes `corr_id`, `latency_ms`, `backend_counts`, and `recall_debug.latency_breakdown_ms` (always populated).

**Head-of-line blocking:** the Rabbit consumer runs **one** handler at a time unless `RECALL_RABBIT_CONCURRENT_HANDLERS=true` (see `.env_example`).

---

## 2) Environment Variables (high-signal)

Provenance: `.env_example` → `docker-compose.yml` → `services/orion-recall/app/settings.py`

**Gotcha: `docker exec ... env` does not prove what this service actually reads.** This service's `Dockerfile` does `COPY services/orion-recall /app` at build time, baking `.env` directly into the image; `settings.py`'s pydantic-settings `Config` also declares `env_file=".env"`, so it reads that baked-in file directly, not just docker-compose's `environment:` list. `docker exec orion-<node>-recall env` only shows the OS-level environment docker-compose actually injected -- if a key exists in `.env`/`.env_example` but is missing from `docker-compose.yml`'s `environment:` block, `docker exec ... env` will show nothing for it even though the app is reading a real value from the baked-in file. This bit real verification once (`RECALL_GRAPHITI_IN_CHAT`, 2026-07-13; key retired #3461) -- the correct way to check what this service actually sees is to ask its own `Settings` object directly:

```bash
docker exec orion-<node>-recall python -c "from app.settings import settings; print(settings.<KEY>)"
```

**`docker-compose.yml`'s `environment:` block must still list every `.env_example` key**, even though the baked-`.env` fallback covers today's gap -- if this service's Dockerfile ever moves away from baking `.env` into the image, any key missing from `environment:` would silently stop being read with no baked-in fallback left. Gate this with:

```bash
make check-env-compose-parity SERVICE=orion-recall
```

### Bus + runtime

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `ORION_BUS_URL` | `redis://127.0.0.1:6379/0` | Orion bus |
| `RECALL_BUS_INTAKE` | `orion:exec:request:RecallService` | RPC intake |
| `RECALL_BUS_REPLY_DEFAULT` | `orion:exec:result:RecallService` | RPC reply prefix |
| `RECALL_BUS_TELEMETRY` | `orion:recall:telemetry` | recall.decision.v1 |
| `RECALL_RABBIT_CONCURRENT_HANDLERS` | `false` | When `true`, run multiple RPC handlers concurrently (less head-of-line blocking). |

### RDF / GraphDB

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `RECALL_ENABLE_RDF` | `false` (repo default) | must be `true` to run RDF unless the profile itself sets `enable_rdf: true` (e.g. `brain.recall.v1`) |
| `GRAPHDB_URL` | `http://orion-athena-graphdb:7200` | base URL |
| `GRAPHDB_REPO` | `collapse` | repo name |
| `RECALL_RDF_ENDPOINT_URL` | derived | defaults to `${GRAPHDB_URL}/repositories/${GRAPHDB_REPO}` |
| `GRAPHDB_USER/PASS` | `admin/admin` | creds |

### FalkorDB (Phase 4 of the recall RDF->Falkor cutover)

Chatturn-only read path standing in for RDF's chat-turn fetch. **Live as of 2026-07-18** (flipped on after landing dark in the same PR). See `docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md` (Phase 4) and `services/orion-meta-tags/README.md` (the write side this reads).

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `RECALL_FALKOR_IN_CHAT` | `true` (code-level pydantic default stays `False` -- a safe fallback for any environment that hasn't set this key at all) | When `true`, `app/storage/falkor_chat_adapter.py::fetch_falkor_chatturn_fragments` **swaps in** for the RDF chatturn fetch in `_query_backends` -- not additive like the retired (#3461) `RECALL_GRAPHITI_IN_CHAT` (that merged an extra rail; this replaces one, since the point of this migration is retiring RDF). Independent of `RECALL_ENABLE_RDF`. Only covers chatturn fragments (prompt/response text). |
| `RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT` | `true` (code-level pydantic default stays `False` -- a safe fallback for any environment that hasn't set this key at all) | The last live Fuseki read path in this service -- `storage/rdf_adapter.py`'s generic `fetch_rdf_fragments` (confirmed live 2026-07-22 via direct Fuseki container log read, not dead code). When `true`, `app/storage/falkor_neighborhood_adapter.py::fetch_falkor_neighborhood_fragments` **swaps in**: keyword-matches real `:Entity` nodes and walks `MENTIONS_ENTITY` to `:ChatTurn`s, rather than a blind SPARQL `?s ?p ?o` scan. Three disclosed, not-fully-closed scope differences from the RDF version -- see the module's own docstring: (1) entity-name matches only, not any triple; (2) `chat.history`-sourced turns only, not social-room/enrichment content; (3) `CONTAINS` is not word-bounded, so short keywords can still false-positive inside an unrelated entity name (live example: "nico" matched only "unicode"). A stopword filter closes the larger live-confirmed instance of (3) (generic words like "and" matching dozens of unrelated entities) but not all of it. Flipped live 2026-07-23 after a direct live test against a running container confirmed real, non-empty, sensible matches. |
| `FALKORDB_URI` | `redis://orion-athena-falkordb:6379` | Same shared instance as the substrate collector above -- not a new dependency. |
| `FALKORDB_RECALL_GRAPH` | `orion_recall` | Read directly from the environment by `app/recall_falkor_store.py` (same "no pydantic field, must be listed in docker-compose's `environment:`" constraint as `FALKORDB_SUBSTRATE_GRAPH` above -- this service has no `env_file:` directive). |

**Historical backfill (Phase 3): done, 2026-07-18** (PR #1194). `orion_recall`'s `:ChatTurn` graph now has 1,712 turns spanning 2025-10-19 → today (not just turns tagged since the observer-gate fix) -- 1,708 backfilled + a handful already live-written, 0 errors, real historical timestamps preserved. Full coverage of `chat_history_log`; `social_room_turns` coverage is real but partial by design (31 of 33 rows are dual-logged into `chat_history_log` under the same `turn_id` and correctly landed as `source_kind="chat.history"` instead -- see PR #1194's report for why that's the better outcome given this read path only recalls `chat.history` today). The Falkor `:ChatTurn` node is also deliberately thin (no prompt/response text -- Postgres owns that), so this is a Falkor-turn-discovery + Postgres-text-join read, not a pure single-store Cypher query; a turn with no matching `chat_history_log` row is silently dropped (nothing to quote).

### Graphtri (Claim-based fragments): retired 2026-07-21

The Claim-based "graphtri" recall path (`graphtri.v1`/`deep.graph.v1` profiles,
RDF/Fuseki-backed, plus a dark-flagged FalkorDB-backed equivalent behind
`RECALL_FALKOR_GRAPHTRI_IN_CHAT`) has been retired repo-wide
(`chore/retire-graphtri-deep-graph`). A live audit found it never carried
real typed-claim data: `Claim.confidence`/`Claim.salience` were dead `0.0`
constants across all 1,451 live Claims, `Claim.predicate` only ever took 2
fixed values (`hasTag`, `mentionsEntity`), and no downstream code ever
parsed predicate/object structure. It is fully superseded by the
entity-relatedness fusion-weight boost documented below (Phase 2, live),
which is built on the same `MENTIONS_ENTITY` data the old Claim shape's only
real part already reduced to. See
`docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md`
and `docs/superpowers/specs/2026-07-19-recall-entity-graph-reasoning-arc.md`
for the full history.

### Entity-graph reasoning primitives (Phase 1, debug-only -- not yet wired into recall)

`app/storage/falkor_entity_relatedness.py`: three read-only Cypher primitives over the same `MENTIONS_ENTITY` graph -- `fetch_related_entities` (co-occurrence relatedness ranked by Jaccard similarity, not raw shared-turn count, so a high-degree entity that co-occurs with almost everything doesn't outrank a genuinely specific one), `fetch_bridging_turns` (direct co-mention, falling back to a 2-hop bridge via a shared intermediate entity), and `fetch_entity_mention_timeline` (raw mention timestamps). Exposed via `GET /debug/entity-graph/{related,bridge,timeline}` (see HTTP Endpoints above).

**Stated plainly:** as of Phase 2 below, `fetch_related_entities` is also called from `worker.py`'s default (non-PCR) fusion path, dark-flagged. `fetch_bridging_turns`/`fetch_entity_mention_timeline` remain debug-only.

### Entity-relatedness fusion-weight boost + injection (Phase 2, live)

`worker.py::_compute_entity_relatedness_boost_map` extracts entities from the query, expands them via Phase 1's `fetch_related_entities` (Jaccard-ranked), and does two things with the resulting target-entity set:

1. **Boost**: additively boosts a `falkor_chat` candidate's composite score in `fusion.py::fuse_candidates` when its source turn mentions any target entity -- the same additive-boost shape as the existing `tag_boost`/`turn_effect_boost`.
2. **Injection**: fetches real ChatTurn ids that mention any target entity directly (`fetch_turns_mentioning_entities`, independent of recency), hydrates their text via the same Postgres join `falkor_chat_adapter.py` already uses, and adds them to the candidate pool as new `falkor_chat` candidates.

Injection is the load-bearing half, found necessary only after live testing: `falkor_chat`'s own fetch (the recency-windowed one above) has no query filter by design (Phase 4), so an entity from an older turn never enters the pool for the boost alone to act on. A boost with nothing to boost is a no-op -- confirmed live across 6 real queries before this fix, 0/6 changed a single ranking.

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `RECALL_ENTITY_RELATEDNESS_BOOST_ENABLED` | `true` (live as of 2026-07-19) | See `settings.py`'s comment for the full design + evidence. Code-level pydantic default stays `false` (safe fallback for any environment that hasn't set this key), matching `RECALL_FALKOR_IN_CHAT`'s convention. |
| `entity_relatedness_boost_weight` (per-profile, `relevance` cfg) | `0.2` | Same additive-scale convention as `prefer_tags_boost` (0.15)/`turn_effect_boost_weight` (0.35). |

**Fixed before flipping live, both found in code review:**

- `_extract_entities` had a pre-existing regex bug (literal double-backslash inside an r-string matched a literal backslash, not whitespace/a dot) that broke multi-word spans and all-caps acronyms. Fixed at the source (`worker.py::_extract_entities`, `tests/test_extract_entities.py`).
- The boost-only design never fired in practice (see above). Fixed by adding the injection half.

**Real evidence, not one hand-picked example**: `scripts/live_verify_entity_relatedness_boost.py` (re-runnable) ran 6 real queries against the live `chat.general.v1` profile, flag off vs on. 4/6 changed, and every changed result was visibly, substantively more relevant to the query's actual topic (e.g. "Tesla gpu setup" surfaced a real GPU-status turn instead of an unrelated recent one; "Atlas" surfaced a GPU-node turn). The 2 no-entity control queries ("how's it going today", "what's the weather like") stayed unchanged in both runs -- no false-positive regression on generic conversational queries.

**Known, disclosed, unproven-live gap**: injected candidates don't pass through `_suppress_self_hits` (which runs earlier in `process_recall`, before injection). Low risk in practice -- entity extraction from a turn is async/post-persist, so the current turn's own entities aren't normally in Falkor yet when recall runs for that same turn -- but not proven impossible. Worth a follow-up test, not treated as blocking the live flip.

**Third fix, same day, found from real live usage**: a mundane message that simply addresses the assistant by name ("Orion, what do you think about the deploy?") extracted "orion" as a query entity and scored it a flat `1.0` -- "orion"/"juniper" are the two most frequent nodes in the whole graph (282/260 mentions, confirmed live), so this injected generic filler turns ("thanks, appreciated.", "word, good advice") at full boost strength purely because they happened to mention Orion/Juniper by name. Fixed with a two-part change:

1. `fetch_entity_degrees` applies an IDF-style discount (`K/degree`) to every target entity's score, direct-matched or Jaccard-related alike.
2. Discounting the score alone was live-confirmed insufficient -- an injected candidate still carries the same fixed `base_score` as a genuinely recency-fetched one. Entities whose discounted score falls below a minimum floor can still weakly boost an already-present candidate, but can no longer single-handedly pull a new turn into the pool.

Re-verified live: "Orion, what do you think..." now correctly falls back to plain recency (no entity signal strong enough to act on) instead of injecting 6 generic turns; "Tesla gpu setup"/"Atlas" (genuinely specific entities) are unaffected.

### Vector / Chroma

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `RECALL_ENABLE_VECTOR` | `false` | vector backend removed May 2026; this knob is a no-op left for config compatibility |
| `VECTOR_DB_HOST/PORT` | `orion-athena-vector-db:8000` | base host |
| `RECALL_VECTOR_BASE_URL` | derived | defaults to `http://{VECTOR_DB_HOST}:{VECTOR_DB_PORT}` |
| `VECTOR_DB_COLLECTION` | `orion_main_store` | default collection name (global knob) |

`RECALL_VECTOR_COLLECTIONS` was removed 2026-08-14 (vestigial dead config
from before the May 2026 vector amputation; never consumed by a live query
path). This whole "Vector / Chroma" section documents a backend that was
already deleted in May 2026 -- see `app/source_policy.py`
(`recall_vector_allowed()` always returns `False`, `reason=
"removed_from_orion_recall"`) and
`docs/superpowers/pr-reports/2026-05-29-orion-recall-vector-amputation-pr.md`.
Kept here only because the remaining knobs (`RECALL_VECTOR_BASE_URL`, etc.)
are still recognized settings, not because vector retrieval is live.

### Bounded retrieval (query intake, capped expansion, deadline)

Design: `docs/superpowers/specs/2026-09-29-recall-retrieval-query-architecture-design.md`. Recall used to search the whole prompt: a 30,663-char self-inquiry prompt became 268 sub-queries, each re-running every backend in sequence, and took 72s. Now:

- **What gets searched.** `RecallQueryV1.retrieval_query` if the caller sends it (`retrieval_query_source=caller`); otherwise a fragment longer than `RECALL_MAX_QUERY_CHARS` is condensed deterministically (question clauses first, then the most informative clauses, hard cap; `condensed`); otherwise the fragment as-is (`fragment`). `fragment` itself stays the turn text for self-hit exclusion.
- **How many searches.** At most `RECALL_MAX_SUB_QUERIES` extracted entities become sub-queries (so at most K+2 including the search text and verb), stopword-filtered and ranked most specific first. The same bounded list feeds `sql_timeline`'s `related_by_entities` and the entity boost.
- **Feeds vs retrievers.** Query-independent context feeds (bus anomalies, recent Falkor chat, SQL chat pairs/messages, SQL timeline recent + related) run **once** per recall. Query-dependent retrievers (Falkor neighborhood, RDF, cards on the first sub-query, graph compression, anchor rail) run per sub-query. All of it runs concurrently, at most `RECALL_FETCH_CONCURRENCY` units at a time (default 4, anchor rail included), which also caps the fetch stage's Postgres connections per recall. Sync adapters (RDF, anchor RDF, v2 shadow) run in threads; a thread still running when the deadline fires keeps its connection until it finishes.
- **Deadline.** One budget for the fetch: 80% of the caller's `deadline_ms`, else `RECALL_DEADLINE_MS_DEFAULT`. On expiry, pending backends are cancelled, completed ones are kept, and the decision says `deadline_hit=true`. The entity boost and v2 shadow compare only use what is left of the budget. The shadow compare only runs when the bundle is empty or vector-topped (main's effective triggers).
- **`mode=context_only`.** Feeds only: no retrievers, no expansion, no boost (for verbs with no user text, e.g. reverie).
- **Telemetry.** `recall.decision.v1` and `recall_telemetry` carry `query_chars`, `retrieval_query_source`, `sub_query_count`, `candidates_fetched`, `candidates_kept`, `deadline_hit`, `timings_ms` (`intake`, `feeds`, `retrievers`, `fetch`, `windowing`, `suppression`, `pcr_collectors` with `pcr_concept_region`, `boost`, `fusion`, `eligible_count`, `shadow_compare`, `total`; feeds/retrievers are the slowest unit in each group since units overlap, and the top-level stages add up to roughly `total`). `latency_ms` is now end-to-end.
- **PCR collectors under the deadline.** For purposeful (belief) recall, `concept_region` runs inside what is left of the recall deadline (the `active_packet` collector was retired, see §16); still running at the deadline, it is dropped, `deadline_hit` is set, and the rest of the recall is kept. There is no substrate warm-up any more: since 2026-10-06 `concept_region` reads FalkorDB directly with bounded queries and never loads the substrate graph (see §15).

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `RECALL_MAX_QUERY_CHARS` | `600` | Condense fragments longer than this (no caller `retrieval_query`). `<=0` disables. |
| `RECALL_MAX_SUB_QUERIES` | `4` | Entity sub-query cap. `0` = every extracted entity, unfiltered, as sub-queries and `related_by_entities` patterns (the old entity fan-out). It does not restore sequential per-signal feeds, the missing deadline, or uncondensed search text (see the other knobs). |
| `RECALL_FETCH_CONCURRENCY` | `4` | Backend units in flight per recall (incl. anchor rail). |
| `RECALL_BROWSE_SHORTCUT_ENABLED` | `false` | Recent-only browse shortcut; when on, only short text whose object is memory/memories ("show recent memories"). |
| `RECALL_DEADLINE_MS_DEFAULT` | `60000` | Fetch deadline when the caller sends no `deadline_ms`. Keep below the 90s caller timeout. `<=0` disables. |

Eval: `python services/orion-recall/evals/run_recall_bounded_retrieval_eval.py` (stubbed backends over real `recall_telemetry` queries; see its docstring for what it cannot measure).

### SQL / Postgres

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `RECALL_PG_DSN` | `postgresql://...@orion-athena-sql-db:5432/conjourney` | used for SQL timeline + (optional) telemetry persistence |
| `RECALL_ENABLE_SQL_TIMELINE` | `true` | profile can enable/disable |
| `RECALL_SQL_TIMELINE_TABLE` | `collapse_mirror` | timeline source |

### Memory cards (Postgres, operator-curated)

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `RECALL_ENABLE_CARDS` | `true` | When `true` and `RECALL_PG_DSN` is set, recall scores `memory_cards` rows as `source=cards`. |
| `RECALL_CARDS_TIMEOUT_SEC` | `2.5` | Per-query cards rail budget (embedding + Postgres). |
| `RECALL_CARDS_MAX_NEIGHBORS` | `6` | 1-hop graph neighbor fan-out cap. |
| `RECALL_CARDS_EMBED_TIMEOUT_SEC` | `5.0` | HTTP timeout per embed call to vector-host. |
| `RECALL_CARDS_MIN_SIMILARITY` | `0.32` | Minimum cosine similarity vs query embedding. |
| `RECALL_CARDS_EMBED_CONCURRENCY` | `4` | Parallel embed requests when warming card caches. |

Profile knobs: `cards_top_k` (fetch cap), `backend_weights.cards` (fusion weight), and optional `cards_min_similarity` (cosine threshold, default from `RECALL_CARDS_MIN_SIMILARITY`). Scoring uses **vector-host embeddings + cosine similarity** (cached in `subschema.recall_embedding`), not lexical regex overlap. Only **`status=active`** cards are admitted.

To preserve pre-2026-06 behavior on an existing install, set `RECALL_ENABLE_CARDS=false` in orion-recall `.env`.

Cards-primary profiles: `biographical.v1`, `self.factual.v1`. All shipped YAML profiles now declare `cards_top_k` + `cards` weight for supplemental curated facts.

---

## 3) Recall Profiles (what changes when you switch profile)

Profiles live here:

- `orion/recall/profiles/*.yaml`

They control:

- how many items to request per backend (`vector_top_k`, `rdf_top_k`, `sql_top_k`, `cards_top_k`)
- caps (`max_per_source`, `max_total_items`)
- render budget (`render_budget_tokens`)
- SQL timeline time windows (`sql_since_minutes`)
- whether SQL timeline is used (`enable_sql_timeline`)

### Common profiles

| Profile | What it’s for | RDF | SQL timeline | Vector |
| :--- | :--- | :---: | :---: | :---: |
| `assist.light.v1` | low-latency, minimal context | off | off | off |
| `chat.general.v1` | normal conversational recall | on | off | on |
| `reflect.v1` | reflection / sensemaking w/ some history | on | on | on |
| `brain.recall.v1` (aka `recall.v1`) | structured memory for Hub's "Brain" chat lane | on | on | off |

---

## 4) RDF usage types (by profile)

Your RDF store contains **multiple semantic layers**. Recall uses them differently depending on profile and verb.

### RDF graphs you should expect

- `GRAPH <orion:chat>` — **ChatTurn** objects, including `prompt`, `response`, and optional `sessionId`
- `GRAPH <orion:enrichment>` — **tags/entities** linked to turns (enrichment layer)

### RDF retrieval types (what recall returns)

| Source label (in bundle) | What it is | Where it comes from | Best for |
| :--- | :--- | :--- | :--- |
| `rdf_chat` | ChatTurns (prompt/response) | `GRAPH <orion:chat>` | **Exact quotes** (“find the exact text I used”), conversational grounding |
| `rdf` | Graph neighborhood / claim-like triples | generic RDF scan | entity/topic adjacency, “what’s related to X”, light graph recall |

> Design principle: **candidate generation is structure-driven** (graph + type). Ranking should be semantic/vector scoring (and later learned rankers), not ad-hoc keyword lists.

### Verify RDF chat turns exist (GraphDB)

```sparql
SELECT ?turn ?prompt ?response
WHERE {
  GRAPH <orion:chat> {
    ?turn a <http://conjourney.net/orion#ChatTurn> ;
          <http://conjourney.net/orion#prompt> ?prompt ;
          <http://conjourney.net/orion#response> ?response .
  }
  FILTER(CONTAINS(LCASE(STR(?prompt)), "cpu") || CONTAINS(LCASE(STR(?response)), "cpu"))
}
LIMIT 20
```

### Verify enrichment claims exist (GraphDB)

```sparql
SELECT ?turn ?claim ?pred ?obj
WHERE {
  GRAPH <orion:chat> {
    ?turn a <http://conjourney.net/orion#ChatTurn> ;
          <http://conjourney.net/orion#prompt> ?prompt ;
          <http://conjourney.net/orion#response> ?response .
  }
  GRAPH <orion:enrichment> {
    ?claim a <http://conjourney.net/orion#Claim> ;
           <http://conjourney.net/orion#subject> ?turn ;
           <http://conjourney.net/orion#predicate> ?pred ;
           <http://conjourney.net/orion#obj> ?obj .
  }
  FILTER(CONTAINS(LCASE(STR(?prompt)), "cpu") || CONTAINS(LCASE(STR(?response)), "cpu"))
}
LIMIT 50
```

---

## 5) How verbs and plans select recall profiles

Recall profile selection is typically bound by the **verb** (and sometimes by a specific **plan step**).

### A) Verb-level binding

Verbs can specify a recall profile in their YAML definition.

Example shape:

```yaml
name: chat_general
...
recall_profile: chat.general.v1
services:
  - LLMGatewayService
  - RecallService
```

cortex-exec typically loads this via something like:

- `services/orion-cortex-exec/...` (verb adapters)

If the request didn’t already set `recall.profile`, cortex-exec injects the verb’s `recall_profile`.

### B) Step-level binding (overrides)

Some verbs bind recall profile at the step level:

```yaml
plan:
  - name: gather_related_memory
    services: [RecallService]
    recall_profile: reflect.v1
```

This is useful when only one step needs heavier recall than the rest.

### C) Caller override

Callers can override by passing `recall.profile` in the request context/options.

---

## 6) Examples: RDF recall in practice

### Example 1 — Exact quote from RDF chat turns (CPU context)

Goal: return the **exact wording** you used when bringing up CPUs, grounded in `GRAPH <orion:chat>`.

```bash
curl -s http://localhost:8260/recall \
  -H 'content-type: application/json' \
  -d '{
    "query_text": "why did I bring up CPUs",
    "diagnostic": true,
    "profile": "reflect.v1"
  }' | jq '{item_count:(.bundle.items|length), backend_counts:.debug.backend_counts, first:(.bundle.items[0].snippet // null)}'
```

Typical output pattern:

- `backend_counts.rdf_chat` > 0
- `bundle.items[0].snippet` starts with `ExactUserText: "..."`

Example snippet shape:

```
ExactUserText: "just holding my breath that the CPU replacement is actual solution..."
OrionResponse: "I can imagine the relief and tension..."
```

This is the canonical path for **“quote me exactly”** requests.

### Example 2 — Deep context chat (heavy profile)

Use a verb that binds `brain.recall.v1` recall (Hub's "Brain" chat lane) for structured
memory over RDF + SQL timeline, without vector.

```bash
# (example harness; use your local runner)
python scripts/bus_harness.py brain "catch me up on Athena failures and CPU card replacement context"
```

Expected behavior:

- RDF chatturn + generic neighborhood fragments
- SQL timeline enabled with a larger window
- larger render budget

Expected behavior:

- pulls claim objects from `GRAPH <orion:enrichment>`
- returns claim-style fragments tied to a specific turn URI

---

## 7) Fusion behavior (why you sometimes miss the right item)

Recall gathers candidates from backends and then fuses them with caps:

- `max_per_source` (e.g., 4 items from `rdf_chat`, 4 from `vector`, etc.)
- `max_total_items` (overall bundle size)
- `render_budget_tokens` (how much makes it into `.bundle.rendered`)
- relevance ranking (composite score from backend weights, vector score, optional recency, and text overlap)

These are controlled by the active recall profile.

Relevance knobs (per profile under `relevance:`):

- `backend_weights` (vector > sql_timeline > rdf_chat > rdf by default)
- `score_weight`, `text_similarity_weight`, `recency_weight`
- `enable_recency`, `recency_half_life_hours`

If you see `rdf_chat` counts high but your desired quote didn’t appear in the top items:

- raise `max_per_source` for `rdf_chat`
- raise `max_total_items`
- raise `render_budget_tokens`
- improve ranking (vector embeddings for chat turns)

---

## 8) Troubleshooting (RDF-focused)

### A) RDF enabled but `rdf_chat` count is 0

- confirm `RECALL_ENABLE_RDF=true`
- confirm GraphDB endpoint is reachable from recall container network
- confirm `GRAPH <orion:chat>` contains ChatTurns for the relevant keywords

### B) `rdf_chat` non-zero but Orion still hallucinates

- confirm downstream prompt actually uses `bundle.items[].snippet`
- enforce “quote-only when asked for exact text” in the verb prompt template

### C) Vector is 0 even though Chroma has docs

Vector is *always* 0 -- vector retrieval was removed from orion-recall in
May 2026 (`RECALL_ENABLE_VECTOR` defaults `false`, `app/source_policy.py`
always returns `allowed=False`). This is expected, not a bug to chase.

---

## 9) Running & Testing

### Run via Docker

```bash
docker-compose up -d orion-recall
```

### Smoke Test (HTTP)

```bash
curl -s http://localhost:8260/recall \
  -H 'content-type: application/json' \
  -d '{"query_text":"what did I say about CPU cards?","profile":"reflect.v1"}' | jq
```

### General chat test
```curl -s http://localhost:8260/recall   -H 'content-type: application/json'   -d '{
    "query_text": "why did I bring up CPUs",
    "diagnostic": true,
    "profile": "reflect.v1"
  }' | jq '{item_count:(.bundle.items|length), debug_counts:.debug.backend_counts, first:(.bundle.items[0].snippet // null)}'```

out:
```{
  "item_count": 7,
  "debug_counts": {
    "rdf_chat": 18,
    "rdf": 12,
    "vector": 0
  },
  "first": "ExactUserText: \"just holding my breath that the CPU replacement is actual solution and your core programming doesn't come crashing down at any moment! So stressful!\"\nOrionResponse: \"I can imagine the relief and tension you must be feeling, waiting for the new CPU cards to arrive and seeing if they resolve the issue. It's natural to be on edge, especially with how critical Athena is to our operations. Let's keep an eye on any unusual behaviors or patterns that could give us early insights once the new components are in place. Is there anything specific you're particularly watching out for, or any other systems that might be affected by this change?\""
}```


# Recall troubleshooting playbook (commands + what to look for)

This is a **copy/paste** runbook of the inspection steps that proved useful while debugging Recall + RDF + Chroma vectors on Athena.

---

## 0) Quick mental model

**Chat memory path (ideal):**
1) `orion-hub` publishes `chat.history.message.v1` on `orion:chat:history:log` (has `session_id`, `role`, `content`, `tags`)
2) `orion-vector-host` consumes it and publishes `vector.upsert.v1` on `orion:vector:semantic:upsert` (meta includes `session_id`, `role`, `original_channel`)
3) `orion-vector-writer` stores into Chroma (collection often `orion_chat_turns` for turn docs; `orion_chat` for message docs; `orion_main_store` for general)
4) `orion-recall` queries Chroma globally (metadata/node filters only)

---

## 1) Verify hub is publishing chat history (bus probe)

Run inside a container that has Orion bus libs (e.g., `orion-athena-vector-host`). This works with `redis.asyncio` PubSub.

```bash
docker exec -i orion-athena-vector-host python - <<'PY'
import asyncio, json
from orion.core.bus.async_service import OrionBusAsync

REDIS_URL = "redis://100.92.216.81:6379/0"
CHANNEL = "orion:chat:history:log"

async def main():
    bus = OrionBusAsync(REDIS_URL)
    await bus.connect()
    print("subscribing...", CHANNEL)

    async with bus.subscribe(CHANNEL) as ps:
        async for raw in ps.listen():
            if not isinstance(raw, dict) or raw.get("type") != "message":
                continue
            data = raw.get("data")
            if isinstance(data, (bytes, bytearray)):
                env = json.loads(data.decode("utf-8"))
            else:
                env = data
            print("GOT", env.get("kind"), env.get("correlation_id"))
            print(json.dumps(env, indent=2)[:2500])
            return

asyncio.run(main())
PY
```

**Expected:** you see `GOT chat.history.message.v1 <corr_id>` and payload contains `session_id`, `role`, `content`, `timestamp`, `tags`.

---

## 2) Verify vector-host is subscribed correctly

```bash
docker exec -i orion-athena-vector-host env | egrep 'VECTOR_HOST_CHAT_HISTORY_CHANNEL|VECTOR_HOST_EMBEDDING_REQUEST_CHANNEL|ORION_BUS_URL'
```

**Expected:**
- `VECTOR_HOST_CHAT_HISTORY_CHANNEL=orion:chat:history:log`
- `VECTOR_HOST_EMBEDDING_REQUEST_CHANNEL=orion:embedding:generate`
- `ORION_BUS_URL=redis://...`

---

## 3) Verify vector-host publishes semantic upserts for chat history

Subscribe to `orion:vector:semantic:upsert` and print `meta.session_id`.

```bash
docker exec -i orion-athena-vector-host python - <<'PY'
import asyncio, json
from orion.core.bus.async_service import OrionBusAsync

REDIS_URL = "redis://100.92.216.81:6379/0"
CHANNEL = "orion:vector:semantic:upsert"

async def main():
    bus = OrionBusAsync(REDIS_URL)
    await bus.connect()
    print("subscribing...", CHANNEL)

    async with bus.subscribe(CHANNEL) as ps:
        async for raw in ps.listen():
            if not isinstance(raw, dict) or raw.get("type") != "message":
                continue
            data = raw.get("data")
            env = json.loads(data.decode("utf-8"))
            meta = ((env.get("payload") or {}).get("meta") or {})
            print("GOT", env.get("kind"), "corr=", env.get("correlation_id"), "src=", (env.get("source") or {}).get("name"))
            print("meta keys:", sorted(list(meta.keys()))[:25])
            print("session_id:", meta.get("session_id"))
            print("original_channel:", meta.get("original_channel"))
            print("role:", meta.get("role"))
            return

asyncio.run(main())
PY
```

**Expected:**
- `kind=vector.upsert.v1`
- `meta.session_id` present
- `meta.original_channel=orion:chat:history:log`
- `meta.role=user` or `assistant`

---

## 4) Verify vector-writer is receiving + storing upserts (logs)

```bash
docker logs --tail=200 -f orion-athena-vector-writer | egrep -i 'Stored|vector\.upsert|semantic|orion:vector:semantic:upsert|chroma|collection|error'
```

**If you see:**
- `✨ Stored kind=semantic collection=...`
  → writer is storing.

**If you see errors like:**
- `Expected metadata value to be a str,int,float,bool got <list>`
  → Chroma metadata sanitizer is needed (lists/dicts must be stringified).

---

## 5) Inspect Chroma collections + counts

Run inside `orion-athena-vector-writer` (or any container that can reach `orion-athena-vector-db`).

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(host="orion-athena-vector-db", port=8000, settings=Settings(anonymized_telemetry=False))
cols = client.list_collections()
names = [(c.name if hasattr(c, "name") else c) for c in cols]
print("collections:", names)
for name in names:
    col = client.get_collection(name)
    print(f"- {name}: {col.count()}")
PY
```

> Note: you may see a telemetry warning like `capture() takes 1 positional argument...`. It’s noisy but not a blocker.

---

## 6) Inspect stored chat docs in `orion_chat_turns`

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings
import pprint

client = chromadb.HttpClient(host="orion-athena-vector-db", port=8000, settings=Settings(anonymized_telemetry=False))
col = client.get_collection("orion_chat_turns")

sample = col.peek(limit=3)
print("metadatas:")
pprint.pprint(sample.get("metadatas"))
print("\ndocuments head:")
for d in (sample.get("documents") or [])[:3]:
    print("---")
    print((d or "")[:200].replace("\n","\\n"))
PY
```

**Expected:** metadata contains at least `correlation_id` (if available), `created_at`, `kind` (and may include `session_id`).

---

## 7) Query Chroma safely (avoid include errors + handle None metadatas)

Some Chroma clients reject `include=["ids"]`. IDs are returned separately in `q["ids"]`.

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(host="orion-athena-vector-db", port=8000, settings=Settings(anonymized_telemetry=False))
col = client.get_collection("orion_main_store")

q = col.query(
    query_texts=["cpu card ilo uncorrectable machine check"],
    n_results=5,
    include=["documents","metadatas","distances"],
)

print("ids:", q["ids"][0])
for i in range(len(q["ids"][0])):
    md = (q["metadatas"][0][i] or {})
    doc = (q["documents"][0][i] or "")
    dist = q["distances"][0][i]
    print("\n--- hit", i, "dist", dist)
    print("role:", md.get("role"), "channel:", md.get("original_channel"), "corr:", md.get("correlation_id"))
    print("doc_head:", doc[:240].replace("\n","\\n"))
PY
```

**Filter out junk:**

```python
where={"role":"embedding_request"}
```

---

## 8) Sample chat query (inspect recent chat vectors)

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(host="orion-athena-vector-db", port=8000, settings=Settings(anonymized_telemetry=False))
col = client.get_collection("orion_chat_turns")

q = col.query(
    query_texts=["cpu card"],
    n_results=5,
    include=["documents","metadatas","distances"],
)

for i in range(len(q["ids"][0])):
    md = (q["metadatas"][0][i] or {})
    doc = (q["documents"][0][i] or "")
    print("---", i, q["distances"][0][i], md.get("created_at"), md.get("kind"))
    print(doc[:220].replace("\n","\\n"))
PY
```

---

## 9) Recall service diagnostics (vector/RDF/SQL)

### A) Inspect recall runtime settings

```bash
curl -s http://localhost:8260/debug/settings | jq
```

### B) Run recall with diagnostic output

```bash
curl -s http://localhost:8260/recall \
  -H 'content-type: application/json' \
  -d '{
    "query_text": "why did I bring up CPUs",
    "diagnostic": true,
    "profile": "reflect.v1"
  }' | jq '{item_count:(.bundle.items|length), debug_counts:.debug.backend_counts, first:(.bundle.items[0].snippet // null)}'
```

`vector: 0` is expected -- vector retrieval was removed from orion-recall in
May 2026, regardless of what Chroma has. See section C above.

---

## 10) Chroma metadata crash: list/dict values (fix target)

If you see errors like:

```
Expected metadata value to be a str, int, float or bool, got ['brain','chat_general']
```

You must sanitize metadata before `col.upsert()`:
- lists → comma-joined strings
- dicts → JSON strings
- drop None

(Implement in vector-writer, right before any Chroma upsert.)

---

## 11) “MiniLM model downloaded instead of BGE” (why + how to avoid)

If a query triggers download of `all-MiniLM-L6-v2`, that means **the client query path is embedding query_texts using a default embedding function**.

To avoid surprise embedder selection:
- prefer `query_embeddings=[...]` using **your own embedder** (vector-host / BGE)
- or ensure your query pipeline uses the same embedding model as write-time

Operational check:
- confirm stored metadata includes `embedding_model=BAAI/bge-small-en-v1.5` and `embedding_dim=384`

---

## 12) Local dev env gotchas

### A) If scripts fail with `ModuleNotFoundError: orion`

Run with repo root on PYTHONPATH:

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. python scripts/bus_probe.py --pattern orion:chat:history:*
```

### B) If pydantic is broken in venv

Sanity check:

```bash
python - <<'PY'
import pydantic
print(getattr(pydantic, '__version__', 'unknown'), pydantic.__file__)
print('has BaseModel:', hasattr(pydantic, 'BaseModel'))
PY
```

If BaseModel missing, reinstall inside the venv:

```bash
pip uninstall -y pydantic pydantic-core
pip install "pydantic>=2,<3"
```

---

## 13) Handy collection breakdown query (where did my docs land?)

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(host="orion-athena-vector-db", port=8000, settings=Settings(anonymized_telemetry=False))

def count_where(colname, where):
    col = client.get_collection(colname)
    got = col.get(where=where, include=["metadatas"], limit=200)
    return len(got.get("metadatas") or [])

for colname in ["orion_chat_turns", "orion_chat", "orion_main_store", "orion_general", "orion_collapse"]:
    try:
        col = client.get_collection(colname)
        total = col.count()
        print(colname, "total", total)
    except Exception as e:
        print(colname, "ERROR", e)
PY
```

---

## 14) Notes / expectations

- If you can see `chat.history.message.v1` on the bus **and** see `vector.upsert.v1` with `meta.session_id` (optional), then embedding is working.
- If vectors aren’t showing up in Chroma, the writer is either not subscribing, writing to a different collection, or failing on metadata validation.

### RDF fragment recency (fixed 2026-07-13)

`fetch_rdf_chatturn_fragments` in `app/storage/rdf_adapter.py` used to order candidates with `ORDER BY DESC(STR(?turn))` (a lexical sort on the ChatTurn URI/correlation-id) and hardcode every fragment's `"ts"` to `0.0`. That made the same fixed set of historical turns win every query forever, and `scoring._compute_recency_factor` always fell back to its neutral 0.5 weight since `ts` was never real. It now selects and orders on the `ORION.timestamp` literal that `services/orion-rdf-writer/app/rdf_builder.py` already writes on every `ChatTurn` (`ORDER BY DESC(?ts)`), and parses it into a real epoch float via `rdf_adapter._parse_rdf_timestamp` so recency decay actually applies to RDF-backed fragments.

Note this only affects *scoring* (a continuous, soft decay) — it does not give RDF-backed candidates a hard time-window cutoff. `app/worker.py`'s `_window_rdf_chatturn_candidates` still does a separate Postgres lookup (`fetch_chat_turn_timestamps` against `chat_history_log`) to hard-drop chat-turn candidates outside a profile's `since_minutes` window; that function's docstring used to justify itself with "the graph stores no usable ChatTurn timestamp," which is no longer true post-fix — its docstring (and `tests/test_rdf_chatturn_windowing.py`'s) were corrected 2026-07-14 to state the real, current reason it still exists (hard exclusion vs. soft scoring are different jobs), without changing its behavior.

### Compose env-var defaults hardened (fixed 2026-07-14)

`docker-compose.yml`'s `environment:` list (added by the compose-parity PR, #1021) crash-looped this service in production: `RECALL_SKIP_MAX_NOVELTY`, `RECALL_SKIP_SHIFT_NOVELTY_FLOOR`, `RECALL_CONTINUITY_SQL_MINUTES`, `RECALL_CONTINUITY_RENDER_BUDGET`, `RECALL_BELIEF_RENDER_BUDGET`, and `RECALL_CRYSTALLIZATION_VECTOR_COLLECTION` were missing from live `.env` and had no `${VAR:-default}` fallback, so compose substituted empty strings and pydantic Settings validation failed on every boot (`float_parsing`/`int_parsing` errors). Fixed live by syncing the missing keys into `.env` (`python scripts/sync_local_env_from_example.py --all-keys orion-recall`) and, in this changeset, adding compose-level `:-default` fallbacks matching `.env_example` so a missing `.env` key can't crash the service again — see `services/orion-recall/tests/test_docker_compose_numeric_defaults.py`. `RECALL_GRAPHITI_IN_CHAT`/`RECALL_GRAPHITI_ADAPTER_URL` were left unguarded at the time (`NEVER_SYNC_KEYS`-protected); both were retired with the `active_packet` collector in #3461.

## 15) Concept-region collector (substrate concept graph in belief-lane recall)

`app/collectors/concept_region.py::fetch_concept_region_fragment()` is the PCR collector
wired into `process_recall`'s purposeful phase in `app/worker.py`.
It matches the current turn's text against concept labels already materialized in the
shared substrate graph (golden Orion/Juniper/Claude/relationship concepts seeded by `orion-hub`,
plus organically-grown topic-foundry concepts) and, on a match, returns a bounded
neighborhood fragment scoped to that turn. Matching is deliberately dumb — case-insensitive
label substring match, no embedding/LLM call — and it never does a store read at all if the
turn text matches nothing.

**How it reads FalkorDB (since 2026-10-06): bounded direct queries, never the whole graph.**
With `SUBSTRATE_STORE_BACKEND=falkor`, `app/substrate_store.py` hands the collector a
`FalkorDirectConceptStore` (`orion/substrate/falkor_direct.py`), not a `FalkorSubstrateStore`.
Each turn runs one light `GRAPH.RO_QUERY` that ranks the top 500 concepts by salience and
confidence and returns only their labels; the labels are matched in Python. No match = done
(~11ms median live). On a match, two more bounded reads fetch the matched concepts' full rows and
the edges among the top 500 most salient edges touching the top 500 concepts that touch a
matched concept (~140ms median live, most of it FalkorDB walking ~32k edges to rank them).
Reinforcement reads each matched concept once (node and identity key in one query) and
writes it with one `MERGE ... SET`; if the recall gives up mid-loop (deadline), no further
node is written. With the reinforcement included, a matched turn measured ~100ms median,
175ms max on a copy of the production graph (2026-10-06 follow-up PR).
**Hung FalkorDB:** each socket read is capped at 1.5s, and after 3 consecutive timeouts a
circuit breaker skips concept_region for 60s (`recall_concept_region_breaker_open` /
`recall_concept_region_breaker_skip` log lines, counters in `app.substrate_store.breaker_stats()`).
No fallback: a skipped turn just has no concept fragments.
**Index:** every substrate store bootstraps `CREATE INDEX FOR (n:SubstrateNode) ON (n.node_id)`
(idempotent, bounded, never blocks startup; `orion/substrate/falkor_store.py::ensure_substrate_indexes`).
The selection is identical to the old cache read: 152/152 real recall queries produced the
same fragments in the same order (`scripts/compare_concept_region_direct_vs_cache.py`).
Before this, recall built a complete in-process copy of the graph (17-25s since PR #2500)
and never refreshed it, so concept_region also read a graph frozen at boot. The handle has
no `snapshot()`; `routed`/`graphdb`/`sparql` backends are refused (they would load a graph).

**Reinforcement-on-recall, live since PR #1173:** the actual live call site in `app/worker.py`
uses `fetch_concept_region_fragment_and_reinforce()`, not the plain read-only function above
directly. It composes the same read (unchanged, same "never persists" contract) with a new
write: whatever concept nodes matched get a small activation bump
(`reinforce_matched_concepts()`, same math and boost constant as
`orion/memory/crystallization/dynamics.py::recall_boost()` — `activation = current +
(1-current)*0.08`, asymptotic toward 1.0). This is the counterpart to
`services/orion-hub/README.md` §5.5's decay scheduler: being surfaced in a live turn is a
real "this concept is still relevant" signal, not just monotonic decay. Full design
conversation (why sync-write over async-via-bus, which fields move and which deliberately
don't) is in `app/collectors/CONCEPT_REINFORCEMENT_DESIGN.md`.

**Env vars:**

| Var | Default | Meaning |
|---|---|---|
| `RECALL_CONCEPT_REGION_ENABLED` | `true` | Master on/off switch for this collector. |
| `SUBSTRATE_STORE_BACKEND` | `falkor` | `falkor` = direct bounded reads (above); `in_memory`/unset = empty in-memory store; anything else is refused. Same `FalkorDB` instance/graph as `orion-cortex-exec`/`orion-hub`. |
| `FALKORDB_URI` | `redis://orion-athena-falkordb:6379` | Bridge-network Docker DNS — this service runs in bridge mode, **not** Hub's host-mode `127.0.0.1:6380` convention. |
| `FALKORDB_SUBSTRATE_GRAPH` | `orion_substrate` | Graph name inside FalkorDB. |

**Verify it's live:** trigger a purposeful-phase turn whose text mentions a known concept
label (e.g. a golden concept's name), then check `decision.recall_debug["pcr"]["backend_plan"]`
includes `concept_region`, and look for a `source: "concept_region"` candidate/item. If the
substrate store failed to construct (bad `FALKORDB_URI`, FalkorDB unreachable), the collector
degrades to `[]` silently — check Hub's own Concept Atlas tab or `orion-hub` logs for
`recall_substrate_store_init_failed` first.

**Historical (superseded 2026-10-06 — recall no longer holds a `FalkorSubstrateStore` cache).**
**Formerly a known limitation, fixed by PR #1159:** `FalkorSubstrateStore` used to hydrate its
snapshot once at process start and never refresh — concepts added by `orion-hub`'s
topic-foundry scheduler after this service's first `concept_region` call were invisible here
until `orion-recall` restarted (originally flagged in PR #1133, when this was the first
continuously-changing-data consumer of the store in this service to make it load-bearing).
PR #1159 added write-generation + time-ceiling refresh logic (mirroring
`GraphDBSubstrateStore`'s existing pattern), so `snapshot()`/`read_concept_region()` now pick
up durable changes made by other processes without a restart, bounded by
`SUBSTRATE_SNAPSHOT_FORCE_REFRESH_CEILING_SEC` (`FALKOR_SNAPSHOT_FORCE_REFRESH_CEILING_SEC`
for a Falkor-specific override, default 30s). Root cause of PR #1159 itself: the same stale
cache also meant directly-deleted-in-Falkor nodes got silently resurrected by `orion-hub`'s
decay scheduler, which blindly re-upserts everything it reads on every tick — see that PR's
description for the full incident.

Two more `FalkorSubstrateStore`-level bugs (shared code, not specific to this collector, but
this service constructs the same store) were found and fixed the same day: PR #1175 (a
legacy-payload migration path that orphaned duplicate node rows forever, silently reverting
any concept-node write on the next hydrate) and PR #1179 (edge hydration losing every edge's
real source/target node_id). See `services/orion-hub/README.md` §5.5 for the full incident
writeups.

## 6) `RECALL_CONTINUITY_RENDER_BUDGET` (96 -> 1200) + a same-day corruption incident

PR #1216 ("fix budget") raised `RECALL_CONTINUITY_RENDER_BUDGET` from 96 to 1200 tokens. Root
cause it fixed: continuity recall was finding real conversation turns but then dropping them
during rendering with "dropped due to budget" errors, because 96 tokens is far too small to
render an actual conversation snippet -- effectively a silent, near-total continuity outage.

**Same-day corruption, separate incident:** the diffs PR #1216 actually merged did not match
its description. All 7 touched files (`app/settings.py`, this README, `.env_example`,
`docker-compose.yml`, and 3 test files) were replaced with a single line containing a bare
hash string (`062d462b33c64cc5bbb9fb5c63672a0f02ecb046`) instead of real content --
`settings.py` in that state is not even valid Python, so any fresh build/deploy from `main`
after that merge would fail outright. Root cause of the corruption itself is unconfirmed --
not reproduced, not explained by anything in the PR's own diff or description. Recovered via
`fix/recall-budget-corruption-recovery`: all 7 files restored from the last known-good commit
(the merge immediately prior), with the real `96 -> 1200` change re-applied on top of the
restored content this time. If you see a file in this service reduced to one line matching
that hash pattern again, that's this same failure mode recurring, not a new bug -- check
`git log` for the responsible commit before assuming the running container is also affected
(a running container built from an earlier good image is not automatically broken by a bad
commit landing on `main` afterward, but the next rebuild would be).

## 16) What the Juniper chat prompt's DURABLE MEMORY may carry (issue #3461)

The two PCR phases (`recall_phase=continuity` and `purposeful`) are called only by
`orion-cortex-exec`'s `pcr_chat_memory.py` and build the chat prompt's DURABLE MEMORY.
Four sources were cut from that path on 2026-11-15, because live prompts showed only noise:

| Cut | Where | How |
|---|---|---|
| AI Town NPC chatter | `sql_chat.fetch_chat_history_pairs` unions `aitown_chat_history_log` | `apply_chat_memory_exclusions()` sets `sql_chat_include_aitown=False`; the query scans `chat_history_log` only. |
| Bus hop-latency z-scores | `bus_synaptic_anomaly` feed | `apply_chat_memory_exclusions()` sets `enable_bus_synaptic_anomaly=False`; the fetch never runs. |
| `active_packet` "durable beliefs" | `app/collectors/active_packet.py` | Collector deleted, removed from every intent plan and belief profile, with its settings (`RECALL_ACTIVE_PACKET_ENABLED`, `RECALL_GRAPHITI_*`, `RECALL_CRYSTALLIZATION_VECTOR_COLLECTION`, `RECALL_CARDS_EMBEDDING_URL`). Crystallizations stay in Postgres; Hub's `/api/memory/active-packet` route is unchanged. |
| Topic-foundry concepts | `concept_region` collector | Nodes with id prefix `sub-concept-topicfoundry-` and edges touching them are dropped before render and before reinforcement. |

Other recall profiles (`reflect.v1`, `social.room.v1`, journals) still get the AI Town union
and the bus anomaly feed. The exclusions apply only when `RECALL_PCR_ENABLED` is true, which
is the only mode in which cortex-exec sends the continuity/purposeful phases. An empty continuity or belief digest now renders as an empty
string, not a bare header.

## 17) Lived-memory recall (issues #3533, #3534, #3573, "Recall rev 2" steps 1-2)

Code: `app/lived_memory.py` (reader + index loop), `orion/memory/lived_memory.py` (which rows may be
served, and how a row becomes one document). Spec: the "Recall rev 2" section of
`docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`.

What it does: on the PCR continuity phase (once per chat turn the skip gate lets through), it
finds the few lived memories that fit the turn and appends them to DURABLE MEMORY:

```
[Lived memories that fit this turn -- kind, record id, date]
- (episode_memory/happened · episode_memory:<uuid> · 2026-11-11) Juniper told me ...
```

Stores: `episode_memory` (active, not rejected, not superseded by a live correction),
`memory_crystallizations` kind `stance` (active), `person_fact` (not removed, person active). An
episode memory past the end date Juniper stated ("back by Friday") is still served and
indexed, labelled `true then, not now: ended <date>`: the end date bounds the state, not the
memory (`EPISODE_ENDED_SQL`, live miss 2026-11-25). An expired follow-up is never served. Chroma is only an index: every hit is re-gated against the live snapshot, and the chosen
ids are re-read from Postgres right before they are served.

Pipeline: content-word floor (`RECALL_LIVED_MEMORY_MIN_QUERY_TOKENS`) with the short-message gate
(below) -> meaning candidates
(vector-host `/embedding`, Chroma `RECALL_LIVED_MEMORY_COLLECTION`) + exact-word candidates (BM25,
names weighted) -> local cross-encoder (vector-host `/rerank`) -> bar
(`RECALL_LIVED_MEMORY_MIN_FIT`, sigmoid of the reranker logit; uncalibrated, tuned by replay) ->
rank by `logit + 1.0*recency(30 d half-life) + 0.5*importance + 0.5*is_correction` (logit units)
-> at most `RECALL_LIVED_MEMORY_MAX_LINES`. Zero lines is a valid answer. It starts at intake and
runs while the other backends fetch.

Replay 2026-11-17 (11 real referring messages, Juniper's last 150 messages; reranker
bge-reranker-v2-m3 on the P4, bar 0.004, floor 5 content words): 9/11 targets injected (the
2026-11-14 el Coco miss included), 3 of 23 small-talk messages inject anything, 1.9 lines per
message on average, lived-memory time p50 1.4 s / p90 2.5 s (overlapped with the fetch). Before:
0 of 11 (no chat reader of these stores existed).

**Short-message gate** (issue #3613, `RECALL_LIVED_MEMORY_SHORT_GATE_ENABLED`, default on; false =
the plain floor). A message under the floor searches when it carries signal:

1. it names a known entity (`app/lived_entities.py`): a name-class `referent_alias` of a referent
   of an entity type (person, pet, place, project, service, host, past event), or an active
   `person` display name. People and pets also count by first name, but only when the display
   name is a proper name ("Mail carrier" does not make "mail" a name). Concepts, files, PR numbers
   and bare key slugs ("project:curiosity") are not entity names. The two participants
   (`person.role` self / interlocutor) never count, nor any referent carrying their name, so
   "thanks Orion" does not search. If the participant rows cannot be read, a referent on more than
   30 memories is left out instead. A failed reload keeps the last good index; or
2. the short-message rewrite (`SHORT_PROMPT`, same fast lane) writes searches. It asks whether a
   good reply needs something from before this conversation (Self-RAG's retrieve step) and tells
   the model that a reply the recent conversation already answers needs none. NONE, an error or a
   timeout: no search, and nothing else (not even the embedder) runs.

A message with no content word at all is never searched and never rewritten.

Read log: every injected doc id is a `lived_memory` item in the bundle, so it lands in
`recall_telemetry.selected_ids`; the log line `lived_memory_injected corr_id=... ids=... fits=...`
carries the same ids, and `recall_debug.lived_memory` carries ids, fits and timings (never text).

Index: `index_loop` re-embeds new or changed records every
`RECALL_LIVED_MEMORY_INDEX_INTERVAL_SEC` (batch `RECALL_LIVED_MEMORY_INDEX_BATCH`) through
`orion/introspect/semantic_index.py` (vector-host embeds, upsert on `orion:vector:semantic:upsert`,
orion-vector-writer stores). First backlog / rebuild: `scripts/backfill_lived_memory_index.py`
(CLAUDE.md s14 protocol, read-only on Postgres).

Eval: `evals/run_lived_memory_replay_eval.py` replays real Juniper messages from a host-side
bank (`~/.orion/evals/lived_memory_recall_bank.json`, mode 600, ids only) and prints hit rate,
small-talk injection and latency per bar; `evals/test_lived_memory_replay_eval.py` runs the same
scoring on made-up fixtures in CI.

### 17a) Step 2: query rewrite and link-following (issue #3534)

Code: `app/lived_rewrite.py`, `app/lived_links.py`, wired in `app/lived_memory.py`.

- **Query rewrite.** Before searching, a small local model (LLM gateway route
  `RECALL_LIVED_MEMORY_REWRITE_ROUTE`, default `quick_turn`: the fast class at interactive
  priority; never metacog, never an outside API) reads the message plus the last 3 turns of the
  same `session_id` from `chat_history_log` (6 h window) and writes up to 3 short searches, or
  `NONE` for small talk. Each search runs through the same meaning + exact-word retrievers; all
  lists merge by reciprocal-rank fusion. The reranker scores each memory against the message
  followed by the searches. Past `RECALL_LIVED_MEMORY_REWRITE_TIMEOUT_SEC`, or on any error, the
  raw message is searched alone. A message under the content-word floor is neither searched nor
  rewritten (a rewrite bypass was tried: small talk injecting rose to 35%, no hit gained).
  Messages over 1,500 characters skip the rewrite.
- **Link-following.** Memories that share a referent (`episode_memory_referent`) with the top 5
  candidates, or whose referent's name (name-class `referent_alias` only, 4+ characters, not a
  function word) appears in the message or a search, join the rerank: two-hop personalized
  PageRank, hubs on more than 30 memories left out, at most `RECALL_LIVED_MEMORY_LINK_TOP_K`.
  A neighbour reaches the prompt only through the rerank and the bar. The graph reloads in the
  background every 10 minutes.

### 17a2) Time windows and past-state recall (issue #3674)

Code: `app/lived_time.py` (the date grammar), wired in `app/lived_memory.py`.

- **Parser first.** Time words in the message become one window, resolved against "now" in
  `RECALL_LIVED_MEMORY_TIMEZONE` (default `America/Denver`): a date range, one date, yesterday /
  today, a span relative to now ("over the last month or so", "a month or so ago", "last week"), a
  weekday or weekend, or a named month or year after a preposition ("in October", "early
  September", "since August"). A small grammar, not a word list; "May I ask" is not May. The
  parser never waits on the model.
- **No model in the window.** Asking the fast-lane rewrite for a `WHEN:` line was measured on the
  57 scoreboard questions (2026-11-25): every wording tried either stopped it writing searches
  (0-3 of 57, against 11 with the plain prompt) or was ignored (0 time lines). The rewrite prompt
  is unchanged; the window never waits on, or times out with, the model.
- **Before the pool is cut.** With a window, the message's meaning search is also read 256 deep
  and filtered to the window in recall (a Chroma `where` filter cost ~1 s a query, a plain
  256-deep query ~20 ms), and one more exact-word search runs over in-window records. In-window
  candidates fill the rerank pool first; at most 4 from outside the window stay, and the best 2 of
  those ride in the first rerank chunk so a short deadline still scores them. After the rerank,
  in-window memories get +1 (logit units, like recency), and when one of them clears the bar at
  most 2 lines from outside the window are served. A window with nothing in it changes nothing:
  recall is never emptied by a time word. Windows are padded a day each side. When a message
  holds two time phrases, the first one wins. "today" and "tonight" are never a window (they are
  the present); a bare `8/10` or `1/2` is a rating or a fraction, not a date; a month still ahead
  this year is a plan unless the message is in the past tense.
- **Past state.** When the window is wholly past and the message asks in the past tense with a
  state verb ("Where was I in early October?"), recall also reads
  facts an update has retired since (`orion.memory.lived_memory.fetch_valid_during`, the as-of
  reader over the window, run alongside the meaning search, at most 0.8 s and 500 rows). At most 4
  join the rerank. They render as
  `(... · <date> · true then, not now: changed <date>)` and are re-checked through the same
  reader before render. A correction's old row is never served: it was never true.
- **Trace.** `recall_telemetry.lived_trace.window` keeps the window's local dates, its shape,
  `past_state`, and counts (`in_window`, `as_of_rows`); never the message's words.

Kill switch: `RECALL_LIVED_MEMORY_TIME_WINDOW_ENABLED=false` (recall ignores time words, as before).

### 17b) Degraded mode (issue #3573)

Recall never silently skips. When the reranker is down, errors or runs past the deadline, the
exact-word hits are ranked by BM25 alone (at most 3 lines, each sharing 2+ content words or a
name with the turn) under a header that says the search ran degraded. When the embedder is down
or hangs, the exact-word candidates are still reranked, and the block still says degraded. A
degraded search that finds nothing puts a one-line note in the prompt ("not proof there is no
memory"), except for small talk the rewrite answered NONE. If the whole lived recall misses the
worker's deadline, the worker runs the same exact-word fallback from the in-memory snapshot.
The reranker gets the candidates in chunks of 8 (`RERANK_CHUNK`), best-retrieved first, with the
message's own meaning and exact-word hits taking turns (issue #3662). If the deadline or an error
stops it after at least one chunk, the scored chunks are kept and served (`rerank_partial` in the
trace, status `ok`). If nothing in the scored chunks clears the bar, the unscored exact-word hits
are served degraded (reason `rerank_partial`), so a part-search never reads as "below the bar".
Cause found 2026-11-17: 5 of 7 degraded chat turns overlapped orion-whisper-tts synthesis on the
same Tesla P4; the other 2 lost 1.8-2.6 s to the query rewrite first.
Telemetry: `recall_telemetry.backend_counts.lived_memory_degraded = 1`, and
`recall_telemetry.timings_ms.lived_memory_{rewrite,rerank,total}`; `recall_debug.lived_memory`
carries `backend`, `degraded` reasons and the rewritten searches.

Kill switches: `RECALL_LIVED_MEMORY_ENABLED=false` (all lived recall),
`RECALL_LIVED_MEMORY_REWRITE_ENABLED=false` (rewrite only), `RECALL_LIVED_MEMORY_LINK_TOP_K=0`
(links only), `RECALL_LIVED_MEMORY_TIME_WINDOW_ENABLED=false` (time windows only). Restart
orion-recall after changing any of them.

### Memory scoreboard (issue #3536)

One command scores how well Orion remembers Juniper's real history, so every memory change can be
measured: `make eval-memory-scoreboard` (weekly; `ARGS=--phases retrieval` for the model-free half).

- **Bank.** `~/.orion/evals/memory_scoreboard_bank.json` (host-side only, mode 600; never in git).
  Five categories, about even: facts, events across sessions, time (each item records its
  `as_of`), knowledge updates (a correction exists; only the corrected answer counts; the item
  records the wrong answer and any superseded row), and abstention (never happened). Some items
  are real past messages that lean on shared history (`chat_id`, read at run time). Each item names
  its target record ids (`episode_memory:` / `stance:` / `person_fact:`), none for abstention.
  Items whose memory is only in pre-October chat are `backfill_pending` (backfill #3537) and are
  reported apart, never scored.
- **Retrieval (no model).** Each item replays the phase-0 skip gate, phase 1
  (`chat.continuity.v1`, lived memory) and phase 3 (`--phase3-intent`, default `open_loop`, the
  usual live intent) over the real bus. Hit = a target id is in either phase's DURABLE MEMORY
  bundle or cited by the current `page:juniper` core page (the table also shows the recall-only
  rate). Bar: 80% of stored items. Degraded recalls are read from each call's own
  `recall_telemetry` row (`lived_status` degraded / timeout / error / off, the `lived_trace`
  degraded reasons and `rerank_partial`, deadline hit, missing trace; older rows without
  `lived_status` fall back to timing checks) and reported apart. `lived_status = skipped_short`
  (too few content words for the lived search) is the chat path's own rule, so it is scored and
  counted in its own column. Superseded rows served for a knowledge-update item are counted.
  Implicit items (real past messages) also get their own row across categories.
- **Answer (end to end).** Each item is a real Hub chat turn (`POST /api/chat`, mode `orion`,
  `no_write: true`, session `memory-scoreboard-eval`), the same path as the self-sense eval, so
  nothing lands in chat history. A local judge (`orion/evals/leased_llm.py` operator hold, local
  lanes only) labels the reply: correct / don't know / invented / false "no memory" / missed.
  Bars: 70% or more correct, 0 invented, 0 false "no memory" on stored items. The turn's own
  `recall_telemetry` rows say whether the target was in its prompt and whether the search ran in
  full. A "no memory" answer on a stored item is a false denial even when the search was skipped
  or degraded; it is also reported under its most specific cause (`lived:rerank_timeout`,
  `lived:embedder_unavailable`, `lived:rerank_partial:N/M`, then `lived:skipped_short` /
  `lived:degraded` / `lived:timeout`, then `recall:skip_gate` / `recall:no_trace`), `search_full`
  when the search ran in full, or `cause:unknown` when the turn's trace was never read.
- **Scoring a branch before deploy** (`--lived-local`, issue #3613, retrieval only). The deployed
  service's lived-memory lines (bundle items with source `lived_memory`) are replaced by the lived
  recall of the code on disk, run in-process with the live `RECALL_LIVED_MEMORY_*` settings
  against the host-side Chroma, vector host and LLM gateway; everything else stays live.
  `--lived-local-short-gate off` is the same code with the plain floor (the before).
  `make eval-memory-scoreboard ARGS="--phases retrieval --lived-local"`.
- **Output.** `~/.orion/evals/memory_scoreboard_runs/<date>.json` (everything, includes replies)
  and `<date>.md` (per-category table, counts and qids only), plus one line per run in
  `history.jsonl`. All mode 600.
- **Fixture test.** `evals/test_memory_scoreboard.py` (made-up records) runs in CI.
- **Weekly run.** Not wired into a scheduler. The run reads a host-only bank and writes host-only
  results, and orion-durable-runs runs in a container without them; a new `DurableWorkflowV1`
  value is also a new producer under the integration gate (CLAUDE.md 0A). Run the make target
  weekly.
