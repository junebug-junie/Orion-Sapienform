# Recall tag/entity Falkor writer — implementation plan

**Date:** 2026-07-18
**Status:** Approved by Juniper (design walked through conversationally, corrected twice mid-review)
**Implements:** Phase 2 of `docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md`, applying the routing/property doctrine in `docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md`.
**Mode:** Implementation (this doc records the design that was reviewed and approved before building; not a fresh proposal)

---

## What this is

A live, dark-shipped Cypher-native write of chat tag/entity data into FalkorDB, alongside (not replacing) the existing Fuseki write. First real Falkor consumer of `orion/graph/persistence_routes.py`-adjacent infrastructure for the recall surface; first real cross-service extraction of the Falkor Cypher client out of `orion/substrate/`.

## Corrected scope (two real corrections made during design review, both load-bearing)

1. **`recall.chat_turn` is not a live-writer job.** `chat.history`/`chat.history.message.v1` RDF writes were killed by PR #1164 (merged concurrently with earlier work in this arc) — Postgres's `chat_history_log` already holds every field the RDF `ChatTurn` node had, plus substantially more (`spark_meta`, memory status/tier/reason, LLM uncertainty telemetry). Porting chat text to Falkor would be a third redundant copy. What Falkor gets instead is a **thin anchor node** (`turn_id`, `session_id`, `ts`, `correlation_id` — no `prompt`/`response` text), created on demand as a side effect of writing tag/entity edges, since those edges need a node to attach to.
2. **This does not live in `orion-rdf-writer`.** That service's own doctrine role is explicitly scoped to "Legacy RDF materialize" (`docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md`'s service-ownership table) and its canonical-writer doc (`docs/architecture/rdf_store_v1_cutover.md`) says its job is RDF materialization from the bus, specifically. Every real Falkor producer in the codebase lives in the originating service instead: `orion-substrate-runtime` for substrate, `orion-graphiti-adapter` for crystallizations, `orion-spark-concept-induction` for concept profiles (that one is the closest precedent — the producer writes directly to Falkor and stops routing through `orion-rdf-writer` for that data). This writer belongs in **`orion-meta-tags`**, the actual producer of tag/entity extraction, not bolted onto the RDF service as a second branch.

## Current architecture (verified this session, not assumed)

- **Producer**: `services/orion-meta-tags/app/main.py::handle_triage_event`. Subscribed to `orion:collapse:triage`, `orion:chat:history:turn`, `orion:chat:social:stored`. For `envelope.kind in {"chat.history", "social.turn.stored.v1"}`, runs spaCy NER (`entities = [ent.text for ent in doc.ents]`, `tags = list(entities)`), a crude keyword-heuristic sentiment tag (`"sentiment:positive"|"sentiment:negative"|"sentiment:neutral"`, appended into `tags`), builds a `MetaTagsPayload` (aliased `Enrichment`), and publishes `kind="tags.enriched"` to `CHANNEL_EVENTS_TAGGED_CHAT` (`orion:tags:chat:enriched`).
- **`MetaTagsPayload`** (`orion/schemas/telemetry/meta_tags.py`) fields: `service_name, service_version, node, timestamp, source_message_id, correlation_id, tags, entities, processing_meta, id, collapse_id, enrichment_type`. **No `session_id` field.**
- **`session_id` is available but currently dropped**: the raw incoming `chat.history` envelope conforms to `ChatHistoryTurnV1` (`orion/schemas/chat_history.py`), which has `session_id: Optional[str]`. `main.py`'s `EventIn` model-validator moves any field not in its `known_fields` set into `extra_data`, and `session_id` isn't in that set — so it's present on the raw payload dict `handle_triage_event` already has in scope, just never carried into the `Enrichment` object it constructs. The new writer reads `session_id` directly from `raw_payload`, not from `Enrichment`/`MetaTagsPayload` — no shared-schema change needed.
- **`ts` provenance** (confirmed from the pre-#1164 `_handle_chat_turn` in `orion-rdf-writer`, since removed): timestamps were always read from the payload's own `timestamp` field, never generated at RDF-write time. `MetaTagsPayload.timestamp` has `default_factory=datetime.now` — meaning **meta-tags itself stamps the enrichment's own processing time**, which is a materially different value from the original chat turn's event time. The Falkor writer must read the turn's real event time from the raw `chat.history` payload (whatever field `ChatHistoryTurnV1`-shaped payloads carry for that — needs a one-line check during implementation, not assumed), not from `MetaTagsPayload.timestamp`.
- **Existing Falkor Cypher client**: `orion/substrate/falkor_store.py::RedisGraphQueryClient` — `redis.Redis` + `redis.commands.graph.Graph`, `.graph_query(cypher, params)`. Zero substrate-specific coupling. `redis[hiredis]==5.0.7` (already in `orion-meta-tags/requirements.txt`) includes `redis.commands.graph` — confirmed by direct import in this session's venv. No new dependency.
- **Existing routing/property infra** (`orion/graph/`): `persistence_routes.py` + `persistence_router.py` (`GraphPersistenceRouter`, workload-keyed primary/shadow resolution from `GRAPH_PERSISTENCE_ROUTES_JSON`) — real, tested, **zero consumers anywhere in the repo**. `property_guard.py::sanitize_metadata` — real, reused by the substrate Falkor store.

## Why this does *not* use `GraphPersistenceRouter`

The router models one service picking a primary (+ optional shadow) backend for *its own* write of one logical operation — that's `RoutedSubstrateGraphStore`'s shape (substrate-runtime choosing falkor vs sparql for its own tick writes). This isn't that shape: `orion-rdf-writer` and `orion-meta-tags` are two **independent services**, each reacting to the same `tags.enriched` event, each deciding independently whether to write. There's no single call site choosing "primary vs shadow" — the Fuseki write (via `orion-rdf-writer`) continues completely unaware that `orion-meta-tags` is now also writing Falkor. Forcing this into the router's primary/shadow model would be adopting an abstraction because it exists, not because it fits. A plain boolean flag (matching the precedent of `RECALL_GRAPHITI_IN_CHAT`, `CONCEPT_PROFILE_GRAPH_BACKEND`) is the honest fit: `RECALL_FALKOR_TAG_ENTITY_ENABLED`, default `false`, ships dark.

## Schema (Cypher-native, confirmed across the design conversation)

Graph name: `orion_recall` (new — separate from `orion_substrate` and `graphiti_temporal`, same shared FalkorDB instance).

```text
(:ChatTurn {turn_id, session_id?, ts, correlation_id?})
(:ChatSession {session_id})
(:ChatSession)-[:HAS_TURN]->(:ChatTurn)          # only when session_id present
(:Entity {name})                                  # canonical, case-fold+trim identity key
(:Tag {name})                                     # canonical, same identity key
(:ChatTurn)-[:MENTIONS_ENTITY {ts}]->(:Entity)
(:ChatTurn)-[:HAS_TAG {ts}]->(:Tag)
```

`ChatTurn.sentiment` — split out of the `"sentiment:*"` string-tag pattern at write time (source: same heuristic already computed in `main.py`, just captured as a field instead of smuggled through `tags`).

**Corrected during code review, after implementation:** the `chat.history` branch in `main.py` builds `tags = list(entities)` then appends one sentiment marker — `tags` and `entities` are the same spaCy NER extraction by construction, not two independently meaningful lists. The call site only passes the sentiment marker as `tags` (consumed entirely by `extract_sentiment()`), so `:Tag`/`HAS_TAG` carries no real data for this producer today — the schema stays general (kept for a future producer with genuinely distinct tag content), but only `:Entity`/`MENTIONS_ENTITY` is populated in practice. Same root cause as the historical Fuseki `hasTag`/`hasEntity` predicates never actually diverging.

**Noise filter** applied to `tags`/`entities` values before writing (per the Phase 0 spec's live-data findings — bare numbers, stopwords, relative-time expressions are not real tags/entities): reject digit-only values, a small stopword list, and a relative-time regex (`"today"`, `"yesterday"`, `"N years/months/weeks ago"`, etc.). Rejected values are logged (`property_cathedral_rejected`-style, matching `orion/graph/property_guard.py`'s convention) not silently dropped.

**No `confidence`/`salience`/`extractor_service` fields** — Phase 0's live audit found these are dead constants (`0.0`/`0.0`/`"meta-tags"` across all 1,449 historical records). Not carried forward as if meaningful.

## Files touched

```text
orion/graph/falkor_client.py                          # NEW — RedisGraphQueryClient extracted here
orion/graph/tests/test_falkor_client.py                # NEW
orion/substrate/falkor_store.py                        # import from orion.graph.falkor_client instead of defining its own
services/orion-meta-tags/app/falkor_recall_writer.py   # NEW — Cypher builder + write path
services/orion-meta-tags/tests/test_falkor_recall_writer.py  # NEW
services/orion-meta-tags/app/main.py                   # wire the writer into handle_triage_event's chat.history branch
services/orion-meta-tags/app/settings.py               # RECALL_FALKOR_TAG_ENTITY_ENABLED, FALKORDB_URI, FALKORDB_RECALL_GRAPH
services/orion-meta-tags/.env_example                  # same, documented, dark default
services/orion-meta-tags/docker-compose.yml            # env passthrough
services/orion-meta-tags/README.md                     # institutional memory
services/orion-rdf-writer/README.md or docs/architecture/rdf_store_v1_cutover.md  # note recall.tag_entity also flows to Falkor now (dual-write during migration ladder, not a replacement yet)
docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md      # service-ownership table: add recall.tag_entity -> orion-meta-tags
services/orion-falkordb/docker-compose.yml             # consumer comment: add orion-meta-tags
docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md # mark Phase 2 status, correct the "in orion-rdf-writer" framing that PR review caught
```

## Non-goals (this patch)

- No read-side change (`orion-recall`'s `rdf_adapter.py` still queries Fuseki — Phase 4).
- No historical backfill (Phase 3).
- No `social.turn.stored.v1` handling — explicitly gated to `envelope.kind == "chat.history"` only; social turns share the same meta-tags code path but are Phase 6, deliberately not touched here.
- No Fuseki write changes — `orion-rdf-writer` is completely untouched by this patch, continues writing exactly as it does today.
- No collapse-mirror (non-chat) tagging changes — the generic `_is_juniper`-gated branch in `handle_triage_event` has no `session_id` and isn't a chat turn at all; out of scope.

## Acceptance checks

- Flag off (default): zero behavior change, zero new Falkor writes, existing tests unaffected.
- Flag on: a real `chat.history` event produces a real `ChatTurn`+`ChatSession`+`Entity`/`Tag` write in Falkor, verified via direct `redis-cli GRAPH.QUERY orion_recall`, not just application-level assertions.
- Noise filter rejects a synthetic digit-only/stopword/relative-time value in a test.
- `Entity`/`Tag` dedup: two mentions of the same normalized name in different turns produce one node with two edges, not two nodes.
- `orion-substrate-runtime`'s existing Falkor tests still pass after the `RedisGraphQueryClient` extraction (import-path change only, no behavior change).
