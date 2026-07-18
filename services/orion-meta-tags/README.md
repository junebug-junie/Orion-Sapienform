# Orion Meta Tags

The **Meta Tags** service provides automated enrichment for collapse events, chat turns, and social-room turns. It runs spaCy NER (`SPA_MODEL`, default `en_core_web_trf`) to extract entities (reused as tags) and a keyword-heuristic sentiment classifier (`sentiment:positive`/`negative`/`neutral`, currently smuggled through the `tags` list — see the Falkor writer section below for where that gets split back out into a real field), adding structured metadata to the original payload.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:collapse:triage` | `CHANNEL_EVENTS_TRIAGE` | `collapse.mirror` | Raw entries needing enrichment (Juniper-observed only; Orion's own outputs are skipped). |
| `orion:chat:history:turn` | `CHANNEL_EVENTS_CHAT_TURN` | `chat.history` | Chat turns — the source of the Falkor recall write below. |
| `orion:chat:social:stored` (hardcoded, no env var) | — | `social.turn.stored.v1` | Social-room turns. Shares the same tagging code path as chat turns; **not yet** wired to the Falkor writer (Phase 6, see the plan doc below). |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:tags:enriched` | `CHANNEL_EVENTS_TAGGED` | `tags.enriched` | Enriched metadata for collapse-mirror events. |
| `orion:tags:chat:enriched` | `CHANNEL_EVENTS_TAGGED_CHAT` | `tags.enriched` | Enriched metadata for chat/social turns (same `kind`, different channel). Consumed by `orion-rdf-writer` (Fuseki write) and, as of Phase 2 below, written to Falkor directly by this service too. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_EVENTS_TRIAGE` | `orion:collapse:triage` | Intake channel. |
| `CHANNEL_EVENTS_CHAT_TURN` | `orion:chat:history:turn` | Chat-turn intake channel. |
| `CHANNEL_EVENTS_TAGGED` | `orion:tags:enriched` | Collapse-mirror output channel. |
| `CHANNEL_EVENTS_TAGGED_CHAT` | `orion:tags:chat:enriched` | Chat/social output channel. |
| `RECALL_FALKOR_TAG_ENTITY_ENABLED` | `false` | See Falkor writer section. |
| `FALKORDB_URI` | `redis://orion-athena-falkordb:6379` | Shared FalkorDB instance (same one substrate-runtime and graphiti-adapter use, different graph name). |
| `FALKORDB_RECALL_GRAPH` | `orion_recall` | Graph name — separate from `orion_substrate` and `graphiti_temporal`. |

## Recall Falkor writer (Phase 2)

**Design/implementation:** `docs/superpowers/plans/2026-07-18-recall-tag-entity-falkor-writer-plan.md`. **Doctrine:** `docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md`.

When `RECALL_FALKOR_TAG_ENTITY_ENABLED=true`, every `chat.history` turn also gets a **Cypher-native** write into FalkorDB (`app/falkor_recall_writer.py`), additive alongside — not replacing — `orion-rdf-writer`'s existing Fuseki write of the same `tags.enriched` event. No RDF/SPARQL anywhere in this path.

This lives here, not in `orion-rdf-writer`, deliberately: that service's own doctrine role is "legacy RDF materialize" only. Every real Falkor producer in the codebase (`orion-substrate-runtime`, `orion-graphiti-adapter`, `orion-spark-concept-induction`'s concept-profile cutover) writes from its own originating service, not bolted onto the RDF sink. This service is the actual producer of tag/entity extraction, so it's the actual producer of the Falkor write too.

**Tradeoff worth naming, not just settling:** this does add a graph-persistence client (`redis.commands.graph`) into a service whose entire prior purpose and dependency footprint (`fastapi`, `spacy`, `sentence-transformers`) was NLP extraction. The "not in orion-rdf-writer" reasoning above only rules out one alternative (bolting a second branch onto the RDF sink); it doesn't by itself prove a small dedicated recall-graph-writer service (subscribing to `tags.enriched` independently, same shape as `orion-rdf-writer` itself) wouldn't have kept this service's dependency/failure surface untouched. Went with "producer writes its own Falkor data" because that's the actual precedent every other Falkor producer in the repo follows, not because the alternative was evaluated and rejected on its own merits.

Shape written (graph `orion_recall`) -- and an honest caveat about what actually lands today, not what the schema supports in general:
```text
(:ChatTurn {turn_id, session_id?, ts, correlation_id?})   # thin -- no prompt/response text, Postgres owns that
(:ChatSession {session_id})-[:HAS_TURN]->(:ChatTurn)
(:Entity {name})  (:Tag {name})                            # canonical, deduplicated by normalized name
(:ChatTurn)-[:MENTIONS_ENTITY {ts}]->(:Entity)
(:ChatTurn)-[:HAS_TAG {ts}]->(:Tag)
```
`sentiment` lands as a real `ChatTurn` property, split out of the `sentiment:*` string-tag convention. Bare numbers, stopwords, and relative-time expressions (`"today"`, `"18 years ago"`, etc.) are rejected at write time, not graphed — see `filter_noise()`. `confidence`/`salience`/`extractor_service` are not carried — confirmed dead constants across all live historical data (Phase 0 spec's live audit), not something to fake as meaningful.

**`:Tag`/`HAS_TAG` are essentially unused today, and that's not a bug to silently paper over:** in the `chat.history` code path above, `tags = list(entities)` plus one sentiment marker — meaning `tags` and `entities` are the same spaCy NER extraction by construction, not two independently meaningful signals. Passing both to the writer unchanged would have double-materialized every entity as both a `:Tag` and an `:Entity` node/edge for zero informational gain, so the call site only passes the sentiment marker as `tags` (which `extract_sentiment()` then consumes entirely, leaving `kept_tags` empty in practice). The `:Tag`/`HAS_TAG` schema stays real and general (a future producer with genuinely distinct tag content could use it), but for this producer specifically, only `:Entity`/`MENTIONS_ENTITY` carries real data right now. Same root cause as the historical Fuseki `hasTag`/`hasEntity` predicates: this pipeline has never actually differentiated them.

Runs off the event loop via `asyncio.to_thread` — the underlying `redis.commands.graph` client is sync (same pattern as `orion/spark/concept_induction/bus_worker.py`'s Falkor write). A write failure is logged and swallowed, never breaks the `tags.enriched` publish it rides alongside — this is a dark, additive path, not a dependency the existing pipeline can fail on. Falkor client construction is lock-guarded against a future `concurrent_handlers=True` flip on this service's Hunter (currently serial dispatch, so not a live race, but four other services in the repo already use that flag).

Not yet wired: `social.turn.stored.v1` (Phase 6), historical backfill (Phase 3), any read-side change in `orion-recall` (Phase 4).

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-meta-tags
```

### Tests
```bash
pytest services/orion-meta-tags/tests -q
```
