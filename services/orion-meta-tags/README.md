# Orion Meta Tags

The **Meta Tags** service provides automated enrichment for collapse events, chat turns, and social-room turns. It runs spaCy NER (`SPA_MODEL`, default `en_core_web_trf`) to extract entities (reused as tags) and a keyword-heuristic sentiment classifier (`sentiment:positive`/`negative`/`neutral`, currently smuggled through the `tags` list — see the Falkor writer section below for where that gets split back out into a real field), adding structured metadata to the original payload.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:collapse:triage` | `CHANNEL_EVENTS_TRIAGE` | `collapse.mirror` | Raw entries needing enrichment (Juniper-observed only; Orion's own outputs are skipped). |
| `orion:chat:history:turn` | `CHANNEL_EVENTS_CHAT_TURN` | `chat.history` | Chat turns — the source of the Falkor recall write below. |
| `orion:chat:social:stored` (hardcoded, no env var) | — | `social.turn.stored.v1` | Social-room turns. Shares the same tagging code path and, as of 2026-07-18, the same Falkor write as chat turns. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:tags:enriched` | `CHANNEL_EVENTS_TAGGED` | `tags.enriched` | Enriched metadata for collapse-mirror events. |

Chat/social-turn tag+entity enrichment is **not published to the bus at all** — it goes straight into FalkorDB (below). See "Historical note" at the end of this section for what used to be here.

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_EVENTS_TRIAGE` | `orion:collapse:triage` | Intake channel. |
| `CHANNEL_EVENTS_CHAT_TURN` | `orion:chat:history:turn` | Chat-turn intake channel. |
| `CHANNEL_EVENTS_TAGGED` | `orion:tags:enriched` | Collapse-mirror output channel. |
| `RECALL_FALKOR_TAG_ENTITY_ENABLED` | `true` | See Falkor writer section. Default is `true` because Falkor is the *only* persistence path for chat/social tags now — `false` means no persistence at all for this data, not "skip a shadow write." |
| `FALKORDB_URI` | `redis://orion-athena-falkordb:6379` | Shared FalkorDB instance (same one substrate-runtime and graphiti-adapter use, different graph name). |
| `FALKORDB_RECALL_GRAPH` | `orion_recall` | Graph name — separate from `orion_substrate` and `graphiti_temporal`. |

## Recall Falkor writer (Phase 2)

**Design/implementation:** `docs/superpowers/plans/2026-07-18-recall-tag-entity-falkor-writer-plan.md`. **Doctrine:** `docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md`.

When `RECALL_FALKOR_TAG_ENTITY_ENABLED=true` (default), every `chat.history` **and** `social.turn.stored.v1` turn gets a **Cypher-native** write into FalkorDB (`app/falkor_recall_writer.py`). As of 2026-07-18 this is the sole persistence path for chat/social-turn tag+entity enrichment — no RDF/SPARQL, no bus publish. See "Historical note" below for the Fuseki dual-write this replaced.

This lives here, not in `orion-rdf-writer`, deliberately: that service's own doctrine role is "legacy RDF materialize" only. Every real Falkor producer in the codebase (`orion-substrate-runtime`, `orion-graphiti-adapter`, `orion-spark-concept-induction`'s concept-profile cutover) writes from its own originating service, not bolted onto the RDF sink. This service is the actual producer of tag/entity extraction, so it's the actual producer of the Falkor write too.

**Tradeoff worth naming, not just settling:** this does add a graph-persistence client (`redis.commands.graph`) into a service whose entire prior purpose and dependency footprint (`fastapi`, `spacy`, `sentence-transformers`) was NLP extraction. The "not in orion-rdf-writer" reasoning above only rules out one alternative (bolting a second branch onto the RDF sink); it doesn't by itself prove a small dedicated recall-graph-writer service (subscribing independently, same shape as `orion-rdf-writer` itself) wouldn't have kept this service's dependency/failure surface untouched. Went with "producer writes its own Falkor data" because that's the actual precedent every other Falkor producer in the repo follows, not because the alternative was evaluated and rejected on its own merits.

Shape written (graph `orion_recall`) -- and an honest caveat about what actually lands today, not what the schema supports in general:
```text
(:ChatTurn {turn_id, source_kind, session_id?, ts, correlation_id?})   # thin -- no prompt/response text, Postgres owns that
(:ChatSession {session_id})-[:HAS_TURN]->(:ChatTurn)
(:Entity {name})  (:Tag {name})                            # canonical, deduplicated by normalized name
(:ChatTurn)-[:MENTIONS_ENTITY {ts}]->(:Entity)
(:ChatTurn)-[:HAS_TAG {ts}]->(:Tag)
```
`source_kind` (`"chat.history"` or `"social.turn.stored.v1"`) exists because both kinds share this same `:ChatTurn`/`:ChatSession` label -- without it, a `turn_id` collision between a Hub chat turn and a social-room turn (both producer-supplied strings, not namespaced against each other) would silently fuse two unrelated conversations into one node with no way to tell them apart afterward. `:ChatSession` itself still has no discriminator (only `session_id`) -- a real, not-yet-closed gap if chat and social sessions ever share an ID space.
`sentiment` lands as a real `ChatTurn` property, split out of the `sentiment:*` string-tag convention. Bare numbers, stopwords, and relative-time expressions (`"today"`, `"18 years ago"`, etc.) are rejected at write time, not graphed — see `filter_noise()`. `confidence`/`salience`/`extractor_service` are not carried — confirmed dead constants across all live historical data (Phase 0 spec's live audit), not something to fake as meaningful.

**`:Tag`/`HAS_TAG` are essentially unused today, and that's not a bug to silently paper over:** `tags`/`entities` are the same spaCy NER extraction by construction, not two independently meaningful signals — see `handle_triage_event`'s chat-kind branch. Passing both to the writer unchanged would have double-materialized every entity as both a `:Tag` and an `:Entity` node/edge for zero informational gain, so the call site only passes the sentiment marker as `tags` (which `extract_sentiment()` then consumes entirely, leaving `kept_tags` empty in practice). The `:Tag`/`HAS_TAG` schema stays real and general (a future producer with genuinely distinct tag content could use it), but for this producer specifically, only `:Entity`/`MENTIONS_ENTITY` carries real data right now. Same root cause as the historical Fuseki `hasTag`/`hasEntity` predicates: this pipeline never actually differentiated them.

Runs off the event loop via `asyncio.to_thread` — the underlying `redis.commands.graph` client is sync (same pattern as `orion/spark/concept_induction/bus_worker.py`'s Falkor write). **A write failure is logged at ERROR and swallowed** (`falkor_recall_write_failed_data_lost`, not a WARNING — bumped 2026-07-18 alongside the Fuseki kill, to reflect that it's no longer a dark/additive path riding alongside a real copy elsewhere): a swallowed failure here means that turn's tag/entity data is not persisted anywhere, full stop. There is no retry and no dead-letter queue on this path (unlike `orion-rdf-writer`'s `RDF_WRITE_QUEUE_MAXSIZE` dead-letter behavior) — a sustained FalkorDB outage silently loses every turn's tags/entities for its duration, one ERROR log line each, with nothing currently alerting on that log pattern. Watch `falkor_recall_write_failed_data_lost` frequency and consumer lag on `orion:chat:history:turn`/`orion:chat:social:stored` if real volume turns out to be high enough for either to matter — this is the natural next thing to instrument if this path proves it needs it, not something built preemptively here. Falkor client construction is lock-guarded against a future `concurrent_handlers=True` flip on this service's Hunter (currently serial dispatch, so not a live race, but four other services in the repo already use that flag).

**Historical backfill (Phase 3): done, 2026-07-18.** `scripts/backfill_recall_falkor_chat_tags_snapshot.py` + `scripts/backfill_recall_falkor_chat_tags_extract_and_write.py` populated `orion_recall` with every historical `chat_history_log`/`social_room_turns` row that predates the live writer -- 1,708 turns written, 0 errors, real historical timestamps preserved (not backdated to "now"). Not a copy of old Fuseki data (there was almost nothing there to copy -- the same observer-gate bug blocked the Fuseki `chat_tagging` extraction too); re-ran the real extraction pipeline (spaCy NER + sentiment heuristic) against raw Postgres text and wrote via this same `write_chat_turn_tags_to_falkor` function. `orion-recall` read-side change (Phase 4): built, PR #1192 (chatturn only; graphtri/Claim-based fragments still need their own redesign, unaffected by this backfill).

### Historical note: the Fuseki dual-write (killed 2026-07-18)

From when this writer first shipped (PR #1180) until 2026-07-18, chat/social-turn tag+entity data was **also** published to the bus (`orion:tags:chat:enriched` / `CHANNEL_EVENTS_TAGGED_CHAT`) for `orion-rdf-writer` to materialize into Fuseki as a `ChatTurn` subject with `hasTag`/`hasEntity` predicates (`rdf_builder.py`, `enrichment_type == "chat_tagging"`) — a *different* path from the raw `chat.history` → `ChatTurn`/`ChatMessage` RDF materialization killed on 2026-07-17.

Both the Fuseki copy and this Falkor writer were effectively dark for ~6 months due to an unrelated observer-gate bug (`fix/meta-tags-chat-history-observer-gate`) that unconditionally skipped every real chat turn before it reached tagging. Fixing that gate brought both stores live simultaneously for the first time; the very next change (this one) killed the now-redundant Fuseki side and extended the Falkor write to cover `social.turn.stored.v1` too (previously chat.history-only — that gap would otherwise have meant social-room turns lost persistence entirely once Fuseki was cut). Verified live before the cut: a real chat turn's `ChatTurn` node queryable directly via `GRAPH.QUERY` against `orion_recall`, matching the same `orion-rdf-writer` Fuseki enqueue log line for the identical `correlation_id`.

Do not re-add the Fuseki side without a real reason — it was a redundant second materialization of the exact same entities/sentiment, not an independent signal.

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-meta-tags
```

### Tests
```bash
pytest services/orion-meta-tags/tests -q
```
