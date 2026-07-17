# Concept Induction (Spark)

Concept Induction consolidates recent Orion experience into profiles, clusters, and deltas, then publishes them on the Titanium bus.

## Flow

```mermaid
flowchart LR
    subgraph Intake
        A["orion:chat:history:log"]
        B["orion:collapse:mirror"]
        C["orion:memory:episode"]
    end
    subgraph ConceptInduction
        X["Extract (spaCy)"]
        Y["Embed (HTTP host)"]
        Z["Cluster (cosine/string)"]
        P["Profile Build + Delta"]
    end
    subgraph Outputs
        O1["orion:spark:concepts:profile<br/>kind=memory.concepts.profile.v1"]
        O2["orion:spark:concepts:delta<br/>kind=memory.concepts.delta.v1"]
        W1["orion:vector:write"]
        W2["orion:rdf:write"]
        W3["orion:collapse:sql-write"]
    end

    A & B & C --> X --> Y --> Z --> P --> O1
    P --> O2
    P --> W1
    P --> W2
    P --> W3
```

## Chat-history text source

For the hub chat-history intake channels (`orion:chat:history:log`, `orion:chat:history:turn`),
extraction reads the canonical `prompt`/`response` row from the `chat_history_log` Postgres table
(pooled connection, same table + DSN pattern `orion-recall` already reads directly) instead of
trusting the bus envelope's own `prompt`/`response` fields. This gives concept induction one
canonical, already-committed text per turn instead of whatever shape a given intake channel's
envelope happens to carry — useful on its own (dedup, no partial/mid-write reads) but **not** a
content-cleaning step: `orion-sql-writer` writes `prompt`/`response` straight from the same
bus-published payload with no stripping, so the Postgres row and the envelope carry identical text.

If a reply contains chat-stance/scaffold vocabulary (e.g. identity-kernel or hazard-label phrasing
from `services/orion-cortex-exec/app/chat_stance.py` recited back into a live response), that text
is present in both sources equally — this change alone does not filter it out. `extractor.py`
(`SpacyConceptExtractor`) still has no stopword/denylist filtering, which is the layer that would
actually need to change to stop that vocabulary from becoming a concept candidate.

The lookup is fail-open — any DB error, or the row not being written yet (bounded retry,
`CONCEPT_CHAT_PG_LOOKUP_RETRIES`, retried on both a miss and a transient connection error), falls
back to the previous envelope-based extraction. See `orion/spark/concept_induction/chat_history_pg.py`.

## Channels and kinds

| Purpose | Default channel | Kind |
| --- | --- | --- |
| Intake | `orion:chat:history:log`, `orion:collapse:mirror`, `orion:memory:episode` | varies (chat.message, collapse.mirror, etc.) |
| Profile out | `orion:spark:concepts:profile` | `memory.concepts.profile.v1` |
| Delta out | `orion:spark:concepts:delta` | `memory.concepts.delta.v1` |
| Optional forward | `orion:vector:write`, `orion:rdf:write`, `orion:collapse:sql-write` | `vector.write`, `rdf.write.request`, `sql.write` |

## Config knobs

| Setting | Default | Notes |
| --- | --- | --- |
| `BUS_INTAKE_CHANNELS` | `["orion:chat:history:log","orion:collapse:mirror","orion:memory:episode"]` | JSON list |
| `BUS_PROFILE_OUT` | `orion:spark:concepts:profile` | Profile channel |
| `BUS_DELTA_OUT` | `orion:spark:concepts:delta` | Delta channel |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model |
| `EMBEDDINGS_BASE_URL` | `http://orion-athena-vector-host:8320` | Base URL; concept induction calls `POST /embedding` using the vector-host `EmbeddingGenerateV1` contract. |
| `USE_CORTEX_ORCH` | `false` | LLM refinement via `concept_induction` verb |
| `CONCEPT_WINDOW_MAX_EVENTS` | `200` | Rolling window size |
| `CONCEPT_WINDOW_MAX_MINUTES` | `360` | Rolling window age |
| `CONCEPT_CHAT_PG_LOOKUP_ENABLED` | `true` | Read chat-history text from Postgres instead of the bus envelope |
| `CONCEPT_CHAT_PG_DSN` | `postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney` | Same DSN pattern as `RECALL_PG_DSN` |
| `CONCEPT_CHAT_PG_LOOKUP_RETRIES` | `3` | Bounded retries for the SQL-writer commit race + transient connection errors |
| `CONCEPT_CHAT_PG_LOOKUP_RETRY_DELAY_SEC` | `0.3` | Delay between retries |

## Local commands

```bash
# Build & run
docker compose -f services/orion-spark-concept-induction/docker-compose.yml --env-file .env up -d orion-spark-concept-induction

# Publish a test event and wait for profile
python -m scripts.test_concept_induction_publish
```
