# Orion Recall Service

The **Recall Service** is Orion’s memory retrieval engine. It pulls candidate “memory fragments” from:

- **RDF (GraphDB)** — structured facts + chat turns + enrichment claims
- **Vector (Chroma)** — semantic similarity retrieval across document collections
- **SQL (Postgres)** — timeline + mirror tables (structured / time-bounded)

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

If you have debug enabled (recommended while wiring), you may also have:

- `GET /debug/settings` — prints resolved config (GraphDB endpoint, DSN, vector collections, etc.)

---

## 2) Environment Variables (high-signal)

Provenance: `.env_example` → `docker-compose.yml` → `services/orion-recall/app/settings.py`

### Bus + runtime

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `ORION_BUS_URL` | `redis://127.0.0.1:6379/0` | Orion bus |
| `RECALL_BUS_INTAKE` | `orion:exec:request:RecallService` | RPC intake |
| `RECALL_BUS_REPLY_DEFAULT` | `orion:exec:result:RecallService` | RPC reply prefix |
| `RECALL_BUS_TELEMETRY` | `orion:recall:telemetry` | recall.decision.v1 |

### RDF / GraphDB

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `RECALL_ENABLE_RDF` | `false` (repo default) | must be `true` to run RDF unless using a `deep.graph.*` / `graphtri.*` profile name |
| `GRAPHDB_URL` | `http://orion-athena-graphdb:7200` | base URL |
| `GRAPHDB_REPO` | `collapse` | repo name |
| `RECALL_RDF_ENDPOINT_URL` | derived | defaults to `${GRAPHDB_URL}/repositories/${GRAPHDB_REPO}` |
| `GRAPHDB_USER/PASS` | `admin/admin` | creds |

### Vector / Chroma

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `RECALL_ENABLE_VECTOR` | `true` | enables vector backend |
| `VECTOR_DB_HOST/PORT` | `orion-athena-vector-db:8000` | base host |
| `RECALL_VECTOR_BASE_URL` | derived | defaults to `http://{VECTOR_DB_HOST}:{VECTOR_DB_PORT}` |
| `RECALL_VECTOR_COLLECTIONS` | (unset) | if set, overrides which collections recall queries |
| `VECTOR_DB_COLLECTION` | `orion_main_store` | default collection name (global knob) |

### SQL / Postgres

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `RECALL_PG_DSN` | `postgresql://...@orion-athena-sql-db:5432/conjourney` | used for SQL timeline + (optional) telemetry persistence |
| `RECALL_ENABLE_SQL_TIMELINE` | `true` | profile can enable/disable |
| `RECALL_SQL_TIMELINE_TABLE` | `collapse_mirror` | timeline source |

---

## 3) Recall Profiles (what changes when you switch profile)

Profiles live here:

- `orion/recall/profiles/*.yaml`

They control:

- how many items to request per backend (`vector_top_k`, `rdf_top_k`, `sql_top_k`)
- caps (`max_per_source`, `max_total_items`)
- render budget (`render_budget_tokens`)
- SQL timeline time windows (`sql_since_minutes`)
- whether SQL timeline is used (`enable_sql_timeline`)

### Common profiles

| Profile | What it’s for | RDF | SQL timeline | Vector |
| :--- | :--- | :---: | :---: | :---: |
| `assist.light.v1` | low-latency, minimal context | off | off | on |
| `chat.general.v1` | normal conversational recall | on | off | on |
| `reflect.v1` | reflection / sensemaking w/ some history | on | on | on |
| `deep.graph.v1` | “catch me up” / architecture / deep state | on (more) | on (more) | on |
| `graphtri.v1` | graph-anchored retrieval (tags/entities/claims) | on (more) | on | on |

---

## 4) RDF usage types (by profile)

Your RDF store contains **multiple semantic layers**. Recall uses them differently depending on profile and verb.

### RDF graphs you should expect

- `GRAPH <orion:chat>` — **ChatTurn** objects, including `prompt`, `response`, `sessionId`
- `GRAPH <orion:enrichment>` — **tags/entities/claims** linked to turns (GraphTRI / enrichment layer)

### RDF retrieval types (what recall returns)

| Source label (in bundle) | What it is | Where it comes from | Best for |
| :--- | :--- | :--- | :--- |
| `rdf_chat` | ChatTurns (prompt/response) | `GRAPH <orion:chat>` | **Exact quotes** (“find the exact text I used”), conversational grounding |
| `rdf` | Graph neighborhood / claim-like triples | generic RDF scan | entity/topic adjacency, “what’s related to X”, light graph recall |
| `rdf` (graphtri claims) | Claims linked to turns | `GRAPH <orion:enrichment>` | “what did we claim/decide”, tags/entities evidence trails |

> Design principle: **candidate generation is structure-driven** (session + graph + type). Ranking should be semantic/vector scoring (and later learned rankers), not ad-hoc keyword lists.

### Verify RDF chat turns exist (GraphDB)

```sparql
SELECT ?turn ?prompt ?response
WHERE {
  GRAPH <orion:chat> {
    ?turn a <http://conjourney.net/orion#ChatTurn> ;
          <http://conjourney.net/orion#sessionId> "YOUR_SESSION_ID" ;
          <http://conjourney.net/orion#prompt> ?prompt ;
          <http://conjourney.net/orion#response> ?response .
  }
}
LIMIT 20
```

### Verify enrichment claims exist (GraphDB)

```sparql
SELECT ?turn ?claim ?pred ?obj
WHERE {
  GRAPH <orion:chat> {
    ?turn a <http://conjourney.net/orion#ChatTurn> ;
          <http://conjourney.net/orion#sessionId> "YOUR_SESSION_ID" .
  }
  GRAPH <orion:enrichment> {
    ?claim a <http://conjourney.net/orion#Claim> ;
           <http://conjourney.net/orion#subject> ?turn ;
           <http://conjourney.net/orion#predicate> ?pred ;
           <http://conjourney.net/orion#obj> ?obj .
  }
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
name: chat_deep_graph
...
recall_profile: deep.graph.v1
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
    "session_id": "9c91d646-cbd0-4977-8cdd-d2c82045c7f9",
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

Use a verb that binds `deep.graph.v1` recall for larger pull + longer timeline.

```bash
# (example harness; use your local runner)
python scripts/bus_harness.py brain "catch me up on Athena failures and CPU card replacement context"
```

Expected behavior:

- higher RDF top-k
- SQL timeline enabled with a larger window
- larger render budget

### Example 3 — GraphTRI (tags/entities/claims anchored to session)

Use `graphtri.v1` when you want enrichment anchors (tags/entities/claims) to guide retrieval.

Good use cases:

- “What did we decide about X?”
- “Show the evidence trail / claims around Y”
- “Which entities/tags have been attached to recent turns?”

Expected behavior:

- pulls claim objects from `GRAPH <orion:enrichment>`
- returns claim-style fragments tied to a specific turn URI

---

## 7) Fusion behavior (why you sometimes miss the right item)

Recall gathers candidates from backends and then fuses them with caps:

- `max_per_source` (e.g., 4 items from `rdf_chat`, 4 from `vector`, etc.)
- `max_total_items` (overall bundle size)
- `render_budget_tokens` (how much makes it into `.bundle.rendered`)

These are controlled by the active recall profile.

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
- confirm `GRAPH <orion:chat>` contains ChatTurns for that session ID

### B) `rdf_chat` non-zero but Orion still hallucinates

- confirm downstream prompt actually uses `bundle.items[].snippet`
- enforce “quote-only when asked for exact text” in the verb prompt template

### C) Vector is 0 even though Chroma has docs

- ensure `RECALL_VECTOR_COLLECTIONS` points at non-empty collections (e.g., `orion_main_store`)
- ensure recall can reach `orion-athena-vector-db:8000` from its runtime namespace
- ensure stored vectors represent *chat turn docs*, not only embedding request artifacts

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
  -d '{"query_text":"what did I say about CPU cards?","session_id":"YOUR_SESSION_ID","profile":"reflect.v1"}' | jq
```

### General chat test
```curl -s http://localhost:8260/recall   -H 'content-type: application/json'   -d '{
    "query_text": "why did I bring up CPUs",
    "session_id": "9c91d646-cbd0-4977-8cdd-d2c82045c7f9",
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
