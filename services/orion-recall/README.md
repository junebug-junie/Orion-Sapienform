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

> **Note:** `session_id` is accepted in recall requests for backwards compatibility, but recall ignores it for retrieval and ranking.

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

- `GRAPH <orion:chat>` — **ChatTurn** objects, including `prompt`, `response`, and optional `sessionId`
- `GRAPH <orion:enrichment>` — **tags/entities/claims** linked to turns (GraphTRI / enrichment layer)

### RDF retrieval types (what recall returns)

| Source label (in bundle) | What it is | Where it comes from | Best for |
| :--- | :--- | :--- | :--- |
| `rdf_chat` | ChatTurns (prompt/response) | `GRAPH <orion:chat>` | **Exact quotes** (“find the exact text I used”), conversational grounding |
| `rdf` | Graph neighborhood / claim-like triples | generic RDF scan | entity/topic adjacency, “what’s related to X”, light graph recall |
| `rdf` (graphtri claims) | Claims linked to turns | `GRAPH <orion:enrichment>` | “what did we claim/decide”, tags/entities evidence trails |

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

### Example 3 — GraphTRI (tags/entities/claims anchored to keywords)

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
3) `orion-vector-writer` stores into Chroma (collection often `orion_chat` for chat docs; `orion_main_store` for general)
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

## 6) Inspect stored chat docs in `orion_chat`

```bash
docker exec -i orion-athena-vector-writer python - <<'PY'
import chromadb
from chromadb.config import Settings
import pprint

client = chromadb.HttpClient(host="orion-athena-vector-db", port=8000, settings=Settings(anonymized_telemetry=False))
col = client.get_collection("orion_chat")

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
col = client.get_collection("orion_chat")

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

If `vector: 0` but you know Chroma has data, check:
- `RECALL_VECTOR_COLLECTIONS` points at the correct collection (often `orion_chat`)
- recall uses global metadata/node filters only

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

for colname in ["orion_main_store", "orion_chat", "orion_general", "orion_collapse"]:
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
