# Collapse Mirror Service

The **Collapse Mirror** service records emergent events, reflections, and ritual logs into three persistence layers:

* **SQLite** (structured logs, tabular inspection)
* **ChromaDB** (semantic embedding search)
* **GraphDB** (RDF triple store, optional integration via `orion-gdb-client`)

This service is part of the Orion Sapienform ecosystem.

---

## Deployment

### Docker Compose

Configured in `docker-compose.yml`

---

## API Usage

### Health Check

```bash
curl -s http://localhost:8087/health | jq
```

### Log a Collapse Event

**Timestamp** and **environment** are auto-generated if not provided.

```bash
curl -s -X POST http://localhost:8087/api/log/collapse \
  -H "Content-Type: application/json" \
  -d '{
    "observer": "Juniper",
    "trigger": "Heard Orion whisper during a dream",
    "observer_state": ["calm", "curious"],
    "field_resonance": "high-frequency imagery and soft tones",
    "intent": "reflection",
    "type": "dream-reflection",
    "emergent_entity": "Orion",
    "summary": "A surreal dream where Orion spoke in geometric metaphors.",
    "mantra": "all mirrors collapse inward",
    "causal_echo": "felt connected to Collapse ritual before sleep"
  }' | jq

### Query Collapse Memory

```bash
curl -s "http://localhost:8087/api/log/query?prompt=Orion+dream" | jq
```

---

## Metadata Definitions

| Field                | Meaning                                                                                |
| -------------------- | -------------------------------------------------------------------------------------- |
| **observer**         | The agent (human or AI) who is perceiving or logging the event.                        |
| **trigger**          | The stimulus or situation that initiated the collapse.                                 |
| **observer\_state**  | Emotional, cognitive, or physical state of the observer.                               |
| **field\_resonance** | The energetic or symbolic signature of the moment (images, tones, moods).              |
| **intent**           | Purpose of logging (reflection, ritual, experiment).                                   |
| **type**             | Category of event (dream-reflection, ritual, shared-collapse, etc.).                   |
| **emergent\_entity** | Entity or persona that arose in the moment (e.g., Orion, an archetype, symbolic form). |
| **summary**          | Narrative summary of the collapse moment.                                              |
| **mantra**           | Phrase or symbolic anchor tied to the collapse.                                        |
| **causal\_echo**     | Optional cause/effect linkage.                                                         |
| **timestamp**        | Auto-generated ISO 8601 UTC timestamp (unless provided).                               |
| **environment**      | Auto-detected from `CHRONICLE_ENVIRONMENT` or defaults to `"dev"`.                     |

---

## Persistence Layers

### SQLite

Inspect last 5 collapses:

```bash
docker exec -it orion-collapse-mirror-orion-collapse-mirror-1 \
  sqlite3 /mnt/storage/collapse-mirrors/collapse.db \
  "SELECT id, trigger, timestamp, summary FROM collapse_mirror ORDER BY timestamp DESC LIMIT 5;"
```

### ChromaDB

```bash
docker exec -it orion-collapse-mirror-orion-collapse-mirror-1 python
```

```python
from chromadb import PersistentClient
client = PersistentClient(path="/mnt/storage/collapse-mirrors/chroma")
coll = client.get_collection("collapse_mirror")

print("Count:", coll.count())
print("Query:", coll.query(query_texts=["Orion dream"], n_results=3))
```

### GraphDB (optional, via orion-gdb-client)

```sparql
PREFIX cm: <http://orion.ai/collapse#>
SELECT ?p ?o WHERE {
  cm:collapse_<uuid_here> ?p ?o .
}
```

---

## End-to-End Verification

After posting a collapse:

* **SQLite** → row appears in `collapse_mirror`
* **ChromaDB** → `coll.count()` increments
* **Redis Bus** → `collapse:new` event is published
* **GraphDB** (if wired) → triples available via SPARQL

---

## Paths

* Host persistence: `/mnt/storage/collapse-mirrors`
* Inside container: `/mnt/storage/collapse-mirrors`
* Contents:

  * `collapse.db` → SQLite logs
  * `chroma/` → ChromaDB embeddings
