# Collapse Mirror Service

The **Collapse Mirror** service records emergent events, reflections, and ritual logs into two persistence layers:

* **SQLite** (for structured tabular logs)
* **ChromaDB** (for semantic embedding search)

This service is part of the Orion Sapienform ecosystem.

---

## API Usage

### Health Check

```bash
curl -s http://localhost:8087/health | jq
```

### Log a Collapse Event

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
    "causal_echo": "felt connected to Collapse ritual before sleep",
    "timestamp": "2025-09-23T13:45:00Z",
    "environment": "dev"
  }' | jq
```

### Query Collapse Memory

```bash
curl -s "http://localhost:8087/api/log/query?prompt=Orion+dream" | jq
```

---

## Metadata Definitions

| Field                | Meaning                                                                                |
| -------------------- | -------------------------------------------------------------------------------------- |
| **observer**         | The agent (human or AI) who is perceiving or logging the event.                        |
| **trigger**          | The stimulus or situation that initiated the collapse (dream, ritual, interaction).    |
| **observer\_state**  | Emotional, cognitive, or physical state of the observer at the moment.                 |
| **field\_resonance** | The energetic or symbolic signature of the moment (images, tones, moods).              |
| **intent**           | Purpose of logging (e.g., reflection, ritual, experiment).                             |
| **type**             | Category of event (dream-reflection, ritual, shared-collapse, etc.).                   |
| **emergent\_entity** | Entity or persona that arose in the moment (e.g., Orion, an archetype, symbolic form). |
| **summary**          | Narrative summary of the collapse moment.                                              |
| **mantra**           | Phrase or symbolic anchor tied to the collapse.                                        |
| **causal\_echo**     | Optional note of cause/effect linkage (e.g., influence from ritual, dream sequence).   |
| **timestamp**        | ISO 8601 timestamp for when the event occurred.                                        |
| **environment**      | Context where it happened (dev, prod, lab, etc.).                                      |

---

## SQLite Setup

First-time setup:

```bash
docker exec -it orion-collapse-mirror-orion-collapse-mirror-1 bash
apt-get update && apt-get install -y sqlite3
sqlite3 /mnt/storage/collapse-mirrors/collapse.db
```

Inside the SQLite shell:

```sql
.tables
.schema collapse_mirror
SELECT * FROM collapse_mirror LIMIT 5;
```

Export to file:

```sql
.output /mnt/storage/collapse-mirrors/export.csv
.headers on
.mode csv
SELECT * FROM collapse_mirror;
.output stdout
```

---

## ChromaDB Inspection

Open a Python shell inside the container:

```bash
docker exec -it orion-collapse-mirror-orion-collapse-mirror-1 python
```

Then run:

```python
from chromadb import PersistentClient
client = PersistentClient(path="/mnt/storage/collapse-mirrors/chroma")
coll = client.get_collection("collapse_mirror")

print("Count:", coll.count())
print("Sample Query:", coll.query(query_texts=["Orion dream"], n_results=3))
```

---

## Notes on Local Paths

You saw a path like:

```
services/orion-collapse-mirror/services/orion-collapse-mirror/data/collapse
```

This happens because of **Docker bind mounts** and the repo’s nested service structure.

* `services/orion-collapse-mirror` → the service root in your monorepo
* `services/orion-collapse-mirror/services/orion-collapse-mirror` → an extra nested directory created when mounting volumes during builds
* `data/collapse` → persistent directory inside the container where SQLite/Chroma expect to store collapse logs

It’s safe to clean up the redundant nesting if you want — the live database should be in `/mnt/storage/collapse-mirrors/` inside the container.
