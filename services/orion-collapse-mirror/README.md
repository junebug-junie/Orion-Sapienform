# Collapse Mirror Service

The **Collapse Mirror** service records emergent events, reflections, and ritual logs into two persistence layers:

* **SQLite** (for structured tabular logs)
* **ChromaDB** (for semantic embedding search)

This service is part of the Orion Sapienform ecosystem.

---

## Deployment

### Docker Compose

The service is configured in `docker-compose.yml`:

```yaml
services:
  orion-collapse-mirror:
    build:
      context: ../..
      dockerfile: services/orion-collapse-mirror/Dockerfile
    image: orion-collapse-mirror:latest
    restart: unless-stopped
    networks:
      - app-net
    environment:
      - POSTGRES_URI=sqlite:////mnt/storage/collapse-mirrors/collapse.db
    volumes:
      - /mnt/storage/collapse-mirrors:/mnt/storage/collapse-mirrors
    ports:
      - "8087:8087"

networks:
  app-net:
    external: true
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first
COPY services/orion-collapse-mirror/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY services/orion-collapse-mirror/app ./app

EXPOSE 8087
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8087", "--reload"]
```

### Launch via rebuild script

`services/rebuild-services.sh` includes:

```bash
# collapse mirror
echo "== Build & start orion-collapse-mirror stack =="
(
  cd "$MIRROR_DIR" && \
  PORT="$MIRROR_PORT" REDIS_URL="${REDIS_URL:-redis://orion-redis:6379/0}" SERVICE_NAME="orion-collapse-mirror" \
  docker compose -f "$(compose_file "$MIRROR_DIR")" up -d --build
)
echo "→ waiting for orion-collapse-mirror on :$MIRROR_PORT ..."
wait_for_http "http://localhost:$MIRROR_PORT/health" 80 0.25 || die "orion-collapse-mirror did not become healthy"
```

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

## SQLite Access

The container already includes `sqlite3`. You can open the database shell with:

```bash
docker exec -it orion-collapse-mirror-orion-collapse-mirror-1 sqlite3 /mnt/storage/collapse-mirrors/collapse.db
```

Inside SQLite:

```sql
.tables
.schema collapse_mirror
SELECT * FROM collapse_mirror LIMIT 5;
```

Export:

```sql
.output /mnt/storage/collapse-mirrors/export.csv
.headers on
.mode csv
SELECT * FROM collapse_mirror;
.output stdout
```

---

## ChromaDB Inspection

Inspect semantic embeddings:

```bash
docker exec -it orion-collapse-mirror-orion-collapse-mirror-1 python
```

```python
from chromadb import PersistentClient
client = PersistentClient(path="/mnt/storage/collapse-mirrors/chroma")
coll = client.get_collection("collapse_mirror")

print("Count:", coll.count())
print("Sample Query:", coll.query(query_texts=["Orion dream"], n_results=3))
```

---

## Notes on Paths

* Host path for persistence: `/mnt/storage/collapse-mirrors`
* Container path (mounted): `/mnt/storage/collapse-mirrors`
* Files inside:

  * `collapse.db` → SQLite database
  * `chroma/` → ChromaDB collection data
