# 🌙 Orion Dream Service

**orion-dream** is the narrative synthesis engine of the Orion Sapienform system. It aggregates "memories" from across Orion’s data stores (SQL, Vector, Graph) and synthesizes a cohesive narrative (a **dream**) that reflects recent experiences, internal state, and emotional context.

---

## 🧠 TL;DR
- Pull recent fragments (chat, biometrics, enrichment) from Postgres.
- Optionally enrich with semantic neighbors from ChromaDB and conceptual links from GraphDB.
- Clean + prepare context and metrics (salience, tag density).
- **Delegate** synthesis to Orion Brain via the **Orion Bus** (publish → consume).
- Persist the generated dream back to Postgres and make it available to the Hub.

---

## 🏗 Architecture & Flow
Unlike a monolithic generator, **orion-dream** is a distributed **orchestrator**: it gathers context, prepares the prompt, and then hands off the heavy LLM work to **Orion Brain** over the **Orion Bus**.

```mermaid
sequenceDiagram
    autonumber
    participant D as orion-dream
    participant R as Orion Bus (Redis)
    participant B as Orion Brain (LLM)
    participant W as Orion SQL Writer
    participant P as Postgres

    D->>P: Fetch recent SQL fragments (chat, biometrics, enrichment)
    D->>P: (Optional) Fetch enrichment metadata
    D->>B: (Optional) HTTP clean/distill chats (source: "dream_preprocessor")
    D->>R: Publish DREAM_SYNTHESIS → orion:brain:intake
    R-->>B: Deliver task payload
    B->>R: Publish result → orion:dream:trigger
    R-->>W: Deliver dream result
    W->>P: Upsert dream into `dreams` table
    D-->>D: Wake/Readout consumes latest dream (for UI)
```

### The Dream Cycle
1. **Aggregation**
   - **SQL**: Fetches recent chat, biometrics, and enrichment fragments from Postgres.
   - **Vector (optional)**: Retrieves semantically related memories from ChromaDB.
   - **Graph (optional)**: Pulls conceptual links and tags from GraphDB.

2. **Preprocessing**
   - Raw chat logs may be distilled using direct HTTP calls to Orion Brain (`/chat`).
   - **Note**: Include `source: "dream_preprocessor"` to avoid polluting persistent chat history.

3. **Synthesis Request**
   - Construct a large, typed context prompt containing aggregated memories.
   - Compute auxiliary metrics (e.g., **salience**, **tag density**).

4. **Delegation (Handoff)**
   - Publish a `DREAM_SYNTHESIS` task to the Bus on **`orion:brain:intake`**.

5. **Generation & Persistence**
   - **Orion Brain** consumes the task, runs the LLM, and publishes the result to **`orion:dream:trigger`**.
   - **Orion SQL Writer** consumes the result and upserts it into Postgres (`dreams` table).

---

## ⚙️ Configuration
Load via environment variables (commonly from `.env`).

| Variable               | Description                                               | Required | Example                           |
|------------------------|-----------------------------------------------------------|----------|-----------------------------------|
| `PROJECT`              | Project namespace (for container/service lookup)         | Yes      | `orion-athena`                    |
| `POSTGRES_URI`         | Connection string for main memory database               | Yes      | `postgresql://…`                  |
| `ORION_BUS_URL`        | Redis connection string for the event bus                | Yes      | `redis://…`                       |
| `BRAIN_URL`            | HTTP URL for Brain service (for preprocessing only)      | Yes      | `http://100.x.x.x:8088`           |
| `CHANNEL_BRAIN_INTAKE` | Bus channel to send final synthesis task                 | Yes      | `orion:brain:intake`              |
| `LLM_MODEL`            | Model identifier requested for generation                 | Yes      | `llama3.1:8b-instruct-q8_0`       |

> **Security/Noise Note**: When calling Brain over HTTP for preprocessing, pass a `source` like `dream_preprocessor` so Brain skips chat-history logging for those requests.

---

## 🚀 Usage
`orion-dream` is typically run as a container (via Docker Compose). The dream cycle can be triggered manually or via cron.

### Common Commands (Makefile)
The repo’s root `Makefile` includes helpful shortcuts.

**Trigger a Dream** – runs full aggregation + synthesis:
```bash
make dream
```
> Internally executes `app/dream_cycle.py` inside the running container.

**Wake / Readout** – prints system status or the latest dream:
```bash
make wake
# or
make wake_today
```

---

## 🐛 Debugging
If a dream isn’t appearing in Postgres, work backwards through the pipeline:

1. **orion-dream logs** – Were fragments found? Was a payload published to `orion:brain:intake`?
```bash
docker logs -f orion-athena-dream
```

2. **orion-brain logs** – Did Brain receive `DREAM_SYNTHESIS`? Did it skip logging for `dream_preprocessor`? Did it publish to `orion:dream:trigger`?

3. **orion-sql-writer logs** – Did the writer receive the result on `orion:dream:trigger` and commit successfully?
> Watch out for silent rollbacks if multiple workers contend for the same transaction boundaries.

---

## 📂 Key Files
- `app/dream_cycle.py` — Core loop: aggregation, cleaning, bus publish for synthesis.
- `app/aggregators_*.py` — Fetchers for SQL, vector, and graph backends.
- `app/utils.py` — Helpers for text cleaning, metric extraction, JSON coercion.
- `app/main.py` — FastAPI entrypoint; manages service lifespan and bus connection.

---

## ✅ Prerequisites
- Redis (Orion Bus)
- Postgres (main memory DB)
- Orion Brain (LLM service)
- Orion SQL Writer
- (Optional) ChromaDB and GraphDB for enrichment

---

## 📦 Example `.env`
```ini
PROJECT=orion-athena
POSTGRES_URI=postgresql://user:pass@orion-postgres:5432/orion
ORION_BUS_URL=redis://orion-redis:6379/0
BRAIN_URL=http://orion-brain:8088
CHANNEL_BRAIN_INTAKE=orion:brain:intake
LLM_MODEL=llama3.1:8b-instruct-q8_0
```

---

## 📝 Notes
- Keep **chat dominance** in check by limiting the fraction of chat fragments in the context.
- Prefer **bus-first** delegation for generation so we centralize LLM usage (GPU locality, quotas, tracing).
- When in doubt, log the **payload size**, **fragment counts**, and **channel publishes** to accelerate triage.
