# Orion LangChain Memory Service

This project packages an **extended Orion-LangChain service** with **multi-layer memory**:
- 🧠 Hot memory (RAM buffer of recent turns)
- 🔍 Warm memory (Chroma semantic search, persisted)
- 🗄️ Cold memory (Postgres archive + RDF triples in GraphDB)
- 🌙 Reflection job (nightly summarization into Warm + RDF dream logs)

---

## 🚀 Components

- **FastAPI service (`app.py`)**
  - `/query` endpoint
  - Calls Orion Brain service (`orion-brain-service`)
  - Injects Hot + Warm memory
  - Logs all turns into Postgres + RDF (GraphDB)

- **Reflection job (`reflect.py`)**
  - Runs daily
  - Pulls last 24h from Postgres
  - Summarizes via Orion Brain
  - Stores embeddings in Chroma
  - Pushes summaries as RDF triples into GraphDB

- **GraphDB**
  - Ontotext GraphDB container
  - Stores RDF structured memory + reflection summaries

---

## 📂 Structure

```
orion-langchain/
├── Dockerfile
├── requirements.txt
├── app.py
├── reflect.py
└── docker-compose.yml
```

---

## 🛠️ Setup

1. **Prepare Postgres**

```sql
CREATE TABLE IF NOT EXISTS conversation_memory (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ DEFAULT NOW(),
    user_msg TEXT NOT NULL,
    bot_msg  TEXT NOT NULL
);
```

2. **Configure Environment**

Edit `docker-compose.yml` or use `.env`:

```env
PG_PASS=yourpassword
```

3. **Start Services**

```bash
docker compose up -d --build
```

- Orion-LangChain API: [http://localhost:8090](http://localhost:8090)
- GraphDB Workbench: [http://localhost:7200](http://localhost:7200)

---

## 📡 Usage

### Send a query

```bash
curl -X POST http://localhost:8090/query \
  -H "Content-Type: application/json" \
  -d '{"message": "Remind me what we discussed about memory layers."}'
```

### Check Postgres

```sql
SELECT * FROM conversation_memory ORDER BY ts DESC LIMIT 5;
```

### Check GraphDB (SPARQL)

```sparql
PREFIX orion: <http://conjourney.net/orion#>

SELECT ?turn ?msg ?reply ?time
WHERE {
  ?turn a orion:ConversationTurn ;
        orion:hasUserMessage ?msg ;
        orion:hasBotMessage ?reply ;
        orion:timestamp ?time .
}
ORDER BY DESC(?time)
LIMIT 5
```

### Check Reflection Summaries

```sparql
PREFIX orion: <http://conjourney.net/orion#>

SELECT ?summary ?text ?time
WHERE {
  ?summary a orion:ReflectionSummary ;
           orion:summaryText ?text ;
           orion:timestamp ?time .
}
ORDER BY DESC(?time)
LIMIT 5
```

---

## 🌙 Reflection Job

The reflection job (`orion-reflection` service) runs once every 24h:
- Fetches new turns from Postgres
- Summarizes via Orion Brain
- Embeds into Chroma (Warm memory)
- Stores RDF summaries into GraphDB

You can tweak the sleep cycle in `docker-compose.yml`:

```yaml
command: ["bash", "-c", "while true; do python reflect.py; sleep 86400; done"]
```

---

## ✨ Future Extensions
- Weekly/monthly summaries
- Cross-link entities in RDF (who, what, where)
- Expose `/reflect` API endpoint to trigger on-demand
- Dashboard for browsing Orion’s dream logs
