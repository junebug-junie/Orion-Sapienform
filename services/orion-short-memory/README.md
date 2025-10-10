# Orion LangChain Memory Service (CPU-only)

This bundle provides a CPU-only LangChain middleware for Orion that:
- 🧠 Maintains **Hot (buffer)**, **Warm (Chroma + embeddings)**, and **Cold (Postgres + RDF)** memory
- 🤝 Delegates all LLM inference to **orion-brain-service** (your GPU router)
- 🌙 Runs a nightly **Reflection Job** that summarizes the day's turns into Warm memory and writes **RDF dream logs**

## Services
- **orion-langchain**: FastAPI service with `/query`
- **orion-reflection**: cron-like loop that runs `reflect.py` every 24h

> Note: GraphDB and Postgres are **external**; this compose expects them on your bus/network.

## Environment Variables
- `ORION_BRAIN_URL` (default `http://orion-brain-service:8088`)
- `LLM_MODEL` (default `mistral:instruct`)
- `CHROMA_DIR` (default `/data/chroma`)
- `EMBED_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`)
- `PG_HOST`, `PG_DB`, `PG_USER`, `PG_PASS`
- `GRAPHDB_URL` (e.g., `http://graphdb:7200/repositories/orion`)

## Postgres Schema
```sql
CREATE TABLE IF NOT EXISTS conversation_memory (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ DEFAULT NOW(),
    user_msg TEXT NOT NULL,
    bot_msg  TEXT NOT NULL
);
```

## Run
```bash
docker compose up -d --build
```

## Test
```bash
curl -X POST http://localhost:8090/query   -H "Content-Type: application/json"   -d '{"message": "What did we decide about warm memory?"}'
```

## Notes
- **No CUDA** needed in this image. All GPU work stays in `orion-brain-service`.
- Embeddings use **HuggingFace** `all-MiniLM-L6-v2` on CPU for speed and portability.
- Reflection cadence can be changed by editing the sleep value in `docker-compose.yml`.
