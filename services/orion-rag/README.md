# 📚 Orion RAG Service

The **Orion RAG (Retrieval-Augmented Generation)** service acts as an intelligent orchestrator that enriches user queries with relevant context from a vector database before delegating the final prompt generation to the LLM Host.

> **Core Purpose:** To provide accurate, context-aware answers by retrieving information from internal documents and combining it with the power of a large language model.

---

## 🧩 How It Works

The service follows a precise, event-driven workflow:

1. **Listens** for incoming requests on a dedicated Redis channel (`orion:rag:request`).
2. **Searches** the `orion-vector-db` service for documents semantically similar to the user's query.
3. **Constructs** a new, context-rich prompt.
4. **Calls** the LLM host (via `orion-llm-gateway` or direct HTTP to `orion-ollama-host`/`llamacpp-host` depending on configuration) to generate a response.
   *(Note: The legacy `orion:brain:intake` flow is deprecated as `orion-brain` has been refactored to `orion-ollama-host`, a dumb host).*
5. **Publishes** the final, context-augmented answer back to the original requester.

---

## 🚀 Quick Start

### 1. Configure the Service

Ensure your `.env` file in this directory (`services/orion-rag/`) is correctly configured, especially the following:

- **PROJECT:** The project name, used for connecting to other services.
- **ORION_BUS_URL:** The connection URL for your Redis instance.
- **SUBSCRIBE_CHANNEL_RAG_REQUEST:** The channel this service listens on.

### 2. Launch the Service

From the project root (`Orion-Sapienform/`), run the standard compose command:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-rag/.env \
  -f services/orion-rag/docker-compose.yml \
  up -d --build
```

---

## ✍️ Ingesting New Documents

This service reads from the vector database. Adding new documents is handled by the `orion-vector-writer` service.

### Step 1: Place Your Source File

Put your text file (e.g., `my-document.txt`) into the shared data directory on your host machine. This path is defined by the `HOST_RAG_FILES_DIR` variable in your root `.env` file.

**Example Host Path:**
```bash
/mnt/storage/rag-files/test-txt/my-document.txt
```

### Step 2: Run the Ingestion Command

From your project root, execute the `ingest.py` script inside the vector-writer container. This command tells the writer to find your file, chunk it, create embeddings, and save them to the database.

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-vector-writer/.env \
  -f services/orion-vector-writer/docker-compose.yml \
  exec vector-writer \
  python -m orion.tier_semantic.ingest "/data/my-document.txt"
```

> Replace `/data/my-document.txt` with the container-side path to your file.

---

## 🧪 Testing the RAG Flow

To test the full end-to-end process, you need two terminals to publish a request and listen for the reply.

### Terminal 1: The Listener

This terminal will subscribe to the reply channel and wait for the final answer. Run this command and leave it open:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-bus/.env \
  -f services/orion-bus/docker-compose.yml \
  exec bus-core redis-cli SUBSCRIBE orion:rag:test-reply
```

### Terminal 2: The Publisher

In a new terminal, run this command to publish a test query. This message tells the RAG service what to search for and where (`orion:rag:test-reply`) to send the final answer.

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-bus/.env \
  -f services/orion-bus/docker-compose.yml \
  exec bus-core redis-cli PUBLISH orion:rag:request '{"query": "Is the narrator insane or unreliable?", "response_channel": "orion:rag:test-reply"}'
```

After running the second command, you should see the complete JSON response — including the LLM's answer and the retrieved context — appear in **Terminal 1**.

---

## ⚙️ Key Dependencies

This service relies on the following other Orion services to function:

- **orion-bus-core:** For all message passing.
- **orion-vector-db:** For document retrieval.
- **orion-llm-gateway** or **orion-ollama-host**: For final prompt generation.
