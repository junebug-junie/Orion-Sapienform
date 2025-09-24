# RAG Voice (Text-first) — Local LLM Starter

Tiny, non‑Docker starter that adds Retrieval‑Augmented Generation (RAG) to a local LLM.
Designed to be simple: ingest a doc, ask questions over it via a FastAPI server.

## What you get

* **Local LLM** via `llama-cpp-python` (point to any `.gguf` model)
* **Embeddings** with `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Plain NumPy retrieval** (no FAISS) for small docs
* **FastAPI** endpoints: `/ingest` (path or raw text) and `/ask`

> This is text-first; you can bolt it onto your voice service by hitting `/ask` from your pipeline.
> GPU is optional—`llama-cpp-python` runs on CPU by default.

---

## Quickstart

### 1) Create venv and install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Put a local `.gguf` model somewhere and set the path

```bash
export MODEL_PATH=/absolute/path/to/your-model.gguf
```

(Any chat-tuned or instruct-tuned small model works for a toy demo.)

### 3) Ingest a sample doc (included)

```bash
# Option A: via endpoint with a file path
uvicorn app.server:app --reload
# in another terminal:
curl -s localhost:8000/ingest -H 'content-type: application/json'   -d '{"path":"data/sources/sample.txt"}' | jq

# Option B: send raw text
curl -s localhost:8000/ingest -H 'content-type: application/json'   -d '{"text":"RAG lets an LLM use retrieved context from documents."}' | jq
```

### 4) Ask a question

```bash
curl -s localhost:8000/ask -H 'content-type: application/json'   -d '{"question":"What is RAG in one sentence?"}' | jq -r '.answer'
```

---

## Endpoints

### `POST /ingest`

Body:

```json
{ "path": "path/to/file.txt" }
```

or

```json
{ "text": "raw text to index" }
```

* Splits into overlapping chunks, embeds, and saves a small index at `data/index.pkl`

### `POST /ask`

Body:

```json
{ "question": "Your question", "k": 4 }
```

* Retrieves top-k chunks and asks the local LLM with a simple RAG prompt

---

## Notes

* For PDFs, install `pypdf` and point `/ingest` to the file path (basic parsing).
* For larger corpora, swap the simple NumPy search for FAISS/Annoy and persist multiple files.
* To integrate with your voice pipeline, POST the transcript to `/ask` and speak back the `.answer`.

---

## Mesh Service Deployment

When running as part of the **Orion mesh** (with Docker Compose):

* **RAG service** is exposed on port `8001` (by default)
* **Data mounting**: `./data` on the host is mounted into the container at `/app/data`
* **Paths for ingest**: when calling `/ingest`, use container-visible paths such as:

  ```json
  { "path": "/app/data/sources/library_of_babel.txt" }
  ```
* **Other services** (voice-app, collapse-mirror, brain) connect over the internal Docker network `app-net`
* **Environment** variables like ports and model paths are managed via `.env` files per service

### Running inside the mesh

From the `services/orion-rag/` folder:

```bash
docker compose up -d --build
```

Service will be live at:

```bash
http://localhost:8001
```

### Mesh smoke tests

Check health:

```bash
curl -s localhost:8001/health | jq
```

Ingest a document:

```bash
curl -s localhost:8001/ingest \
  -H 'content-type: application/json' \
  -d '{"path":"/app/data/sources/sample.txt"}' | jq
```

Ask a question:

```bash
curl -s localhost:8001/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Summarize the sample."}' | jq -r '.answer'
```

---

## Dev tips

* If `llama-cpp-python` wheel fails, try: `pip install --upgrade pip setuptools wheel` first.
* Tune `n_ctx`, `n_threads`, and `n_batch` in `app/llm.py` for your machine.
* If your model is not chat-tuned, adjust the prompt template in `app/rag.py` accordingly.
