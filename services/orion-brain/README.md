# 🧠 Orion Brain Service

A lightweight, event-aware gateway that fronts one or more Ollama backends and integrates with the Orion Bus for message emission and telemetry.

> **Core Purpose:** To decouple applications from the underlying GPU node topology while providing a centralized point for LLM orchestration, routing, and monitoring within the Orion Mesh. The Brain service itself does not run models; it intelligently routes requests to dedicated Ollama backends.

---

## ⚙️ Key Features

- **Dynamic Backend Routing:** Automatically routes requests to healthy Ollama backends based on the configured balancing policy (`least_conn` or `round_robin`).
- **Automatic Health Checks:** Continuously probes each backend's `/api/tags` endpoint to ensure it only sends requests to responsive nodes.
- **Orion Bus Integration:** Publishes detailed model-response and telemetry events to Redis, allowing other services in the mesh to observe and react to cognitive tasks.

---

## 🚀 Quick Start

### 1. Configure the Service

Create a `.env` file in this directory (`services/orion-brain/`) or use the provided `.env.example`. The most important variables to check are:

- **PROJECT:** The project name, used for naming containers and networks.
- **BACKENDS:** A comma-separated list of the Ollama backend URLs (e.g., `http://orion-janus-brain-llm:11434`).
- **ORION_BUS_URL:** The connection URL for your Redis instance.

### 2. Launch the Service

From the project root (`Orion-Sapienform/`), run the following command:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-brain/.env \
  -f services/orion-brain/docker-compose.yml \
  up -d --build
```

---

## 🖥️ Usage & API

### Health Check

Check the status of the Brain service and its connection to backend LLMs.

**Endpoint:** `GET /health`

```bash
curl -s http://localhost:8088/health | jq
```

### Chat Inference

Send a standard Ollama-compatible chat request. The Brain service will select a healthy backend, forward the request, and return the response.

**Endpoint:** `POST /api/chat`

#### Example Request

```bash
curl -s http://localhost:8088/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mistral:instruct",
    "messages": [
      {
        "role": "user",
        "content": "Explain the significance of the Library of Babel."
      }
    ],
    "stream": false
  }' | jq
```

#### Example Response

```json
{
  "trace_id": "a1b2c3d4-...",
  "backend": "http://orion-janus-brain-llm:11434",
  "response": "The Library of Babel is a short story by Jorge Luis Borges, exploring concepts of infinity, reality, and the universe as a vast, exhaustive library containing all possible books...",
  "meta": {
    "model": "mistral:instruct",
    "latency_ms": 1234.56,
    "done_reason": "stop"
  }
}
```

---

## ⚙️ Environment Configuration

The service is configured via environment variables defined in `.env`.

| Variable | Description | Default |
|-----------|-------------|----------|
| `SERVICE_NAME` | The logical name for this service. | `brain` |
| `PORT` | The port the FastAPI server will listen on. | `8088` |
| `BACKENDS` | Comma-separated URLs of the backend Ollama servers. | `http://${PROJECT}-brain-llm:11434` |
| `SELECTION_POLICY` | Backend routing policy (`least_conn` or `round_robin`). | `least_conn` |
| `HEALTH_INTERVAL_SEC` | How often (in seconds) to probe backends. | `5` |
| `ORION_BUS_URL` | The connection URL for the Orion Bus (Redis). | `redis://${PROJECT}-bus-core:6379/0` |
| `CHANNEL_BRAIN_INTAKE` | The Redis channel this service listens on for requests. | `orion:brain:intake` |
| `CHANNEL_BRAIN_OUT` | The Redis channel for publishing final LLM results. | `orion:brain:out` |

---

## 🛠️ Developer Guide (Makefile)

The included `Makefile` provides convenient shortcuts for common development tasks.

| Command | Description |
|----------|-------------|
| `make start-prod` | Builds and starts the Brain service and its dependencies. |
| `make stop-prod` | Stops all containers associated with the Brain service. |
| `make restart` | Rebuilds the Docker image and restarts the service. |
| `make status` | Runs a health check and lists available models. |
| `make env.print` | Prints the effective environment variables being used. |

For more advanced targets, please see the `Makefile` itself.
