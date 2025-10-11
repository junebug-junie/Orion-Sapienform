Orion Brain Service (Cognitive Router for Ollama)

A lightweight gateway + event‑aware router that fronts one or more Ollama backends and integrates with the Orion Bus for message emission.

Purpose: decouple applications from GPU node topology while streaming telemetry into the Orion Mesh.

Routes: /health, /chat

Balancing: least_conn (default) or round_robin

Health Checks: probes each backend’s /api/tags

Bus Integration: publishes model‑response and telemetry events to Redis Streams (orion:evt:gateway, orion:bus:out)

The Brain performs orchestration and routing; GPUs live in the Ollama backends.

## Quick Start (Compose)

cp .env.example .env
# Adjust BACKENDS and PROJECT if needed
docker compose up ‑d ‑‑build

### Smoke Tests

# Health
curl -s http://localhost:8088/health | jq


# Chat (test request)
curl -s http://localhost:8088/chat \
  -H 'content-type: application/json' \
  -d '{
   "model":"mistral:instruct",
   "messages":[{"role":"user","content":"Say hi then stop."}],
   "user_id":"u1","session_id":"s1"
  }' | jq

## API Overview

 Route 	 Method 	 Description 
 /health 	 GET 	 Mesh + backend status 
 /chat 	 POST 	 Send conversation‑style LLM request 

### /chat Body Schema

{
  "model": "mistral:instruct",
  "messages": [
    {"role": "user", "content": "Say hi then stop."}
  ],
  "user_id": "u1",
  "session_id": "s1",
  "options": {"temperature": 0.2, "num_predict": 64}
}

### Response

{
  "response": "Hello there!"
}

## Environment (.env)

PROJECT=orion-janus
BACKENDS=http://llm-brain:11434
SELECTION_POLICY=least_conn
HEALTH_INTERVAL_SEC=5
CONNECT_TIMEOUT_SEC=10
READ_TIMEOUT_SEC=600
PORT=8088
REDIS_URL=redis://orion-janus-bus-core:6379/0
EVENTS_ENABLE=true
BUS_OUT_ENABLE=true

## Notes - Use make start-prod to start Brain with the bus and Ollama auto‑checks.
- When you later re‑enable /models or /generate, this README can revert to the original gateway shape.
- Health and chat are now fully event‑integrated with Orion Bus.
