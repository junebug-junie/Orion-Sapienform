# Orion Cortex Gateway

Simple HTTP gateway for interacting with the Cortex Orchestrator via the Orion Bus.

## Run with Docker

```bash
docker compose -f services/orion-cortex-gateway/docker-compose.yml up --build
```

## API Usage

### Health Check

```bash
curl http://localhost:8090/health
```

### Chat Request

```bash
curl -X POST http://localhost:8090/v1/cortex/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I like cats",
    "mode": "brain"
  }'
```

With overrides:

```bash
curl -X POST http://localhost:8090/v1/cortex/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Plan a trip to Mars",
    "mode": "agent",
    "verb": "plan_trip",
    "packs": ["travel_pack"],
    "recall": {
        "enabled": false
    }
  }'
```

## Configuration

See `.env_example` and `docker-compose.yml`. Key variables:

* `ORION_BUS_URL`: Redis URL for Orion Bus.
* `ORCH_REQUEST_CHANNEL`: Channel where Cortex Orch listens (default: `orion-cortex:request`).
* `ORCH_RESULT_PREFIX`: Prefix for reply channels (default: `orion-cortex:result`).
* `CORTEX_GATEWAY_RPC_TIMEOUT_SEC`: Timeout for RPC calls.
