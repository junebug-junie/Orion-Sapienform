# orion-state-service

Canonical read-model service for Orion's latest internal state (Spark/Tissue).

## Responsibilities

- Consume `spark.state.snapshot.v1` from the bus (real-time).
- Maintain an in-memory + Redis cache of the latest state:
  - per-node
  - global (most-recent wins, or authoritative node if configured)
- On startup, hydrate the cache from Postgres (`spark_telemetry.metadata.spark_state_snapshot`)
  so restarts are non-events.
- Serve a single retrieval interface to consumers:
  - Bus RPC: `state.get_latest.v1` -> `state.latest.reply.v1`
  - Optional HTTP endpoint for debugging: `GET /state/latest`

## Notes

This service is intentionally a *read model* (latest snapshot oracle). It is not part of recall/RAG.

## 🔍 Testing: Bus RPC

This test verifies that **`orion-state-service`** can respond to a bus RPC request
(`state.get_latest.v1 → state.latest.reply.v1`) using Redis pub/sub.

### Prerequisites
- `orion-state-service` running
- Redis / Orion bus reachable
- Python 3.10+
- `redis` Python package installed:

---

### One-liner RPC Test (Python)

This script:
- Publishes a `state.get_latest.v1` request
- Subscribes to a temporary reply channel
- Prints the first reply it receives
- Times out cleanly if no response arrives

```bash
python - <<'PY'
import json
import time
import uuid
import redis

# --- Configuration ---
BUS_URL = "redis://100.92.216.81:6379/0"
REQUEST_CHANNEL = "orion-state:request"
REPLY_CHANNEL = f"orion:tmp:state-reply:{uuid.uuid4()}"

# --- Connect to Redis ---
redis_client = redis.Redis.from_url(BUS_URL, decode_responses=True)
pubsub = redis_client.pubsub()
pubsub.subscribe(REPLY_CHANNEL)

# --- Build RPC request envelope ---
request = {
    "schema": "orion.envelope",
    "schema_version": "2.0.0",
    "kind": "state.get_latest.v1",
    "source": {
        "name": "readme-test",
        "node": "athena"
    },
    "reply_to": REPLY_CHANNEL,
    "payload": {
        "scope": "global"
    }
}

# --- Send request ---
redis_client.publish(REQUEST_CHANNEL, json.dumps(request))

# --- Await reply ---
deadline = time.time() + 3
while time.time() < deadline:
    message = pubsub.get_message(timeout=1)
    if message and message.get("type") == "message":
        print("Received reply:\n")
        print(message["data"])
        break
else:
    raise SystemExit(
        "❌ No reply received (timeout). "
        "Check STATE_REQUEST_CHANNEL and state-service logs."
    )
PY
```

---

### Expected Result

You should see a JSON envelope printed containing:

- `kind: state.latest.reply.v1`
- `status: fresh | stale | missing`
- `snapshot` (if available)
- `age_ms` and `as_of_ts`

Example (truncated):

```json
{
  "kind": "state.latest.reply.v1",
  "payload": {
    "ok": true,
    "status": "fresh",
    "age_ms": 9123,
    "snapshot": {
      "source_node": "athena",
      "phi": { "coherence": 0.73, "novelty": 0.79 }
    }
  }
}
```

## 🛰️ Spark signal sanity check
Use this tiny snippet to publish a `spark.signal.v1` frame (e.g., distress from equilibrium) and watch subscribers on `orion:spark:signal`:

```bash
python - <<'PY'
import json, uuid, datetime, redis
BUS_URL = "redis://100.92.216.81:6379/0"
CHANNEL = "orion:spark:signal"
msg = {
    "schema": "orion.envelope",
    "schema_version": "2.0.0",
    "kind": "spark.signal.v1",
    "source": {"name": "readme-signal", "node": "athena"},
    "payload": {
        "signal_type": "equilibrium",
        "intensity": 0.42,
        "valence_delta": -0.05,
        "coherence_delta": -0.02,
        "as_of_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ttl_ms": 15000,
        "source_service": "readme-signal",
        "source_node": "athena"
    }
}
r = redis.Redis.from_url(BUS_URL)
r.publish(CHANNEL, json.dumps(msg))
print(f"published spark.signal.v1 to {CHANNEL}")
PY
```
