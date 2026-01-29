# orion-state-service

Canonical read-model service for Orion's latest internal state (Spark/Tissue).

## Responsibilities

- Consume `spark.state.snapshot.v1` from the bus (real-time).
- Consume biometrics summaries/inductions (and optional cluster aggregates).
- Maintain an in-memory + Redis cache of the latest state:
  - per-node
  - global (most-recent wins, or authoritative node if configured)
- On startup, hydrate the cache from Postgres (`spark_telemetry.metadata.spark_state_snapshot`)
  so restarts are non-events.
- Serve a single retrieval interface to consumers:
  - Bus RPC: `state.get_latest.v1` -> `state.latest.reply.v1`
  - Optional HTTP endpoint for debugging: `GET /state/latest`

Biometrics are surfaced under `payload.biometrics` with freshness metadata so callers can avoid missing-key failures.

## Biometrics Inputs

| Channel | Kind | Usage |
| :--- | :--- | :--- |
| `orion:biometrics:summary` | `biometrics.summary.v1` | Cached per-node + global summary. |
| `orion:biometrics:induction` | `biometrics.induction.v1` | Cached per-node + global induction. |
| `orion:biometrics:cluster` | `biometrics.cluster.v1` | Cached latest cluster aggregate (optional). |

Environment:
- `BIOMETRICS_STALE_AFTER_SEC` (default `90`) controls biometrics freshness.

## Notes

This service is intentionally a *read model* (latest snapshot oracle). It is not part of recall/RAG.

## ✅ Spark snapshot ACKs

When a snapshot envelope includes `reply_to`, the service emits an ACK reply:

- Canonical kind: `spark.state.snapshot.ack.v1`
- Legacy kind (optional): `spark.state.snapshot.ack`

The legacy kind is controlled by `STATE_SERVICE_EMIT_LEGACY_SNAPSHOT_ACK` (default `true`).

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
import os
import redis

# --- Configuration ---
BUS_URL = os.getenv("ORION_BUS_URL", "redis://100.92.216.81:6379/0")
REQUEST_CHANNEL = os.getenv("STATE_REQUEST_CHANNEL", "orion:state:request")
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
- `biometrics` (summary/induction + freshness metadata)

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
