# orion-bus-tap (UI)

A live “bus activity dashboard” for the Orion Titanium bus. This is the *primary* way to see what’s happening on the bus without memorizing channels.

---

## Quick start (copy/paste)

### 1) Run the service
```bash
docker compose up -d orion-bus-tap
docker logs -f orion-bus-tap
```

By default it starts:
- HTTP: `0.0.0.0:8101`
- WebSocket: `/ws/tap`

### 2) Expose it via Tailscale (your current setup)
You’re serving the UI under a path prefix:

```bash
sudo tailscale serve --bg --set-path=/tap 8101
```

Open:
- `https://athena.tail348bbe.ts.net/tap`

> Internally the UI still connects to `ws(s)://<host>/ws/tap` and should work through the same base host.

---

## What it does
- Observes the bus (Redis pub/sub)
- Streams a live feed to the browser over WebSocket
- (Optionally) computes lightweight stats (channels, rates, last-seen)

This is observability only — it should not mutate messages or write to databases.

---

## When to use it
Use bus-tap whenever you find yourself reaching for:

- `redis-cli ... PSUBSCRIBE "orion:*"` (too noisy)
- wondering “is anything publishing?”
- debugging why a downstream service is quiet

---

## CLI alternatives (only when you need them)

### Set your bus URL once
```bash
BUS=redis://100.92.216.81:6379/0
```

### If bus-tap publishes stats
```bash
redis-cli -u "$BUS" SUBSCRIBE "orion:bus:tap:stats"
```

### Don’t wildcard the whole bus
Use targeted lenses instead:

```bash
redis-cli -u "$BUS" PSUBSCRIBE \
  "orion:equilibrium:*" \
  "orion:event:equilibrium:*" \
  "orion:state:*" \
  "orion:spark:*" \
  "orion:pad:*"
```

---

## Recommended UX conventions (future stubs)
These are quality-of-life improvements we should standardize across Orion services:

- **Startup log banner**: prints bus URL, subscribed patterns, output channels, and UI URL.
- **Optional bus summary logging**:
  - `ORION_LOG_BUS_IN=true`
  - `ORION_LOG_BUS_OUT=true`
  - `ORION_LOG_BUS_FULL=false` (default)
- **Health endpoints**:
  - `GET /healthz` (liveness)
  - `GET /readyz` (deps)
  - `GET /stats` (counts/rates)

---

## Common failure modes
- UI loads but no stream: bus URL wrong or patterns too strict
- WebSocket fails behind a proxy: path/host mismatch (check browser console)
- Channel enforcement errors: bus-tap trying to publish to a channel not in the catalog
