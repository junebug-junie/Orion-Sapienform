# Orion Bus (Redis)

Shared Redis stack for the Orion mesh. Provides:
- **Redis** on `6379` (AOF enabled)
- **Redis Commander** UI on `http://localhost:8085`
- **Redis Exporter** for Prometheus on `http://localhost:9121/metrics`
- Attached to Docker network `app-net` with DNS alias `orion-redis`

## Prereqs

- Docker / Docker Compose v2
- A shared Docker network (created once):
  ```bash
  docker network inspect app-net >/dev/null 2>&1 || docker network create app-net
  ```

## Bring Up / Down

```bash
# start
make up

# status + ports
make status

# follow logs
make logs
make logs-all

# stop
make down
```

## Ports

- Redis: `localhost:6379`
- Redis Commander: `http://localhost:8085`
- Redis Exporter: `http://localhost:9121/metrics`

## Streams convention (used by Orion services)

- Internal events: `orion:evt:gateway`
- Outbound bus: `orion:bus:out`

Tail them (latest 20):

```bash
make tail.events
make tail.out
```

or raw monitor (noisy; dev only):

```bash
make tail
```

## Using from other services

Point services to this Redis via Docker DNS:

```
REDIS_URL=redis://orion-redis:6379/0
```

Ensure the service is attached to the same network:

```yaml
services:
  my-service:
    # ...
    networks:
      - app-net

networks:
  app-net:
    external: true
    name: app-net
```

### Example: Orion Brain Service

In `orion-brain-service/compose.yaml`:

```yaml
environment:
  REDIS_URL: ${REDIS_URL:-redis://orion-redis:6379/0}
  EVENTS_ENABLE: ${EVENTS_ENABLE:-"true"}
  EVENTS_STREAM: ${EVENTS_STREAM:-orion:evt:gateway}
  BUS_OUT_ENABLE: ${BUS_OUT_ENABLE:-"true"}
  BUS_OUT_STREAM: ${BUS_OUT_STREAM:-orion:bus:out}
networks:
  app-net:
    aliases: ["brain-service"]
```

## Managed Redis (Prod)

If you use a hosted/managed Redis, skip this stack and set:

```
REDIS_URL=rediss://<user>:<password>@<host>:<port>/<db>
```

Your Orion services will publish/consume the same streams without changes.

## Troubleshooting

- **Commander 404 / empty UI**: confirm `make up` succeeded and Redis is healthy:
  ```bash
  docker compose ps
  docker compose exec -T orion-redis redis-cli PING
  ```
- **Other services can’t connect**: verify they’re on `app-net` and use `redis://orion-redis:6379/0`.
- **High CPU / MONITOR spam**: stop `make tail` (CTRL-C). Prefer `XREVRANGE` in `tail.events` / `tail.out`.
