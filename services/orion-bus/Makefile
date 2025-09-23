NET ?= app-net

.PHONY: up down status ps logs tail cli tail.stream

up:
	@docker network inspect $(NET) >/dev/null 2>&1 || docker network create $(NET)
	@docker compose up -d
	@echo "✔ Redis: tcp://localhost:6379"
	@echo "✔ Redis Commander: http://localhost:8085"
	@echo "✔ Redis Exporter (Prometheus): http://localhost:9121/metrics"

down:
	@docker compose down --remove-orphans

status: ps
ps:
	@docker compose ps
	@docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}' \
	  | awk 'NR==1||/orion-redis|redis-commander|redis_exporter/'

logs:
	@docker compose logs -f --tail=100

# Raw MONITOR stream (dev only — can be noisy)
tail:
	@docker compose exec -T orion-redis redis-cli MONITOR

# Tail any stream (default = events)
STREAM ?= orion:evt:gateway
tail.stream:
	@docker compose exec -T orion-redis redis-cli XREVRANGE $(STREAM) + - COUNT 20

# Shortcut to drop into redis-cli
cli:
	@docker compose exec -it orion-redis redis-cli
