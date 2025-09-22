NET ?= app-net

.PHONY: up down status tail tail.events tail.out ps logs

up:
	@docker compose up -d
	@echo "✔ Redis: tcp://localhost:6379"
	@echo "✔ Redis Commander: http://localhost:8085"
	@echo "✔ Redis Exporter (Prometheus): http://localhost:9121/metrics"

down:
	@docker compose down --remove-orphans

status: ps
ps:
	@docker compose ps
	@docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}' | awk 'NR==1||/orion-redis|redis-commander|redis_exporter/'

logs:
	@docker compose logs -f --tail=100

# Raw MONITOR stream (be careful in prod)
tail:
	@docker compose exec -T orion-redis redis-cli MONITOR

# Tail Orion event streams (works with default stream names)
tail.events:
	@docker compose exec -T orion-redis redis-cli XREVRANGE orion\\:evt\\:gateway + - COUNT 20

tail.out:
	@docker compose exec -T orion-redis redis-cli XREVRANGE orion\\:bus\\:out + - COUNT 20
