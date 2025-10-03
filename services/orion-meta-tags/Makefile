# Orion Makefile
BUS_CONTAINER ?= orion-bus-orion-redis-1
MIRROR_SERVICE ?= orion-collapse-mirror-orion-collapse-mirror-1
TAGS_SERVICE ?= orion-meta-tags

.PHONY: up down logs rebuild \
        curl-mirror sub-raw sub-tags monitor \
        restart-mirror restart-tags

# --- Core orchestration ---
up:
	@docker compose up -d
	@echo "✔ Services started"

down:
	@docker compose down --remove-orphans

logs:
	@docker compose logs -f $(MIRROR_SERVICE) $(TAGS_SERVICE)

rebuild:
	@docker compose build $(MIRROR_SERVICE) $(TAGS_SERVICE)
	@docker compose up -d $(MIRROR_SERVICE) $(TAGS_SERVICE)

restart-mirror:
	@docker compose restart $(MIRROR_SERVICE)

restart-tags:
	@docker compose restart $(TAGS_SERVICE)

# --- Collapse Mirror helpers ---
curl-mirror:
	@curl -s -X POST http://localhost:8087/api/log/collapse \
	  -H "Content-Type: application/json" \
	  -d '{"observer":"DEMO","trigger":"makefile test","observer_state":["curious","reflective"],"summary":"Testing collapse via makefile","environment":"dev"}' \
	  | jq

# --- Redis bus monitoring ---
sub-raw:
	@docker exec -it $(BUS_CONTAINER) redis-cli SUBSCRIBE collapse.events.raw

sub-tags:
	@docker exec -it $(BUS_CONTAINER) redis-cli SUBSCRIBE orion.tags

monitor:
	@docker exec -it $(BUS_CONTAINER) redis-cli MONITOR
