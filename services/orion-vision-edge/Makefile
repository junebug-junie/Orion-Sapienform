# Orion Vision Makefile
SERVICE        ?= orion-vision
CONTAINER      ?= orion-vision
BUS_CONTAINER  ?= orion-bus-orion-redis-1

.PHONY: help up down rebuild logs ps shell nvidia events brain-events monitor health snapshot

help:
	@echo "Targets:"
	@echo "  up            -> build and start Vision service"
	@echo "  down          -> stop Vision service"
	@echo "  rebuild       -> rebuild Vision image and restart"
	@echo "  logs          -> tail logs from Vision service"
	@echo "  ps            -> list running containers"
	@echo "  shell         -> bash into Vision container"
	@echo "  nvidia        -> run nvidia-smi inside Vision container"
	@echo "  events        -> subscribe to vision.events (raw detections)"
	@echo "  brain-events  -> subscribe to brain.events (enriched)"
	@echo "  monitor       -> monitor all Redis activity on Orion bus"
	@echo "  health        -> curl Vision /health endpoint"
	@echo "  snapshot      -> grab a snapshot.jpg from Vision"

up:
	@docker compose up -d --build
	@echo "✔ $(SERVICE) running on http://localhost:8000"

down:
	@docker compose down --remove-orphans

rebuild: down
	@docker compose build --no-cache
	@$(MAKE) up

logs:
	@docker logs -f $(CONTAINER)

ps:
	@docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

shell:
	@docker exec -it $(CONTAINER) bash

nvidia:
	@docker exec -it $(CONTAINER) nvidia-smi || echo "⚠️ No NVIDIA runtime inside $(CONTAINER)"

events:
	@docker exec -it $(BUS_CONTAINER) redis-cli SUBSCRIBE vision.events

brain-events:
	@docker exec -it $(BUS_CONTAINER) redis-cli SUBSCRIBE brain.events

monitor:
	@docker exec -it $(BUS_CONTAINER) redis-cli MONITOR

health:
	@curl -s http://localhost:8000/health | jq

snapshot:
	@curl -s http://localhost:8000/snapshot.jpg --output snapshot.jpg && echo "✔ saved snapshot.jpg"

gpu-check:
	@docker compose exec -T orion-vision python3 -c "import torch; \
print('Torch CUDA available:', torch.cuda.is_available()); \
print('CUDA device count:', torch.cuda.device_count()); \
[print(f'Device {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]; \
from ultralytics import YOLO; \
model = YOLO('yolov8n.pt'); \
model.to('cuda:1'); \
print('YOLO device:', model.device)"
