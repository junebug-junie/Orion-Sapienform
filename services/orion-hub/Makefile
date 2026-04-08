# ───────────────────────────────────────────────────────────────
# 🎧 Orion Hub Makefile
# Voice + Web Gateway | Node-aware | Mesh-integrated
# ───────────────────────────────────────────────────────────────

ROOT_ENV = /mnt/services/Orion-Sapienform/.env
NET      ?= app-net
MODE     ?= dev
LOG_TAIL ?= 200
PROJECT  ?= $(shell grep -E '^PROJECT=' $(ROOT_ENV) | cut -d= -f2)

SVC ?= hub-app

.PHONY: up down restart build status logs log shell env test-smoke-browser

# ───────────────────────────────────────────────────────────────
# Lifecycle
# ───────────────────────────────────────────────────────────────

## 🟢 Start Orion Hub (dev|prod)
up:
	@echo "→ Starting Orion Hub ($(PROJECT)) on $(NET) network..."
	@MODE=$(MODE) docker compose up -d --build
	@echo "✔ Orion Hub ($(PROJECT)) started in $(MODE) mode"

## 🔴 Stop and clean up
down:
	@docker compose down --remove-orphans

## ♻️ Restart cleanly
restart: down up

## 🧱 Build images only
build:
	@docker compose build

# ───────────────────────────────────────────────────────────────
# Monitoring
# ───────────────────────────────────────────────────────────────

## 📋 Container status
status:
	@docker compose ps

## 🧾 Follow all logs
logs:
	@docker compose logs -f --tail=$(LOG_TAIL)

## 🔍 Follow logs for a specific service: make log SVC=hub-app
log:
	@docker compose logs -f --tail=$(LOG_TAIL) $(SVC)

# ───────────────────────────────────────────────────────────────
# Debugging
# ───────────────────────────────────────────────────────────────

## 🐚 Shell into main container
shell:
	@docker compose exec -it $(SVC) /bin/bash || docker compose exec -it $(SVC) /bin/sh

## 🌎 Show effective environment context
env:
	@echo "PROJECT=$(PROJECT)"
	@echo "MODE=$(MODE)"
	@echo "NET=$(NET)"
	@echo "LOG_TAIL=$(LOG_TAIL)"
	@echo "Loaded from $(ROOT_ENV) + .env"

## 🧪 Run browser smoke test lane (CI-friendly)
test-smoke-browser:
	@python -m pip install playwright
	@python -m playwright install chromium
	@pytest -q tests/test_schedule_panel_browser_smoke.py
