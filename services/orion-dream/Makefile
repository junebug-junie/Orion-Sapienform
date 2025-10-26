# ───────────────────────────────────────────────
# Orion Dream Makefile
# ───────────────────────────────────────────────

up:
	@docker compose up -d --build

down:
	@docker compose down

logs:
	@docker compose logs -f dream

rebuild:
	@docker compose build --no-cache dream && docker compose up -d dream

# ───────────────────────────────────────────────
# Dream Cycle Commands
# ───────────────────────────────────────────────

wake:
	@docker exec -it $$(docker ps -qf name=$${PROJECT:-orion-janus}-dream) python -c 'from app.wake_readout import build_readout, write_readout; ro=build_readout(on_date=None); print(ro); write_readout("manual_dream_test", ro)'

wake_today:
	@docker exec -it $$(docker ps -qf name=$${PROJECT:-orion-janus}-dream) python -c 'import datetime as dt; from app.wake_readout import build_readout, write_readout; ro=build_readout(on_date=dt.date.today(), use_latest_if_missing=False); print(ro); write_readout("manual_dream_test", ro)'
dream:
	@docker exec -it $$(docker ps -qf name=$${PROJECT:-orion-janus}-dream) python -c 'import asyncio; import logging; logging.basicConfig(level=logging.INFO, format="[DREAM_CMD] %(levelname)s - %(name)s - %(message)s"); from app.context import initialize_bus; from app.dream_cycle import run_dream; print("[DREAM_CMD] Initializing bus..."); initialize_bus(); print("[DREAM_CMD] Running dream cycle..."); asyncio.run(run_dream())'
debug-env:
	@echo "--- Checking ENV inside container (via docker exec) ---"
	@docker exec $$(docker ps -qf name=$${PROJECT:-orion-janus}-dream) env | grep ORION_BUS_ENABLED
	@echo "--- Checking ENV with Python (what settings.py sees) ---"
	@docker exec $$(docker ps -qf name=$${PROJECT:-orion-janus}-dream) python -c 'from app.settings import settings; print(f"settings.ORION_BUS_ENABLED = {settings.ORION_BUS_ENABLED}")'

debug-bus-verbose:
	@echo "--- Advanced Bus Debug (with logging) ---"
	@docker exec $$(docker ps -qf name=$${PROJECT:-orion-janus}-dream) python -c '\
import os; \
import logging; \
\
print("--- 0. Configuring logging ---"); \
logging.basicConfig(level=logging.INFO, format="[PY_SCRIPT] %(levelname)s - %(name)s - %(message)s"); \
\
from app.settings import settings; \
from app.context import bus, initialize_bus; \
\
print("\n--- 1. Checking settings ---"); \
print(f"settings.ORION_BUS_ENABLED = {settings.ORION_BUS_ENABLED}"); \
print(f"settings.ORION_BUS_URL = {settings.ORION_BUS_URL}"); \
\
print("\n--- 2. Checking bus *before* init ---"); \
print(f"bus is {bus}"); \
\
print("\n--- 3. Calling initialize_bus() ---"); \
try: \
    initialize_bus(); \
    print("initialize_bus() finished."); \
except Exception as e: \
    print(f"ERROR during initialize_bus(): {e}"); \
\
print("\n--- 4. Checking bus *after* init ---"); \
print(f"bus is {bus}"); \
\
if bus: \
    print("\n--- 5. Inspecting bus object ---"); \
    print(f"hasattr(bus, 'enabled'): {hasattr(bus, 'enabled')}"); \
    if hasattr(bus, 'enabled'): \
        print(f"bus.enabled = {bus.enabled}"); \
    if hasattr(bus, 'url'): \
        print(f"bus.url = {bus.url}"); \
else: \
    print("\n--- 5. Bus is still None ---"); \
'
