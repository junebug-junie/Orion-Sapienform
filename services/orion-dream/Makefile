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
	@docker exec -it $$(docker ps -qf name=$${PROJECT:-orion-athena}-dream) python -c 'from app.wake_readout import build_readout, write_readout; ro=build_readout(on_date=None); print(ro); write_readout("manual_dream_test", ro)'

wake_today:
	@docker exec -it $$(docker ps -qf name=$${PROJECT:-orion-athena}-dream) python -c 'import datetime as dt; from app.wake_readout import build_readout, write_readout; ro=build_readout(on_date=dt.date.today(), use_latest_if_missing=False); print(ro); write_readout("manual_dream_test", ro)'
dream:
	@docker exec -it $$(docker ps -qf name=$${PROJECT:-orion-athena}-dream) python -c 'import asyncio; import logging; logging.basicConfig(level=logging.INFO, format="[DREAM_CMD] %(levelname)s - %(name)s - %(message)s"); from app.context import initialize_bus; from app.dream_cycle import run_dream; print("[DREAM_CMD] Initializing bus..."); initialize_bus(); print("[DREAM_CMD] Running dream cycle..."); asyncio.run(run_dream())'
debug-env:
	@echo "--- Checking ENV inside container (via docker exec) ---"
	@docker exec $$(docker ps -qf name=$${PROJECT:-orion-athena}-dream) env | grep ORION_BUS_ENABLED
	@echo "--- Checking ENV with Python (what settings.py sees) ---"
	@docker exec $$(docker ps -qf name=$${PROJECT:-orion-athena}-dream) python -c 'from app.settings import settings; print(f"settings.ORION_BUS_ENABLED = {settings.ORION_BUS_ENABLED}")'

