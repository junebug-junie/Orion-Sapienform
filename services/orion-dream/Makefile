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
	@curl --fail-with-body -sS -X POST "http://localhost:$${PORT:-8620}/dreams/run" -H "Content-Type: application/json" && echo " Dream trigger published (cortex-orch will run dream_cycle)."
debug-env:
	@echo "--- Checking ENV inside container (via docker exec) ---"
	@docker exec $$(docker ps -qf name=$${PROJECT:-orion-athena}-dream) env | grep ORION_BUS_ENABLED
	@echo "--- Checking ENV with Python (what settings.py sees) ---"
	@docker exec $$(docker ps -qf name=$${PROJECT:-orion-athena}-dream) python -c 'from app.settings import settings; print(f"settings.ORION_BUS_ENABLED = {settings.ORION_BUS_ENABLED}")'

