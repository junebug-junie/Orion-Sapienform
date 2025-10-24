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
	@docker exec -it $$(docker ps -qf name=$${PROJECT:-orion-janus}-dream) python -c 'import asyncio; from app.dream_cycle import run_dream; asyncio.run(run_dream())'
