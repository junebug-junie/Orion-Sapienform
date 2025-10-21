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

dream:
	@docker exec -it $$(docker ps -qf name=$${PROJECT:-orion-janus}-dream) \
	python -c "import asyncio; from app.dream_cycle import run_dream; asyncio.run(run_dream())"

wake:
	@docker exec -it $$(docker ps -qf name=$${PROJECT:-orion-janus}-dream) \
	python -c "import datetime as dt; from app.wake_readout import build_readout, write_readout; ro=build_readout(dt.date.today()); print(ro); write_readout('manual_dream_test', ro)"
