.PHONY: test packs rdf plan lint

VENVDIR?=.venv
PYTHON?=python3

install:
	$(PYTHON) -m pip install -e .

test:
	pytest -q

packs:
	$(PYTHON) cli.py list-packs

rdf:
	$(PYTHON) tools/generate_rdf.py > /tmp/orion_verbs.ttl
	@echo "Wrote TTL to /tmp/orion_verbs.ttl"

plan:
	$(PYTHON) cli.py plan introspect

lint:
	flake8 planner/ packs_loader.py tools/ tests/
