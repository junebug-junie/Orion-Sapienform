.PHONY: test test-hub test-actions bootstrap-test-envs

SERVICE ?=
ARGS ?=

bootstrap-test-envs:
	@./scripts/bootstrap_test_envs.sh $(if $(SERVICE),--service $(SERVICE),)

test:
	@if [ -z "$(SERVICE)" ]; then \
		echo "usage: make test SERVICE=<service-name> [ARGS='...']"; \
		exit 1; \
	fi
	@./scripts/test_service.sh "$(SERVICE)" $(ARGS)

test-hub:
	@./scripts/test_hub.sh $(ARGS)

test-actions:
	@./scripts/test_orion_actions.sh $(ARGS)
