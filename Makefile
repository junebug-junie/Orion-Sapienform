.PHONY: test test-hub test-actions bootstrap-test-envs check-inner-state-registry check-single-consumer-channels check-activation-saturation

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

# NOTE: CLAUDE.md §17 describes a `make agent-check` target chaining
# check_env_template_parity.py, check_schema_registry.py, check_bus_channels.py,
# and this check -- confirmed 2026-07-12 that `agent-check` itself and the
# first two of those three scripts do not exist in this repo. Not built here
# (out of scope for this patch); this target is the one real piece of that
# promised chain, added standalone until Juniper decides whether to build the
# rest.
check-inner-state-registry:
	@python scripts/check_inner_state_registry.py

# Live-bus gate: every channel marked single_consumer: true in
# orion/bus/channels.yaml must have exactly one live subscriber
# (Redis pub/sub duplicates execution otherwise -- see PR #994).
# Requires ORION_BUS_URL=redis://<tailscale-ip>:6379/0.
check-single-consumer-channels:
	@python scripts/check_single_consumer_channels.py

# Standing gate from docs/superpowers/specs/2026-07-13-memory-recall-reinforcement-decay-
# wiring-spec.md acceptance check 1: recall_boost()+decay() must not grow the fraction of
# active crystallizations pinned at the activation ceiling over time. No persisted baseline
# by design (see the script's own docstring) -- re-run by hand and compare against a prior
# run's fraction; pass FAIL_ABOVE=<prior-fraction> to fail automatically on regression.
# Requires POSTGRES_URI (see services/orion-hub/.env).
check-activation-saturation:
	@python scripts/check_activation_saturation.py $(if $(FAIL_ABOVE),--fail-above $(FAIL_ABOVE),)
