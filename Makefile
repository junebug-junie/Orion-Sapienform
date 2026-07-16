.PHONY: test test-hub test-actions bootstrap-test-envs check-inner-state-registry check-single-consumer-channels check-activation-saturation concept-relation-digest check-concept-relation-digest-liveness check-env-compose-parity check-journal-dispatch-registry check-daily-schedule-collisions bus-core-health-watchdog worktree-status worktree-status-summary worktree-status-stale prune-merged-worktrees

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

# Runs the concept-relation decision digest (see services/orion-memory-consolidation/
# README.md, "Cross-window concept-relation resolution"). This is the actual cron
# entry point -- see that README's "Scheduled maintenance" section for the crontab
# line. Requires POSTGRES_URI.
concept-relation-digest:
	@python scripts/concept_relation_digest.py

# Fail-safe for the above: fails if the oldest undigested
# memory_concept_relation_decisions row is older than --max-age-hours (default 3h),
# which only happens if the digest cron entry died, was dropped after a host
# migration, or the job is crashing. Requires POSTGRES_URI.
check-concept-relation-digest-liveness:
	@python scripts/check_concept_relation_digest_liveness.py $(if $(MAX_AGE_HOURS),--max-age-hours $(MAX_AGE_HOURS),)

# Diffs a service's .env_example keys against its docker-compose.yml environment:
# list. A missing key is a working accident today only if the service's Dockerfile
# bakes .env into the image directly (see services/orion-recall's history) -- this
# gate exists so that accident can't silently rot further.
check-env-compose-parity:
	@if [ -z "$(SERVICE)" ]; then \
		echo "usage: make check-env-compose-parity SERVICE=<service-name>"; \
		exit 1; \
	fi
	@python scripts/check_service_env_compose_parity.py $(SERVICE)

# Completeness gate for orion/journaler/dispatch_registry.py: fails if any
# trigger_kind in orion.journaler.worker._TRIGGER_TO_MODE has no matching row in
# JOURNAL_DISPATCH_REGISTRY (see services/orion-actions/app/main.py's
# _dispatch_journal_notifications, which resolves policy off this registry --
# an unregistered trigger_kind silently sends nothing at runtime by design,
# fail-closed, but that gap should be loud in CI, not silent).
check-journal-dispatch-registry:
	@python scripts/check_journal_dispatch_registry.py

# Report-only: flags orion-actions daily cadences (Daily Pulse, World Pulse, Daily
# Metacog, and Daily Journal -- which has no env var of its own and reuses Daily
# Pulse's hour/minute, see services/orion-actions/app/main.py's journal_should_run
# call) that land within --threshold-minutes of each other. Always exits 0 unless
# THRESHOLD/FAIL_ON_COLLISION make it a real gate -- see the script's docstring for why
# this isn't a hard gate today.
check-daily-schedule-collisions:
	@python scripts/check_daily_schedule_collisions.py $(if $(THRESHOLD_MINUTES),--threshold-minutes $(THRESHOLD_MINUTES),) $(if $(FAIL_ON_COLLISION),--fail-on-collision,)

# Host-level crash-loop detector for bus-core (Redis, services/orion-bus/docker-
# compose.yml). Reads container health/restart-count via `docker inspect` ONLY --
# no Redis connection, no Postgres connection -- so it still works when both are
# down at the same time (a confirmed, not hypothetical, dev failure mode). Writes
# local JSON state and, on a crash-loop signature, a plain marker file (this repo
# has no notify-send/osascript/desktop-notification mechanism to reuse -- see
# scripts/bus_core_health_watchdog.py's docstring). Intended to run via host cron
# or a systemd timer (see that script's docstring / scripts/README.md for install
# instructions), not from inside a container.
bus-core-health-watchdog:
	@python scripts/bus_core_health_watchdog.py $(if $(PROJECT),--project $(PROJECT),)

# Reconciled worktree view -- path, branch, merged-into-main status, open PR,
# disk size -- regardless of which of this repo's several worktree location
# conventions (sibling dir, .worktrees/, .claude/worktrees/agent-<id>) each
# one uses. See scripts/worktree_status.py.
# BASE overrides the branch merge status is compared against (default:
# origin/main) -- e.g. `make worktree-status BASE=origin/release`.
worktree-status:
	@python3 scripts/worktree_status.py $(if $(BASE),--base $(BASE),)

worktree-status-summary:
	@python3 scripts/worktree_status.py --summary $(if $(BASE),--base $(BASE),)

worktree-status-stale:
	@python3 scripts/worktree_status.py --stale-only $(if $(BASE),--base $(BASE),)

# Dry-run by default; pass YES=1 to actually remove merged worktrees. Never
# force-removes a worktree with uncommitted changes -- see
# scripts/prune_merged_worktrees.py.
prune-merged-worktrees:
	@python3 scripts/prune_merged_worktrees.py $(if $(YES),--yes,) $(if $(BASE),--base $(BASE),)
