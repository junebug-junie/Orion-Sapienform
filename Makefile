.PHONY: test test-hub test-actions bootstrap-test-envs check-inner-state-registry check-metric-lineage check-single-consumer-channels check-activation-saturation concept-relation-digest check-concept-relation-digest-liveness check-env-compose-parity check-journal-dispatch-registry check-daily-schedule-collisions check-substrate-projection-schema-drift check-service-hostname-refs check-scripts-dir-no-stdlib-shadow bus-core-health-watchdog worktree-status worktree-status-summary worktree-status-stale prune-merged-worktrees

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

# Metric semantic layer (phases 1+2 of docs/superpowers/specs/
# 2026-08-12-metric-semantic-layer-design.md). Joins the four metric-bearing
# registries into one URN space and mechanically discovers each metric's
# downstream blast radius. Read-only reporting; enforcement is phase 4.
#   make check-metric-lineage                    # summary
#   make check-metric-lineage METRIC=cpu_pressure  # one lineage card
check-metric-lineage:
	@python scripts/check_metric_lineage.py $(if $(METRIC),--metric $(METRIC),) $(if $(JSON),--json,)

# Live-bus gate: every channel marked single_consumer: true in
# orion/bus/channels.yaml must have exactly one live subscriber
# (Redis pub/sub duplicates execution otherwise -- see PR #994).
# Requires ORION_BUS_URL=redis://<tailscale-ip>:6379/0.
check-single-consumer-channels:
	@python scripts/check_single_consumer_channels.py

# Fails if anything under scripts/ collides with a Python stdlib module
# name (e.g. `platform`, `json`, `types`) -- Python auto-inserts scripts/
# at sys.path[0] for any `python3 scripts/<name>.py` invocation, so a
# collision silently shadows the real stdlib module for every script run
# that way. See scripts/platform_audits/README.md for the incident this
# gate exists to prevent recurring (scripts/platform/, renamed
# 2026-08-12, cost an extended live investigation to trace).
# python3, not the bare `python` every other check-* target above uses --
# deliberate, not an inconsistency: this patch found live that `python`
# isn't guaranteed on PATH (confirmed on this exact host), and a portability
# gate using an interpreter name that isn't itself portable would be an odd
# thing to ship in the same patch. Not fixing the older targets' `python`
# here -- that's a separate, repo-wide cleanup, out of scope.
check-scripts-dir-no-stdlib-shadow:
	@python3 scripts/check_scripts_dir_no_stdlib_shadow.py

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

# Repo-wide gate: flags any services/*/.env_example URL that hardcodes another
# service's services/<dirname> directory name as an HTTP hostname instead of its
# real Docker Compose service key. Found live 2026-07-28 -- that directory-name
# style hostname only resolves by accident (depends on container_name:
# ${PROJECT}-<name> happening to equal orion-<name>, i.e. PROJECT having no host
# suffix); it silently broke orion-notify-digest's daily email since inception,
# plus 15 other references across 11 other services. The compose service key is
# the one hostname Docker's default network guarantees regardless of PROJECT.
check-service-hostname-refs:
	@python scripts/check_service_hostname_refs.py

# Report-only: flags orion-actions daily cadences (Daily Pulse, World Pulse, Daily
# Metacog, and Daily Journal -- which has no env var of its own and reuses Daily
# Pulse's hour/minute, see services/orion-actions/app/main.py's journal_should_run
# call) that land within --threshold-minutes of each other. Always exits 0 unless
# THRESHOLD/FAIL_ON_COLLISION make it a real gate -- see the script's docstring for why
# this isn't a hard gate today.
check-daily-schedule-collisions:
	@python scripts/check_daily_schedule_collisions.py $(if $(THRESHOLD_MINUTES),--threshold-minutes $(THRESHOLD_MINUTES),) $(if $(FAIL_ON_COLLISION),--fail-on-collision,)

# Detection gate for the 2026-07-24 orion-substrate-runtime crash-loop incident
# (PR #1331 renamed TransportBusStateV1 fields; the one stale persisted row still
# had the old names, and extra="forbid" turned every tick's load into a hard
# ValidationError crash-loop for ~10 hours undetected). For each of the seven
# persisted, fixed-projection_id singleton rows in
# services/orion-substrate-runtime/app/store.py, loads the CURRENT live row
# against the CURRENT schema and fails if it does not validate -- see
# scripts/check_substrate_projection_schema_drift.py's docstring for the full
# incident writeup and scope rationale. Skips cleanly (exit 0) if POSTGRES_URI is
# unset or Postgres is unreachable -- deliberately different from
# check-activation-saturation/check-concept-relation-digest-liveness's exit-2
# convention, see that script's "DB unavailability" note for why. Requires
# POSTGRES_URI (see services/orion-substrate-runtime/.env).
check-substrate-projection-schema-drift:
	@python scripts/check_substrate_projection_schema_drift.py $(if $(JSON),--json,)

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
	@python3 scripts/bus_core_health_watchdog.py $(if $(PROJECT),--project $(PROJECT),) \
		$(if $(TELEMETRY_ROOT),--telemetry-root $(TELEMETRY_ROOT),) \
		$(if $(UNHEALTHY_STREAK_THRESHOLD),--unhealthy-streak-threshold $(UNHEALTHY_STREAK_THRESHOLD),) \
		$(if $(RESTART_COUNT_THRESHOLD),--restart-count-threshold $(RESTART_COUNT_THRESHOLD),) \
		$(if $(RESTART_WINDOW_MINUTES),--restart-window-minutes $(RESTART_WINDOW_MINUTES),)

# Host-level disk-usage threshold watchdog for /mnt/docker, /mnt/scripts, and
# /mnt/telemetry (each a distinct physical mount on this host). Publishes an
# orion-notify /attention/request (Hub Pending Attention card) the first time
# any monitored path crosses --threshold-pct (default 90), debounced via local
# state so it doesn't refire every tick while already confirmed-notified and
# still breached -- but DOES retry every tick if the prior notify attempt
# never actually confirmed success (orion-notify down/unreachable), so a
# breach can never be silently swallowed. See scripts/disk_threshold_watchdog.py's
# docstring for the full design and scripts/README.md for cron install
# instructions. Requires PYTHONPATH=. (it imports orion.notify.client) and
# orion-notify reachable at NOTIFY_BASE_URL.
disk-threshold-watchdog:
	@PYTHONPATH=. python3 scripts/disk_threshold_watchdog.py $(if $(PATHS),--paths $(PATHS),) \
		$(if $(THRESHOLD_PCT),--threshold-pct $(THRESHOLD_PCT),) \
		$(if $(PROJECT),--project $(PROJECT),) \
		$(if $(TELEMETRY_ROOT),--telemetry-root $(TELEMETRY_ROOT),) \
		$(if $(NOTIFY_BASE_URL),--notify-base-url $(NOTIFY_BASE_URL),)

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
