.PHONY: check-metric-generic-consumers check-metric-unwritten test test-hub test-actions bootstrap-test-envs check-inner-state-registry check-metric-lineage check-metric-lineage-cache-refresh check-metric-lineage-gate check-definition-drift check-single-consumer-channels check-activation-saturation concept-relation-digest check-concept-relation-digest-liveness check-env-compose-parity check-journal-dispatch-registry check-daily-schedule-collisions check-substrate-projection-schema-drift check-service-hostname-refs check-scripts-dir-no-stdlib-shadow bus-core-health-watchdog worktree-status worktree-status-summary worktree-status-stale prune-merged-worktrees check-sql-migrations-applied check-sql-migrations-applied-quiet

SERVICE ?=
ARGS ?=

# Interpreter for the metric-lineage targets below. Two live problems this
# solves, both confirmed on this host 2026-08-13:
#   - bare `python` does not exist here (only python3), so a target invoking
#     `python` dies with "No such file or directory" / exit 127. The
#     check-metric-lineage target shipped in PR #1603 was broken this way.
#   - system python3 has no pydantic, which every registry import needs, so
#     python3 alone fails at import time.
# Prefer the repo venv when present, else fall back to python3 so the target
# still runs anywhere the deps happen to be installed globally.
#
# The venv lookup has to reach the MAIN checkout: linked worktrees have no
# .venv of their own, so a plain `[ -x .venv/bin/python ]` resolves to python3
# inside every worktree -- i.e. it would fail exactly where this repo does most
# of its work (CLAUDE.md 2 requires worktrees for implementation). git's
# common-dir points at the main checkout's .git, whose parent holds the venv.
#
# Scoped to these targets deliberately -- switching every older check-* target
# off bare `python` is a separate repo-wide cleanup, out of scope here.
METRIC_PYTHON ?= $(shell \
	if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	else _c=$$(git rev-parse --git-common-dir 2>/dev/null); \
	     if [ -n "$$_c" ] && [ -x "$$_c/../.venv/bin/python" ]; then echo "$$_c/../.venv/bin/python"; \
	     else echo python3; fi; fi)

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

# One owner per tuned env key. Some numbers are restated in a service's
# .env_example, its compose default, a Field(...) default, and prose deriving a
# budget from them -- every restatement is a copy, and copies drift. Measured
# 2026-08-26: HARNESS_FCC_TIMEOUT_SEC was live at 1600 while six other places
# still said 900, and nothing failed because nothing was checking. The gate
# never hardcodes the number; it reads the owner file, so retuning a key stays
# a one-line change.
check-env-key-single-source:
	@$(METRIC_PYTHON) scripts/check_env_key_single_source.py

# Metric semantic layer (phases 1+2 of docs/superpowers/specs/
# 2026-08-12-metric-semantic-layer-design.md). Joins the four metric-bearing
# registries into one URN space and mechanically discovers each metric's
# downstream blast radius. Read-only reporting; enforcement is phase 4.
#   make check-metric-lineage                    # summary
#   make check-metric-lineage METRIC=cpu_pressure  # one lineage card
check-metric-lineage:
	@$(METRIC_PYTHON) scripts/check_metric_lineage.py $(if $(METRIC),--metric $(METRIC),) $(if $(JSON),--json,)

# Phase 3: edit-time PreToolUse nudge (scripts/hooks/metric_lineage_nudge.py,
# registered in .claude/settings.json). It reads .cache/metric_lineage.json
# rather than recomputing the ~13-14s repo scan on every Edit/Write -- this
# target builds that cache by hand. Gitignored (.cache/), not committed. The
# hook also self-refreshes it in the background (detached, non-blocking) the
# first time it's missing or once it's over an hour old, so this target is
# for forcing a fresh one immediately, not a hard prerequisite.
check-metric-lineage-cache-refresh:
	@$(METRIC_PYTHON) scripts/refresh_metric_lineage_cache.py

# The two reports that answer "is this metric safe to retire?" -- the question
# the blast radius alone gets wrong in both directions.
#
#   make check-metric-generic-consumers   # who reads whole vectors, unnamed
#   make check-metric-unwritten           # declared, with no discovered writer
#
# Given their own targets rather than left as CLI flags: the blast-radius
# surface they correct is the one people actually run, and a flag nobody knows
# about corrects nothing.
check-metric-generic-consumers:
	@$(METRIC_PYTHON) scripts/check_metric_lineage.py --generic-consumers

check-metric-unwritten:
	@$(METRIC_PYTHON) scripts/check_metric_lineage.py --unwritten

# Phase 4 CI gate over the same layer. Three checks, all provable from repo
# state -- no naming heuristics:
#   1. registry integrity (everything resolves, no dangling upstream URNs)
#   2. declared-consumer existence -- catches a registry claiming a consumer
#      that was deleted (found orion-spark-introspector, orion-timeline, and
#      orion-evidence-index on first run)
#   3. orphan ratchet -- registered metrics with no consumer may shrink, never
#      grow; a metric that names something but feeds nothing is a keyword
#      cathedral (CLAUDE.md 0A)
# Pre-existing debt lives in config/metrics/orphan_baseline.json so it stays
# visible instead of being silently waived. Regenerate deliberately:
#   make check-metric-lineage-gate UPDATE_BASELINE=1
# UPDATE_BASELINE is matched against explicit true values, not tested for
# non-emptiness: `$(if $(UPDATE_BASELINE),...)` treats ANY value as true, so
# UPDATE_BASELINE=0 and UPDATE_BASELINE=no -- the obvious ways to disable a
# flag -- silently rewrote the baseline to whatever the tree contained,
# ratcheting the debt UPWARD and exiting 0 without ever gating.
check-metric-lineage-gate:
	@$(METRIC_PYTHON) scripts/check_metric_lineage.py $(if $(filter 1 true yes TRUE YES,$(UPDATE_BASELINE)),--update-baseline,--gate)

# Definition-change alert (R4): fails when a metric's resolved DEFINITION --
# meaning, schema, producers, consumers, upstream -- changed without the lock
# being regenerated. The lock's diff is the alert; see
# scripts/check_definition_drift.py.
#
#   make check-definition-drift              # report
#   make check-definition-drift GATE=1       # exit 1 on drift (what CI runs)
#   make check-definition-drift UPDATE=1     # re-lock and print the deltas
#
# GATE/UPDATE use the same explicit-true matching as UPDATE_BASELINE above,
# for the same reason: UPDATE=0 must not rewrite the lock.
#
# GATE=1 UPDATE=1 together is rejected by the script itself (argparse exits 2),
# because --update returns before the gate block -- so that combination would
# rewrite the lock and exit 0 with the gate silently skipped. The space between
# the two $(if ...) expressions is load-bearing for producing that error rather
# than the uninterpretable `--update--gate`.
check-definition-drift:
	@$(METRIC_PYTHON) scripts/check_definition_drift.py $(if $(filter 1 true yes TRUE YES,$(UPDATE)),--update,) $(if $(filter 1 true yes TRUE YES,$(GATE)),--gate,)

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

# Ratchet gate for merge domination: fails if a merge point NOT in
# config/metrics/merge_domination_baseline.json has one source winning the
# majority of non-tied merges while carrying an order of magnitude fewer
# informative distinct values than a contender that never wins. Two instances
# are on record (thermal_pressure 2026-08-11, node:substrate.codebase
# 2026-08-14) and BOTH were caught by a person reading a diff. Requires
# POSTGRES_URI. Not in orion-static-gates -- that workflow has no infra.
# Run on a schedule via host crontab, same as concept-relation-digest.
check-merge-domination:
	@$(METRIC_PYTHON) scripts/check_merge_domination.py --gate $(if $(TICKS),--ticks $(TICKS),)

# Fail-safe for the above: fails if the oldest undigested
# memory_concept_relation_decisions row is older than --max-age-hours (default 3h),
# which only happens if the digest cron entry died, was dropped after a host
# migration, or the job is crashing. Requires POSTGRES_URI.
check-concept-relation-digest-liveness:
	@python scripts/check_concept_relation_digest_liveness.py $(if $(MAX_AGE_HOURS),--max-age-hours $(MAX_AGE_HOURS),)

# Labels attention_salience_trace loops that were scored, never explicitly
# closed by a human (Resolve/Dismiss in the Hub), and then stopped being
# re-scored -- writes attention_loop_outcome verdict=decayed_unattended and
# suppresses the theme out of the Hub's pending-attention panel (never out of
# live reverie selection -- see the script's own docstring). This is the actual
# cron entry point; run on a schedule via host crontab, same as
# concept-relation-digest. Requires POSTGRES_URI.
attention-loop-decay-digest:
	@python scripts/attention_loop_decay_digest.py $(if $(DRY_RUN),--dry-run,)

# Fail-safe for the above: fails if the most-overdue decay-eligible loop
# exceeds its own min-silence threshold by more than --max-overshoot-hours
# (default 3h), which only happens if the digest cron entry died, was dropped
# after a host migration, or the job is crashing. Requires POSTGRES_URI.
check-attention-loop-decay-liveness:
	@python scripts/check_attention_loop_decay_liveness.py $(if $(MAX_OVERSHOOT_HOURS),--max-overshoot-hours $(MAX_OVERSHOOT_HOURS),)

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

# Field tension report -- admission rate, rank discrimination, channel liveness,
# and producer liveness over real substrate_field_state history. Read-only:
# opens a read-only Postgres session, writes nothing, publishes nothing.
#
#   make field-tension-report                      # last 24h, human-readable
#   make field-tension-report HOURS=72             # wider window
#   make field-tension-report Z=3.5                # tighter admission threshold
#   make field-tension-report JSON=1               # machine-readable
#   make field-tension-report LIMIT=10000          # newest N ticks only (fast)
#
# POSTGRES_URI defaults to the in-container host; from the host machine pass:
#   POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney
#
# See orion/attention/tension/README.md for what each number means.
field-tension-report:
	@$(METRIC_PYTHON) scripts/analysis/measure_field_tension_admission.py \
		$(if $(HOURS),--hours $(HOURS),) \
		$(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(Z),--z-threshold $(Z),) \
		$(if $(ALPHA),--alpha $(ALPHA),) \
		$(if $(JSON),--json,)

# Layer 5 attention input starvation -- is attention choosing between competing
# inputs, or idling for lack of any? Read-only. Companion to
# field-tension-report; the two rates are NOT directly comparable (different
# table, cadence, and kind of event -- the script says so in its own output).
#
#   make attention-starvation-report            # last 72h
#   make attention-starvation-report HOURS=24
#   make attention-starvation-report JSON=1
attention-starvation-report:
	@$(METRIC_PYTHON) scripts/analysis/measure_attention_input_starvation.py \
		$(if $(HOURS),--hours $(HOURS),) \
		$(if $(JSON),--json,)

# Attention-loop outcome coverage -- the label half of the salience refit's
# input/label join, and the outcome measure the drives program never had.
# Read-only by default; --emit writes derived implicit labels (snapshots first,
# never overwrites a human verdict, idempotent).
#
#   make attention-outcome-coverage             # coverage + derivable labels
#   make attention-outcome-coverage SWEEP=1     # horizon sensitivity
#   make attention-outcome-coverage JSON=1
attention-outcome-coverage:
	@$(METRIC_PYTHON) scripts/analysis/measure_attention_outcome_coverage.py \
		$(if $(SWEEP),--sweep,) \
		$(if $(HOURS),--min-silence-hours $(HOURS),) \
		$(if $(JSON),--json,)

# Which hand-applied SQL migrations actually reached the live database?
#
# services/orion-sql-db/*.sql is applied BY HAND. There is no migration table, no version
# stamp, and no ordering guarantee, so a migration that was written, reviewed, merged and
# never applied looks identical in git to one that is live.
#
# Needs a reachable Postgres, so this is an operator/agent command rather than a CI gate.
# Exit 1 = drift found; exit 2 = could not connect (deliberately distinct, so an infra
# failure cannot be mistaken for a pass).
check-sql-migrations-applied:
	python3 scripts/check_sql_migrations_applied.py

check-sql-migrations-applied-quiet:
	python3 scripts/check_sql_migrations_applied.py --quiet
