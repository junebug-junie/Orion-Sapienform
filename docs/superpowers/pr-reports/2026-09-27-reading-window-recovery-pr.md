# Reading Window Recovery

## Root cause

Hub Stage 2 used an 08:00-22:00 America/Denver window while local Stage 1
ran around the clock. From 04:04 UTC its ticks returned outside_window.
Durable run reading-1ac6bc6f-ffc0-52fe-8c3f-19fd70734e3b completed at
04:23 UTC, but the gate prevented Hub from collecting the result. Its
unconsumed binding retained queue precedence over arXiv 2310.19279.

## Current architecture

- Service: Hub, Stage 1 and Stage 2 tick entry points.
- Config: app/settings.py, local .env, existing docker-compose.yml.
- Queue: world_pulse_read_seed and reading_durable_turn in PostgreSQL.
- Existing bus lifecycle and journal channels unchanged; no schema changes.
- Admission and generation remain on the existing held durable reading graph.
- Gap: submission gates ran before durable recovery and result collection.

## Changes

- Parameterized active-only SQL claim mode for each stage.
- Closed window/cap/cooldown/backoff permits only an unconsumed binding for
  that same stage. Row locking and immutable binding reuse remain intact.
- Disabled workers still stop; fresh work remains gated.
- Recovery may retry delivery of an already-bound request whose original
  acceptance was uncertain. It is not restricted to completed results.
- Local Stage 2 hours changed to 0/0, matching local Stage 1; operator
  template remains 8/22. No counters, attempts or GPU policies reset.

## Verification

- 174 tests/evals passed, including disposable real PostgreSQL.
- Gate matrix covers both stages, all five gate reasons, active/absent/
  consumed/opposite-stage bindings.
- Includes 11 handoff eval cases; isolated Hub Docker build passed.
- Synced local env with the existing script, preserving intentional overrides.
- No new metric definitions.

## Review findings fixed

- Independent review: no material findings.
- Minor coverage request: consumed and opposite-stage bindings.
- Fixed with expanded real PostgreSQL gate matrix.

## Rollout

Hub recreated from this worktree using safe_docker_build.sh, project orion-hub.
Live Stage 2 environment now reports 0/0. Result-collection verification follows.
Rollback: restore prior Hub image and Stage 2 window 8/22; durable state and
queue history remain intact. Do not reset wallets or delete bindings.
