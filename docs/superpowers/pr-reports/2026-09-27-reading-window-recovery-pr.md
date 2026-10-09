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
- Both stages default to 0/0 in settings and the operator template, as
  requested. Local configuration matches. No counters, attempts or GPU
  policies reset; execution timeouts and cooldowns remain unchanged.
- Regression tests cover all 24 hours for both default windows and template
  parity. Explicit operator window overrides remain supported.

## Verification

- 174 tests/evals passed, including disposable real PostgreSQL.
- Gate matrix covers both stages, all five gate reasons, active/absent/
  consumed/opposite-stage bindings.
- Includes 11 handoff eval cases; isolated Hub Docker build passed.
- Around-the-clock follow-up: 53 route/wallet tests and handoff eval cases
  passed. Live Hub reports all four window values as 0; no restart needed.
- Synced local env with the existing script, preserving intentional overrides.
- No new metric definitions.

## Review findings fixed

- Independent review: no material findings.
- Minor coverage request: consumed and opposite-stage bindings.
- Fixed with expanded real PostgreSQL gate matrix.

## Rollout

Hub recreated from this worktree using safe_docker_build.sh, project orion-hub.
Live Stage 2 environment reports 0/0 and both caps remain 12.
An operator recovery tick collected the predecessor's completed result.
The stored-artifact verifier now reports verified_complete=true, gaps=[] for
https://www.youtube.com/watch?v=BgD62MIINo8.

One subsequent operator tick used the existing force option to bypass only
the newly set cooldown: preflight required enabled=true, count below cap,
no retry backoff, and arXiv 2310.19279 next in the normal queue. No Redis
counters, timestamps or queue priorities were manually rewritten.

The arXiv Stage 2 binding is reading-08364abe-2492-59b9-bb5c-471e76aabebc.
Its durable trace records run.started at 2026-09-27T06:01:33.477536Z, with a
granted GPU0/chat lease and Qwen3.6-35B-A3B-UD-Q5_K_M.gguf. Work is started;
paper completion is not yet verified. GPU0 lending needed no policy change.
Initial PR CI: all three checks passed.

Rollback: restore prior Hub image and Stage 2 window 8/22; durable state and
queue history remain intact. Do not reset wallets or delete bindings.
