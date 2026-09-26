# FCC Messages transport repair

## Summary

Fix the pinned FCC proxy's non-streaming response contract and silent-stream
liveness without increasing timeouts or changing model reasoning settings.

## Runtime evidence

- Paper: https://arxiv.org/abs/2310.19279
- Seed: `reading:b0aec83b-0458-5a06-b934-ffa6934d1566`.
- Earlier controlled Stage 2 correlation:
  `6d45c25b-bf2c-4a0a-ba76-e1957b252e4a`.
- Its stance assessment passed; three WebFetch calls then failed. Claude
  reported a 300000ms idle watchdog and explicitly rejected an HTTP 200 SSE
  body returned to its non-streaming retry.
- Installed `api/handlers/messages.py:78` unconditionally wraps provider output
  in SSE; `api/routes.py` calls this handler regardless of `request.stream`.
  No transport heartbeat exists in its StreamingResponse wrapper.
- Current pre-patch verifier: Stage 1 done, Stage 2 failed with
  `turn_error:fcc_stream_stalled`, no Stage 2 journal or landing. Automatic
  daytime retries occurred after the earlier controlled attempt.

## Current architecture

- Service: `services/orion-fcc`, pinned upstream free-claude-code 2.4.4.
- Entry: `entrypoint.sh` -> `fcc-server` -> `api.routes.create_message`.
- Config: mounted `~/.fcc/.env`, thin local `.env`, upstream Settings.
- Transport: Claude -> FCC -> gateway `/v1/messages` -> pool-selected llama.cpp.
- Docker: service Dockerfile/Compose. No bus/schema/config contract changes.
- Tests/evals: previously absent in this wrapper; now route/transport unittests
  and opt-in, read-only Claude WebFetch eval.

## Changes

- Build-time, upstream-source-hash-guarded route adapter.
- Explicit false/null/omitted stream requests produce complete JSON or 502.
- True stream requests get immediate and idle 15-second pings, never fake
  cognitive output. Provider output/routing/recovery/deadlines are unchanged.
- Cancellation closes the pending provider read and completes async cleanup.
- Dedicated CI executes against the actual pinned FCC route, not a route mock.
- No env templates changed; worktree local env symlinks use existing ignored
  operator configuration. No secrets committed.

## Verification

- Docker build: passed through safe wrapper in isolated Compose project.
- Eleven focused tests passed (including actual patched-route HTTP requests).
- Same route regression against original deployment image: fails as expected
  (`application/json` expected, `text/event-stream` received).
- Real WebFetch candidate: successful source receipt at 455.2 seconds overall,
  388.4 seconds after tool invocation (past the old 300-second idle watchdog).
  Receipt contains the correct paper title and two concrete abstract claims.
  Claude session: `0ae56752-cbfe-4661-9372-1b8a900a7d26`. No retry/error receipt.
  Final CLI completion: IN PROGRESS. Separate live non-streaming probe timed
  out waiting for response headers at 240 seconds under shared agent load;
  live non-streaming completion remains UNVERIFIED, despite deterministic
  wire-format regression coverage. No retry was launched for that probe.
- PR #2368: all three checks passed (transport, static gates, browser smoke);
  mergeable. Independent review has no remaining material findings.
- Full reading completion: UNVERIFIED. This patch does not claim journal landing.

## Review findings fixed

- Finding: Starlette's cancelled AnyIO scope interrupts awaited provider cleanup.
  - Fix: shield cleanup while explicitly cancelling the pending read once.
  - Evidence: AnyIO task-group cancellation regression waits inside the provider
    finalizer and verifies completion. All eleven tests pass.
- Finding: buffering a non-streaming reply inside the route no longer inherits
  StreamingResponse's disconnect monitor.
  - Fix: poll the real request's disconnect callback while collecting in an
    owned task; cancel once with shielded cleanup on disconnect/cancellation.
  - Evidence: two non-streaming cancellation regressions; independent re-review
    found no remaining material issues.

The live candidate uses image `27633922d515` (wire-format/heartbeat fix).
The final rebuilt image `4719044bae08` adds the reviewed cancellation shield
and non-streaming disconnect monitor; all eleven tests passed against that built image. Production FCC
and the reading queue were not modified during these isolated checks.

## Deployment / rollback

FCC-only rebuild/restart, after review/approval, from this worktree using
`scripts/safe_docker_build.sh orion-fcc -p orion-fcc up -d --build` (the verified
deployment Compose project is `orion-fcc`). No Hub/Cortex/model worker restart required.
An in-flight FCC client will be interrupted, so check active callers first.

Rollback by rebuilding the preceding revision without the adapter and restarting
FCC. Do not requeue unrelated papers or erase stored Stage 1 artifacts. A paper
retry requires explicit approval and its own persisted-evidence verification.
