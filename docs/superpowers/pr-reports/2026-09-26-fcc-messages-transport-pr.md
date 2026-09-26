# FCC Messages transport repair

## Summary

Fix the pinned FCC proxy's non-streaming response contract and silent-stream
liveness, and stop converting provider errors into successful assistant text.
No timeouts or model reasoning settings are changed.

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
- A second source-hash guard patches the shared ledger's provider-error emitter:
  early and midstream errors stay SSE errors rather than normal answer text.
  Non-streaming responses preserve those error payloads with status 502.
- Dedicated CI executes against the actual pinned FCC route, not a route mock.
- No env templates changed; worktree local env symlinks use existing ignored
  operator configuration. No secrets committed.

## Verification

- Docker build: passed through safe wrapper in isolated Compose project.
- Thirteen focused tests passed (including actual patched-route HTTP requests
  and the actual upstream early/midstream error emitters).
- Same route regression against original deployment image: fails as expected
  (`application/json` expected, `text/event-stream` received).
- Real WebFetch candidate: successful source receipt at 455.2 seconds overall,
  388.4 seconds after tool invocation (past the old 300-second idle watchdog).
  Receipt contains the correct paper title and two concrete abstract claims.
  Claude session: `0ae56752-cbfe-4661-9372-1b8a900a7d26`. No retry/error receipt.
  The final CLI response was a GPU-pool HTTP 503 rendered as assistant text
  (`is_error=false`, provider request `req_63d8f1a3a42c`). This exposed the error
  emitter bug fixed in the last patch. The initial eval only required a source
  receipt and printed PASS; it now also requires a grounded final reply and
  would reject this captured run. Full live eval is NOT passed.
  Separate live non-streaming probe timed
  out waiting for response headers at 240 seconds under shared agent load;
  live non-streaming completion remains UNVERIFIED, despite deterministic
  wire-format regression coverage. No retry was launched for that probe.
- PR #2368 runs transport, static gates, and browser smoke; current CI status
  is recorded on the PR. No merge conflicts at handoff.
- Full reading completion: UNVERIFIED. This patch does not claim journal landing.

## Review findings fixed

- Finding: Starlette's cancelled AnyIO scope interrupts awaited provider cleanup.
  - Fix: shield cleanup while explicitly cancelling the pending read once.
  - Evidence: AnyIO task-group cancellation regression waits inside the provider
    finalizer and verifies completion.
- Finding: buffering a non-streaming reply inside the route no longer inherits
  StreamingResponse's disconnect monitor.
  - Fix: poll the real request's disconnect callback while collecting in an
    owned task; cancel once with shielded cleanup on disconnect/cancellation.
  - Evidence: two non-streaming cancellation regressions; independent re-review
    found no remaining material issues.
- Finding: live GPU capacity rejection was reported as successful assistant text.
  - Fix: emit the existing typed top-level SSE error at the producing ledger
    method, rather than trying to recognize error prose downstream.
  - Evidence: actual early-provider and midstream error emitter regressions;
    non-streaming error payload retains `gpu_pool_unavailable` provenance.
    Independent final review found no remaining material issues; all 13 tests pass.

The live candidate uses image `27633922d515` (wire-format/heartbeat fix).
Later rebuilt images add the reviewed cancellation shield, non-streaming
disconnect monitor, and explicit provider errors. Production FCC and the
reading queue were not modified during these isolated checks. The candidate
was stopped and removed after the bounded probes completed.

## Remaining concerns

The source-fetch transport succeeded, but live generation still encounters
`gpu_pool_unavailable: deadline` on the shared agent route. This change does
not invent capacity, change pool admission, unload unrelated work, override
thermal guards, or lower model reasoning. Full Stage 2 completion remains
UNVERIFIED and the paper remains failed until an explicitly approved retry.
Deployment/retry approval was requested; no response received during this work.

## Deployment / rollback

FCC-only rebuild/restart, after review/approval, from this worktree using
`scripts/safe_docker_build.sh orion-fcc -p orion-fcc up -d --build` (the verified
deployment Compose project is `orion-fcc`). No Hub/Cortex/model worker restart required.
An in-flight FCC client will be interrupted, so check active callers first.

Rollback by rebuilding the preceding revision without the adapter and restarting
FCC. Do not requeue unrelated papers or erase stored Stage 1 artifacts. A paper
retry requires explicit approval and its own persisted-evidence verification.
