## Summary

- Long agent turns lost the task after compaction. Every compaction summary since 2026-10-21 was the single word "Conversation".
- Cause: #2802 strips tools from compaction requests. With no tools, FCC's built-in "skip chat-title generation" shortcut matched Claude Code's system prompt and answered "Conversation" in ~20 ms, without calling the model.
- Fix: `guard_local_optimizations()` stops FCC's local shortcuts from answering any compaction request. The patched routes module installs it at startup.

## Outcome moved

Compaction summaries reach the model again, so the handoff carries the task forward. This was the "losing stream" on agent turns: after each compaction Orion forgot the question and drifted (correlations ff5b59ad, 656cfc8b, 6df2c432, d8917251 all repair-flagged `misaligned`).

## Current architecture

orion-fcc wraps the pinned upstream FCC proxy. `prepare_request` (#2802) strips tools from compaction requests. Upstream `MessagesHandler._intercept_local_optimization` then calls `try_optimizations`, whose `try_title_skip` fired.

## Live evidence (before)

- Before 10-21: compaction `duration_ms` 4,626-62,695, real summaries.
- From 10-21 05:35: 19 of 19 compactions took 150-216 ms; all summaries "Conversation".
- `/tmp/fcc-server.log`: every compaction lines up to the millisecond with "Optimization: Skipped title generation request".
- The harness emitted `context_compaction_degenerate` 21 times since 10-18; nothing acted on it.

## Files changed

- `services/orion-fcc/orion_fcc_messages_transport.py`: `guard_local_optimizations()`; corrected the `prepare_request` docstring.
- `services/orion-fcc/install_transport_patch.py`: patched routes call the guard at import.
- `services/orion-fcc/tests/test_messages_transport.py`: regression tests.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
docker run ... orion-fcc-transport-test-fcc -m unittest discover -s /tests
Ran 22 tests ... OK
- unguarded try_optimizations returns "Conversation" for a no-tools compaction (bug reproduced)
- guarded returns None for compaction; real title requests still get "Conversation"
- fresh-process check: importing api.app installs the guard
```

## Evals run

None for this service. Live check after deploy is below.

## Docker/build/smoke checks

`scripts/safe_docker_build.sh orion-fcc -p orion-fcc-transport-test build`: built.

## Review findings fixed

- Finding: the guard-install test could not fail, because earlier tests installed the guard in the same process.
  - Fix: it now runs in a subprocess.
  - Evidence: 22 OK.

## Restart required

Rebuild/redeploy orion-fcc after merge (post-pull rebuild covers it).

Live check after deploy:
- No "Skipped title generation" line at a compaction timestamp.
- The next compact_boundary `duration_ms` is in the thousands.
- The summary text contains "Summary:".

## Risks / concerns

- Severity: low. Concern: a degenerate compaction still continues silently (the harness only logs `context_compaction_degenerate`). Mitigation: follow-up to re-send the task anchor or mark the turn degraded.
- Severity: info. Background outreach turns on the 27B lane sometimes time out after 210 s while reading turns hold its single slot, then fall back to chat. That is by design; changing it is a policy call.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
