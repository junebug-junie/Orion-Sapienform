# fix: a worker HTTP 500 reaches callers as a failure, not a blank success

## Summary

- The LLM gateway now answers an upstream failure with its standard failure reply: empty content, a typed `raw.error`, and the worker's own message in `raw.details.message`. This covers non-2xx responses, a 2xx with an error body, an empty `/apply-template` prompt, timeouts and refused connections. Before, a worker 500 came back as text `[Error: llamacpp failed: Server error '500 ...']` with `raw={}`, and cortex-exec recorded it as a successful step.
- cortex-exec's step failure now carries the worker's words, e.g. `upstream_http_5xx:http_500: No user query found in messages.`
- The pool lease release detail and stage 6.2 inference telemetry carry the typed class and the message.
- When every self-sense eval question comes back empty, the run now fails with the turns' own errors. It no longer publishes four non-measurements and reports `completed`.
- The vision-council foveal probe reports a typed gateway failure as `gateway_error` with the worker's message, not as `empty_response`.

## Outcome moved

Live 2026-10-02..06: the gpu2 Bonsai worker (circe:8016) answered 44/44 agent-lane `/v1/chat/completions` calls with HTTP 500 (Jinja template: "No user query found in messages").

- The gateway logged each 500 but replied ok with error text.
- cortex-exec agent steps passed.
- Hub blanked the error-shaped text.
- 4 self-sense runs reported "completed" with 16/16 empty answers.
- Curiosity turns died as `HarnessTurnFailed: no_final_frame` with the real cause invisible.

After this patch, the same 500 fails the cortex-exec step by name, and that name reaches durable runs (`verb_not_ok:...`), the harness (finalize/stance), and the admitted self-sense run's `last_error`.

## Current architecture (before)

- `llm_backend._execute_openai_chat` / `_execute_llamacpp_native_completion`: `r.raise_for_status()` sent HTTP errors to a catch-all that returned text `[Error: <backend> failed: <httpx message>]` with `raw={}`. httpx's message does not include the response body, so the worker's own error text was lost.
- `main._result_error` sniffed the text to release the pool lease `upstream_error`. `grammar_emit.classify_outcome` sniffed the text to get `upstream_http_5xx`. So the gateway itself counted the failure correctly.
- cortex-exec `gateway_error_step_failure` only fails a step when content is empty AND `raw.error` is set. Neither held, so the step was `success`.
- Overflow/retry: `_dispatch_on_pool` re-leases only on context overflow or a min_ctx clamp. Any other upstream error returns at once with one lease, no retry, and no move to another card. This is unchanged: a 500 now neither retries nor overflows elsewhere (pinned by a test asserting one lease and one upstream call).

## Architecture touched

- `services/orion-llm-gateway` (bus reply shape for upstream failures; no new fields, it reuses the existing empty-content + `raw.error` convention that pool-unavailable/recalled/deadline replies have used since 2026-09-24).
- `services/orion-cortex-exec` (failure string includes `details.message`).
- `services/orion-durable-runs` (self-sense no-answer runs fail; runner honours a graph's own terminal failure).
- `services/orion-vision-council` (foveal probe reads `raw.error`).

## Files changed

- `services/orion-llm-gateway/app/llm_backend.py`: new `_upstream_http_failure` / `_upstream_failure_result` / `_upstream_exception_result` / `_error_body_message`, wired into the chat, native and ollama paths.
- `services/orion-llm-gateway/app/main.py`: `_failure_summary`, so the lease release detail reads `upstream_http_5xx: <message>`.
- `services/orion-llm-gateway/app/grammar_emit.py`: typed upstream classes pass through `classify_outcome`.
- `services/orion-llm-gateway/README.md`: documents the failure shape.
- `services/orion-llm-gateway/tests/test_upstream_error_reply.py` (new), `tests/conftest.py` (FakePool records release detail).
- `services/orion-cortex-exec/app/executor.py`, `tests/test_llm_gateway_overloaded_reply.py`.
- `services/orion-durable-runs/app/self_sense_graph.py`, `app/admitted_self_sense_graph.py`, `app/runner.py`, plus tests `test_self_sense_graph.py`, `test_admitted_self_sense_graph.py`, `test_finish_timing.py`.
- `services/orion-vision-council/app/foveal_probe.py`, `tests/test_foveal_probe.py`.

## Schema / bus / API changes

- Added: none. No schema field and no channel. `raw.error` / `raw.details` already exist on `ChatResultPayload.raw: Dict` and are already the gateway's failure convention.
- Behavior changed: for an upstream failure, `llm.chat.result.content` is now `""` (was `[Error: ...]` text) and `raw.error` is set. The class names are those already in `orion/schemas/llm_inference_projection.py`.
- Compatibility: every caller already handles this shape, because pool-unavailable, recalled and deadline replies use it. Callers that only checked for the `[Error:` text prefix now see empty content; they treat it as an empty/failed answer.

## Env/config changes

None. `.env_example` is untouched, and env parity passes.

## Tests run

```text
services/orion-llm-gateway: pytest tests -> 407 passed
services/orion-durable-runs: PYTHONPATH=<repo> pytest tests -> 276 passed, 72 skipped (Postgres-DSN-only)
services/orion-vision-council: pytest tests -> 110 passed
services/orion-cortex-exec: tests/test_llm_gateway_overloaded_reply.py -> 18 passed alone.
  The full suite has 142 pre-existing failures/collection errors on origin/main (verb double-registration and similar).
  Diffed against a clean origin/main worktree, the only new entry is the new
  test_agent_step_fails_by_name_on_upstream_500_reply. It fails in the full suite for the same
  pre-existing pollution as its 3 sibling _run_step tests (which also fail on main), and passes alone.
New gateway regression tests run against the origin/main gateway code: 17 failed / 1 passed (they catch the bug).
scripts/check_env_template_parity.py PASS; check_metric_lineage.py --gate PASS; check_chat_route_poachers.py PASS
```

## Evals run

No eval harness covers the gateway reply path. The regression tests drive the real request path (`run_llm_chat` -> httpx -> MockTransport fake worker returning the live 500 Jinja body) through `handle_chat`, pool lease and telemetry. A live check after deploy: the next 500 from any worker should show `llm_gateway_error_reply ... reason=upstream_http_5xx:http_500: <message>` in cortex-exec logs and `upstream_failed` in gateway logs.

## Docker/build/smoke checks

Not run; deploy is out of scope for this task. Live behavior is UNVERIFIED until deployed.

## Review findings fixed

- Finding: the foveal probe reported a typed gateway failure as `empty_response` and dropped the worker message.
  - Fix: it reads `raw.error`/`raw.details.message` and raises `gateway_error`.
  - Evidence: `test_run_foveal_probe_names_a_typed_gateway_failure`.
- Finding: the runner's `next_node` rule also matched a last node that *raised*, which really is resumable, and the test stubbed `_emit_state`.
  - Fix: an explicit `terminal=True` is passed from the reached-END branch only.
  - Evidence: the test asserts the real published `DurableRunStateV1` (`next_node is None`).
- Finding: nothing pinned that context overflow beats the new non-2xx check.
  - Fix: one test per path (chat, native) asserting `context_overflow`.
- Finding: gateway post-processing bugs were blamed on the worker (`upstream_error`).
  - Fix: non-httpx, non-ValueError exceptions are now `gateway_exception` (telemetry already treats it as unattributed).
  - Evidence: `test_a_gateway_bug_after_a_good_reply_is_not_blamed_on_the_worker`.
- NIT: a native `/apply-template` transport failure was reported against the completion URL. Fixed: the URL in flight is tracked, with a test.
- NIT: the admitted test now pins a single `attempt_failed` release.
- Not fixed (pre-existing, out of scope): a context overflow that cannot be re-leased still returns non-empty `[Error: context overflow ...]` text, so cortex-exec still passes it as a step success. Same bug class; follow-up.

## Restart required

Rebuild and restart in this order. The gateway goes last so consumers are ready for the shape, though every consumer already handles it:

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-vision-council up -d --build
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
```

(Production deploys from the primary checkout on main after merge.)

## Risks / concerns

- Severity: medium. Calls that used to "succeed" with blank or error text will now fail loudly whenever a worker 5xx/4xx/404s, times out, or refuses a connection:
  - cortex-exec agent/brain steps on any route;
  - durable-runs verb steps (`orion_day`, `journal_compose` and the rest);
  - harness stance/finalize;
  - curiosity turns;
  - self-sense eval (whole run fails when every answer is empty);
  - direct bus callers of the gateway: orion-mind metacog, topic-foundry, memory-consolidation, dream, agent-council, graph-compression, juniper-affective-state, hub, context-exec, vision-council.

  Those direct callers now get empty content instead of `[Error: ...]` text. Mitigation: this is the intended visibility, and the shape is the one they already handle for pool-unavailable replies.
- Severity: low. The cortex-exec step error string now contains free text from the worker (truncated to 240 chars), so anything using it as a metric label or dedup key sees higher cardinality. The prefix parsers (`is_transient_failure`, `is_capacity_deferral`) were checked and still classify correctly.
- Severity: low. A self-sense run with partial answers still completes and publishes "none"-source rows for the empty questions (the existing contract). Only the all-empty case fails.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
