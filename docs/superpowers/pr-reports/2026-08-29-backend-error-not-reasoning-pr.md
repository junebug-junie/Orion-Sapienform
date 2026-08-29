# PR: a backend failure is not Orion's reasoning

Fallout fixes from the 2026-08-29 circe outage. Unlike PR #1935 (which is dormant by
design), **both changes here alter live behavior.**

## Summary

- `orion-cortex-exec` no longer publishes a metacognitive trace when its fallback content
  is a gateway error sentinel. It was recording transport failures as
  `trace_role="reasoning"`.
- The gate uses the repo's **existing canonical detector**,
  `looks_like_error_text()` (`orion/cognition/cortex_payload_extract.py`). No new detector
  is added and `orion-llm-gateway` is not touched at all.
- `orion-gpu-cluster-power` gains the `service_version` setting its own `api.py` has always
  read -- the service has been crash-looping its heartbeat on `AttributeError` every tick.

## Outcome moved

**936 rows** of `orion_metacognitive_trace` currently hold
`content="[Error: llamacpp timed out after waiting]"`, `trace_role="reasoning"`,
`trace_stage="pre_answer"`, `model="unknown"`. Daily history: 08-16:9, 08-17:32, 08-18:9,
**08-19:760**, 08-20:82, 08-23:8, 08-25:6, 08-29:30. During the 00:00Z hour of the outage
that was 30 of 59 rows -- **51% of Orion's recorded reasoning for that hour was a transport
error message.** No new rows can be written after this patch.

`orion-athena-gpu-cluster-power` emitted `Heartbeat failed: 'Settings' object has no
attribute 'service_version'` once per tick, continuously, publishing no heartbeat at all --
including throughout a 45-minute outage of a GPU host.

## Current architecture

`services/orion-cortex-exec/app/main.py:206-236` selects the first non-empty
`MetacognitiveTraceV1` from `res.metacog_traces`, and **falls back to `res.final_text`** when
there is none. On a gateway failure `final_text` *is* the error sentinel, because
`llm_backend.py`'s failure branches return a normal result dict rather than raising. The
code already knew: it sets `metadata["fallback_from_final_text"] = True` and then labelled it
`reasoning` anyway.

## Files changed

- `services/orion-cortex-exec/app/main.py`: gate the fallback publish
- `services/orion-gpu-cluster-power/app/settings.py`: add `service_version`
- `tests/test_backend_error_not_reasoning.py`: 13 tests (new)

## Schema / bus / API changes

None. `run_llm_chat()`'s return contract is **deliberately untouched** -- vetoed by Juniper,
and correctly: every caller consumes a dict and making the failure path raise would change
the whole chat path. This patch fixes what is *recorded*, not what is *returned*.

Dropped rather than relabelled because `TraceRole` (`orion/schemas/metacognitive_trace.py`)
is a closed `Literal` with no honest value for a backend failure, and widening it would be a
contract change to carry a non-event. Nothing is lost: the failure is already on the
gateway's ERROR lines, in `rpc_health` latency, and as an `rpc_transport_timeout` grammar
atom.

## Env/config changes

None. `SERVICE_VERSION` was **already** in `services/orion-gpu-cluster-power/.env_example`
and already passed through `docker-compose.yml`; only the `Settings` class failed to declare
it. Verified live: `docker exec orion-athena-gpu-cluster-power printenv SERVICE_VERSION` ->
`0.1.0`, so the fix resolves at runtime with no operator action.

## Tests run

```text
pytest tests/test_backend_error_not_reasoning.py -q      -> 13 passed
pytest tests/test_attention_runtime_store.py tests/test_backend_error_not_reasoning.py -q
                                                         -> 24 passed
pytest tests/test_backend_error_not_reasoning.py tests/test_receipt_pruner.py -q
                                                         -> 17 passed
pytest services/orion-cortex-exec/tests -q --continue-on-collection-errors
                                                         -> 787 passed, 96 failed, 14 errors
```

The two mixed-file runs are review finding 1's own repros, which failed before the fix.

The cortex-exec failures/errors are **pre-existing and environmental** (verb registry +
missing services). Verified by reverting `main.py` in place and re-running the identical
invocation: **96 failed / 787 passed / 14 errors both ways**, byte-identical. `git stash`
deliberately avoided -- it is shared across worktrees.

Mutation-tested: removing `service_version` fails
`test_gpu_cluster_power_settings_expose_service_version`; swapping the canonical detector for
a narrow `startswith("[Error: ")` check fails `test_cortex_exec_gates_on_the_canonical_detector`.

## Docker/build/smoke checks

```text
Not run. No Dockerfile, compose file, dependency, or port changed.
```

## Restart required

```bash
# gpu-cluster-power: required for the heartbeat fix to take effect
docker compose --env-file .env --env-file services/orion-gpu-cluster-power/.env \
  -f services/orion-gpu-cluster-power/docker-compose.yml up -d --build

# cortex-exec: required to stop new poisoned rows
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

Prefer `scripts/safe_docker_build.sh <service> ...`; do not run these from the shared
checkout.

## Review findings fixed

- Finding 2 (medium) -- **the serious one.** My first cut added
  `orion/llm/backend_errors.py` with its own `[Error: ` constant and predicate, and the PR
  claimed it gave the sentinel "one owner". That was false and I made the problem worse:
  `looks_like_error_text()` was moved into `orion/cognition/cortex_payload_extract.py` on
  2026-08-19 expressly to be the single home for this exact check (5 live call sites), and
  `services/orion-vision-council/app/llm_reply.py:27` holds a second constant. Mine was a
  third, and strictly narrower -- it would have let `[error:` (lowercase), `[Error:` (no
  space), `Traceback ...` and `Internal Server Error` through the gate.
  - Fix: deleted my module, reverted `llm_backend.py` entirely (the gateway is now untouched
    by this PR), and pointed the gate at the canonical detector.
  - Evidence: `test_gateway_failure_text_is_detected` now covers all four variants the
    private version would have missed. I did not run the existing-mechanism check before
    building; that is the process failure here, not just the duplicate.
- Finding 1 (high) -- the settings test used `sys.path.insert` + `from app.settings import
  Settings`, which collides with ~20 root tests binding a top-level `app` to a *different*
  service. Review showed it failing beside `test_receipt_pruner.py` and passing **vacuously**
  beside `test_attention_runtime_store.py` (whose Settings also declares `service_version`),
  and leaving a poisoned `sys.modules['app']` behind.
  - Fix: load by explicit file path via `importlib.util.spec_from_file_location` under a
    unique name, never registered in `sys.modules`.
  - Evidence: both of review's repro orderings now pass (24 passed / 17 passed), and the
    mutation still fails the test.
- Finding 3 (low) -- the pinning test grepped source text for 2 of 16 producer sites, so a
  change at any other site stayed green while detection silently broke.
  - Fix: dissolved by the finding-2 fix. The gateway is no longer modified, so there is
    nothing to pin there; the replacement test asserts *which detector the gate calls*, which
    is the property that actually matters, and covers behavior rather than source text.

## Known remaining duplication (not fixed here)

`services/orion-vision-council/app/llm_reply.py:27`'s `GATEWAY_ERROR_PREFIX = "[Error:"` is
still a second definition alongside the canonical `looks_like_error_text`. Consolidating it
crosses a service boundary for a different concern (deciding whether a *reply* is an error,
not whether to persist reasoning), so it is deliberately left as a follow-up rather than
scope-crept into this patch.

## Risks / concerns

- Severity: low. `looks_like_error_text()` matches error *framing*, not the word "error";
  real prose reflecting on an error is not swallowed (tested). The gate is additionally
  narrowed to `reasoning_trace is None`, so a genuine reasoning trace is never suppressed.
- Severity: low. Two services in one PR. The gpu-cluster-power fix is independent and
  bundled deliberately -- both are one-line fallout from the same incident.
- Severity: medium, NOT addressed. The **936 existing rows are untouched.** They remain
  labelled `reasoning` and continue to reach any consumer reading that table. Relabelling or
  quarantining them is a data migration and needs its own decision.
- Severity: low, NOT addressed. `notify_attempts` has 0 rows ever and `notify_requests` is
  100% `status='pending'` across 10,671 rows. Delivery itself appears to work (in-app publish
  + `maybe_send_email`); it is the *accounting* that is dead, so delivery cannot be verified.
  Separate patch.
