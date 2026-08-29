# PR: a backend failure is not Orion's reasoning

Fallout fixes from the 2026-08-29 circe outage. Unlike PR #1935 (which is dormant by
design), **both changes here alter live behavior.**

## Summary

- `orion-cortex-exec` no longer publishes a metacognitive trace when its fallback content
  is a gateway error sentinel. It was recording transport failures as
  `trace_role="reasoning"`.
- New `orion/llm/backend_errors.py` gives the gateway's `[Error: ...]` sentinel one owner
  (`BACKEND_ERROR_PREFIX` + `is_backend_error_text()`), replacing an unowned literal
  repeated at ~10 sites.
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

- `orion/llm/backend_errors.py`: new; one owner for the sentinel
- `services/orion-llm-gateway/app/llm_backend.py`: two failure branches use the shared
  constant. **Output is byte-identical** -- this is deduplication, not a behavior change, so
  the 936 already-persisted rows stay matchable by the same predicate.
- `services/orion-cortex-exec/app/main.py`: gate the fallback publish
- `services/orion-gpu-cluster-power/app/settings.py`: add `service_version`
- `tests/test_backend_error_not_reasoning.py`: 12 tests (new)

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
pytest tests/test_backend_error_not_reasoning.py -q      -> 12 passed
pytest services/orion-llm-gateway/tests -q               -> 285 passed
pytest services/orion-cortex-exec/tests -q --continue-on-collection-errors
                                                         -> 787 passed, 96 failed, 14 errors
```

The cortex-exec failures/errors are **pre-existing and environmental** (verb registry +
missing services). Verified by reverting `main.py` in place and re-running the identical
invocation: **96 failed / 787 passed / 14 errors both ways**, byte-identical. `git stash`
deliberately avoided -- it is shared across worktrees.

Mutation-tested: removing `service_version` fails
`test_gpu_cluster_power_settings_expose_service_version`; reverting the gateway to its inline
literal fails `test_prefix_matches_what_the_gateway_actually_emits`.

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

## Risks / concerns

- Severity: low. `is_backend_error_text()` is a prefix match over a single-producer format,
  not a text classifier. A real answer that merely mentions an error is not flagged (tested).
  The gate is additionally narrowed to `reasoning_trace is None`, so a genuine reasoning
  trace is never suppressed.
- Severity: low. Three services in one PR. The gpu-cluster-power fix is independent and
  bundled deliberately -- both are one-line fallout from the same incident.
- Severity: medium, NOT addressed. The **936 existing rows are untouched.** They remain
  labelled `reasoning` and continue to reach any consumer reading that table. Relabelling or
  quarantining them is a data migration and needs its own decision.
- Severity: low, NOT addressed. `notify_attempts` has 0 rows ever and `notify_requests` is
  100% `status='pending'` across 10,671 rows. Delivery itself appears to work (in-app publish
  + `maybe_send_email`); it is the *accounting* that is dead, so delivery cannot be verified.
  Separate patch.
