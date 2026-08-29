# PR: system.error replies were silently rejected by the bus's own schema enforcement

## Summary

- Root-caused a live incident (corr=`d0627642...`): whisper-tts hit a CUDA OOM synthesizing a TTS reply, logged the failure and tried to publish a `system.error` envelope back on the RPC reply channel within 42s -- but that channel is schema-locked to the *success* payload (`TTSResultPayload`, requires `audio_b64`), so the error publish itself raised `ValueError` inside `OrionBusAsync.publish()`, and the worker's own `except Exception: pass` around the error-publish swallowed it. The hub never got a reply and burned the full `HUB_TTS_TIMEOUT_SEC=180s` before reporting a bare timeout with no real cause.
- Fixed the bus-level root cause in `orion/core/bus/async_service.py`'s `_validate_payload()`: an envelope whose `kind` is `system.error`/`system.error.v1` is now validated against `SystemErrorV1` instead of the channel's own declared schema, rather than being rejected outright.
- First-round code review caught that this exemption, once error replies could actually reach consumers, exposed two pre-existing **silent-success** bugs in RPC clients that never checked `envelope.kind` before treating a reply as real output: `orion/harness/cortex_client.py` (returned the raw `{"error": ...}` dict as plan-execution output) and `services/orion-cortex-exec/app/clients.py` (validated an error payload into a hollow `content=None` "successful" chat reply, since every field on that model is `Optional` with `extra="ignore"`). Both fixed to raise with the real error message instead.
- Widened `SystemErrorV1.details` from `Dict[str, Any]` to `Union[str, Dict[str, Any]]` -- 3 of the 4 real `system.error` producers in the repo send a plain exception string, not a dict, so a strict-dict schema would have rejected the very payloads this fix exists to let through.
- Added a shared `SYSTEM_ERROR_KINDS` constant (`orion/core/bus/bus_schemas.py`) and migrated 4 other pre-existing hardcoded `kind == "system.error"` checks (`orion/harness/substrate_client.py`, `services/orion-hub/scripts/bus_clients/tts_client.py` x2, `services/orion-hub/scripts/thought_client.py`) to it, closing a review-flagged gap and giving all of them the `"system.error.v1"` variant for free.

## Outcome moved

- A TTS/STT/harness-cortex/LLM-gateway synthesis or RPC failure now surfaces to the caller within the producer's own real handling time (tens of seconds) instead of always burning the full RPC timeout with a generic, causeless "timed out" message.
- Two previously-silent data-corruption paths (a raw error dict returned as "plan output"; an empty chat reply returned as "success") now raise with the real error message instead.

## Current architecture

`OrionBusAsync.publish()` (`orion/core/bus/async_service.py`) calls `_validate_payload()` on every publish, which resolves the target channel's `schema_id` from `orion/bus/channels.yaml` and validates the payload against it unconditionally -- with no awareness of envelope `kind`. Every RPC-style service (whisper-tts, orion-substrate-runtime, orion-llm-gateway, orion-harness-governor) shares one convention for reporting a mid-request failure on its own reply channel: publish an envelope with `kind="system.error"` (or `"system.error.v1"` for `orion/harness/finalize.py`) and an error-shaped payload. Because the reply channel's catalog entry is schema-locked to the channel's *success* payload, that error envelope always failed validation -- and because every producer wraps its error-publish in a broad `except Exception: pass`, that failure was always invisible.

## Architecture touched

- `orion/core/bus/async_service.py` -- `_validate_payload()`'s redirect logic (the fix's core).
- `orion/core/bus/bus_schemas.py` -- new `SYSTEM_ERROR_KINDS` shared constant.
- `orion/schemas/platform.py` -- `SystemErrorV1.details` type widened to match real producer payloads.
- `orion/harness/cortex_client.py`, `services/orion-cortex-exec/app/clients.py` -- new `kind`-aware error handling (silent-success fix).
- `orion/harness/substrate_client.py`, `services/orion-hub/scripts/bus_clients/tts_client.py`, `services/orion-hub/scripts/thought_client.py` -- migrated to the shared `SYSTEM_ERROR_KINDS` constant.
- `orion/bus/channels.yaml` -- top-of-file note documenting the `system.error` validation exception so a channel's `schema_id` isn't misread as covering 100% of its real traffic.

## Files changed

- `orion/core/bus/async_service.py`: redirect `system.error`/`system.error.v1` payload validation to `SystemErrorV1` instead of the channel's declared schema (previously: unconditional validation against the channel schema, which an error payload never matches).
- `orion/core/bus/bus_schemas.py`: added `SYSTEM_ERROR_KINDS = frozenset({"system.error", "system.error.v1"})`, the shared, documented literal every consumer below now imports.
- `orion/schemas/platform.py`: `SystemErrorV1.details: Union[str, Dict[str, Any]]` (was `Dict[str, Any]`) -- real producers send a plain string.
- `orion/harness/cortex_client.py`: `execute_plan()` now raises `RuntimeError` with the real error detail when the reply's `kind` is a system-error kind, instead of falling through to `return payload`.
- `services/orion-cortex-exec/app/clients.py`: `LLMGatewayClient.chat()` now raises `RuntimeError` with the real error detail before `ChatResponsePayload.model_validate(payload)` would have silently accepted it.
- `orion/harness/substrate_client.py`, `services/orion-hub/scripts/bus_clients/tts_client.py`, `services/orion-hub/scripts/thought_client.py`: `kind == "system.error"` → `kind in SYSTEM_ERROR_KINDS`.
- `orion/bus/channels.yaml`: top-of-file comment documenting the `system.error` exception to per-channel `schema_id` enforcement.
- `orion/core/bus/tests/test_validate_payload_system_error_exemption.py` (new): 5 tests -- the original incident payload now publishes; a non-error payload with the wrong schema is still rejected; the dedicated `orion:system:error`/`SystemErrorV1` channel's own enforcement is untouched; a genuinely malformed error payload is still rejected even on a mismatched-schema channel; `"system.error.v1"` is recognized too.
- `orion/harness/tests/test_cortex_client_finalize_timeouts.py`: added a regression test for the silent-success fix.
- `services/orion-cortex-exec/tests/test_llm_gateway_client_errors.py` (new): regression test for the silent-success fix, plus a control test confirming a real successful reply is unaffected.

## Schema / bus / API changes

- Added: `orion.core.bus.bus_schemas.SYSTEM_ERROR_KINDS`.
- Behavior changed: `orion/core/bus/async_service.py`'s `_validate_payload()` now validates `system.error`/`system.error.v1` envelopes against `SystemErrorV1` instead of the target channel's declared schema. `SystemErrorV1.details` accepts `str` in addition to `dict`.
- Consumer behavior changed: `HarnessCortexClient.execute_plan()` and `LLMGatewayClient.chat()` now raise on an error reply instead of silently returning a hollow/wrong "successful" result. **Known, intentional downstream effect**: `services/orion-cortex-exec/app/executor.py:3255`'s `MetacogDraftService` step calls `llm_client.chat()` outside its own inner `try` (which starts at line 3263) -- a `system.error` reply there now propagates to the per-service `except Exception` at line 4545 and returns `StepExecutionResult(status="fail", ...)` for the whole step, where it previously degraded silently into a fabricated fallback draft reported as `ok=True`. This is the correct behavior per this repo's explicit "no empty-shell cognition" rule (a masked LLM-gateway failure was exactly that: fallback text masquerading as generated cognition), and `status="fail"` is the same, already-established outcome every other service in that dispatch loop already returns on its own exceptions -- not a new failure mode, just this one call site finally using it. No test was added for this specific call site's now-correct hard-fail; flagged as a concern below.
- Compatibility notes: `SystemErrorV1`'s `extra="allow"` and now-widened `details` type make this a backward-compatible schema relaxation, not a break. The 4 migrated clients' behavior is unchanged for the `"system.error"` kind they already checked; they gain recognition of `"system.error.v1"` as a bonus.

## Env/config changes

None.

## Tests run

```text
# New/updated regression suites
python -m pytest orion/core/bus/tests/ orion/harness/tests/test_cortex_client_finalize_timeouts.py -q
  -> 8 passed

cd services/orion-cortex-exec && python -m pytest tests/test_llm_gateway_client_errors.py tests/test_context_exec_client.py -q
  -> 3 passed

# Broader bus/RPC/catalog regression sweep
python -m pytest tests/test_bus_pubsub_timeout.py tests/test_bus_async_rpc_worker.py \
  tests/test_bus_rpc_fork_client.py tests/test_exec_result_channel_catalog_specificity.py \
  tests/test_mind_llm_bus_catalog.py services/orion-substrate-runtime/tests/test_finalize_appraisal_rpc.py \
  services/orion-llm-gateway/tests/test_handle_chat_meta.py -q
  -> 68 passed, 2 failed (both pre-existing, unrelated -- see below)

# Migrated-client regression check (needs 2 env keys this fresh worktree's .env lacks)
CHANNEL_COLLAPSE_INTAKE=orion:collapse:intake CHANNEL_COLLAPSE_TRIAGE=orion:collapse:triage \
  python -m pytest services/orion-hub/tests/test_tts_client_errors.py services/orion-hub/tests/test_thought_client.py -q
  -> 6 passed

cd services/orion-whisper-tts && python -m pytest tests -q
  -> 53 passed, 2 failed (pre-existing, unrelated -- see below)
```

**Pre-existing, unrelated failures** (confirmed via mutation testing -- reverted the fix to true `HEAD`, byte-identical failure set before and after):

- `tests/test_exec_result_channel_catalog_specificity.py::test_exec_result_channels_have_specific_wildcards` -- asserts `orion:exec:result:PadRpc:*` is in `channels.yaml`; it is not, on `main`, independent of this branch.
- `services/orion-whisper-tts/tests/test_tts_worker_replies.py::test_typed_reply_includes_metadata` / `test_legacy_reply_includes_metadata_and_mime_type` -- both construct a `BaseEnvelope` with a non-UUID `correlation_id` string (`"cid-legacy"`); pre-existing test-fixture bug, unrelated to this diff.
- `services/orion-llm-gateway/tests/test_handle_chat_meta.py::test_handle_chat_meta_includes_llm_uncertainty` -- passes in isolation (confirmed); fails only when run in the same pytest session as the wider bus/catalog sweep above, due to cross-test-file `Settings()` env pollution (`POSTGRES_URI` required by a different test's import). Pre-existing test-isolation issue, unrelated to this diff.
- `services/orion-cortex-exec/tests/`: 14 files fail to *collect* (`ValueError: Verb already registered: legacy.plan`) whenever the full suite runs together; confirmed present with or without this branch's new test file. A full 128-failure baseline diff (with-fix vs `HEAD`) for the rest of that service's suite was byte-identical, confirming zero new failures from this patch.
- No `.env` exists for `orion-hub`/`orion-cortex-exec` in this fresh worktree at all (a known gap -- `scripts/sync_local_env_from_example.py` only diffs an *existing* `.env`, it doesn't bootstrap one), which is why `services/orion-hub/tests/test_tts_client_errors.py` needed 2 env vars injected by hand above just to collect.

Every mutation-tested new/updated regression test (the 5 in `test_validate_payload_system_error_exemption.py`, the 1 in `test_cortex_client_finalize_timeouts.py`, the 1 in `test_llm_gateway_client_errors.py`) was confirmed to fail against the pre-fix code and pass against the fixed code.

## Evals run

No eval harness exists for `orion/core/bus/`, `orion/harness/`, or the touched services beyond their `tests/` directories.

## Docker/build/smoke checks

Not run -- this is a pure Python library/schema change with no new dependency, port, or compose wiring. `orion-athena-whisper-tts` and `orion-athena-hub` (the two containers involved in the live incident) are already running on this host; a restart is required for the fix to take effect there (see below).

## Review findings fixed

**Round 1** (initial `_validate_payload()` exemption reviewed):

- Finding: `orion/harness/cortex_client.py:66` -- `system.error` reply silently returned as plan-execution output (no `kind` check).
  - Fix: raise `RuntimeError` with the real error detail when `kind in SYSTEM_ERROR_KINDS`.
  - Evidence: `orion/harness/tests/test_cortex_client_finalize_timeouts.py::test_system_error_reply_raises_instead_of_returning_as_result`, mutation-tested.
- Finding: `services/orion-cortex-exec/app/clients.py:118` -- `system.error` reply silently validated into a hollow `content=None` "successful" `ChatResponsePayload` (every field `Optional`, `extra="ignore"`).
  - Fix: raise `RuntimeError` with the real error detail before `model_validate()`.
  - Evidence: `services/orion-cortex-exec/tests/test_llm_gateway_client_errors.py::test_system_error_reply_raises_instead_of_a_hollow_success`, mutation-tested; live repro (`ChatResponsePayload.model_validate({"error": "..."})` → `ChatResultPayload(content=None, ...)`, no exception) confirmed pre-fix.
- Finding: the exemption was an unconditional skip, not a validation against `SystemErrorV1` -- a malformed error payload would sail onto the bus unchecked.
  - Fix: redirect validation to `SystemErrorV1` instead of skipping it. Required widening `SystemErrorV1.details` to `Union[str, Dict]` first, since the strict `Dict` type would otherwise reject the real `str`-typed payloads from 3 of 4 producers.
  - Evidence: `test_malformed_system_error_reply_is_still_rejected_on_a_different_schema_channel`.
- Finding: `orion/harness/finalize.py` uses `kind="system.error.v1"`, not matched by the original exact-string check.
  - Fix: `SYSTEM_ERROR_KINDS` recognizes both.
  - Evidence: `test_system_error_v1_kind_is_also_recognized`.
- Finding: `channels.yaml`/contract docs untouched despite the enforcement-semantics change (CLAUDE.md section 6).
  - Fix: added a top-of-file note in `orion/bus/channels.yaml`.
- Finding: `"system.error"`/`"SystemErrorV1"` literals duplicated across sites with no shared constant; `kind`/`payload` threaded as separate, unpaired variables.
  - Fix: `SYSTEM_ERROR_KINDS` constant in `bus_schemas.py`; `model is not SystemErrorV1` compares against the imported class, not a bare string; `kind, payload` are now assigned together in a single expression per branch.

**Round 2** (after the above fixes were applied):

- Finding: `services/orion-cortex-exec/app/executor.py:3255`'s `MetacogDraftService` step now hard-fails instead of masking an LLM-gateway error as a fabricated draft, with no test added for that call site.
  - Fix: not code-changed -- verified this is the correct, intended behavior (matches CLAUDE.md's explicit "no empty-shell cognition" rule) and that `status="fail"` is the same, already-established outcome every other service in that dispatch loop already returns on its own exceptions. Documented explicitly above under "Schema / bus / API changes" and in Risks below as a known, deliberate behavior change rather than adding a test that would just assert the existing generic exception-handling contract.
- Finding: `detail or payload` / `payload.get('error') or payload` silently discards a producer-sent empty-string error and falls back to dumping the whole payload.
  - Fix: `detail if detail is not None else payload` in both `cortex_client.py` and `clients.py`.
- Finding: `SYSTEM_ERROR_KINDS`'s docstring claimed it's checked by "every RPC client that must not treat an error reply as a successful result", but 4 pre-existing hardcoded `kind == "system.error"` checks weren't migrated.
  - Fix: migrated `orion/harness/substrate_client.py`, `services/orion-hub/scripts/bus_clients/tts_client.py` (x2), `services/orion-hub/scripts/thought_client.py` to `SYSTEM_ERROR_KINDS`.
  - Evidence: `services/orion-hub/tests/test_tts_client_errors.py` + `test_thought_client.py`, 6/6 passing post-migration.
- Finding: `_extract_kind_and_payload()` had exactly one call site and double-wrapped its own `ValueError` only to have the caller immediately discard the inner message and raise a differently-worded one.
  - Fix: inlined the extraction back into `_validate_payload()` (paired `kind, payload = ...` assignment per branch, same "can't desync" property, no pointless double-wrap).

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-whisper-tts/.env \
  -f services/orion-whisper-tts/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build
```

Every other service touched (`orion-harness-governor`, `orion-cortex-exec`, `orion-substrate-runtime`, `orion-llm-gateway`) imports the shared `orion.core.bus`/`orion.schemas` package and needs a rebuild+restart to pick up the fix, but none of them were mid-incident at review time -- restart on the next normal deploy is fine unless Juniper wants it sooner.

## Risks / concerns

- Severity: low
  Concern: `services/orion-cortex-exec/app/executor.py:3255`'s `MetacogDraftService` step now hard-fails (`status="fail"`) on an LLM-gateway `system.error` reply instead of silently degrading to a fabricated draft. This is the intended fix (matches this repo's explicit "no empty-shell cognition" rule), but it's a real behavior change with no dedicated test.
  Mitigation: verified the failure path is the same, already-established `except Exception -> StepExecutionResult(status="fail")` contract every other service in that dispatch loop already uses -- not a new/untested code path, just this call site correctly reaching it for the first time. Flagging for Juniper's awareness rather than adding a test that would only assert pre-existing generic exception handling.
- Severity: low
  Concern: this worktree's `services/orion-hub` and `services/orion-cortex-exec` have no local `.env` at all (pre-existing gap, not introduced by this patch), so several of their test files can't collect without hand-injected env vars, and the full 100+-test sweeps show pre-existing, unrelated failures from that gap and from cross-test `Settings()` pollution. All were diffed byte-for-byte against `HEAD` to confirm zero new failures from this patch (see Tests run).

## PR link

Branch pushed: `fix/tts-error-reply-schema-block`. Open a PR from this branch against `main` (`gh pr create` was not run in this session -- paste-ready body above).
