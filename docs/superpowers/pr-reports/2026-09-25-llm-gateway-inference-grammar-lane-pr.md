# PR: the LLM gateway reports on its own inference calls (llm_inference grammar lane)

## Summary

- Until now, Orion's sense of "can I think right now" (`capability:llm_inference`) was
  guessed entirely from the outside: circe's GPU and memory load, plus how heavy
  recent reasoning runs were. The part that actually runs the models never reported
  anything about itself.
- The LLM gateway now sorts every bus-RPC reply by what really happened: answered,
  the model backend failed (timeout, refused connection, HTTP error), the gateway
  itself declined, or the request was unusable. It counts these per serving node and
  publishes one small grammar trace per minute (`llm_gateway.inference:`) on
  `orion:grammar:event`. Counts only, never prompt or reply text.
- A new substrate reducer (`orion/substrate/llm_inference_loop/`) turns each minute
  into one reading per node: the share of calls that reached a backend and came back
  with no answer (`inference_failure_pressure`).
- The field digester writes that onto `node:circe`, and the circe edge carries it to
  `capability:llm_inference` `reliability_pressure`, a channel on that capability
  that has had no input at all until now.
- Three switches, all **off by default**: deploying this changes nothing live until
  Juniper flips them in order (below).

## Outcome moved

`capability:llm_inference` could only ever say "busy", never "broken". A failed
backend call comes back to the caller as a normal reply whose text is
`[Error: llamacpp failed: ...]` (`llm_backend.py` has ~16 such returns), so:

- the caller's RPC health counts it as a success, because a reply arrived;
- an idle, crashed llama.cpp worker reads as *low* GPU pressure, which looks calm;
- `reasoning_load` only exists for runs that produced output.

Every existing input read "calm" in exactly the failure it most needs to catch. After
the flip, a backend that answers with errors pushes `capability:llm_inference`
`reliability_pressure` up, with provenance `node:circe`.

## Current architecture

- `capability:llm_inference` came from `node:circe` over one edge: `gpu_pressure` and
  `memory_pressure` fed `pressure`, and `reasoning_load` fed `reasoning_pressure`.
  `reasoning_load` is attributed to the serving node via `llm_serving_node`, which
  cortex-exec scrapes from the gateway's reply `meta.served_by`
  (`services/orion-field-digester/app/ingest/state_deltas.py:317-327`).
- The gateway published nothing about its own outcomes. `admission_ledger.py` is
  in-memory and only reachable over HTTP at `GET /admission`.
- orion-gpu-pool emits grammar only for lease exceptions (`gpu_pool.lease:`); it never
  sees backend latency, tokens, errors or `served_by`.
- The signal registry has an `llm_gateway` organ entry, but no adapter feeds it
  (registry only). It was left alone.

## Architecture touched

- **Producer**: `services/orion-llm-gateway/app/grammar_emit.py` (new), hooked into
  `main.py::handle_chat` after dispatch. A window publisher task is started in
  `main()` only when the flag is on.
- **Contract**: `orion/schemas/llm_inference_projection.py`. It holds the wire
  constants (trace prefix, roles, outcome classes) and the projection models. It sits
  outside `orion/substrate` on purpose, so the gateway never imports the substrate
  package.
- **Reducer**: `orion/substrate/llm_inference_loop/` with constants, extract,
  reducer and pipeline.
- **substrate-runtime**: `REDUCER_SPECS[5]` `llm_inference`, its own cursor
  `llm_inference_grammar_reducer`, a poll loop, `_llm_inference_tick`, store
  fetch/advance/load/save, and grammar_truth maps.
- **field-digester**: new node channel `inference_failure_pressure` (written with
  `mode="replace"`, deliberately does not decay), plus a `llm_inference_node` branch in
  `delta_to_perturbations` behind a per-lane gate `delta_digestion_enabled()`.
- **Topology**: `node:circe -> capability:llm_inference` gains
  `inference_failure_pressure: reliability_pressure` in both lattice files.
- **sql-writer**: the `GRAMMAR_LANES` retention mirror gains the new lane.

### Design choices worth knowing

- **One atom per serving node per window, not one per call.** Live volume is about
  1,207 calls in 3.2 h (~9k/day). One atom per node keeps grammar at 2-3 events per
  minute. Each node's reading is also self-contained, so a reducer batch boundary can
  never split one node's counts.
- **"Not measured" is never written as calm.** A node with no upstream traffic in a
  window gets no hint, so nothing reaches the field. An all-refused window gets the
  same treatment: the backend was never asked.
- **Refusals are counted but never wired.** `gateway_overloaded`,
  capacity/lease rejects, operator-closed and `llm_route_unavailable` stay in the
  projection for inspection. Admission and lease belong to the GPU pool
  (gpu-pool spec stage 3 moves or deletes them), so they are not duplicated into the
  field.
- **Only the backend's fault counts.** Upstream 4xx (e.g. an oversized prompt),
  image-on-text-route refusals and unreadable attachments count as `request_invalid`.
  Model prose that merely starts with "Error:" is `served`; only `[Error: ...`
  gateway framing is ever a failure.
- **The reading holds instead of decaying.** Decay would fade a real failure to a
  fake calm 0.0 whenever callers stop calling circe (the `node:substrate.route`
  decayed-to-zero artifact). With replace-mode writes, the next window that has
  traffic moves it back down, so holding does not ratchet.
- **`upstream_empty` is not counted as a failure.** A reply that is empty with no
  reasoning content is recorded but excluded, because how common empty completions are
  at rest has not been measured live.
- **Node attribution** uses the same `{node}-worker...` convention and the same known
  node set as cortex-exec's `_normalize_served_by_to_node`. Unknown labels
  (`atlas-*`, `None`) count as unattributed and never create a phantom field node.

## Metric quality gate: `inference_failure_pressure`

1. **Provenance.** `grammar_emit.classify_outcome()` runs on the exact reply dict
   `handle_chat` returns (`main.py`, right after `_dispatch_chat`). From there:
   - `_NodeBucket` counts it;
   - `build_window_events()` puts it in the atom summary;
   - `llm_inference_loop/extract.py::inference_failure_pressure()` computes
     `upstream_failed / (served + upstream_failed)`;
   - `state_deltas.py` turns it into a perturbation;
   - `diffusion.py` carries it over the circe edge.
2. **Independence.**
   - vs `gpu_pressure`/`memory_pressure`: those come from a different sensor (host
     biometrics). In the target failure (worker crashed or refusing) they move the
     *opposite* way, because an idle card reads low.
   - vs `reasoning_load`: that comes from cortex-exec run shape, and only for runs that
     produced output.
   - vs rpc-health (`RpcHealthSnapshotV1`) and `rpc_transport_timeout` grammar: those
     are caller-side deadline misses. They count an `[Error: ...]` reply as a success,
     and a caller timing out says nothing about whether the backend later answered.
   - No short causal chain links it to anything already in the model.
3. **Theory anchor.** This is the error rate of a request-serving system: the "Errors"
   signal of the SRE golden signals / RED method. It is measured at the only hop that
   sees the backend's answer, and it measures exactly the thing it names.
4. **Live sanity.**
   - Rest state reached: replaying the live gateway log
     (`evals/run_inference_outcome_eval.py`, 3.2 h, 1,207 calls, 131 one-minute
     windows with traffic) gives 0.0 in 131 of 131 windows. It is a real, measured
     zero, not a decayed one.
   - The failure path exists in production: `chat_history_log` holds 2 replies from
     2026-08-14 starting `[Error: llamacpp failed: Client error '4..`. Those would
     classify as `upstream_http_4xx`.
   - **Movement to a nonzero reading live: UNVERIFIED.** No backend failure happened
     inside the retained log window. It is proven only by tests, and for real only once
     the flags are flipped. The gateway's older logs are gone, and non-chat lanes'
     error replies were never persisted anywhere.
5. **Existing-mechanism check.**
   - gpu-pool grammar covers lease exceptions only.
   - rpc-health is the caller's view.
   - The admission ledger is HTTP-only, and its admission data is not duplicated here.
   - The signal-registry `llm_gateway` organ has no adapter.
   - No existing mechanism reports backend outcomes from the gateway.
6. **Reversibility.** It is cheap to undo:
   - three flags off stops it at any stage;
   - the channel is one line in `channels.py` and in each lattice file, plus one
     glossary entry;
   - the projection table is standalone;
   - the anomaly autoencoder reads a fixed trained manifest, so this channel never
     enters its input width.
   - The metric-definition lock was re-locked: one routing change (new
     `orion:grammar:event` producer) and one added field channel.

Channels considered and **not** wired: latency / time-to-first-token and throughput.
Latency is dominated by `max_tokens` and prompt size. Without a per-workload
normalization there is no theory anchor for "high", so a pressure built on it would be
a knob, not a finding. The chat path is non-streaming, so TTFT does not exist here.
Latency p50/p95 and token totals are kept in the projection for inspection only.

## Existing signals this does not duplicate (and what it leaves alone)

- GPU-pool lease/queue wait (`gpu_pool:<class>#gpu_pool_wait` rpc-health, and
  `gpu_pool.lease:` grammar): untouched. Gateway refusals are counted, not wired.
- Caller-side RPC timeouts: untouched. For the record, grammar shows 62, 147 and 32
  `LLMGatewayService` RPC timeouts on 09-22, 09-23 and 09-24, not ~2,700/day.
- `llm_serving_node` scrape for `reasoning_load`: kept. It is a per-run join key,
  not a health reading, so it is not redundant with this lane. Stage 3 of the gpu-pool
  plan ("`served_by` comes from the grant") changes the source of `served_by` for both
  the scrape and this lane at once. Neither needs code changes for that, since both
  read `result["served_by"]`.
- **orion-heartbeat drops this source.** `routing.py`'s `ORGAN_SITE_MAP` holds five
  fixed organs on five fixed boundary sites. An unknown `source_service` raises
  `UnroutableOrganError`, and `service.py:201-203` skips the event, only incrementing
  `events_skipped_organ` with no log line. Adding a sixth organ means changing the
  site layout, so it is not cheap and was not done.

## Overlap with open PRs

- **#2324** deletes `config/substrate-lattice/grammar_producer_registry.v1.yaml`
  because nothing reads it. This PR therefore does not touch that file. #2324 also
  adds a static gate saying `orion:grammar:event` producers must match the code's
  `GrammarProvenanceV1(source_service=...)` sites. The emitter uses a module-level
  literal `SOURCE_SERVICE = "orion-llm-gateway"` so that gate can resolve it, and it
  should pass once both PRs merge.
- **#2323** touches `state_deltas.py` and `channels.py` in different hunks. Whichever
  PR merges second rebases.

## Sequencing note (gpu-pool spec)

`docs/superpowers/specs/2026-09-24-gpu-pool-design.md` puts "gateway per-call
telemetry and grammar reducers" at stage 6, after the gateway, durable-runs and gpu2
cutovers ("per Juniper"). Only stages 1-2 have landed (#2318-#2321). This PR builds
that lane early, at Juniper's explicit "hit it all". It keeps to what survives the
cutovers:

- it attributes by `result["served_by"]`, whatever source that field has;
- it wires only backend failures;
- it leaves admission and lease signals to the pool.

The spec's planned rpc-health hop `llm:<role>#call` for model latency is still open,
and it complements this lane rather than replacing it.

## Files changed

- `services/orion-llm-gateway/app/grammar_emit.py`: classifier, window recorder, event
  builder, publisher (new).
- `services/orion-llm-gateway/app/main.py`: record each reply when the flag is on;
  start the publisher.
- `services/orion-llm-gateway/app/settings.py`, `.env_example`, `README.md`: flag and
  window keys, and docs.
- `services/orion-llm-gateway/tests/test_grammar_emit.py`: classifier pinned to the
  real error strings, aggregation, no-text-leak, publisher, `handle_chat` hook.
- `services/orion-llm-gateway/evals/run_inference_outcome_eval.py` and
  `test_inference_outcome_eval.py`: log-replay eval (new).
- `orion/schemas/llm_inference_projection.py`, `orion/schemas/registry.py`: contract
  and projection schemas.
- `orion/substrate/llm_inference_loop/*`: reducer package.
- `services/orion-substrate-runtime/app/{worker,store,settings,grammar_truth}.py`,
  `.env_example`, `docker-compose.yml`, `README.md`: lane wiring.
- `services/orion-substrate-runtime/tests/test_worker_llm_inference_tick.py`: new.
  `test_worker_independent_reducers.py` had a poll-task count already stale and failing
  on main; it is pinned to the real set now.
- `services/orion-sql-db/manual_migration_llm_inference_substrate_loop.sql`: projection
  table and cursor seed.
- `services/orion-field-digester/app/{tensor/channels.py,digestion/decay.py,ingest/state_deltas.py,settings.py,worker.py}`,
  `.env_example`, `docker-compose.yml`, `README.md`: channel, gate, perturbation.
- `services/orion-field-digester/tests/test_field_llm_inference_perturbations.py`:
  new, including the diffusion check against both real lattice files.
- `config/field/orion_field_topology.v1.yaml`, `config/field/biometrics_lattice.yaml`:
  circe edge channel_map.
- `config/field/field_channel_glossary.v1.yaml`, `tests/test_field_channel_glossary.py`:
  glossary entry and count.
- `orion/bus/channels.yaml`: `orion-llm-gateway` added as a producer of
  `orion:grammar:event`.
- `services/orion-sql-writer/app/grammar_truth.py` and
  `tests/test_grammar_retention_periodic.py`: retention lane mirror.
- `scripts/sync_local_env_from_example.py`: prefixes `LLM_GATEWAY_GRAMMAR_`,
  `ENABLE_LLM_INFERENCE_`, `LLM_INFERENCE_`. None of the new keys was synced before.
- `scripts/check_substrate_projection_schema_drift.py`: the new singleton projection
  row is covered.
- `config/metrics/metric_definitions.lock.json`: re-locked.
- `tests/test_llm_inference_substrate_reducer.py`: round trip that uses the gateway's
  real emitter to build its input.

## Schema / bus / API changes

- **Added:**
  - `LlmInferenceNodeStateV1` and `LlmInferenceProjectionV1` (registered);
  - grammar trace prefix `llm_gateway.inference:` with roles
    `llm_inference_window_observed` and `llm_gateway_window_completed`;
  - `StateDeltaV1.target_kind="llm_inference_node"`;
  - node channel `inference_failure_pressure`;
  - `orion-llm-gateway` as a producer on `orion:grammar:event`;
  - table `substrate_llm_inference_projection`.
- **Removed / renamed:** none.
- **Behavior changed:** none while the flags are off. The gateway's reply payload is
  unchanged either way.
- **Compatibility:** the existing channel and envelope kind (`grammar.event.v1`)
  are reused. sql-writer already persists every source.

## Env/config changes

- **Added keys:**
  - `LLM_GATEWAY_GRAMMAR_ENABLED=false` and `LLM_GATEWAY_GRAMMAR_WINDOW_SEC=60`
    (orion-llm-gateway);
  - `ENABLE_LLM_INFERENCE_REDUCER=false` and `LLM_INFERENCE_GRAMMAR_BATCH_LIMIT=200`
    (orion-substrate-runtime);
  - `ENABLE_LLM_INFERENCE_FIELD_DIGESTION=false` (orion-field-digester).
- **Removed / renamed:** none.
- **`.env_example` updated:** yes, all three services. Compose `environment:` updated
  for substrate-runtime and field-digester; the gateway uses `env_file`.
- **Local `.env` synced:** yes, with `python scripts/sync_local_env_from_example.py`.
  All 5 keys landed in the primary checkout's service `.env` files.
- **Skipped keys requiring operator action:** none.

## Tests run

```text
services/orion-llm-gateway/tests + evals               401 passed
services/orion-llm-gateway/evals                      2 passed
tests/test_llm_inference_substrate_reducer.py         19 passed
services/orion-field-digester/tests                   236 passed (from repo root)
services/orion-substrate-runtime/tests (non-integration, POSTGRES_URI dummy)
    branch 9 failed / 350 passed vs parent 10 failed / 349 passed -- identical
    failure set minus the stale poll-task test this PR fixes (all env/DB-bound)
services/orion-sql-writer/tests (non-integration)     identical pass/fail set to parent
tests/test_metric_definition_drift.py                 47 passed (after re-lock)
tests/test_check_substrate_projection_schema_drift.py 18 passed
tests/test_field_* (digester-path suites)             identical 7 pre-existing failures as parent
scripts/check_metric_lineage.py --gate                PASS
scripts/check_definition_drift.py --gate              PASS after re-lock
scripts/check_inner_state_registry.py, check_sentience_instruments.py --static-only,
check_system_health_producers.py, check_env_key_single_source.py,
check_compose_no_relative_mounts.py, check_async_routes_not_blocking.py   all OK
check_service_env_compose_parity.py orion-field-digester / orion-llm-gateway   OK
check_service_env_compose_parity.py orion-substrate-runtime   18 pre-existing BRAIN_FRAME_* gaps, none new
```

## Evals run

```text
docker logs --timestamps orion-llm-gateway | python services/orion-llm-gateway/evals/run_inference_outcome_eval.py
  134 one-minute windows with traffic, circe: 134/134 at 0.0, max 0.0 (rest state reached;
  nonzero movement UNVERIFIED live -- no backend failure in the retained window)
```

## Docker/build/smoke checks

```text
Import smoke inside the live gateway image (worktree code bind-mounted read-only, --network none,
no image rebuilt or retagged): app.main imports, flag reads True, a timeout reply classifies to
"node=circe calls=1 served=0 upstream_failed=1 ... classes=upstream_timeout:1".
docker compose config (worktree compose + live env files): substrate-runtime renders
ENABLE_LLM_INFERENCE_REDUCER="false", LLM_INFERENCE_GRAMMAR_BATCH_LIMIT="200"; field-digester
renders ENABLE_LLM_INFERENCE_FIELD_DIGESTION="false".
No image was built and nothing was deployed.
```

## Review findings fixed

The review ran as a subagent against `origin/main...HEAD` and found no blockers.

- **Finding:** model prose ("Connection refused usually means...", "Error: that
  value...", "A read timeout happens when...") was classified as a backend failure,
  because the canonical detector matches markers anywhere in the head.
  - **Fix:** only `[Error:`-framed text can be a failure. Everything else is `served`.
  - **Evidence:** three prose cases were added to `test_classify_outcome`; they pass.
- **Finding:** image-on-text-route refusals were counted as `upstream_error`.
  - **Fix:** they now map to `request_invalid`.
  - **Evidence:** two new classifier cases.
- **Finding:** upstream 4xx (oversized prompt) raised circe's failure pressure.
  - **Fix:** `upstream_http_4xx` moved to `REQUEST_INVALID_CLASSES`.
  - **Evidence:** `test_upstream_4xx_is_a_bad_request_not_a_node_failure`.
- **Finding:** decay would fade a real failure to a fake calm 0.0 during idle time.
  - **Fix:** removed from `NODE_DECAY_CHANNELS`, so the reading holds.
  - **Evidence:** `test_failure_reading_survives_idle_minutes_and_clears_on_next_window`
    (300 idle ticks leave 1.0; the next window's 0.0 clears it).
- **Finding:** calls that raised inside dispatch were never counted.
  - **Fix:** they are recorded as `gateway_exception` (unattributed), then re-raised
    unchanged.
- **Finding:** the log-replay eval turned `served_by=None` into a fake node.
  - **Fix:** it now maps to `unrouted`. The eval docstring also states its low-bias
    gaps.
- **Finding:** a mid-window publish failure dropped the rest silently.
  - **Fix:** the log line now carries `dropped=N of M`.
- **Documented, not changed:**
  - timeouts caused by a short caller budget;
  - latency includes admission wait;
  - one gateway per node assumed;
  - no shutdown flush;
  - a window split across reducer batches mis-states `window_sec` and
    `last_unattributed_calls` (cosmetic; the pressure value is correct).

## Restart required

Merging and deploying with the defaults changes nothing. To turn the lane on, run
these in order and check each stage before starting the next. Run them from the
primary checkout after merge and `git pull`, because the live containers are deployed
from there (compose `working_dir` labels). Deploying from a worktree would pin that
worktree as production. Juniper runs these; nothing here was executed:

```bash
cd /mnt/scripts/Orion-Sapienform
# 0. table + cursor
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_llm_inference_substrate_loop.sql

# 1. gateway starts publishing (services/orion-llm-gateway/.env: LLM_GATEWAY_GRAMMAR_ENABLED=true)
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-llm-gateway up -d --build
#    proof, after ~2 min: rows from the new source, one trace per minute
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "select count(*), count(distinct trace_id), max(created_at) from grammar_events
   where source_service='orion-llm-gateway' and trace_id like 'llm_gateway.inference:%'
   and created_at > now()-interval '10 minutes'"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "select event_json->'atom'->>'summary' from grammar_events
   where source_service='orion-llm-gateway' order by created_at desc limit 3"

# 2. reducer (services/orion-substrate-runtime/.env: ENABLE_LLM_INFERENCE_REDUCER=true)
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
#    proof: cursor moves, receipts carry llm_inference_node deltas, projection row exists
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "select last_event_created_at from substrate_reduction_cursor where cursor_name='llm_inference_grammar_reducer'"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "select created_at, d->>'target_id', d->'after'->'pressure_hints', d->'after'->'outcome_classes'
   from substrate_reduction_receipts, jsonb_array_elements(receipt_json->'state_deltas') d
   where d->>'target_kind'='llm_inference_node' and created_at > now()-interval '10 minutes'
   order by created_at desc limit 5"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "select generated_at, projection_json->'nodes'->'llm_node:circe'->>'calls' from substrate_llm_inference_projection"

# 3. field (services/orion-field-digester/.env: ENABLE_LLM_INFERENCE_FIELD_DIGESTION=true)
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-field-digester up -d --build
#    proof: node channel present and capability provenance names node:circe
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "select field_json->'node_vectors'->'node:circe'->>'inference_failure_pressure',
          field_json->'capability_vectors'->'capability:llm_inference'->>'reliability_pressure',
          field_json->'capability_provenance'->'capability:llm_inference'->>'reliability_pressure'
   from substrate_field_state order by created_at desc limit 1"
```

At rest the stage-3 proof should read `0.0 | 0.0 | node:circe`. The provenance
entry is what separates "measured healthy" from "never measured". Nonzero movement
only shows when a backend actually fails.

## Risks / concerns

- **Severity: medium.** Sequencing: this is the gpu-pool spec's stage-6 work, built
  before stages 3-5. It depends only on `result["served_by"]`, so the cutovers should
  not break it. Juniper should still confirm the ordering is acceptable.
- **Severity: medium.** Nonzero readings have not been seen live (see gate item 4).
  The classifier is pinned to today's error strings. A new error return with different
  wording still reads as `upstream_error` via the canonical `looks_like_error_text`,
  never as served.
- **Severity: low.** If callers stop calling a dead backend, the last failing reading
  holds indefinitely. It is stale, but never falsely calm; its age is
  `node_vector_updated_at`.
- **Severity: low.** `upstream_timeout` also counts timeouts where the read timeout was
  the caller's own short leftover budget, so a short-budget caller on a busy but
  healthy lane can register one. Documented in the README; not split out yet.
- **Severity: low.** State is keyed by serving node only. This assumes one gateway
  reports on a node, which is true today. A second gateway would overwrite the first
  gateway's windows.
- **Severity: low.** Model output that itself begins with the literal `[Error:`
  would count as a failure. Plain "Error: ..." prose does not, and a test pins that.
- **Severity: low.** The OpenAI/Anthropic HTTP passthroughs (AI Town, operator tools)
  are not counted, only the bus path.
- **Severity: low (pre-existing, not this PR).** `sync_local_env_from_example.py`
  prints diverged values in plain text, including one service's GitHub token. Worth a
  follow-up that masks secret-looking values.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2327

🤖 Generated with [Claude Code](https://claude.com/claude-code)
