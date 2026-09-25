# GPU pool stage 4.1 — contracts + config

## Summary

This PR lays down the message shapes and config that the rest of stage 4 builds on. It changes no
behavior: nothing sends the new messages or fields yet.

- **Lease verbs.** A durable run's LLM calls can now be described as children of the run's hold:
  `GpuLeaseRequestV1` gains `attach` (with `hold_lease_id` + `hold_generation`) and a read-only
  `status`. There is also a new lease reference, `GpuLeaseRefV1 {lease_id, generation, role, holder}`,
  with its HTTP header codec `X-Orion-Gpu-Lease` (`orion/llm/resource_lease.py`).
- **Actuation.** There is now a way for the pool to tell a host "load role X on cards Y":
  `GpuActuateV1` / `GpuActuateResultV1`, registered on `orion:gpu_pool:actuate:request` and
  `:actuate:result` in `channels.yaml` and in both schema registries. These replace the stage-1
  placeholder, which had no producer and no consumer.
- **Pool events and card state.** New events `swap_started`, `swap_failed` and `actuate_refused`,
  and a new card state `fault`.
- **Config.** `config/gpu_pool.yaml` can now say how to start each role: `actuators`, a card
  `index`, and `launch` blocks. The bridge verbs `swap.load`/`unload` become optional (they must
  come as a pair), and there are optional per-seat `swap.after_wait_sec`/`guards` plus three new
  defaults. `launch_digest()` lets the pool and the actuator agree that they are reading the same
  launch config.
- **Gate.** `scripts/check_gpu_pool_config.py` now checks every `launch` block against the real
  compose file and its committed `.env_example`: the service exists, its profile, `LLM_ROLE` and
  port match, `cuda_env` is set, and the device matches the card's `index`.
- **Consumer-first fix in the pool.** The pool's lease dispatcher sent every verb other than
  acquire/heartbeat/release to **cancel**. With `status` in the schema, a durable run asking about
  its own hold would have ended it. `attach`/`status` now get `unavailable
  reason=verb_not_supported:<verb>` and touch nothing, until 4.3 implements them.

## Outcome moved

- The 4.2 (controller), 4.3 (pool engine) and 4.4 (gateway/Hub/field) PRs can each be written
  against a fixed, tested contract instead of inventing shapes on both sides at once.
- A latent failure mode is closed before it can fire: a `status` read would have cancelled a live
  lease.
- The one config fact the stage-4 actuator relies on, "which compose service starts this role and on
  which CUDA device", is now checked in CI instead of trusted.

## Current architecture

- `GpuLeaseRequestV1` verbs: acquire/heartbeat/release/cancel. The pool's `_on_lease` fell through
  to `cancel` for anything else.
- `GpuActuateV1 {action_id, target, role, cards}` and `GpuActuateResultV1 {action_id, ok, ...}`
  existed as unused placeholders. The request-channel constant existed, but neither channel was in
  `channels.yaml` and neither schema was registered.
- `config/gpu_pool.yaml`'s `swap` required `load` + `unload`. It had no way to say how a role is
  started.
- gpu2's 27B is still loaded by durable-runs' elastic decider via circe's
  `orion-gpu-lane-controller` HTTP route (unchanged here).

## Architecture touched

- Contracts: `orion/schemas/gpu_pool.py`, `orion/schemas/registry.py` (`_REGISTRY` and
  `SCHEMA_REGISTRY`, verified via `resolve()`), `orion/bus/channels.yaml`.
- Config: `config/gpu_pool.yaml` (copied into the pool and gateway images; each image carries a
  matching parser) and `orion/gpu_pool/config.py`.
- Pool service: `services/orion-gpu-pool/app/main.py` lease dispatch.
- Gate/CI: `scripts/check_gpu_pool_config.py`, `.github/workflows/orion-gpu-pool-tests.yml` trigger
  paths, and the metric-definitions lock (two new bus-channel metrics).

## Files changed

- `orion/schemas/gpu_pool.py`: attach/status verbs + hold fields with validator; `GpuLeaseRefV1`;
  `GpuActuateV1`/`GpuActuateResultV1` replaced per spec; new events; `fault`;
  `GPU_POOL_ACTUATE_RESULT_CHANNEL`.
- `orion/schemas/registry.py`: register both actuation schemas in both maps.
- `orion/bus/channels.yaml`: `orion:gpu_pool:actuate:request` and `:actuate:result`.
- `orion/llm/resource_lease.py`: `GPU_LEASE_HEADER`, `GPU_LEASE_OPTION`,
  `encode_gpu_lease_header`/`decode_gpu_lease_header`.
- `orion/gpu_pool/config.py`: `ActuatorSpec`, `CardSpec.index`, `LaunchSpec`/`DrainSpec`, optional
  bridge verbs, `after_wait_sec`/`guards`, new defaults, cross-ref rules, `swap_after_wait_sec()`,
  `launch_digest()`, `check_launch()`.
- `config/gpu_pool.yaml`: `actuators: {circe}`, `gpu2.index: 2`, and `launch` for `diffusion` and
  `agent-gpu2`.
- `scripts/check_gpu_pool_config.py`: runs `check_launch`.
- `services/orion-gpu-pool/app/main.py`: `dispatch_lease()`, with explicit answers for
  unsupported verbs.
- `services/orion-gpu-pool/README.md`: verb and channel table.
- `orion/gpu_pool/tests/test_stage4_contracts.py`: 60 test cases (round trips, rejections, the shipped
  config, compose-drift mutations, and the spec's gpu4 example).
- `services/orion-gpu-pool/tests/test_dispatch_stage4_verbs.py`: a regression test proving
  `status`/`attach` never reach cancel.
- `.github/workflows/orion-gpu-pool-tests.yml`: also triggers on the diffusion compose/env template
  and on `resource_lease.py`, which the gate and tests now read.
- `config/metrics/metric_definitions.lock.json`: re-locked (2 added bus-channel metrics).

## Schema / bus / API changes

- Added:
  - `GpuLeaseRequestV1.verb` values `attach` and `status`, plus the fields `hold_lease_id` and
    `hold_generation` (only valid on attach, and required there). `attach` also requires
    `request_id` and `work_class`, and `status` requires `lease_id`.
  - `GpuLeaseRefV1`.
  - `GpuPoolEventV1.event` values `swap_started`, `swap_failed` and `actuate_refused`.
  - `GpuCardStateV1.swap_state` value `fault`.
  - The channels `orion:gpu_pool:actuate:request` and `orion:gpu_pool:actuate:result`.
- Removed: the placeholder `GpuActuateV1.target` and `GpuActuateResultV1.ok`. Nothing produced or
  consumed them (grep-verified).
- Renamed: none.
- Behavior changed: the pool answers `attach`/`status` with `unavailable` instead of treating them
  as cancel. No producer sends either verb today.
- Compatibility notes (consumer-first, since every model is `extra="forbid"`):
  - `GpuLeaseRequestV1` is consumed by **orion-gpu-pool**.
  - `GpuPoolEventV1` is validated by **orion-sql-writer** (`MODEL_MAP`). The other listeners read
    raw dicts (`orion/gpu_pool/client.py`).
  - Hub's panel reads state and events as raw JSON and does not render `swap_state`. That makes the
    `fault` display a 4.3 item, since acceptance check 7 needs it.
  - The pool and the gateway each `COPY` `config/gpu_pool.yaml` into their image, so the YAML and
    its parser always ship together.

## Env/config changes

- Added keys: none (no `.env_example` touched).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.
- `config/gpu_pool.yaml` gained `actuators`, `cards.gpu2.index`, and two `launch` blocks. The new
  defaults and per-seat `after_wait_sec`/`guards` are **deliberately not written into the live
  file**. The parser accepts them with the spec's defaults, but nothing reads them until the 4.3
  scheduler. Writing `after_wait_sec: 1200` today would make the file claim a 1200s trigger while
  the observe-mode scheduler still uses 30s.

## Tests run

```text
PYTHONPATH=. pytest orion/gpu_pool/tests -q                          132 passed (72 before + 60 new cases)
cd services/orion-gpu-pool && pytest tests -q                       43 passed, 5 skipped (Postgres-only)
cd services/orion-sql-writer && pytest tests/test_gpu_pool_event_sql_shape.py -q   4 passed
cd services/orion-hub && pytest tests/test_gpu_pool_routes.py -q    10 passed, 1 skipped
cd services/orion-llm-gateway && pytest tests -q                    311 passed
pytest orion/llm/tests -q                                           10 passed
pytest tests/test_agent_trace_schema_registry.py tests/test_grammar_event_producer_catalog.py \
       tests/test_field_topology_edges.py tests/scripts/test_schema_skew_discovery.py -q   29 passed
python scripts/check_gpu_pool_config.py      ok (4 cards, 8 roles, 7 classes, 2 launch blocks)
python scripts/check_metric_lineage.py --gate     PASS
python scripts/check_definition_drift.py --gate   PASS (after --update: 2 added bus_channel metrics)
python scripts/check_bus_reply_channels.py        13 resolved, 0 uncovered
python scripts/check_inner_state_registry.py      OK
git diff --check                                  clean
```

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py      VERDICT: PASS
```

No eval covers hold or actuation behavior yet, because there is none to measure. The spec assigns
the hold/actuation eval extension (35-min and 7.8h holds, a failed load) to 4.3.

## Docker/build/smoke checks

```text
Not run. Not deployed (per instruction). No runtime path moves: the only live-code change is the
pool dispatcher's handling of two verbs nothing sends. Live effect is UNVERIFIED until deployed.
```

## Review findings fixed

Code review ran in a subagent. Every material and minor finding was fixed except #5 and #8, which
are recorded under Risks.

- Finding: the gate let the compose `${VAR:-d}` default beat the `.env_example` value, which is
  backwards from Compose. It missed real drift (template 9999 vs pool 8016 passed). One drift test
  also passed for the wrong reason.
  - Fix: `_resolve` now follows Compose precedence (`:-` means unset-or-empty falls back; `-` means
    unset falls back). The false-positive test now uses an unset var. Added "template beats
    default" and "default overridden to the right port is not drift".
  - Evidence: `test_gate_env_template_value_beats_compose_default`,
    `test_gate_template_default_equal_to_role_port_is_not_drift`.
- Finding: `parent_lease_id` would collide with the pool's replay-lineage key in the stored request
  dict.
  - Fix: renamed the fields to `hold_lease_id`/`hold_generation` (spec erratum 4).
  - Evidence: the schema and both test files were updated; 132 + 43 pass.
- Finding: `launch_digest` left out the role's port/kind, the actuator host, and the evicted roles'
  card indices.
  - Fix: all of them are now included. The docstring says the digest is over the parsed model, on
    purpose.
  - Evidence: `test_launch_digest_is_stable_and_moves_with_launch` (seat port change moves it).
- Finding: port parsing missed `ip:host:container`, `/tcp`, and long-form `published: ${VAR}`.
  - Fix: tokenize around `${...}` and resolve `published`.
  - Evidence: `test_gate_port_forms` (3 forms).
- Finding: `status` without `lease_id`, and `attach` without `request_id`/`work_class`, were
  accepted.
  - Fix: the validator now requires them.
  - Evidence: `test_status_needs_lease_id`, `test_attach_needs_idempotency_key_and_class`.
- Finding (nits):
  - Guard names were listed twice.
  - `restored` was allowed on a failed unload.
  - `export KEY=` in templates was not handled.
  - `LLM_ROLE` was compared unresolved.
  - Unpadded base64 was rejected.
  - Fix: each is fixed (`SwapGuard` Literal + `get_args`; failed-load-only; export stripped;
    `LLM_ROLE` resolved; re-pad before decode).
  - Evidence: covered in `test_stage4_contracts.py`.
- Finding: the sql-writer must be rebuilt before 4.3 emits the new events, and the Hub does not
  render `fault`.
  - Fix: recorded in Restart required and in spec erratum 2. The Hub `fault` rendering belongs to
    4.3 (acceptance check 7).

## Restart required

No restart is needed for correctness. This PR's consumers must be deployed **before** 4.2/4.3 start
producing:

```bash
# athena, from a worktree at the merged main, in this order (any order among the two is fine):
scripts/safe_docker_build.sh orion-gpu-pool up -d --build        # accepts attach/status safely; new YAML + parser
scripts/safe_docker_build.sh orion-sql-writer up -d --build      # accepts swap_started/swap_failed/actuate_refused
```

The gateway and Hub need no rebuild for 4.1. The gateway rebuilds anyway in 4.4, where it starts
reading `GpuLeaseRefV1`.

## Risks / concerns

- Severity: low
  - Concern: `check_launch` resolves `${VAR}` from the committed `.env_example`, not circe's live
    `.env`. A live override (e.g. `HOST_PORT`) can differ from the template.
  - Mitigation: the template is the operator contract. The 4.2 actuator should re-check against its
    own environment at start.
- Severity: low
  - Concern: `launch_digest` covers, for the seat and each evicted role, the kind, port, card
    indices, launch and actuator host, plus the seat's bridge verbs. It does not cover the compose
    file's *contents*, so an edit to a compose service does not move it.
  - Mitigation: the actuator runs from its own checkout. The digest guards the pool/actuator
    agreement about *which* service, not the service body.
- Severity: medium, but it must be decided before 4.2
  - Concern: `atlas-agent-burst` hard-codes `CUDA_VISIBLE_DEVICES_OVERRIDE=2` as a literal. A
    stage-5 actuator that "sets `cuda_env` from the cards' index" through the shell environment
    would have no effect on that service. Diffusion's `${CUDA_VISIBLE_DEVICES}` does work.
  - Mitigation: stage 4's bridge uses compose as-is, and the gate already checks that the literal
    equals `index`. Before stage 5, either make the compose value interpolated or make the gate
    require `${...}`.
- Severity: low
  - Concern: no CI workflow runs the llm-gateway test suite. `orion/llm/resource_lease.py` now
    imports `orion.schemas.gpu_pool`, so a schema change could break a gateway import without that
    suite running.
  - Mitigation: the gpu-pool workflow now triggers on `resource_lease.py` and imports it in
    `test_stage4_contracts.py`. The gateway suite (311 passed) was run locally.

## Spec errata found

1. **gpu4 worked example fails validation as written.** `fast2` is given `owner: [metacog, fast]`
   but only class `fast` lists it. The existing rule "owner class must list the role" refuses that.
   Fix: add `fast2` to `classes.metacog.roles` too. Pinned by
   `test_spec_gpu4_example_as_written_is_rejected_because_metacog_does_not_list_fast2`.
2. **4.1 deploy list is wrong.**
   - The spec says "pool, gateway, Hub".
   - The real forbid-model consumers of changed shapes are **orion-gpu-pool** (lease requests) and
     **orion-sql-writer** (it validates every `GpuPoolEventV1`, so `swap_started` from a 4.3 pool
     would be rejected by an old writer).
   - The gateway and Hub parse neither changed model.
3. **The pool's cancel fall-through** was not in the spec. Adding `status` to the schema without the
   dispatcher change would have turned a read into a cancel.
4. **`parent_lease_id` on the request would collide with replay lineage.** The spec adds
   `GpuLeaseRequestV1.parent_lease_id`/`parent_generation` for the hold. But the pool's stored
   request dict already uses `parent_lease_id` for dead-letter replay lineage
   (`runtime.py` replay path and `_row()`), and `acquire` stores `req.model_dump()` in that dict.
   This PR names the fields **`hold_lease_id`/`hold_generation`**. The hold id is still expected to
   land in the `gpu_pool_leases.parent_lease_id` *column* for children (4.3 decides how that column
   coexists with replay lineage). `lease_id` is forbidden on attach, because the pool mints the
   child's id.
5. **`cuda_env` assumes the actuator can set the device.** For `atlas-agent-burst` it cannot (the
   value is a compose literal). See Risks.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
