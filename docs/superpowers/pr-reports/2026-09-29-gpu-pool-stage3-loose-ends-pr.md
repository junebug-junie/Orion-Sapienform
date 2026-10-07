# GPU pool stage 3 loose ends: the FCC latency key and Orion's "which model am I" line follow the granted role

## Summary

- Since the GPU pool went live, one request for the agent lane can be served by a different card and model when agent is busy (this is called spill). Two readers still assumed the route table was the whole truth. This patch fixes both, and closes a third item with evidence and no code.
- **FCC latency key (spec item 2):** the harness governor used to record the FCC motor's wall time under the name of whatever model the CLI echoed back (`fcc:<served_model>`). It now records it under the GPU pool role the turn was granted (`fcc:<role>`). A turn with no pool hold is recorded under the route it asked for (`fcc:route:<route>`). The old model-name key is retired.
- **Self-knowledge (spec item 5):** Orion used to be told "Backend model currently serving this turn: X", and X came from the gateway's `/routes` default. That line is now built from the turn's lease: granted role, then the model the pool discovered on that role. With no lease, the prompt names the route's default model and says outright that it is a default, not a confirmation. The Hub situation brief gets the same fix.
- **exec→gateway bus-synaptic reset (spec item 1):** closed with no code. The persisted data shows a small mesh-wide bump around the cutover that was back to baseline within about 3 hours. The live edge is not anomalous now. Along the way this turned up a real deviation from the spec (see "Risks"): pool events travel on the turn's correlation id.

## Outcome moved

- Under spill, Orion is no longer told a false model name. Tests cover both a spilled held turn (agent class granted `agent-gpu2`, and granted `chat`) and an unheld turn.
- FCC hop keys no longer grow with spill. They are capped by the roles and routes in `config/gpu_pool.yaml`, and junk keys like `fcc:<synthetic>` can no longer be minted.

## Current architecture

- `services/orion-harness-governor/app/bus_listener.py::fcc_hop_key` keyed the hop by the CLI-echoed model, falling back to the `/routes` model probed before the turn (`HarnessMotorResult.probed_served_model`).
- `orion/harness/runner.py` probed `GET /routes` for the label's route and passed `current_served_model` to `orion/harness/prefix.py`, which rendered "Backend model currently serving this turn: X".
- `orion/situational/context.py::_fetch_runtime_context` read `/routes` for `ORION_SITUATION_RUNTIME_ROUTE` (default `chat`) and rendered "You are currently running on model: X (route=chat)". This happened even on agent-mode turns and on held turns running on another role.

## Architecture touched

- New `orion/gpu_pool/placement.py`, which does four things:
  - holds a `ServingPlacement` with two shapes: `from_lease` and `route_default`;
  - renders its single self-line;
  - provides `discovered_role(state, role)`;
  - provides `fetch_pool_state(bus)`, a single `orion:gpu_pool:state:request` RPC with `include_leases=false`, a 2 s timeout and fail-open behaviour.
- Harness runner: for a held turn it reads pool state for `gpu_lease.role`; otherwise it uses the old `/routes` probe, worded as a default. It also records `serving_role` and `fcc_route` on `HarnessMotorResult`.
- Situation brief: when `ctx["gpu_placement"]` is present it wins and no network read happens. Hub fills it from its already-running `orion:gpu_pool:state` feed, so there is no extra RPC. The per-session brief cache key now includes the placement.

## Files changed

- `orion/gpu_pool/placement.py` (new): the placement model, its self-line, and the pool-state read.
- `orion/harness/runner.py`: probes the placement and records `serving_role`/`fcc_route`. Retires `probed_served_model`.
- `orion/harness/prefix.py`: renders `serving_placement` in place of `current_served_model`.
- `orion/harness/fcc_motor.py`: `resolve_fcc_route_key()` is split out of `probe_route_runtime` (same logic, reused for the hop key).
- `services/orion-harness-governor/app/bus_listener.py`: new `fcc_hop_key(serving_role, fcc_route)`.
- `orion/schemas/situation.py`: `RuntimeContextV1` gains `placement`, `granted_role` and `profile_name`, all optional with defaults.
- `orion/situational/context.py`: adds `_runtime_from_gpu_placement`, `_runtime_line` and the cache-key change.
- `orion/hub/turn_orchestrator.py`: new `_turn_gpu_placement(payload)`. The only hunk is in `_build_situation_prompt_fragment`, plus one import. This keeps the file clear of the concurrent ResourceLeaseV1 removal.
- `orion/bus/channels.yaml`: harness-governor is now a producer on `orion:gpu_pool:state:request` and a consumer of `orion:gpu_pool:state:reply:*`.
- `config/metrics/metric_definitions.lock.json`: re-locked (see below).
- Docs and comments: `orion/core/bus/rpc_health.py`, `orion/core/bus/async_service.py`, `orion/schemas/telemetry/rpc_health.py`, harness-governor `README.md`/`settings.py`/`main.py`, cortex-exec `README.md`.
- Tests:
  - `orion/gpu_pool/tests/test_placement.py` (new)
  - `services/orion-hub/tests/test_unified_turn_gpu_placement.py` (new)
  - `orion/harness/tests/test_harness_{runner,prefix}.py`
  - `services/orion-harness-governor/tests/test_rpc_health_publish.py`
  - `services/orion-cortex-exec/tests/test_situation_provider.py`
- `.github/workflows/orion-gpu-pool-tests.yml`: the tests above run in the `consumers` job, and the job's path filter now covers the touched files.

## Schema / bus / API changes

- Added: `RuntimeContextV1.placement` (`"route_default"` | `"lease"`, default `"route_default"`), `granted_role` and `profile_name` (both optional). `SituationBriefV1` is only built in-process; nothing re-validates it from JSON, so an old/new deploy mix is safe.
- Removed: `HarnessMotorResult.probed_served_model`. This is an in-process dataclass and its only reader was the hop key.
- Bus: `orion:gpu_pool:state:request` gains a producer (harness-governor), and `orion:gpu_pool:state:reply:*` gains a consumer. The payload is `GpuPoolStateRequestV1(include_leases=False)`, the existing schema and field. A pool that predates the field would reject it (`extra=forbid`), but the pool has had `include_leases` since stage 1.
- Behaviour changed: the prompt wording for the model line, in both the harness prefix and the situation brief.
- Compatibility: `compile_harness_prefix(current_served_model=...)` and `build_harness_prompt(current_served_model=...)` are replaced by `serving_placement=`. Every caller in the repo is updated (grep confirms none are left).

## Metric quality gate: FCC hop key re-key (`fcc:<served_model>` → `fcc:<role>`)

1. **Provenance.**
   - The value is `HarnessMotorResult.fcc_elapsed_sec`: the FCC leg's wall time, measured from `fcc_started` at the top of `HarnessRunner.run()` (`orion/harness/runner.py`). It is recorded by `record_fcc_hop` (`bus_listener.py`) into `channel_latency` on the governor's dispatch-bus RPC-health aggregator.
   - `orion-equilibrium-service` folds it into the transport baseline (`orion/metacog/transport_baseline.py`), keyed `(service, instance, hop)`.
   - Only the key changes. The value, the success/timeout/skip rules and the producer are unchanged.
2. **Independence.** This is a re-key of one existing signal, not a new one. It does not duplicate any other hop. The gateway-side passthrough latency is a different leg and is not in this service.
3. **Theory anchor.** The transport baseline learns "normal latency per hop". One baseline has to mean one latency population, which is the rule in `rpc_health.py`: "one EWMA baseline per key means the same thing everywhere". A held turn runs every call on its granted role, so the role is the thing that decides which population a sample belongs to. The echoed model name is not: it split one role into several keys, and it also produced `<synthetic>`.
4. **Live data (2026-09-29 ~05:02 UTC, `equilibrium:transport_baseline_state:v1` on the bus Redis).** There are exactly three FCC keys today:
   - `fcc:Qwen3.6-35B-A3B-UD-Q5_K_M`
   - `fcc:Qwen3.8-27B-UD-Q4_K_XL`
   - `fcc:<synthetic>` (last real sample 2026-09-27 11:11 UTC)

   All three have `fast_count=0`, `level_count=0`, no floor and no episodes. **In practice the transport baseline has never evaluated an FCC hop.** One FCC turn lasts minutes (the live `pend_max_ms` is 172 s). The reducer needs 5 successes inside 20 pooled 30-second windows, and FCC traffic never reaches that. This is a separate degenerate-signal finding, not fixed here; see Risks. It also means the re-key throws away no learned baseline.
5. **Existing mechanism.** The same role-based keying already exists on the pool side: `gpu_pool:<class>#gpu_pool_wait`. No other FCC keying mechanism exists.
6. **Reversibility.** Cheap. It is a pure key-string function plus an in-process dataclass field. No schema, manifest or training default depends on it.

**Retiring the old key completely.** After the governor restarts, the producer no longer emits `fcc:<model>` keys at all. `RpcHealthAggregator` only re-reports hops that the process has recorded, so the restart is the kill. The three frozen equilibrium state rows carry no learned statistics and no open episodes. They stop receiving windows, and `_evict` drops them after `max_idle_s` (7 days since their last success). Nothing else reads FCC hop keys: grep for `fcc:` across `orion/` and `services/` finds only the producer, its tests and docs. The metric lock (`scripts/check_definition_drift.py`) does not track RPC-health hop keys, so the lock does not change for this. It changed only for the `channels.yaml` edit below.

**Unheld turns.** Without a hold, each FCC call gets its own lease inside the gateway, and the Claude CLI never sees which role granted it. Live, 4 days of `http:anthropic` leases:
- 2,420 calls ran under a hold (role known);
- 653 calls ran without a hold, spread across `agent`, `agent-gpu2`, `chat` and none.

So an unheld turn is keyed `fcc:route:<route>`. The `route:` prefix stops "asked for agent" from sharing a baseline with "ran on agent". A label that points at a non-pool backend (live: `MODEL_HAIKU=nvidia_nim/...`) is keyed `fcc:backend:<backend>`, so a remote API's latency never lands in `fcc:unknown`.

## Metric lock

`python scripts/check_definition_drift.py --update` records three entries, all from the `channels.yaml` edit:
- `high removed metric://bus_channel/orion-hub/orion:gpu_pool:state:request`
- `medium added metric://bus_channel/orion-harness-governor/orion:gpu_pool:state:request`
- `high routing_changed …state:reply:*` (consumers + `orion-harness-governor`)

The removed/added pair is one channel whose name moved. The name is built from the alphabetically first producer (`orion/metrics/lineage.py:301`), and adding `orion-harness-governor` changed that first producer. Nothing else refers to the old name (grep finds it only in the lock). `check_metric_lineage.py --gate` passes.

## Item 3 evidence: exec→gateway bus-synaptic baseline at the stage-3 cutover (no code)

- **Where the history lives.** FalkorDB keeps only the current edge value (`CAUSALLY_FOLLOWED_BY.latency_ewma_sec`), with no history. `substrate_field_state` retention starts 2026-09-26 04:55, after the cutover. The one persisted series that spans the cutover is `substrate_attention_self_model.self_model_json.prediction_error_by_domain.bus_synaptic`, from 2026-09-22 onward. That is the mesh-wide fraction of anomalous edges, `bus_synaptic_prediction_error`.
- **30-minute means on 2026-09-25 (UTC).**
  - Before cutover, 02:00–03:30: 0.043 / 0.043 / 0.038.
  - After, 03:30–07:00: 0.114 / 0.086 / 0.078 / 0.069 / 0.075 / 0.089 / 0.082.
  - From 07:00 on: 0.052 / 0.043 / 0.050 / 0.040.
- **3-hour means for context.** 09-22 to 09-24 ranged 0.040–0.066. 09-26 to 09-29 ranged 0.027–0.061. The bump lasted about 3.5 hours and then returned to the pre-cutover range. There is no lasting shift.
- **Downstream.** Bus-synaptic episode triggers (`metacog_trigger`, `transport:bus_synaptic:episode_start`) per day:
  - 09-22: 53; 09-23: 56; **09-24: 155** (before cutover); 09-25: 111; 09-26: 70; 09-27: 84; 09-28: 64.
  - During the cutover window, 03:00–08:00 on 09-25, they fired about 6 per hour. That is no higher than the 8–11 per hour seen on 09-24.
  - The `telemetry_anomaly` burst that night started at 02:00, before the cutover, so deploy activity is the more likely cause.
  - Nothing persisted past about 07:00.
- **The edge today.** `cortex-exec → llm-gateway`: count 487,153, `latency_ewma_sec` 6.25, **z = -0.12** at 05:25 UTC on 09-29 (1.58 at 04:24). Both are well under the |z| ≥ 3 anomaly line. Verdict: re-learned, no lingering harm, **no reset done.** The one-off "reset at cutover" step in the spec was never done. It is now pointless, because EWMA alpha 0.2 has absorbed the step.
- **Finding: the spec claim about pool events is false live.** The spec says bus-mirror never sees the pool inside a turn's chain. The pool's lifecycle events (`orion:gpu_pool:event`, `services/orion-gpu-pool/app/runtime.py::_emit`) are published with `correlation_id = turn_correlation_id`. Bus-mirror therefore puts `orion-gpu-pool` into turn chains. The edges are live now:
  - `cortex-exec → orion-gpu-pool`: 18,695
  - `orion-gpu-pool → llm-gateway`: 18,998
  - `http:anthropic → orion-gpu-pool`: 17,860
  - `orion-durable-runs → orion-gpu-pool`: 21,144
  - and more.

  One side effect helps: repeated pool events on the same organ (admitted→granted) are absorbed as a same-organ repeat, so queue wait sits in no edge value. That matches "waiting in line is not transport". The costs:
  - the exec→gateway edge now updates only about 3 times per 30 minutes;
  - about 20 pool edges join the anomaly population;
  - it contradicts the spec.

  **Not changed here.** Giving pool events a fresh correlation id would put the queue wait back into `exec→gateway`. That is a design choice for Juniper; see Risks.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no. Local `.env` sync: not needed (no template changed).
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/harness/tests orion/situational/tests orion/gpu_pool/tests            703 passed
services/orion-harness-governor tests                                       58 passed
services/orion-cortex-exec tests/test_situation_provider.py                 19 passed (6 new)
services/orion-hub unified-turn + cockpit-hop tests                        34 passed (4 new)
orion/gpu_pool/tests/test_placement.py                                      6 passed (new)
services/orion-durable-runs tests (CI deps venv, throwaway postgres:16)     213 passed
  -- includes the held-turn acceptance tests that drive the real HarnessRunner under a gpu_lease
gpu-pool "consumers" CI step, reproduced in a clean venv with that job's pip installs: all green
services/orion-hub -k "turn_orchestrator or unified_turn or situation or gpu": 183 passed, 1 failed
  -- test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness fails identically
     on untouched main in this environment (pre-existing, unrelated)
python scripts/check_definition_drift.py --gate      PASS (after --update)
python scripts/check_metric_lineage.py --gate        PASS
python scripts/check_bus_reply_channels.py           0 uncovered
python scripts/check_async_routes_not_blocking.py    clean
git diff --check                                     clean
```

## Evals run

```text
No eval harness covers harness prompt self-context or the transport-baseline key; the
behaviour is pinned by deterministic tests (spilled held turn -> granted role's model; unheld
-> worded as default; per-turn cache isolation; hop keys by role). Follow-up: none proposed --
the FCC hop's real eval gap is the sparse-key degeneracy in Risks.
```

## Docker/build/smoke checks

```text
Not deployed (task scope: do not deploy). No image built. Runtime claims below are UNVERIFIED
until the restart: the new prompt line and fcc:<role> keys have not been observed live.
```

## Review findings fixed

The code-review subagent found no must-fix issues. All should-fix items and the material nits are fixed:

- Finding: Hub could name a model from a pool snapshot of any age, because its feed keeps the last one when the pool goes quiet.
  - Fix: `_pool_state_is_fresh` treats a snapshot older than 30 s (six broadcasts) as absent. The brief then names the role without a model.
  - Evidence: `test_stale_pool_snapshot_never_names_a_model`.
- Finding: every non-llamacpp FCC run (e.g. `MODEL_HAIKU` → nvidia_nim) was keyed `fcc:unknown`, mixing a remote API into the unknown bucket.
  - Fix: `resolve_fcc_backend()`. The runner carries `fcc_backend`, and the key becomes `fcc:backend:<backend>`.
  - Evidence: `test_fcc_hop_non_pool_backend_keeps_its_own_key`, `test_harness_runner_resolves_route_or_backend_for_the_hop_key`.
- Finding: on an unheld Hub turn, the brief (route `chat`) and the harness prefix (route `harness`) could each name a different "default" model.
  - Fix: for an unheld unified turn, Hub sets `runtime_line_owner="harness"` and the brief omits its model line (`RuntimeContextV1.placement="harness"`), so only the harness line names a model. The brief's `route` for a lease is now the lease route (`agent`), not `chat`.
  - Evidence: `test_no_hold_hands_the_model_line_to_the_harness`, `test_runtime_line_is_omitted_when_the_harness_owns_it`, `route == "agent"` assertion.
- Finding: for a `mismatch` role the wording said the model "could not be read", but the pool did read it; it is just unconfirmed.
  - Fix: discovery status carried through (`ServingPlacement.role_status`, `RuntimeContextV1.role_status`). The text now says "the pool reports that role as mismatch, not a confirmed model".
  - Evidence: `test_runtime_context_mismatch_role_says_why_it_names_no_model`, `test_placement.py`.
- Finding: the route-default probe used the raw label while the hop key used the default label.
  - Fix: both use `request.fcc_model_label or DEFAULT_FCC_MODEL_LABEL`.
  - Evidence: the probe is awaited with `"MODEL_SONNET"` when no label is given.
- Finding: the workflow's `push:` path filter did not cover the touched files.
  - Fix: added.
- Finding: are old `fcc:<model>` keys held anywhere else?
  - Fix: none needed. `channel_latency` has two readers. Equilibrium's transport baseline is one. `orion/substrate/rpc_delivery.py` is the other, but its `counted_hop` only counts bus-RPC hops, so it ignores `fcc:*` entirely (it also keeps only an in-memory rolling window). Its three frozen rows are covered in the retirement section.
- Not changed: `resolve_fcc_backend` reads `~/.fcc/.env` synchronously on each turn. The previous probe did the same; this is a small local file read.

## Restart required

```bash
# Order: none required between these (no schema forbid-migration; pool already serves include_leases).
scripts/safe_docker_build.sh orion-harness-governor up -d --build   # fcc:<role> hop + lease-based prompt line
scripts/safe_docker_build.sh orion-hub up -d --build                # situation brief from the turn's lease
scripts/safe_docker_build.sh orion-cortex-exec up -d --build        # situation brief wording (route default)
```

Live check after the restart:
- the Redis key `equilibrium:transport_baseline_state:v1` gains `fcc:agent`, `fcc:chat` or `fcc:route:*` entries, and no new `fcc:<model>` ones;
- the cockpit motor-boot prompt of a held turn shows "Backend model serving this turn: … (GPU pool role …)".

## Risks / concerns

- **Severity: medium. Pool events sit inside turn causal chains (spec deviation).**
  - `runtime.py::_emit` publishes on `turn_correlation_id`.
  - Mitigation: documented above with live edge counts.
  - Juniper to choose one of two options:
    - (a) fresh ids for pool events: matches the spec, but queue wait returns to exec→gateway;
    - (b) keep as is and amend the spec.
- **Severity: medium. The FCC transport baseline has never evaluated.** Every FCC key shows `fast_count=0`, because turns are too sparse for `min_calls=5` inside `max_aggregate_windows=20`. The re-key keeps the key honest, but the hop still produces no baseline. A fix belongs in the transport-baseline sparse-key policy, not here.
- **Severity: low. The FCC leg includes pool queue wait.** Unheld calls wait for their lease inside the gateway, and held calls may wait behind an interleaved higher-priority inference, all within `fcc_elapsed_sec`. That breaks "waiting in line is not transport" for this hop. It predates this patch and has no effect today, because the key is never evaluated (previous item).
- **Severity: low. Unheld turns cannot name their model.** Each call's grant is invisible to the CLI, so the prompt says "default, not confirmed". An exact answer would need the gateway to return the granted role to the harness, and the Claude CLI does not surface response headers.
- **Severity: low. Held turns make one extra bus RPC.** One `orion:gpu_pool:state:request` per held turn, 2 s timeout, run in parallel with the existing pre-turn reads. On failure the prompt names the role without a model.

## PR link

- https://github.com/junebug-junie/Orion-Sapienform/pull/2395 (merged 2026-09-29 06:03 UTC, before the review fixes landed)
- Review fixes (the "Review findings fixed" section above): follow-up PR on branch `fix/gpu-pool-stage3-review-followups`
