# GPU2 elastic admission evidence — 2026-09-14

Evidence collected before code edits from main `75f5d9e17` (PR #2213), the
#2038/#2205/#2209/#2210/#2211 reports and current service code. No production
mutation, env sync, migration, restart or generation was performed.

## Physical topology and live observations

Read-only HTTP at 01:00 UTC: Circe `100.112.254.99:8090/v1/gpu-lane/status`
reported agent running/healthy (`orion-circe-atlas-llamacpp-agent`) and affect
exited (`orion-circe-affectgpt-worker`). Controller targets are compose
`affectgpt-worker` in `services/orion-affectgpt-worker/docker-compose.yml` and
`atlas-agent`, profile `agent-split`, in
`services/orion-llamacpp-host/docker-compose.atlas-workers.yml`. Controller
`_targets()` explicitly overrides `ATLAS_AGENT_CUDA_VISIBLE_DEVICES=1`.
Affect's checked-in HOST_PORT is 32798 (internal 6610); live listener mapping
was not available while its container was exited.
GPU1 assignment is supported by the September 2 live deployment report;
current device-level verification is UNVERIFIED: Tailscale SSH denied this user.

Diffusion `/health` and `/ready` at `100.112.254.99:8014` reported model-loaded
and ready, `YuCollection/FLUX.1-schnell-Diffusers`. Compose service
`diffusion-host`, container `${PROJECT}-diffusion-host` (Circe convention:
`orion-circe-diffusion-host`), internal port 6700, template physical
`CUDA_VISIBLE_DEVICES=2`. Current Docker GPU mapping is UNVERIFIED.

Live Gateway `localhost:8210/routes` returned:

| Routes | Circe port | Model | Context per slot | Vision |
| --- | --- | --- | --- | --- |
| chat, harness | 8011 | Qwen3.6-35B-A3B-UD-Q5_K_M.gguf | 131072 | true |
| metacog, metacog_background | 8012 | Qwen_Qwen3-8B-Q5_K_M.gguf | 4096 | false |
| quick, quick_background | 8013 | Qwen_Qwen3-8B-Q4_K_M.gguf | 4096 | false |
| agent | 8015 | Qwen3.8-27B-UD-Q4_K_XL.gguf | 131072 | false |

All upstream URLs are `http://100.112.254.99:<port>`. Agent `/props`
independently confirmed model, 131072 context and one slot. Its checked-in
profile is `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex` in
`config/llm_profiles.yaml`. The new slot must use this same effective contract.
Port 8016 is a proposed reservation, NOT verified free on Circe. Operator must
check listeners before rollout. No existing checked-in worker target uses it.
Proposed unique service `atlas-agent-burst`, profile `agent-burst`, container
`${PROJECT}-atlas-llamacpp-agent-burst`, GPU2, upstream Circe:8016. GPU1 unchanged.

## Existing seams and gaps

* Controller `app/main.py` authenticates mutations with
  `GPU_LANE_CONTROLLER_TOKEN`, fails closed when empty, and serves read-only
  status. Hub/Cortex callers use the old GPU1 endpoints. `lane_control.py`
  uses fixed settings, exact service arguments, a single-process flip lock,
  stop-before-start and currently **builds on every flip**. GPU2 will use
  prebuilt `up -d --no-build`; preserve the GPU1 adapter.
* Diffusion `generate()` rejects concurrent work with 429, `/ready` checks
  `_pipe`, and a dedicated single-thread executor owns GPU work. There is no
  drain contract. Cancelling an awaiting coroutine does not stop its GPU
  thread; lifecycle ownership must outlive that cancellation. Power intent
  is published before generation; drain must wait through the actual work.
* Thought `visual_chain.call_diffusion_generate()` currently converts every
  HTTP error (including 429) to generation failure. `main.run_visual` emits
  acknowledged execution receipts. `orion/reverie/baseline.py` and existing
  action execution consumer retain outstanding baseline through deferrals;
  use these receipts, not a second obligation model. Resource displacement
  needs a distinct defer reason and must bypass failure artifacts/posterior.
* `orion/autonomy/thermal_gate.py` owns cabinet thermal classification.
  Thought reads Hub `/api/cabinet/sensors/latest`, environment.temp_c and
  age_sec. Existing visual policy allows degraded readings; optional elastic
  spending will fail closed on unknown/stale data, without changing reverie.
* `AdmissionRuntime.refresh_lanes()` discovers real Gateway routes and probes
  `/slots`. `ResourceBroker.tick()` arbitrates FIFO under Postgres
  `ADMISSION_LOCK`, preserving first-assigned lane. `decide_lane()` requires
  configured healthy routes plus explicit capability/compatibility policy.
  Inactive elastic capacity cannot be marked healthy to obtain a lease.
* `PostgresCapacityStore` shares the admission transaction lock. Leases and
  request permits are distinct: a released lease is not proof upstream work
  stopped. Restoration needs admission closure under that same authority,
  then permit and actual upstream-idle evidence before container stop.

## Contradictions and verification limits

PR #2213 has already enabled durable admission, capacity and widening in
operator templates, while code/compose fallbacks remain off. Policy is still
`{}`. Preserve this; all NEW elastic flags default off. Its report found
missing production tables before activation; do not assume migrations ran.
Prompt's blanket env-sync requirement is superseded by its explicit prohibition
on modifying production ignored `.env` files. No production path has been
exercised. No transition/cold-load duration is measured here; conservative
configured budgets must be labelled declarations, not measurements.

## Operational metric gate

Use existing SQL ownership/event timestamps, HTTP readiness and controller
transition phases as bounded operational evidence. Queue ages, estimates and
transition durations are causally dependent, not independent cognition inputs.
Their anchors are elapsed time and exclusive resource conservation. Existing
live observations above confirm readiness, not future transition timing.
No new cognitive detector/learned metric will be wired; new transition timings
remain UNVERIFIED until operator smoke and are removable operational fields.
