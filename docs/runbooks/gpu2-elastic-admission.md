# GPU2 elastic admission: operator rollout and rollback

Do not execute until this branch is reviewed/merged and the operator authorizes
production activation. No deployment or production migration was performed for
this patch. Worktree build configs are validation-only copies of safe templates.

## Verify physical mapping first

On Circe, inspect `nvidia-smi`, `docker ps`, `docker inspect` and `ss -ltn`.
Confirm GPU1 agent/affect, GPU2 diffusion, and free host port 8016. Confirm GPU2
has room for the same 131072-context Qwen3.8-27B UD-Q4_K_XL profile. If not, stop
rollout; do not silently lower context or use a different capability contract.
The evidence note's live HTTP checks do not prove GPU device placement/headroom.

| Slot | Target | Compose service | Container with PROJECT=orion-circe | Host port |
| --- | --- | --- | --- | --- |
| GPU1 | affect | affectgpt-worker | orion-circe-affectgpt-worker | 32798 (template) |
| GPU1 | agent | atlas-agent / agent-split | orion-circe-atlas-llamacpp-agent | 8015 |
| GPU2 | diffusion | diffusion-host | orion-circe-diffusion-host | 8014 |
| GPU2 | agent-burst | atlas-agent-burst / agent-burst | orion-circe-atlas-llamacpp-agent-burst | 8016 |

ATLAS_AGENT_PROFILE_NAME must remain the established
`qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex`. Use the current deployed
LLAMACPP_IMAGE_TAG supporting this model, not an older default image. Burst shares
LLM_CACHE_DIR; pre-cache the model before borrowing. Restrict direct backend
access to participating Gateways/controller with the existing tailnet/network
boundary; uncoordinated direct llama.cpp clients are outside capacity fencing.

## Flags and declarations

All new enable flags default false. `DURABLE_RUNS_ELASTIC_SHADOW` defaults true.
Existing admission/widening/capacity flags are preserved (PR #2213 template
values are already true). Never restart those services until their existing
migrations are installed. Apply the new additive migration in the same checkpoint
DB **after** durable_resource_admission_v1 and gateway_capacity_v1:

```bash
psql "$POSTGRES_URI" -v ON_ERROR_STOP=1 -f services/orion-sql-db/manual_migration_gpu2_elastic_v1.sql
```

GPU2 activation, diffusion drain, and elastic intent use the existing internal
service/tailnet boundary. No bearer token or shared secret is required.
Keep these control-plane endpoints inside that boundary. GPU1 retains its existing contract.

Controller: `GPU2_ENABLED=false`; diffusion/agent URLs are Circe:8014/:8016;
`GPU2_AUTHORITY_URL=http://100.92.216.81:8124`; drain timeout 300s and model-ready
poll timeout 600s. Targets/profile/GPU overrides are fixed in code/compose, not
caller-editable API parameters. Burst host port declaration is
`ATLAS_AGENT_BURST_HOST_PORT=8016`; changing it requires all URL owners to agree.

Durable: `ELASTIC_ENABLED=false`, `ELASTIC_SHADOW=true`,
`ELASTIC_ASSIGNMENTS=false`, `ELASTIC_RESTORATION=false`,
`ELASTIC_THERMAL_ENABLED=false` (all prefixed `DURABLE_RUNS_`). URLs are
controller Circe:8090, backend Circe:8016, cabinet Hub
`http://100.92.216.81:8080/api/cabinet/sensors/latest`. Hub uses host
networking, so its container name does not resolve from durable-runs' bridge
network; use Athena's Tailscale address. Budgets:
`ELASTIC_DRAIN_BUDGET_SEC=300`, `ELASTIC_TRANSITION_BUDGET_SEC=60`,
`ELASTIC_COLD_BUDGET_SEC=600`; idle grace 300s, minimum residency 600s,
maximum borrow 3600s. These are declarations, not measured timing.

Thought: `ORION_VISUAL_ELASTIC_STATUS_ENABLED=false`, controller URL Circe:8090.
Hub: `HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED=false`, producing the explicit
per-run opt-in on newly accepted runs. Existing submissions/candidate sets remain
immutable; changing flags does not retroactively enroll old studies.

The shipped lane policy remains `{}`. After capability review, stage this
explicit policy for shadow decisions (use exact `/routes` model value):

```json
{
  "agent": {"capabilities": {"structured_output": true}},
  "agent-burst": {
    "compatible_with": ["agent"],
    "activatable": true,
    "activation_model": "/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf",
    "capabilities": {"structured_output": true},
    "quality_drop": 0,
    "switching_cost_seconds": 120
  }
}
```

Set as `DURABLE_RUNS_LANE_POLICY_JSON`. This declaration permits estimation;
assignment additionally requires the separate ELASTIC_ASSIGNMENTS gate and
real matching Gateway model/context/vision health. No route name implies
structured-output compatibility. Audit that capability before declaring it.

## Consumer-first commands (operator only)

Run from reviewed worktrees on the indicated host. Sync each production env
intentionally from its reviewed template while preserving secrets/host values.
All bus URLs must be `redis://100.92.216.81:6379/0` on this deployment.

1. Circe: build/cache burst without starting it:

```bash
scripts/safe_docker_build.sh orion-llamacpp-host -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml --profile agent-burst build atlas-agent-burst
```

2. Deploy consumers first: Thought, feedback-runtime, execution-dispatch-runtime,
proposal-runtime and relevant schema consumers before any resource_deferred
receipt can be emitted. Set Thought's status gate true only with the new controller
status endpoint present; until then keep borrowing off. Deploy diffusion's
additive drain API.

```bash
# Athena
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-feedback-runtime up -d --build
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
# Circe
scripts/safe_docker_build.sh orion-diffusion-host up -d --build
scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
```

3. Athena: apply additive schema, deploy durable authority with elastic disabled,
then all Gateway replicas with the new system route while burst remains stopped.
Deploy Hub opt-in support, initially off. Preserve FCC lease propagation consumers.

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build hub-app
curl -fsS http://100.112.254.99:8090/v1/gpu-lane/status
curl -fsS http://100.112.254.99:8090/v1/gpu-slots/circe-gpu2/status
curl -fsS http://127.0.0.1:8210/routes
```

4. Enable controller GPU2 management and Thought resource status handling. Enable
elastic authority + thermal eligibility + restoration, with shadow true and
assignments false. Inspect `/elastic/status`, `/admission`, real cabinet history
and fresh `/visual-chain/activity`. Stage the audited compatibility policy and
Hub opt-in; observe fake-free shadow predictions on new waiting runs.

5. Controlled manual round-trip: shadow false, assignments still false. The runtime records the intent;
the controller will refuse unrecorded/direct target requests. Poll both status
endpoints until ready before the reverse command. No model request is required.

```bash
curl -fsS -H 'Content-Type: application/json' \
  -d '{"target":"agent-burst"}' http://100.92.216.81:8124/elastic/target
curl -fsS http://100.92.216.81:8124/elastic/status
curl -fsS http://100.112.254.99:8090/v1/gpu-slots/circe-gpu2/status
curl -fsS -H 'Content-Type: application/json' \
  -d '{"target":"diffusion"}' http://100.92.216.81:8124/elastic/target
```

Repeat the same target for an idempotent operator retry. Observe real transition
and cold-start durations; update declared budgets conservatively. Urgent baseline
or thermal suppression should prevent borrowing, not be overridden for a smoke.

6. Enable automatic borrowing without assignments and inspect restoration after
minimum residency. Finally enable ELASTIC_ASSIGNMENTS for a newly submitted,
opted-in self-inquiry contention smoke. Observe checkpoint wait, committed intent,
drain latch, model readiness, fenced lease, full FCC/Gateway permit ownership,
nonempty run artifacts, terminal release, and automatic return to diffusion.
Verify baseline need survives deferral and no posterior update/image-failure
artifact appears. This production acceptance is currently UNVERIFIED.

## Rollback and failure

Close new admissions by requesting diffusion through `/elastic/target` while the
authority remains enabled. This does not kill an active lease/permit. Wait for
controller diffusion ready and `/ready` true before disabling controller GPU2,
Hub opt-in, elastic assignments/activation and Thought status handling. Preserve
additive DB schema/history. Do not DROP/TRUNCATE or disable the authority before
protected requests drain. Keep existing GPU1 behavior unchanged.

If restoration fails, inspect `/elastic/status` and controller status/logs.
The row stays failed/closed; diffusion is not claimed available. Retry the same
diffusion target. If `/slots` cannot prove a resident burst worker idle, restoration
fails closed and requires operator investigation; do not force-stop an unknown
active upstream. No destructive automatic rollback is provided.
