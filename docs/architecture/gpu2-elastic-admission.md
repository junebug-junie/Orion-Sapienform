# ADR: borrow a fixed GPU2 slot through durable admission

Status: additive implementation; production activation UNVERIFIED.
See [pre-edit evidence](gpu2-elastic-evidence.md) and
[operator runbook](../runbooks/gpu2-elastic-admission.md).

## Decision and ownership

Preserve GPU1 affect↔agent and its HTTP adapter. Add only GPU2
 diffusion↔agent-burst. A distinct llama.cpp service uses physical GPU2,
port 8016 and the same ATLAS_AGENT_PROFILE_NAME/model cache. The canonical
route vocabulary marks agent-burst system-only; the Gateway template configures
it as a distinct, normally stopped upstream. There is no fallback to it.
Gateway rejects unleased burst calls even with capacity enforcement off.

The existing LangGraph resource_wait checkpoint releases the workflow task.
The existing FIFO broker evaluates explicit opt-in/compatibility, hard
requirements, original 1200s queue age and hysteresis. Activation estimates
include declared 300s diffusion drain + 60s transition + 600s cold load +
older candidate queue budgets + declared switching cost. These are conservative
operator budgets, not measurements. Preferred lane saturation comes from the
existing capacity/occupancy estimate; an unknown preferred route is not grounds
to steal GPU2. Pins and previous assignments remain authoritative. A previously
burst-assigned retry can request reactivation without changing its assigned lane.

One additive `durable_elastic_slot` row records desired ownership, stable
operation ID, generation, admission closure, timestamps and bounded diagnostics.
It is not a workflow or another queue. The broker writes the intent and existing
`durable.resource.event.v1` outbox record under ADMISSION_LOCK before any HTTP.
AdmissionRuntime's existing reconciliation drives the controller outside this
transaction, under the existing advisory claim pattern. No graph worker, original
Cortex/Hub RPC or inference timeout waits for this operation. Activation consumes
no inference attempt. Redis wakeups are hints; SQL reconciliation recovers restart
and lost replies using the same intent ID. Stale controller requests must match
the current authoritative target, generation and operation ID.

## Admission and restoration

Controller success is not a grant. The runtime refreshes Gateway discovery and
requires its real configured URL, healthy model, exact audited model/context/vision
contract and idle `/slots`. The broker then grants one ordinary fenced durable
lease in original FIFO order. No priority accrues to the activation requester.
Compatibility names alone never authorize the new route. Live preferred model
must match the explicit activation_model declaration; inactive capability checks
use that audited same-model contract. Once ready, actual burst model, context
and vision must match the preferred route.

Idle grace keeps a newly ready lane open long enough for the broker's first
allocation. Existing eligible queue work can retain residency. Maximum borrowing
or urgent baseline closes new lease admissions immediately; idle queue closure
occurs after grace. Minimum residency prevents premature switching. Restoration
requires no active durable lease AND no active Gateway request permit. Both
broker and capacity acquisition share ADMISSION_LOCK. Physical backend aliases
cannot bypass closure. Existing valid lease owners may continue sequential FCC
requests while new admissions are closed; permit renewal remains legal after
lease release until the real upstream call finishes. Controller independently
requires `/slots` idle before stopping burst. Unknown occupancy fails closed.

Rollback/failure never fabricates availability. The SQL row retains desired
ownership and error state, and retries the same operation. A failed burst startup
restores diffusion when safe. A failed drain merely unlatches diffusion; it never
recreates a container while generation is active. An exhausted activation window
requests restoration. Explicit internal operator retry uses the same intent
when the target has not changed. Additive tables remain on rollback.

## Diffusion and visual meaning

The diffusion lifecycle drain latch and generation admission execute without an
intervening await on its single uvicorn event loop. Drain rejects new work and
reports the actual executor future as in-flight even if its HTTP waiter was
cancelled. The controller waits for the existing generation before stop. Power
intent publication remains inside actual admitted generation, so deferred calls
open no measurement window; a running generation completes before displacement.

Thought's opt-in resource preflight reads controller status. Intentional absence,
429 busy and 503 model-unready become explicit resource deferrals. HTTP 500 and
real generation exceptions remain generation failures. No image artifact or
production acknowledgement is fabricated. The existing execution receipt,
activity projection, baseline checkpoint/debt reducer and feedback outcome path
preserve the obligation. `visual_non_observation` prevents non-produced actions
from updating the effect posterior. Controller status unavailable and diffusion
transport loss are conservatively resource-unknown while this consumer gate is
on; they are not proof of deliberate displacement.

The existing cabinet thermal classifier and acknowledged visual activity API are
reused. Hot, stale, unavailable or nonfinite thermal readings suppress optional
borrowing, as do urgent baseline and unavailable/stale baseline history. This
stricter policy for optional extra capacity does not change visual reverie's
existing degraded-sensor behavior. There is no new learned thermal score.

## Contracts, security and operational evidence

New HTTP: typed `/v1/gpu-slots/activate`, fixed-slot status,
`/v1/lifecycle/drain`, lifecycle status, `/elastic/status`, `/elastic/target`.
Every mutation requires a configured bearer secret. Controller callers cannot
supply Docker paths/services/profiles/commands/GPU/env. Compose operators remain
trusted; known targets come from local code. Every subprocess names exactly one
service. GPU2 uses `up -d --no-build --no-deps`; GPU1 retains its existing build
behavior for compatibility. Single uvicorn process per controller and diffusion
service is a deployment invariant. GPU1/GPU2 locks are independent.

ResourceRequirementV1 gains optional `allow_elastic_activation=false`.
Visual outcome/terminal vocabulary gains `deferred_resource`/`resource_deferred`.
Existing registered resource lifecycle envelope/channel is reused for bounded
operational IDs, phases, suppression reasons and durations. No new bus channel,
prompt, model output or image is included in operational events. The evidence
note records metric provenance/independence/theory and unverified timing limits.

Non-goals: affect on GPU2; new scheduler; mid-run model migration; replacing the
preferred lane; direct-backend clients outside Gateway fencing; production
enablement; GPU/device benchmarking; new cognition scores or obligation models.
