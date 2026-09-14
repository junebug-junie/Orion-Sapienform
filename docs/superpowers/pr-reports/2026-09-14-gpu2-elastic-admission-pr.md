# Fixed GPU2 elastic admission

## Summary

- Borrow Circe GPU2 from diffusion for explicitly opted-in, long-waiting durable agent work.
- Add a fixed slot API, authenticated atomic diffusion drain, and distinct stopped agent-burst worker/system route.
- Persist activation intent and restoration under the existing Postgres capacity authority; retain FIFO, fencing and full FCC request ownership.
- Preserve acknowledged visual resource deferral and baseline debt without false images or posterior lessons.
- Ship disabled new gates, consumer-first runbook, isolated acceptance/evals and independent review fixes.

## Outcome moved

A stopped compatible lane can now be activated without holding a graph worker or
inference RPC open. Ready capacity still requires a real Gateway discovery check
and a fenced lease. Restoration cannot stop an active lease, permit or observed
upstream request. **Production actuation and inference are UNVERIFIED** and were
explicitly excluded from this implementation task.

## Current architecture

[Pre-edit evidence](../../architecture/gpu2-elastic-evidence.md) records live
controller/Gateway/diffusion HTTP and repository truth at main `75f5d9e17`.
Contradiction: PR #2213 already enables widening in operator templates; policy
remains `{}`. Those flags are preserved. GPU1 agent is Circe:8015, Qwen3.8-27B
UD-Q4_K_XL, one slot, 131072 context; diffusion is Circe:8014, FLUX.1-schnell.
Tailscale SSH denied current device-level verification. Port 8016 is a proposed
reservation, not a live-verified free listener.

## Architecture touched

[ADR](../../architecture/gpu2-elastic-admission.md) and
[rollout/rollback](../../runbooks/gpu2-elastic-admission.md) document ownership,
privacy, exact topology, flags, lifecycle, failure behavior and operator commands.
GPU1 affect↔agent stays unchanged. GPU2 diffusion↔agent-burst uses an additional
fixed service/profile and port; no generic scheduler or Docker-control API.
Operational facts reuse the registered resource event outbox; metric gate findings
are recorded in the evidence note. No new cognition metric/detector is wired.

## Files changed

The PR diff contains the complete file inventory. Principal seams:

- `orion/durable_admission/elastic.py`: durable slot intent/status/completion.
- `orion/durable_admission/broker.py`, `capacity.py`, `policy.py`: atomic closure, compatibility, opt-in and FIFO.
- `services/orion-durable-runs/app/elastic_runtime.py`: reconciliation, thermal/baseline checks and restoration.
- `services/orion-gpu-lane-controller/app/gpu2.py`: fixed container actuation and readiness.
- Diffusion lifecycle, Thought visual receipts/activity, Hub opt-in, canonical route vocabulary and Gateway lease-only admission.
- Additive SQL/HTTP schema, service config/READMEs, deterministic tests, eval and CI workflows.

## Schema / bus / API changes

- Added: `durable_elastic_slot` table via `manual_migration_gpu2_elastic_v1.sql`; typed GpuSlotRequestV1 HTTP contract.
- Added: fixed-slot activate/status, diffusion drain/status, durable elastic target/status HTTP endpoints.
- Added: optional ResourceRequirementV1 `allow_elastic_activation=false`; `resource_deferred` visual terminal reason and `deferred_resource` outcome.
- Removed/Renamed: none.
- Behavior changed: resource refusal is no longer recorded as generation failure; ordinary unleased burst traffic is rejected.
- Compatibility: old GPU1 endpoints retained; existing registered resource event envelope/channel reused; additive SQL retained on rollback; old serialized requests normalize the new false opt-in default.

## Env/config changes

- Added keys: GPU2 controller enable/URLs/token/drain/readiness timeouts; diffusion drain token; durable elastic enable/shadow/assignment/restoration/thermal gates, controller token/URLs, declared budgets, idle/minimum/maximum timing; Thought status gate/URL; Hub per-run opt-in; burst port.
- Removed/Renamed keys: none.
- `.env_example` updated: seven affected services. The Gateway template adds the stopped system route; lane policy remains `{}`.
- New enable gates default false; elastic shadow defaults true. Existing enabled admission flags remain unchanged.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: **not run**, per explicit prohibition on production env writes. Only validation worktree envs were created from safe tracked templates; all are ignored.
- Skipped keys requiring operator action: all new production keys and privately provisioned tokens, intentionally deferred to rollout. No token was committed or logged.

## Tests run

Using `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest ... -q --disable-warnings`:

| Scope | Result |
| --- | --- |
| GPU controller (legacy + fixed slot) | 41 passed |
| Diffusion | 41 passed |
| Durable runs, isolated real Postgres, six connected FCC/Gateway cases | 82 passed |
| Gateway + canonical route vocabulary | 355 passed |
| Thought activity/chain/thermal + baseline/policy/feedback | 140 passed |
| Hub admitted Curiosity + opt-in | 5 passed |

Total: **664 passed**. Test Postgres was a disposable local container, bound to
127.0.0.1:32770; no production DSN, migration or container was used. Physical GPU,
Docker runners, Gateway model responses and external cognition were fixtures.
Known Pydantic namespace/deprecation warnings remain.

## Evals run

- `services/orion-durable-runs/evals/elastic_fairness.py`: PASS; 20/20 FIFO from inactive burst through restoration.
- `services/orion-durable-runs/evals/admission_fairness.py`: PASS; 20/20 served, zero remaining queue/leases.
- `services/orion-durable-runs/evals/gateway_capacity.py`: PASS; 40 two-client contention rounds, zero remaining permits/leases.
- `services/orion-proposal-runtime/evals/test_baseline_replay.py`: 1 passed.
- `services/orion-thought/evals/test_visual_chain_honesty_eval.py`: 1 passed.

## Docker/build/smoke checks

Seven affected service images built through `scripts/safe_docker_build.sh` with
isolated `gpu2-validation` image names: controller, diffusion, durable-runs,
Gateway, Thought, Hub and agent-burst llama.cpp. No service was deployed.
The wrapper's documented `ORION_ALLOW_ENV_DRIFT=1` was needed because its parity
checker resolves production main envs even from a worktree; changing those was
expressly prohibited. Worktree templates were checked separately. The llama
build used a validation-only image override, protecting the production image tag.

Passed: env/compose parity for affected canonical service files, hostname refs,
async route blocking gate, bus reply coverage, metric lineage static gate and
git diff whitespace check. Atlas worker compose was rendered by the targeted
burst build. Scripts named `check_schema_registry.py` and `check_bus_channels.py`
are absent; existing schema/transport suites and reply/catalog gates were used.

## Review findings fixed

- First activation could close before allocation; added a transactional first-broker-pass handshake, tested even with zero idle grace.
- Retained burst retry could strand after restoration; allowed same-lane reactivation without migration.
- Backend alias/per-run opt-in gaps; physical closure and explicit permission now gate grants and permits.
- Paused/restarting/invalid Docker state could look absent; fail closed before starting the other tenant.
- Diffusion cancellation could hide real GPU occupancy; shielded future remains visible through drain.
- Failed startup/restoration idempotency and thrashing; persist verified rollback/residency, match durable intent, preserve errors without stale evidence.
- Restoration depended on Gateway availability; only burst readiness now depends on Gateway.
- Visual activity and preflight could mislabel/block restored diffusion; typed resource history and verified-rollback generation tests added.
- Evidence: controller, real-Postgres, visual and connected acceptance regression suites above. Final independent review found no outstanding material issues.

## Restart required

No restart was performed. Exact host-specific commands and consumer-first ordering
are in [the runbook](../../runbooks/gpu2-elastic-admission.md). Deploy schema
consumers and visual deferral handling before borrowing; build/cache burst,
deploy drain/controller, migrate authority, deploy Gateway with burst stopped,
shadow, manual round-trip, borrowing without assignment, then controlled widening.

## Risks / concerns

- Production path UNVERIFIED by design; operator must check GPU placement/headroom, port 8016, model image revision, migrations and actual transition timing.
- Direct backend clients outside participating Gateway/network fencing are outside the lease guarantee. Unknown `/slots` blocks restoration and requires investigation.
- Single uvicorn process is required for controller and diffusion locks/latch.
- Conservative declared cold/transition budgets may overestimate benefit; inspect shadow timing before activation.
- No destructive database rollback. Request protected restoration before disabling the authority.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2215

GitHub CI is tracked in the PR checks; mergeability was confirmed on creation.
