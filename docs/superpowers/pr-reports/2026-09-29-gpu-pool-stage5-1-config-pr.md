# GPU pool stage 5.1: config + contracts

## Summary

This is the first of the stage 5 PRs. It changes the rules and the checks, and it leaves live
behaviour alone except for one number: the 27B seat on gpu2 may now stay loaded for about 2.5 hours
instead of 1 hour.

- **A new YAML rule, `serialize_with`, keeps world and image generation off gpu2 at the same time.**
  `world: serialize_with: [diffusion]`. The scheduler places nothing on a role while a lease is
  active on its partner, in either direction. It never recalls or pre-empts across the pair.
- **The waiting lease keeps its place in line.** If a lease is blocked only by this rule, the
  partner role is reserved, so younger work cannot keep taking it. Without that, world's two slots
  could starve an older image hold forever. The pool says why once, with a `queued` event carrying
  `reason=serialized:<role>`. It is dormant today, because the world model does not
  take pool leases until 5.4.
- **`launch.cuda_env` now names the compose variable the actuator will set**, and there is a CI gate
  for it. `scripts/check_gpu_pool_config.py` refuses any launch service whose GPU is a literal. Each
  device variable (`CUDA_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES_OVERRIDE` or
  `NVIDIA_VISIBLE_DEVICES`) must be exactly `${cuda_env}` or `${cuda_env:-<card index>}`. A literal
  `device_ids` pin is also refused.
- **The agent-burst compose service is migrated:** `CUDA_VISIBLE_DEVICES_OVERRIDE=${ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES:-2}`
  and `LLM_PROFILE_NAME=${ATLAS_AGENT_BURST_PROFILE_NAME:-${ATLAS_AGENT_PROFILE_NAME}}`. Both still
  resolve to today's device 2 and today's profile (checked with `docker compose config`, below).
- **`launch.profile_var` / `launch.profiles` (a model choice per role)** are parsed and gated. The
  allow-list is checked against `config/llm_profiles.yaml`, and `LLM_PROFILE_NAME` must interpolate
  `profile_var`. Nothing produces a profile until 5.3; `agent-gpu2` gets `profile_var` only.
- **Class `world` is now `on_unavailable: wait`.** `experiment` lost its dead bridge verbs. The
  validator exempts an operator-only seat that has no launch block and no bridge (deferred, stage 5
  Decision 3). `agent-gpu2` `max_hold_sec` went from 3600 to 9000 (Juniper, 2026-09-29).

## Outcome moved

- **Seat limit:** a 27B load on gpu2 now drains after 9000 s instead of 3600 s. Curiosity and
  self-sense holds that land there keep the card longer, and image generation may wait up to about
  2.5 h for gpu2. Juniper accepted that cost.
- **Adding a card or a model option is now a config edit that CI can check.** Spec worked examples A
  (add gpu4) and B (a second model for gpu2's seat) pass the validator with YAML and compose edits
  only (tests below). A hard-coded device, the 4.1-5 correction, now fails CI instead of silently
  pinning a seat the stage 5 actuator could never move.

## Current architecture

- `config/gpu_pool.yaml` roles had no ordering constraint between roles that share a card. The only
  world/diffusion mutex was durable-runs' `/capacity` permit, where both use the same backend key.
- `LaunchSpec.cuda_env` named the *container* variable. The gate only checked that its resolved value
  matched the card index. `atlas-agent-burst` hard-coded `CUDA_VISIBLE_DEVICES_OVERRIDE=2`.
- Class `world` was `backlog`. `experiment` carried the bridge verbs `circe/experiment`/`circe/restore`,
  which no controller maps (`BRIDGE_TARGETS` only knows `gpu2/*`).

## Architecture touched

- `orion/gpu_pool/config.py`:
  - new fields `RoleSpec.serialize_with`, `LaunchSpec.profile_var` and `LaunchSpec.profiles`;
  - new cross-reference checks;
  - `PoolConfig.serialized_with()`, built once at validation time;
  - the interpolation parser now understands nested defaults (`${A:-${B}}`);
  - `check_launch` gains the device and profile gates.
- `orion/gpu_pool/scheduler.py`: rule Z1 (`_Ctx.serialized_by` checked in `placeable`, plus a
  queue-order reservation of the blocking partner) and a new report-only decision `Serialized`.
- `services/orion-gpu-pool/app/runtime.py`: `Serialized` → an edge-triggered `queued` event.
- `services/orion-llamacpp-host`: compose interpolation plus two `.env_example` keys.

## Files changed

- `config/gpu_pool.yaml`: `world.serialize_with`, class world `wait`, `agent-gpu2` `max_hold_sec: 9000`
  plus `cuda_env`/`profile_var`, and experiment bridge verbs removed.
- `orion/gpu_pool/config.py`: the schema fields, validation and gates listed above.
- `orion/gpu_pool/scheduler.py`: Z1 and `Serialized`.
- `services/orion-gpu-pool/app/runtime.py`: reports `Serialized` once per (lease, blocker).
- `services/orion-gpu-pool/evals/run_pool_day_eval.py`: tolerates and counts `Serialized`.
- `scripts/check_gpu_pool_config.py`: docstring (the gate logic lives in `check_launch`).
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`: agent-burst device and profile
  interpolation.
- `services/orion-llamacpp-host/.env_example`: `ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES=2`,
  `ATLAS_AGENT_BURST_PROFILE_NAME=`.
- `orion/gpu_pool/tests/test_stage5_config.py` (new): live config, gates, worked examples A and B,
  and the Z1 scheduler rule.
- `orion/gpu_pool/tests/{test_scheduler,test_scheduler_holds,test_stage4_contracts}.py` and
  `services/orion-gpu-pool/tests/{test_runtime,test_client_roundtrip}.py`: updated for world `wait`,
  9000 s and the new gate message. The gpu4 fixture now uses interpolated devices. Added a runtime
  test for the serialized report.
- `services/orion-gpu-pool/README.md` and `services/orion-gpu-lane-controller/README.md`: stage 5.1
  notes, and the rule that the controller's checkout and image must move together.

## Schema / bus / API changes

- **Added:** YAML keys `roles.<r>.serialize_with`, `roles.<r>.launch.profile_var` and
  `roles.<r>.launch.profiles`.
- **Added:** a pool event value `event=queued`, `reason=serialized:<role>`, `detail={"serialized": true}`
  on the existing `orion:gpu_pool:event`. `GpuPoolEventV1.reason` is free text, so there is no model
  change.
- **Removed:** `experiment.swap.load`/`unload` (YAML values, not schema).
- **Behaviour changed:** `cuda_env` now means the compose interpolation variable. No code reads it
  at runtime yet; the controller's `gpu2.py` still hard-codes its own targets until 5.2 and 5.3.
- **Compatibility:**
  - The `launch_digest` of `agent-gpu2` changes (`a38205b331d0…` → `4efbb051879a…`), and so does
    diffusion's (`1c0f607f7133…` → `1db8c5aef416…`). `LaunchSpec` gained fields, so the digest moves
    even for identical YAML.
  - The circe controller must run this commit (see Restart required). Otherwise every actuation is
    refused with `launch_digest_mismatch` (old YAML) or `config_unloadable:*` (new YAML read by old
    code). Both are safe, but no 27B load happens until they match.
  - No channel or registry changes.

## Env/config changes

- **Added keys:** `ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES=2`, `ATLAS_AGENT_BURST_PROFILE_NAME=` (empty,
  which falls back to `ATLAS_AGENT_PROFILE_NAME`).
- **Removed / renamed keys:** none.
- **`.env_example` updated:** `services/orion-llamacpp-host/.env_example`.
- **Local `.env` synced:** `python scripts/sync_local_env_from_example.py` alone skipped both keys,
  because `ATLAS_` is outside `SYNC_PREFIXES`. `python scripts/sync_local_env_from_example.py
  orion-llamacpp-host --all-keys` added exactly these two to athena's
  `services/orion-llamacpp-host/.env`. It reported one pre-existing divergence and left it alone:
  `ATLAS_AGENT_HOST_PORT` local 8014, example 8015.
- **Operator action on circe (not done here):** in circe's primary checkout, run
  `python scripts/sync_local_env_from_example.py orion-llamacpp-host --all-keys`, or add the two keys
  by hand. This is optional for correctness: with the keys absent, the compose defaults give device 2
  and `ATLAS_AGENT_PROFILE_NAME`.

## Tests run

```text
PYTHONPATH=. python scripts/check_gpu_pool_config.py
  check_gpu_pool_config: ok (4 cards, 8 roles, 7 classes, 2 launch blocks, digest e58f5238846b0bb7)
PYTHONPATH=. python -m pytest orion/gpu_pool/tests -q                 290 passed
cd services/orion-gpu-pool && python -m pytest tests -q               91 passed, 7 skipped (Postgres, run in CI)
cd services/orion-gpu-lane-controller && python -m pytest tests -q    77 passed
cd services/orion-durable-runs && python -m pytest tests -q           188 passed, 64 skipped (Postgres)
cd services/orion-llm-gateway && python -m pytest tests -q            304 passed
python -m pytest orion/llm/tests/test_routes.py tests/test_record_lane_occupancy.py -q   69 passed
static gates: check_metric_lineage --gate PASS, check_definition_drift --gate PASS (no metric
  definition changed, so no re-lock), check_env_template_parity PASS, check_service_hostname_refs OK,
  check_compose_no_relative_mounts PASS, check_chat_route_poachers PASS,
  tests/scripts/test_sync_local_env_from_example.py + test_check_env_template_parity.py +
  tests/test_agent_trace_schema_registry.py 39 passed
```

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py   VERDICT: PASS (before and after)
```

The eval's day of traffic includes world and diffusion together, so the mutex is visible in it.
Compared with origin/main on the same simulation:

- world system p95 wait went from 2 s to 48 s (world now waits while a diffusion call runs);
- diffusion system p95 wait went from 117 s to 17 s;
- backlog decisions went from 111 to 1 (world no longer backlogs);
- owner starvation, lost leases and small-role violations stay at 0;
- the scheduler counted 8839 `serialized:diffusion` and 18 `serialized:world` reports, once per
  tick. The runtime emits one per lease.

In production (5.4) world's 2 s deadline turns that wait into today's `gpu_contended`, as the permit
does now.

## Docker/build/smoke checks

```text
docker compose --env-file <tmp> -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml \
  --profile agent-burst config atlas-agent-burst
  env file has only ATLAS_AGENT_PROFILE_NAME=p27b      -> CUDA_VISIBLE_DEVICES_OVERRIDE "2", LLM_PROFILE_NAME p27b
  plus BURST_PROFILE_NAME= and BURST_CUDA=2            -> "2", p27b
  plus BURST_PROFILE_NAME=other and BURST_CUDA=3       -> "3", other
  process env ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES=5 -> "5" (process env beats --env-file: the
                                                          mechanism the 5.2 actuator relies on)
```

No image was built. Nothing was deployed.

## Review findings fixed

A code-review subagent reviewed commit 6f1fb8ec7. It found no blockers. Every finding is fixed
except where noted.

- **Finding (should-fix): younger world work could jump an older diffusion hold and starve it.**
  world has 2 slots, so overlapping calls kept diffusion blocked. That contradicted "keeps its
  place".
  - Fix: a lease blocked only by serialize reserves the blocking partner role. No later lease in
    queue order (priority, then age) is granted there, in either grant loop.
  - Evidence: `test_an_older_waiter_keeps_its_place_against_a_stream_on_the_other_side` and
    `test_higher_priority_still_goes_first_across_the_pair`. The first test fails when the
    reservation is removed (mutation checked).
- **Finding (should-fix): the deploy ordering was under-specified.** The digest changes, and the
  controller's image and checkout are coupled.
  - Fix: the Restart section and the controller README spell out pull + rebuild on circe and a
    rebuild on athena at the same commit, done while gpu2 has diffusion resident.
- **Finding (should-fix): `PoolConfig.not_actuatable()` had no consumer** (a keyword cathedral).
  - Fix: the method is removed. Only the validator exemption for an operator-only seat without a
    launch block stays. The Hub "not_actuatable" surface belongs with 5.7, where operator holds
    become possible.
  - Not fixed here, and pre-existing: an operator lease for `experiment` still drains everything the
    seat evicts, although nothing can load it. It is unreachable today, because operator holds are
    refused in observe mode. Follow-up for 5.7.
- **Finding (nit): `_report_serialized` marked the event as reported before publishing it.**
  - Fix: the key is now added after `_emit` returns.
- **Finding (nit): the report loop could name the wrong cause** for a lease held back by the urgent
  cap, or on a borrowed role whose owners come first.
  - Fix: both cases are skipped.
  - Evidence: `test_urgent_capped_lease_is_not_reported_serialized`, which fails without the fix
    (mutation checked).
- **Finding (nit): an owner blocked only by serialize dropped out of `owners_waiting`**, which could
  let a borrower in.
  - Fix: `owners_waiting` now uses `usable and fits`, without the serialize check.
- **Finding (nit): `serialized_with()` rescanned every role on every call.**
  - Fix: it is built once, in a private attribute, at validation time.
- **Finding (nit): a nested device default (`${X:-${Y}}`) was compared as a literal** (a false
  positive).
  - Fix: the literal comparison is skipped for an interpolated default. The resolved-value check
    still applies.
  - Evidence: `test_gate_nested_device_default_is_not_read_as_a_literal`.
- **Finding (nit): `device_ids: ["${OTHER}"]` passed the gate.**
  - Fix: `device_ids` entries must interpolate `cuda_env`.
  - Evidence: `test_gate_accepts_a_device_ids_pin_through_cuda_env_and_rejects_another_var`.
- **Finding (nit): the profile check fell back to the repo's own `llm_profiles.yaml`** when the
  checked tree had none.
  - Fix: it now checks only the tree under test, and reports "not found" otherwise.
  - Evidence: `test_gate_checks_profiles_against_the_tree_it_checks`.
- **Finding (nit): test gaps.**
  - Fix: the child-blocks test now asserts the reason, and the queue-order and urgent-cap tests
    were added.

## Spec corrections found while implementing

1. **The 5.1 deploy row says "circe pull". That alone breaks the controller.** Its YAML comes from
   the host checkout, but its parser is baked into the image, and the models reject unknown keys.
   New keys plus the old image means every action refuses `config_unloadable:*`. The correct step is
   pull **and** `safe_docker_build.sh orion-gpu-lane-controller up -d --build`.
2. **"The waiting lease keeps its place" needs an explicit reservation.** A plain "not placeable
   while the partner is active" rule lets a two-slot role starve its partner. 5.1 implements the
   reservation (see above).
3. **The diffusion `launch_digest` also changes**, not only agent-gpu2's. Any new `LaunchSpec` field
   changes every launch digest.
4. **The default `sync_local_env_from_example.py` run skips the new `ATLAS_` keys**, because they are
   outside `SYNC_PREFIXES`. It needs `orion-llamacpp-host --all-keys`. The spec says "then sync".
5. **The Decision 3 "validator marks experiment `not_actuatable`" has no consumer until 5.7.** 5.1
   ships only the exemption.

## Restart required

Do not deploy until Juniper says so. Order when deploying (athena and circe **at the same commit**,
because of the `launch_digest` fence):

```bash
# 1. athena: confirm gpu2 idle (no actuation in flight): curl -s :8127/v1/pool | jq '.cards[]|select(.card=="gpu2")'
git -C /mnt/scripts/Orion-Sapienform pull --ff-only
# 2. circe: pull AND rebuild the controller together. Its YAML comes from the host checkout, but
#    its parser is baked into the image; new YAML keys + old image = config_unloadable:* refusals.
ssh circe 'cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && \
  python scripts/sync_local_env_from_example.py orion-llamacpp-host --all-keys && \
  scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build'
# 3. athena: rebuild the pool (config is baked into its image)
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
# 4. optional, same YAML baked in, no behaviour depends on it yet: llm-gateway, durable-runs images
```

`atlas-agent-burst` itself needs no restart. The next pool load starts it with the same resolved
device and profile.

## Risks / concerns

- **Medium: a deploy on one host only stalls 27B loads.** The digest fence refuses safely, but
  `agent-gpu2` cannot load until athena and circe match. Mitigation: the deploy order above, and the
  new controller README note.
- **Low: the 2.5 h seat limit delays image generation.** Accepted by Juniper on 2026-09-29.
- **Low: a merge conflict with PR #2407** (fresh pool-event correlation ids). Both append to
  `services/orion-gpu-pool/tests/test_runtime.py`. The code hunks do not overlap, and
  `_report_serialized` goes through `_emit`, so it inherits #2407's envelope id.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2408

🤖 Generated with [Claude Code](https://claude.com/claude-code)
