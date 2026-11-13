## Summary

- Replace Circe GPU3's separate fast and Qwen3 metacog workers with one Unsloth Qwen3.5 9B UD-Q5_K_XL worker on port 8013.
- Give the shared worker ten 32,768-token slots (`--ctx-size 327680 --parallel 10`).
- Keep fast and metacog route names, priorities, backlog behavior, and request policies. Both classes own the one physical GPU pool role.
- Default thinking off per call; preserve explicit thinking on, and retain metacog's JSON-oriented sampling/output defaults in the gateway.
- Retire the `atlas-metacog` compose service and the duplicate cortex-exec profile pin. Add a live deployment and acceptance runbook.

## Outcome moved

The pool can place up to ten fast/metacog calls on one GPU3 worker with 32k per request. The prior four 4k fast slots spilled prompts up to 22.9k tokens to agent roles. Real 10 × 32k VRAM, throughput, JSON quality, and caller artifacts remain **UNVERIFIED until deployment**.

## Current architecture

Before this patch, `atlas-fast` served Qwen3.5 Q5 on four 4k slots and `atlas-metacog` served Qwen3 8B Q5 on four 4k slots. Both shared GPU3 with Kev. GPU pool classes and priorities were already separate; gateway metacog and cortex-exec each pinned the old profile.

## Architecture touched

- `config/llm_profiles.yaml`: one new ten-slot Qwen3.5 profile. The old four-slot profile key remains solely for Athena-first discovery during cutover.
- `config/gpu_pool.yaml`: one GPU3 LLM role owned by both classes; operator metacog evals use the same physical role.
- `services/orion-llm-gateway`: request-level thinking and metacog defaults for bus and OpenAI HTTP paths; profile pin selects the shared model.
- `services/orion-cortex-exec`: leaves an omitted metacog profile for the gateway to select.
- `services/orion-llamacpp-host`: no standalone metacog compose service; Circe fast worker selects the new profile and b10398 image.

## Files changed

- `config/gpu_pool.yaml`, `config/llm_profiles.yaml`: shared role and ten-slot profile.
- `services/orion-llm-gateway/app/fast_thinking.py`, `llm_backend.py`, `passthrough_proxy.py`: per-class request defaults and explicit thinking override.
- `services/orion-cortex-exec/app/executor.py`, `app/settings.py`: remove duplicate metacog profile pin.
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`: remove the old worker.
- Service `.env_example`, compose, and README files: update profile, CUDA binding, image tag, and retired keys.
- `orion/gpu_pool/tests/`, affected service tests, and `services/orion-llamacpp-host/evals/run_qwen35_fast_eval.py`: test single-role placement, profile discovery, thinking behavior, 32k slots, and live ten-seat overlap.
- `docs/operations/qwen35-gpu3-consolidation.md`: host order, manual worker cutover, and live checks.

## Schema / bus / API changes

- Added: no schema or bus event.
- Removed: physical GPU pool role `metacog` and its port 8012 worker.
- Renamed: no caller route.
- Behavior changed: `metacog*` and `operator_metacog` target physical role `fast`; gateway applies metacog sampling/output defaults and thinking off unless explicitly enabled.
- Compatibility notes: old four-slot fast profile key remains in the registry for Athena-first rolling discovery. Existing route names and priority classes remain.

## Env/config changes

- Added keys: none.
- Removed keys: `ATLAS_METACOG_SERVICE_NAME`, worker `ATLAS_METACOG_PROFILE_NAME`, `ATLAS_METACOG_CUDA_VISIBLE_DEVICES`, and `ATLAS_METACOG_HOST_PORT`; cortex-exec's duplicate `ATLAS_METACOG_PROFILE_NAME`; gateway's unused metacog CUDA/port keys.
- Renamed keys: none.
- `.env_example` updated: fast selects the shared Qwen3.5 profile, b10398, and CUDA device 3; gateway metacog pin selects the same profile.
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: command run after changes; touched worktree service `.env` files copied from the final examples and ignored by git. The script reads the primary checkout, where cortex-exec and GPU pool have no `.env`, so it reported those skips. Athena and Circe live `.env` values require the manual edits in the runbook before restart.
- Skipped keys requiring operator action: Athena gateway's divergent scalar `ATLAS_METACOG_PROFILE_NAME`, Circe's `ATLAS_FAST_PROFILE_NAME`, and removal of cortex-exec's retired key. No secret or `.env` file is committed.

## Tests run

```text
PYTHONPATH=$PWD python3 -m pytest orion/gpu_pool/tests: 572 passed
PYTHONPATH=$PWD/services/orion-gpu-pool:$PWD python3 -m pytest services/orion-gpu-pool/tests: 202 passed, 15 skipped
PYTHONPATH=$PWD/services/orion-llm-gateway:$PWD python3 -m pytest services/orion-llm-gateway/tests: 492 passed
LLM_PROFILE_NAME=<shared-profile> PYTHONPATH=$PWD/services/orion-llamacpp-host:$PWD python3 -m pytest services/orion-llamacpp-host/tests: 165 passed
PYTHONPATH=$PWD/services/orion-gpu-lane-controller:$PWD python3 -m pytest services/orion-gpu-lane-controller/tests: 110 passed
PYTHONPATH=$PWD/services/orion-cortex-exec:$PWD python3 -m pytest services/orion-cortex-exec/tests/test_metacog_route_profile.py: 3 passed
python3 scripts/check_gpu_pool_config.py: PASS
python3 scripts/check_circe_worker_refs.py: PASS
git diff --check: PASS
```

## Evals run

```text
run_qwen35_fast_eval.py --help and py_compile: PASS
Live direct worker and mixed gateway eval: UNVERIFIED until approved production cutover. The eval now requires ten simultaneously occupied /slots and exercises a 24k prompt, JSON, tools, logprobs, and thinking on/off.
```

## Docker/build/smoke checks

```text
safe_docker_build.sh orion-llamacpp-host --compose-file docker-compose.atlas-workers.yml build atlas-fast: PASS (b10398)
safe_docker_build.sh orion-llm-gateway build: PASS
safe_docker_build.sh orion-gpu-pool build: PASS (local primary has no pool .env; build-only ORION_ALLOW_ENV_DRIFT=1)
safe_docker_build.sh orion-gpu-lane-controller build: PASS (same build-only env exception)
docker build -f services/orion-cortex-exec/Dockerfile: PASS; local compose tool rejects its pre-existing `gpus` field
No production worker was restarted; live /health, /props, /slots, VRAM, and caller artifacts: UNVERIFIED
```

## Review findings fixed

- Finding: A `fast`-only owner on the shared role let fast system calls jump ahead of metacog perception calls.
  - Fix: Both classes own the one physical role.
  - Evidence: `test_shared_gpu3.py` grants ten mixed leases, with all perception calls ahead of lower-priority fast calls; full pool suites pass.
- Finding: The first eval version sent ten short requests without proving overlap.
  - Fix: barrier-synchronized long requests, `/slots` active sampling, a peak-ten assertion, and an optional mixed gateway pass.
  - Evidence: eval code compiled and independently reviewed; live run is pending deployment.
- Finding: A fresh worker env could expose every GPU because the CUDA binding was blank.
  - Fix: set `ATLAS_FAST_CUDA_VISIBLE_DEVICES=3` in the example, quickstart, and runbook.
  - Evidence: env parity and independent review.
- Finding: Post-pull does not rebuild the Atlas compose file; Athena's live gateway pin remains the old Qwen3 profile unless manually changed.
  - Fix: runbook orders Athena's env edit/pull first, retains the old fast profile as a transition alias, then requires explicit Circe worker restart and old-container removal.
  - Evidence: reviewer inspected the live post-pull helper and host env; no remaining material review findings.

## Restart required

See `docs/operations/qwen35-gpu3-consolidation.md` for exact host order and acceptance. After merge and explicit production-write approval, update the Athena/Circe local env values, pull clean primary main on each host, stop/remove only the old metacog container, and run:

```bash
scripts/safe_docker_build.sh orion-llamacpp-host --compose-file docker-compose.atlas-workers.yml up -d --build atlas-fast
```

Also confirm or deploy Athena `orion-gpu-pool`, `orion-llm-gateway`, `orion-cortex-exec` and Circe `orion-gpu-lane-controller` from clean main. Run the live eval and inspect real caller artifacts before reporting success.
