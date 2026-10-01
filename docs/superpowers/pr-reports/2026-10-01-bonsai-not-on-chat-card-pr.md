## Summary

GPU pool stage 7.0. Bonsai (the Ternary-Bonsai-2-27B bake-off worker) can no longer default onto gpu0, which is chat's card. Juniper's rule (2026-09-30): Bonsai runs on gpu1 or gpu2 only.

- The compose file no longer has a default card. If `BONSAI_CUDA_VISIBLE_DEVICES` is missing, `up` fails with "set to 1 or 2 - never 0, which is chat's card" instead of picking one.
- The `.env_example` template says `2`, and the profile's doc-only `device_ids` says `[2]`.
- README: the "stop chat, start Bonsai on gpu0" swap is gone. It now documents gpu1/gpu2 only, the gpu2 bake-off path (wait for `agent-gpu2` to unload, then pause pool actuation), and that this service is a bake-off tool retired in stage 7.6.
- New gate: a test fails if any Bonsai config (compose, template, any `*bonsai*` profile) points at chat's card. It reads card indices from `config/gpu_pool.yaml`, so it follows a card move.

## Outcome moved

Before: a fresh `.env` from the template, or a compose run with no value, put Bonsai on gpu0 next to (or instead of) the 35B chat worker. After: the repo has no path to gpu0, and CI fails if one comes back. Host `.env` files written before this change still need the one-line fix below; the test cannot see them.

## Current architecture

PR #2434 (merged 2026-09-30 20:32 UTC) moved Bonsai's default from gpu2 to gpu0 and documented stopping the chat worker for a bake-off. The service is manual-only (no auto-rebuild list), not a pool role (`LLM_ROLE=bonsai-bakeoff`, listed under `unclaimed`), and `restart: "no"`.

## Architecture touched

`services/orion-llamacpp-bonsai-host` (compose, template, README, tests) and the Bonsai profile in `config/llm_profiles.yaml`. No pool, gateway, bus or schema change.

## Files changed

- `services/orion-llamacpp-bonsai-host/docker-compose.yml`: `${BONSAI_CUDA_VISIBLE_DEVICES:-0}` -> `${BONSAI_CUDA_VISIBLE_DEVICES:?...}`; header comment.
- `services/orion-llamacpp-bonsai-host/.env_example`: `0` -> `2`, with a never-0 comment.
- `config/llm_profiles.yaml`: Bonsai profile `device_ids: [0]` -> `[2]` (doc-only; the real pin is the env var).
- `services/orion-llamacpp-bonsai-host/README.md`: swap path replaced; cards section; 7.6 retirement note. The gpu0 VRAM numbers stay, labelled as measured before this rule.
- `services/orion-llamacpp-bonsai-host/tests/test_bonsai_contract.py`: `test_no_bonsai_config_targets_chats_card`, `test_compose_refuses_to_start_without_a_card`.

Required vs default 2: the spec allows either. Required was chosen because gpu2 is not free either (the pool lends it to `agent-gpu2` and diffusion), so picking a card should be a deliberate operator act, not a silent default.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none at runtime (manual-only service, nothing deployed).
- Compatibility notes: `docker compose ... config/up` for this service now fails without `BONSAI_CUDA_VISIBLE_DEVICES`.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes, `BONSAI_CUDA_VISIBLE_DEVICES` 0 -> 2 (meaning: required, gpu1/gpu2 only).
- local `.env` synced with `python scripts/sync_local_env_from_example.py orion-llamacpp-bonsai-host --all-keys`: the script reported the key as diverged (`local='0' example='2'`) and left it. `--force` rewrites the whole file, so athena's line was fixed by hand (one line plus its comment); a re-run reports no changes needed.
- skipped keys requiring operator action: **circe** `/mnt/scripts/Orion-Sapienform/services/orion-llamacpp-bonsai-host/.env` line 26 is `BONSAI_CUDA_VISIBLE_DEVICES=0`. Change it to `BONSAI_CUDA_VISIBLE_DEVICES=2`. No Bonsai container exists on circe (running or stopped), so nothing needs stopping.

## Tests run

```text
cd services/orion-llamacpp-bonsai-host && python -m pytest tests -q -p no:cacheprovider
7 passed (docker compose test ran, not skipped)

Mutation checks (each reverted after):
  .env_example =0                        -> 1 failed
  .env_example =2,0                      -> 1 failed
  compose back to :-0                    -> 2 failed
  profile device_ids [1, 0]              -> 1 failed

python scripts/check_gpu_pool_config.py   -> ok (4 cards, 8 roles, 7 classes, 2 launch blocks)
python scripts/check_env_template_parity.py -> PASS (94 services)
git diff --check                          -> clean
```

## Evals run

```text
None. Config/docs-only change to a manual-only service; no eval harness applies.
The contract test is the gate.
```

## Docker/build/smoke checks

```text
docker compose config (inside the test): fails with no card ("never 0" in stderr),
renders CUDA_VISIBLE_DEVICES_OVERRIDE "2" with BONSAI_CUDA_VISIBLE_DEVICES=2.
No image built, no container started (per task: do not deploy).
```

## Review findings fixed

Code review ran in a subagent: no must-fix findings. Compose `:?` syntax, mutation coverage and CI behavior (`docker compose config` needs no daemon or network) were checked.

- Finding (should): README pause command would not run as written (script not executable, needs bus URL and PYTHONPATH).
  - Fix: copied the script's own documented invocation for `pause` and `resume`.
  - Evidence: `scripts/gpu_pool_pause.py` docstring lines 9-10.
- Finding (should): `test_never_auto_deployed` docstring still said Bonsai defaults to gpu0.
  - Fix: reworded to the gpu1/gpu2 pool-card collision reason.
  - Evidence: 7 passed after the change.
- Finding (nit): "`nvidia-smi -i 2` shows no process" may never be true (world-model lane lives on gpu2).
  - Fix: README now says no agent or diffusion process; world-model may stay.
  - Evidence: `config/gpu_pool.yaml` roles on gpu2.
- Finding (nit): `:?` blocks `down`/`logs` too, not only `up`.
  - Fix: README says every compose command fails, including `down`.
  - Evidence: compose interpolation applies to all subcommands.
- Not fixed (nit, out of scope): no runtime guard rejects an explicit `0` in a host `.env`. The service is retired in 7.6; the README grep step and the circe line below cover it.

## Restart required

```text
No restart required. Nothing is running. On circe, edit the .env line above before the next manual Bonsai `up`.
```

## Risks / concerns

- Severity: low
- Concern: the gate only covers the repo. A host `.env` with `0` still lands on chat's card (circe's does today).
- Mitigation: README tells the operator to grep the value before `up`; circe line called out above.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2447

🤖 Generated with [Claude Code](https://claude.com/claude-code)
