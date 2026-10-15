## Summary

- Kev, the small System One model that gates Orion's self-started curiosity, moves from athena's T10 to circe GPU 3.
- GPU 3 already runs the fast (:8013) and metacog (:8012) lanes. They use ~15GB of 32GB. Kev uses ~3.9GB.
- orion-substrate-runtime on athena now calls Kev at `http://100.112.254.99:8009` over Tailscale.
- `config/gpu_pool.yaml` gains a `kev` service role on gpu3. The pool lists and probes it. The pool never schedules on it.
- A contract test pins Kev's GPU index, port and host to the pool role.

## Outcome moved

- Athena's T10 is 13.5/16GB full across four processes, and Kev shares it with whisper. This move frees ~4GB there.
- Kev gets its own room on a card with ~17GB free.
- No lane is killed. Live gateway logs (about 12h) show that both lanes are still busy:
  - metacog: ~2,440 leases. vision-council sends 1,694. Live Hub turns send `metacog_turn`.
  - fast: ~835 leases. orion-topic-foundry sends 715.
  - A merge would make Hub turns wait behind the vision-council load.

## Current architecture

- Kev ran as `orion-athena-kev` on athena GPU 1 (T10).
- Substrate reached Kev by Docker DNS on `app-net`.
- The GPU pool did not know about Kev.

## Architecture touched

- `services/orion-kev`: env template now targets circe (`PROJECT=orion-circe`, `KEV_GPU_DEVICE_ID=3`).
- `services/orion-substrate-runtime`: the Kev URL now points at circe's Tailscale IP.
- `config/gpu_pool.yaml`:
  - adds a `kev` role: `kind: service`, `cards: [gpu3]`, `port: 8009`, `vram_gb: 4`, `health: /docs`
  - the role has no class and no route

## Files changed

- `config/gpu_pool.yaml`: the `kev` role. The header now says a service can have no class.
- `services/orion-kev/.env_example`: circe project, GPU 3, comments.
- `services/orion-kev/README.md`: circe bring-up, move order, rollback note.
- `services/orion-kev/Dockerfile`: comment only.
- `services/orion-kev/tests/test_orion_kev_contract.py`: two drift tests.
- `services/orion-substrate-runtime/.env_example`: Kev URL. Removed the stale comment "no behavioral consumers".
- `services/orion-substrate-runtime/README.md`: Kev location.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Kev is on a new host.
  - The pool probes `http://100.112.254.99:8009/docs`.
- Compatibility notes: if Kev is down, the curiosity gate fails open. This is the same as before.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- Changed values:
  - `SUBSTRATE_SYSTEM_ONE_BASE_URL` → `http://100.112.254.99:8009`
  - `PROJECT` → `orion-circe` (orion-kev)
  - `KEV_GPU_DEVICE_ID` → `3`
- `.env_example` updated: yes.
- Local `.env` synced:
  - The sync script reported the values as diverged and did not change them.
  - Athena's substrate `.env` URL was set by hand, for that one key only.
  - Athena's `services/orion-kev/.env` was left with the athena values on purpose, for rollback.
  - Circe has no `services/orion-kev/.env` yet. The operator creates it at deploy time.
- Skipped keys requiring operator action: create circe's orion-kev `.env` (see restart).

## Tests run

```text
services/orion-kev/tests                   5 passed
orion/gpu_pool/tests                       454 passed
services/orion-gpu-pool/tests              177 passed, 15 skipped
services/orion-llm-gateway/tests           455 passed
tests/test_check_circe_worker_refs.py etc  145 passed
tests/test_system_one_appraisal.py + substrate worker system-one tests  13 passed
scripts/check_gpu_pool_config.py           ok (10 roles)
Mutation checks:
  KEV_GPU_DEVICE_ID=1 → 1 failed
  substrate URL moved to hecate's IP → 1 failed
```

## Evals run

```text
No eval change. This patch moves where Kev is placed. It does not change Kev's behavior.
scripts/analysis/eval_system_one_appraisal.py is not affected.
```

## Docker/build/smoke checks

```text
circe: app-net exists; /mnt/scripts/kev/.hf-cache created (owned by circe).
circe:8009 is not deployed yet: live move is UNVERIFIED until the restart steps below run.
```

## Review findings fixed

- Finding: the docs said the pool "counts VRAM", but `check_vram` sums only service footprints.
  - Fix: the YAML and README now say "listed and probed" and name the gap.
  - Evidence: `config/gpu_pool.yaml` kev comment.
- Finding: the URL test checked only the port. A wrong IP or host passed.
  - Fix: the test now asserts that the URL equals `load_pool_config().url("kev")`.
  - Evidence: a mutation to hecate's IP fails the test.
- Finding: the YAML header said every service needs a class.
  - Fix: the header now notes the direct-call exception.
- Finding: the move order and the stop of the old athena container were not documented.
  - Fix: the README has a "Moving from athena" section.

## Restart required

```bash
# 1. circe, primary main checkout, after merge + pull:
ssh circe@circe "cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && cp services/orion-kev/.env_example services/orion-kev/.env && scripts/safe_docker_build.sh orion-kev up -d --build"
# 2. athena: check that circe answers
curl -fsS http://100.112.254.99:8009/docs >/dev/null && echo ok
# 3. athena primary checkout: restart substrate so it reads the new URL
scripts/safe_docker_build.sh orion-substrate-runtime up -d
# 4. athena: stop the old Kev
scripts/safe_docker_build.sh orion-kev down
```

## Risks / concerns

- Severity: low
  - Concern: substrate restarts before circe Kev is up.
  - Mitigation: the gate fails open, and the README gives the order.
- Severity: low
  - Concern: GPU 3 VRAM headroom is not enforced at runtime.
  - Mitigation: ~17GB is free now. The README says to move the GPU index and the pool role together.
- Severity: low
  - Concern: the first start downloads the Kev weights into the new circe cache.
  - Mitigation: the healthcheck allows a 180s start.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2695

🤖 Generated with [Claude Code](https://claude.com/claude-code)
