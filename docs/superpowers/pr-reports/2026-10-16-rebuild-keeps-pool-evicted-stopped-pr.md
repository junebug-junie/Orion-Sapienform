## Summary

- Orion has not rendered a picture (waking painting or dream hop) since 2026-10-16 03:02 UTC. Root cause: a routine `git pull` on circe auto-rebuilt `orion-diffusion-host` and **started it**, while the GPU pool had stopped it on purpose to lend gpu2 to the agent-gpu2 seat.
- With both containers running on gpu2, the lane controller refuses every load and unload (`seat_and_evicted_both_running`). The pool marks gpu2 `fault`, and every diffusion lease backlogs `no_serviceable_role` until its deadline.
- Fix: the post-pull rebuild now asks `rebuild_affected_services.py --pool-held-services <svc>` which GPU-pool launch targets (`config/gpu_pool.yaml` `launch:`) are stopped right now. It builds their image and leaves them stopped. The lane controller starts them with the new image on its next load.
- Docker unreadable means "hold every target": unknown state never starts something on a pool card.
- The live gpu2 state still needs a one-time operator fix (below). This PR prevents the next pull from causing it again.

## Outcome moved

Failure mode removed: "a pull that touches shared `orion/` code restarts a pool-evicted service and wedges its GPU card". Live evidence of the failure (all times UTC, 2026-10-16):

- 03:17:54 pool loads agent-gpu2 on gpu2 (`observed: agent-gpu2 running, diffusion exited`).
- 04:01:57 circe `pull: Fast-forward` (reflog), `.orion/post-merge-rebuild-enabled` present.
- 04:02:39 new diffusion-host image built; 04:02:46 diffusion-host container started.
- 04:01:19 pool `swap_failed reconcile_ambiguous:boot`, `observed: agent-gpu2 running, diffusion running`.
- 04:16:15 and 18:17:04 controller `launch_exec_load_failed role=agent-gpu2 reason=seat_and_evicted_both_running`; gpu2 `swap_state=fault`.
- 04:34 onward: every `reverie-visual-*` diffusion lease `backlogged:no_serviceable_role` then `released:deadline` (5 runs). Runs that started while another was stuck failed `attempt_unresolved` at `prepare` (2 runs).
- nvidia-smi on circe: the gpu2 23.8 GB process is the agent-burst llama-server (pid 3846941); diffusion-host holds no GPU memory, yet its `/health` answers 200.

## Current architecture

`scripts/git_hooks/post-merge` -> `scripts/rebuild_services_from_git_diff.sh` -> `safe_docker_build.sh <svc> up -d --build` for every affected service. Profile-gated services were already protected ("never start a profile service that is not already running"). Plain services that the GPU pool starts and stops itself (today: `diffusion-host`) had no such protection.

## Architecture touched

- Post-pull rebuild only. No pool, controller, schema, bus, or env change.

## Files changed

- `scripts/rebuild_affected_services.py`: `pool_launch_targets`, `pool_held_services`, `--pool-held-services` CLI.
- `scripts/rebuild_services_from_git_diff.sh`: hold branch (`build <held>` + `up -d --build --no-deps <others>`), dry-run output.
- `tests/scripts/test_rebuild_affected_services.py`: helper tests, including one pinned to the real `config/gpu_pool.yaml`.
- `tests/scripts/test_rebuild_services_from_git_diff.py`: shell behavior tests (fail on the old script, verified).

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: post-pull rebuild no longer starts a stopped GPU-pool launch target.
- Compatibility notes: a running launch target keeps the old `up -d --build` path.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest -q tests/scripts/test_rebuild_affected_services.py tests/scripts/test_rebuild_services_from_git_diff.py
89 passed
# New shell tests against the pre-fix script: 3 failed (regression confirmed).
```

## Evals run

```text
No eval harness for the rebuild scripts; behavior is deterministic and covered by the gate tests above.
```

## Docker/build/smoke checks

```text
python3 scripts/rebuild_affected_services.py --pool-held-services orion-diffusion-host --docker-ps-file <empty>   -> hold	diffusion-host
python3 scripts/rebuild_affected_services.py --pool-held-services orion-diffusion-host --docker-ps-file <running> -> (nothing)
No live rebuild run (would deploy on circe).
```

## Review findings fixed

See the review section appended below.

## Restart required

The code fix needs no restart (it runs at the next pull). The live wedge needs one operator step on circe — stop the stray diffusion-host so gpu2 matches what the pool thinks (agent-gpu2 loaded, diffusion evicted). Pool discovery then clears the fault by itself and swaps diffusion back in on the next picture request:

```bash
ssh circe@circe docker stop orion-circe-diffusion-host
```

If gpu2 still reads `swap_state: fault` in `curl -fsS localhost:8127/v1/pool` a minute later, clear it from the Hub GPU panel (pool control verb `clear_fault`, card gpu2). Expected path after the stop: discovery sees agent-gpu2 alive and diffusion down, adopts "agent-gpu2 loaded", and the next diffusion lease triggers an owner-reclaim unload. UNVERIFIED until it runs.

## Risks / concerns

- Severity: medium. Concern: the pool cannot recover from "seat and evicted both running" by itself; it clears the fault, retries the load, and faults again. Mitigation: follow-up (controller could stop the evicted resident when the seat is the one actually holding VRAM, or the pool could alert).
- Severity: medium. Concern: the pool probes diffusion `/health`, which answers 200 while the model holds no GPU memory; `launch.ready` is `/ready`. Mitigation: follow-up.
- Severity: low. Concern: a running launch target is still recreated by a pull without a pool drain (unchanged behavior).

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
