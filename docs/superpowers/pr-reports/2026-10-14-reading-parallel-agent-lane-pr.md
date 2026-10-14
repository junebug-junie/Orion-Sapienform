## Summary

- Reading now runs up to 3 reads at once per stage (`HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT=3`, shipped ON). Before, it ran exactly one at a time.
- The reread backfill gains `run --concurrency N`. Every requeue is now written down before the script waits on it, so stopping and restarting the script never queues a read twice and never loses one. It also recognizes reads that the current pre-change run requeued without writing anything down.
- The GPU pool gets **warm standby** (`swap.standby: warm`, ON for `agent-gpu2`). gpu2 keeps the 27B loaded whenever diffusion is idle, instead of unloading it after 5 minutes. Diffusion still evicts it exactly as today.
- Together, reading work can land on gpu1 and gpu2 at the same time with no 20-minute wait.

## Outcome moved

Before this change, the live pool (`/v1/pool`, 2026-10-14 04:02Z) showed one reading hold granted on `agent` (gpu1). `agent-gpu2` was unloaded, and the queue and backlog were empty. Reading never asked for a second card, and the gpu2 seat had been unloaded since 2026-10-11 15:08 (reason=idle).

With this change:

- Hub keeps 3 durable turns per stage in flight. Tests pin "3 in flight, never 4" against real SQL.
- The warm seat takes 2 of those 3 holds. A scheduler test pins the placement: one hold on `agent`, two on `agent-gpu2`.

The live effect is UNVERIFIED until deploy.

## Current architecture

**What serialized reading.** `WorldPulseReadPipeline.tick` and `WorldPulseReadStage2Pipeline.tick` each claimed ONE seed per tick (300 s). `CLAIM_SQL` sorts seeds that already have an unconsumed durable turn first, so every tick did the same thing:

1. Re-claimed the read already in flight.
2. Polled it. `poll_turn` is non-blocking.
3. Got `ReadingPending` and released the claim.

No second durable turn was ever bound, so durable-runs (`MAX_CONCURRENT_DRIVERS=4`) and the pool never saw a second agent hold.

**Pool behaviour before this change** (`orion/gpu_pool/scheduler.py`):

- `agent` (gpu1) takes one hold (`max_holds` defaults to 1).
- `agent-gpu2` loads only after a hold has waited `after_wait_sec: 1200`.
- `agent-gpu2` unloads after `swap_idle_unload_sec: 300`.
- gpu0 (`chat`) is lent only by the operator `lend` verb. Nothing looks at whether chat is idle. That is policy, unchanged here.

**Circe controller check.** The controller probe ran from this branch's YAML at 2026-10-14 ~04:40Z:

- `status`: OK, `observed={'agent-gpu2': 'exited', 'diffusion': 'running'}`, nothing in flight.
- `digest`: OK, launch digests agree.

So a load can actuate: `launch_exec.start` runs `docker compose up -d`, which starts an Exited container.

## Architecture touched

- **orion-hub reading workers.** Claim loop and claim SQL (an `exclude` list).
- **Reread backfill script.**
- **gpu-pool.** Scheduler rule W1/W2, `SwapSpec.standby*`, and `CardLive.role_active_at`. That last one is in memory and not persisted. The runtime sets it at boot and on every granted lease of a non-swap role.
- **`config/gpu_pool.yaml`.**

## Files changed

- `orion/world_pulse_read/queue.py`: `CLAIM_SQL`/`CLAIM_STAGE2_SQL` take an exclusion list. Order unchanged.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`, `world_pulse_read_stage2.py`: the tick claims up to `max_in_flight` seeds. The per-seed body was moved into `_read_claimed`/`_pass_claimed` unchanged.
- `services/orion-hub/app/settings.py`, `.env_example`, `scripts/main.py`: `HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT` (default 3).
- `services/orion-hub/scripts/reading_reread_backfill.py`: `--concurrency`, `in_flight` records, adopting unrecorded requeues, own-lease-aware pool gate, Hub parallelism refusal, tolerant JSONL loader, report treats unsettled reads as pending.
- `orion/gpu_pool/config.py`, `scheduler.py`, `config/gpu_pool.yaml`, `services/orion-gpu-pool/app/runtime.py`: warm standby.
- Tests: `services/orion-hub/tests/test_world_pulse_read_parallel_postgres.py` (new), `test_world_pulse_read_reread_backfill_postgres.py`, `orion/gpu_pool/tests/test_warm_standby.py` (new), `services/orion-gpu-pool/tests/test_warm_standby_runtime.py` (new). Older pool tests now pin `standby: off` explicitly through `cold(CFG)`.
- READMEs: `services/orion-hub/README.md`, `services/orion-gpu-pool/README.md`.

## Schema / bus / API changes

- Added: `SwapSpec.standby` (`off|warm`), `standby_after_idle_sec` (900), `standby_retry_after_fail_sec` (3600).
- Added: the `SwapLoad`/`swap_started` reason `standby`.
- Added: `SwapBlocked` reasons `shed:<name>` and `standby_backoff`, reported through the existing `swap_requested`.
- No bus channel or registry change.
- Behavior changed: a warm seat is never idle-unloaded or max_hold-drained.
- Compatibility: `standby` is not in `launch_digest` (a test pins this), so the pool and the controller need no lockstep. **However**, the circe controller re-parses `config/gpu_pool.yaml` from its checkout on every request with `extra="forbid"`. Circe's controller image must be rebuilt in the same step as circe's `git pull`.

## Env/config changes

- Added keys: `HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT=3` (orion-hub).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`services/orion-hub/.env:898`).
- Skipped keys: none.

## Tests run

```text
RUN_READING_POSTGRES=1 pytest services/orion-hub/tests/test_reading_*.py test_world_pulse_read_*.py test_curiosity_investigation.py orion/world_pulse_read/tests tests/test_world_pulse_read_*.py  -> 770 passed
pytest services/orion-field-digester/tests/test_queue_contention_digestion.py -> 18 passed
pytest orion/gpu_pool/tests -> 449 passed
(services/orion-gpu-pool) pytest tests -> 177 passed, 15 skipped
(services/orion-gpu-lane-controller) pytest tests -> 81 passed
(services/orion-durable-runs) 413 passed; (orion-llm-gateway) 455 passed; (orion-mesh-guardian) 100 passed
scripts/check_gpu_pool_config.py -> ok
scripts/check_env_template_parity.py -> PASS
Mutation check: forcing one claim per tick fails test_world_pulse_read_parallel_postgres (2 failed).
```

## Evals run

```text
pytest services/orion-hub/evals/test_reading_*_eval.py -> 38 passed
```

There is no live-pool eval for warm standby. Live evidence is the post-deploy check below.

## Docker/build/smoke checks

```text
Not deployed (CLAUDE.md: deploy only from clean primary main).
scripts/gpu_pool_actuator_probe.py --role agent-gpu2 --check status digest -> both OK (read-only).
```

## Review findings fixed

See the PR comment / follow-up commit.

## Restart required

Order matters. Deploy from the primary checkout on main, after merge.

1. **circe** (controller first: it must accept the new YAML keys):

   ```bash
   ssh circe@circe "cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build"
   ```

2. **athena**, pool, then Hub:

   ```bash
   cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-gpu-pool up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
   ```

3. **Reread backfill** (after Hub is up on this code). Stop the old run any time: its requeues are committed and the new code adopts them. Then restart:

   ```bash
   pkill -f 'reading_reread_backfill.py run'
   cd /mnt/scripts/Orion-Sapienform/services/orion-hub && PYTHONPATH=.:../.. ../../.venv/bin/python scripts/reading_reread_backfill.py run --yes --concurrency 3 >> /tmp/reading-reread-backfill/run.out 2>&1 &
   ```

**Live checks after deploy:**

- `curl -s localhost:8127/v1/pool`: within ~15 min of pool boot, `agent-gpu2` should go `loading`, then `confirmed`, with a `swap_started reason=standby` event. That assumes diffusion stayed idle and nothing was shed.
- Hub logs show `world_pulse_read_tick_claims n=3`.
- The backfill `progress.log` shows `in_flight=2` and `in_flight=3`.

## Risks / concerns

- **Severity: medium.**
  - Concern: removing the max_hold drain on warm seats. Juniper's earlier stage-5 answer (2026-09-29, `max_hold_sec: 9000`) drained gpu2 every 2.5 h. A warm seat now gives the card back only when diffusion asks.
  - Mitigation: diffusion's owner-demand drain is unchanged (tested). The rollback is `standby: off`.
- **Severity: medium.**
  - Concern: the circe controller rejects the new YAML until it is rebuilt, because it parses with `extra="forbid"`.
  - Mitigation: the deploy order above does pull and rebuild in one step.
- **Severity: low.**
  - Concern: gpu0 is still never lent automatically. That is policy (the operator `lend` verb), not changed here. The pool can therefore use gpu1 and gpu2 by itself. gpu0 is used only when Juniper lends it.
- **Severity: low.**
  - Concern: the backfill's `--concurrency 3` is refused until Hub runs `HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT>=3`. This is deliberate: an older Hub would serialize the reads and run out their timers.

## PR link

(filled in on the PR)
