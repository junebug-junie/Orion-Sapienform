# Runbook: GPU pool stage 5.7 — enforce mode, config decides actuation, one emergency stop

Spec: `docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md` (Decision 6, the
5.7 row, "Corrections from building 5.1" item 5). PR report:
`docs/superpowers/pr-reports/2026-09-30-gpu-pool-stage5-7-enforce-pr.md`.

## What this changes, in plain words

- **The pool decides which models it may load from the YAML alone.** Any swap seat with a `launch:` block in
  `config/gpu_pool.yaml` is loaded and unloaded by the pool (today: the second 27B on gpu2, `agent-gpu2`). The old
  env list `GPU_POOL_ACTUATE_ROLES` is gone; nothing reads it.
- **enforce is the default mode.** On every pool start, the pool asks circe's controller once, read-only, what gpu2
  really holds and believes the answer. If someone started or stopped the 27B by hand, the pool adopts that instead of
  reloading or fighting it. It no longer guesses from "the worker answers".
- **Operator holds work for seats that can load.** A hold on `experiment` (nothing can load it yet) is refused as
  `not_actuatable:experiment` and drains nothing. Before this, that hold would have emptied every card and loaded
  nothing (unreachable only because observe mode refused all holds).
- **One emergency stop.** The Hub GPU pool panel has "Emergency stop: pause all model loading/unloading". While
  paused, nothing is loaded or unloaded and nothing is drained for a swap; the pause survives a pool restart. See
  "Emergency stop" at the bottom.

What does **not** change: which model loads (same profile), when it loads (same thresholds), the circe controller
(no rebuild; launch digests are unchanged), leases and grants.

## Conventions

- athena commands run from a **worktree** of merged `main` (`scripts/safe_docker_build.sh` refuses the shared
  checkout); `$WT` is that worktree. `.env` files live only in the primary checkout
  (`/mnt/scripts/Orion-Sapienform`); link them into the worktree before a deploy so the container gets the live values:
  `for s in orion-gpu-pool orion-sql-writer orion-hub; do ln -sf /mnt/scripts/Orion-Sapienform/services/$s/.env $WT/services/$s/.env; done; ln -sf /mnt/scripts/Orion-Sapienform/.env $WT/.env`
- Postgres reads: `PSQL() { docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "$1"; }`
- **Every step that changes production is marked [GO] and needs Juniper's explicit go.** Unmarked steps are read-only.

## Live state when this was written (2026-09-30 ~06:50 UTC)

- athena pool: `GPU_POOL_MODE=observe`, `GPU_POOL_ACTUATE_ROLES=agent-gpu2` (container env and primary `.env`
  lines 21 and 37).
- gpu2: `swapped_in={agent-gpu2}`, `swap_state=idle`, `swap_generation=79`: the first real **generic-path load**
  succeeded at 06:32:26 (`swap_started` 06:32:10 → `swapped` 06:32:26, reason `demand`).
- The first real **generic-path unload** has **not happened yet** (the 27B is still loaded). That is precondition P1
  below: do not deploy 5.7 until it has.

## Deploy order and why

1. **sql-writer first.** It validates every pool event against `GpuPoolEventV1`, whose event list gains
   `actuation_paused`/`actuation_resumed`. An old sql-writer would drop those rows (the pool keeps working; only the
   history row is lost).
2. **Migration v3 before the pool** (recommended). Until 2026-09-30 a 5.7 pool refused to boot without the two
   new `gpu_pool_cards` columns, and a pool that will not boot is a full LLM outage (every gateway call leases
   through it) -- that happened 2026-09-30 09:01-09:09. Since the boot self-heal (services/orion-gpu-pool/README.md,
   "Boot schema self-heal") the pool adds them itself; if it cannot get the lock it serves degraded and
   `curl -s localhost:8127/health | jq .schema` says so. Running the file first just makes the boot a no-op.
3. **The pool**, with `GPU_POOL_MODE=enforce`.
4. **Hub last.** The Hub builds `GpuPoolControlV1` locally; its new verbs are refused by an old pool's validator.
5. **circe: nothing.** No change to any launch block or to any config model field, so launch digests are unchanged
   (`scripts/check_gpu_pool_config.py` prints the same digest as main before this PR).

## Step 0 — preconditions (read-only; all must hold)

**P1. The first real generic unload has succeeded** (after the 06:32 generic load):

```bash
PSQL "SELECT generated_at, event, role, reason, detail->>'action' AS action, detail->>'action_id' AS action_id
      FROM gpu_pool_events
      WHERE role='agent-gpu2' AND event IN ('swap_started','swapped','swap_failed','actuate_refused')
        AND generated_at > '2026-09-30 06:32' ORDER BY generated_at"
# need: a swap_started action=unload followed by swapped action=unload (reason idle, max_hold or owner_reclaim).
# A swap_failed or actuate_refused for the unload = STOP; investigate before 5.7.
PSQL "SELECT card, swapped_in, swap_state, swap_generation, swap_action->>'outcome' FROM gpu_pool_cards WHERE card='gpu2'"
# need: swap_state=idle, outcome=succeeded
```

**P2. The actuator probe agrees** (from a checkout at the commit the pool runs):

```bash
cd $WT && ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. /mnt/scripts/Orion-Sapienform/.venv/bin/python \
    scripts/gpu_pool_actuator_probe.py --role agent-gpu2 --check status digest
# need: status succeeded, in_flight false; digest -> "launch digests agree" (profile_not_allowed)
```

**P3. Nothing in flight, no operator lease:**

```bash
curl -fsS localhost:8127/v1/pool | python3 -c "
import json,sys; d=json.load(sys.stdin)
for c in d['cards']: print(c['card'], c['swap_state'], c['swapped_in'], (c.get('actuation') or {}).get('outcome'))"
# need: every card swap_state idle
PSQL "SELECT lease_id, work_class, status FROM gpu_pool_leases WHERE operator AND status NOT IN ('released','unavailable')"
# need: no rows
```

## Step 1 [GO] — sql-writer

```bash
cd $WT && scripts/safe_docker_build.sh orion-sql-writer up -d --build
docker logs --tail 50 orion-athena-sql-writer | grep -iE "error|gpu_pool" | tail
```

## Step 2 [GO] — migration v3 (additive, two nullable columns, lock_timeout 5 s; re-run if it times out)

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < $WT/services/orion-sql-db/manual_migration_gpu_pool_v3_actuation_pause.sql
PSQL "SELECT card, actuation_paused_at, actuation_paused_by FROM gpu_pool_cards ORDER BY card"
# need: four rows, both columns NULL (the migration pauses nothing)
```

## Step 3 [GO] — primary `.env`: flip the mode, delete the dead key

```bash
# in /mnt/scripts/Orion-Sapienform/services/orion-gpu-pool/.env:
#   line 21  GPU_POOL_MODE=observe          -> GPU_POOL_MODE=enforce
#   line 37  GPU_POOL_ACTUATE_ROLES=agent-gpu2   -> delete the line (nothing reads it any more)
grep -nE '^GPU_POOL_(MODE|ACTUATE_ROLES)=' /mnt/scripts/Orion-Sapienform/services/orion-gpu-pool/.env
# need: only GPU_POOL_MODE=enforce
```

(`python scripts/sync_local_env_from_example.py orion-gpu-pool --all-keys` reports `GPU_POOL_MODE` as diverged
and leaves it: flipping it is this step, not a sync.)

## Step 4 [GO] — the pool

```bash
cd $WT && scripts/safe_docker_build.sh orion-gpu-pool up -d --build
```

## Step 5 — verify the pool (read-only)

```bash
curl -fsS localhost:8127/health | python3 -m json.tool
# need: "mode": "enforce", "actuation": {"seats": ["agent-gpu2"], "paused": false}
docker exec orion-athena-gpu-pool env | grep -E '^GPU_POOL_(MODE|ACTUATE_ROLES)='
# need: GPU_POOL_MODE=enforce only  (acceptance 9: the old key is absent)
docker logs orion-athena-gpu-pool 2>&1 | grep -E "gpu_pool_actuation mode|gpu_pool_reconcile" | tail
# need: "gpu_pool_actuation mode=enforce seats=['agent-gpu2']", then ONE "gpu_pool_reconcile_send seat=agent-gpu2
#       ... why=boot" and either "gpu_pool_reconcile_agrees" (no event) or a swapped event reason=adopted:boot
PSQL "SELECT generated_at, event, reason FROM gpu_pool_events WHERE role='agent-gpu2'
      AND generated_at > now() - interval '10 minutes' ORDER BY generated_at"
# need: no swap_started (no reload). adopted:boot only if the card had changed by hand.
```

Acceptance 9, "a restart with the 27B loaded adopts it via one `status`, no reload": if the 27B happens to be loaded
at deploy time, this step is that check. If it is not, run it on the next load: after `swapped action=load`,
`docker restart orion-athena-gpu-pool` **[GO]**, then the same log/event reads (one `status`, no `swap_started`,
`swapped_in` still `{agent-gpu2}`).

## Step 6 [GO] — Hub

```bash
cd $WT && scripts/safe_docker_build.sh orion-hub up -d --build
```

Verify in a browser (GPU pool tab): the header reads `mode: enforce`; Controls shows "Emergency stop: pause all
model loading/unloading"; the experiment hold button is greyed out with `not_actuatable:experiment: nothing can load
it ...`.

## Step 7 [GO, optional] — exercise the emergency stop once

Pick a quiet moment (no gpu2 swap in flight). Click "Emergency stop", confirm; then "Resume", confirm.

```bash
PSQL "SELECT generated_at, event, holder, detail FROM gpu_pool_events
      WHERE event IN ('actuation_paused','actuation_resumed') ORDER BY generated_at DESC LIMIT 2"
PSQL "SELECT card, actuation_paused_at FROM gpu_pool_cards"          # all NULL again after the resume
docker logs orion-athena-gpu-pool 2>&1 | grep -E "gpu_pool_actuation_(paused|resumed)|reconcile_send .*why=resume" | tail
```

## Rollback

Pick the smallest one that fixes the problem.

1. **Only the boot/resume reconcile misbehaves** (e.g. a wrong `adopted:*` or a fault it should not have raised):
   `GPU_POOL_MODE=observe` in the primary `.env` + `cd $WT && scripts/safe_docker_build.sh orion-gpu-pool up -d`
   **[GO]**. Harmless: every seat is still actuated; observe turns the reconcile off and refuses operator holds. It
   gives gpu2 **no** adoption path at all (its liveness shortcut only covers a seat the pool has never acted on, and
   agent-gpu2 has), so after a hand change to gpu2 in observe, use the Hub "Clear fault" / wait for the pool's own
   next action. Clear a fault the reconcile left with the Hub "Clear fault" button.
2. **Stop all model moves now, investigate later:** the emergency stop (below). No deploy needed.
3. **The 5.7 pool itself is wrong:** redeploy the previous pool image **[GO]**: a worktree at the pre-5.7 commit
   (`git worktree add ../Orion-Sapienform-pool-rollback d3c09c9cb`), put `GPU_POOL_MODE=observe` and
   `GPU_POOL_ACTUATE_ROLES=agent-gpu2` back in the primary `.env` (that image still reads it), then
   `scripts/safe_docker_build.sh orion-gpu-pool up -d --build` from that worktree. The v3 columns can stay (an old pool
   never reads them; a leftover pause is simply not honoured by it). The sql-writer and Hub can stay on 5.7 (an old
   pool refuses the Hub's two new verbs with a validation error; nothing else differs).

## Emergency stop (keep this with the pool's docs)

**To stop every model load and unload at once:** Hub → GPU pool tab → Controls → **"Emergency stop: pause all model
loading/unloading"** → confirm. From a shell instead (e.g. the Hub is down), from any worktree of main:

```bash
cd $WT && ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. /mnt/scripts/Orion-Sapienform/.venv/bin/python \
    scripts/gpu_pool_pause.py pause          # prints: PAUSED since ... by ...   (resume: ... gpu_pool_pause.py resume)
curl -fsS localhost:8127/health | python3 -c "import json,sys; print(json.load(sys.stdin)['actuation'])"
```

(The script sends the same control verb as the Hub button; its round trip through the pool's control path is tested
in `services/orion-gpu-pool/tests/test_stage5_7_enforce.py`. Not yet run against the live bus: UNVERIFIED live.)

What it does: the pool sends nothing more to circe's controller, drains nothing for a swap (the 27B, if loaded, keeps
serving; image generation waits for gpu2), refuses operator holds, and stays paused through restarts until Resume.
Every blocked swap is still visible as `swap_requested reason=actuation_paused`.

What it does **not** do: stop a load or unload that is already running on circe (the controller owns it; the pool
keeps following it). To stop that too: on circe, `docker stop orion-circe-gpu-lane-controller` **[GO]**; the pool then
faults the card when the action times out, and "Clear fault" reconciles it after you restart the controller.

Resume first asks the controller what each card holds (enforce), adopts it, then acts on queued demand.
