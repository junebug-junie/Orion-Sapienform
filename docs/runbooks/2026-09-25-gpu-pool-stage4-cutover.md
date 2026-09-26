# Runbook: GPU pool stage 4 cutover (4.5) — durable runs onto pool holds, the pool decides gpu2

Spec: `docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md`
(PR #2348), including every "Corrections from building 4.x" section. PR: stage 4.5.

## What this changes, in plain words

Today two schedulers hand out the agent card. Durable-runs' own broker gives one run at a time an
exclusive "agent seat" and, on its own, decides when to swap gpu2 from diffusion to a second 27B.
After this cutover the GPU pool is the only decider: each durable run asks the pool for a *hold*,
every LLM call the run makes attaches to that hold, and the pool (not durable-runs) decides when
to load gpu2. circe's controller still does the physical load; only *who asks* changes.

The order below is a safety dependency, not a preference. The pool cannot see durable-runs'
legacy leases, and once the controller trusts the pool it stops re-checking thermal and the
visual baseline itself.

Conventions: athena commands run from a **worktree** of the merged `main` (the deploy wrapper
refuses the shared checkout); `$WT` is that worktree. Postgres reads use
`docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "..."` (alias `PSQL` below).

**Every step that changes production needs Juniper's explicit go before it runs**, and each such
step is marked **[GO]**: the P2 migration (DDL), the pause/resume calls (run control rows), the env
flips and restarts in steps 2–5, and the step-6 row update. The read-only checks are not marked.

**Env files.** `.env` files exist only in the primary checkout (`/mnt/scripts/Orion-Sapienform`),
but `scripts/safe_docker_build.sh` reads `.env` and `services/<svc>/.env` relative to the worktree
it runs in. Before any deploy from `$WT`, link them, so a flip made in the primary copy is the one
the container gets:

```bash
for f in .env services/orion-durable-runs/.env services/orion-gpu-pool/.env; do
  ln -sf /mnt/scripts/Orion-Sapienform/$f $WT/$f; done
```

After every flip, check the value INSIDE the container (commands given per step) -- a flip that did
not reach the container is the failure this runbook exists to prevent.

```bash
PSQL() { docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "$1"; }
```

## Preconditions (all must hold before step 1)

### P1. 4.1–4.4 are merged and deployed, in their own order

| PR | deployed where | check |
| --- | --- | --- |
| 4.1 contracts (#2349) | pool + sql-writer (athena) | `curl -fsS localhost:8127/health` answers; sql-writer running the 4.1 image |
| 4.2 actuator (#2350) | circe controller, `GPU2_AUTHORITY=durable` | see P3 |
| 4.3 holds (#2352) | v2 migration, then the pool with `GPU_POOL_ACTUATE_ROLES=` (empty) | see P2 |
| 4.4 consumers (#2351) | gateway, cortex-exec (all four), thought, harness-governor, Hub, field-digester | see P4 |

### P2. The 4.3 migration is applied and the 4.3 pool is running with its guards live

```bash
PSQL "SELECT column_name FROM information_schema.columns WHERE table_name='gpu_pool_leases' AND column_name='hold_lease_id'"
# expect: hold_lease_id   (empty = migration NOT applied -- stop. A 4.3 pool refuses to boot without it)
# (2026-09-25 read: the column is ABSENT on live -- the migration has not been applied yet.)

# [GO] apply it (additive DDL, lock_timeout 5s) if missing:
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < $WT/services/orion-sql-db/manual_migration_gpu_pool_v2_holds.sql

docker logs --since 10m orion-athena-gpu-pool 2>&1 | grep -c gpu_pool_guard_read_failed   # expect 0
curl -fsS localhost:8127/health          # mode + config_digest of the 4.3+ config
```

### P3. circe runs 4.1+ (launch_digest) and the 4.2 controller with its three new env keys

The controller computes `launch_digest` over **circe's own** `config/gpu_pool.yaml`. If circe's
checkout predates 4.1, every pool action is refused (`role_not_on_this_actuator`) — safe, but the
cutover would silently never load gpu2.

```bash
ssh circe@circe
cd /mnt/scripts/Orion-Sapienform && git log --oneline -1 && git merge-base --is-ancestor <4.1 merge sha> HEAD && echo ok
grep -E '^(GPU2_AUTHORITY|GPU_POOL_ACTUATOR_NAME|GPU2_POOL_FENCE_STATE_PATH)=' services/orion-gpu-lane-controller/.env
# expect exactly:
#   GPU2_AUTHORITY=durable
#   GPU_POOL_ACTUATOR_NAME=circe
#   GPU2_POOL_FENCE_STATE_PATH=/state/gpu2_pool_fence.json
docker logs --tail=200 orion-circe-gpu-lane-controller 2>&1 | grep -E "gpu_pool_actuator_started authority=durable"
docker volume inspect orion-gpu-lane-controller-state >/dev/null && echo "fence volume present"
```

Never `docker compose down -v` the controller: the fence state lives on that named volume and
`-v` resets it to generation 0.

### P4. All six 4.4 consumers carry the hold ref (grep each container)

An old thought or governor silently drops the ref (their request models accept unknown fields),
and the run's call then queues behind its own hold. Each line must print a non-zero count:

```bash
for c in orion-llm-gateway orion-athena-cortex-exec orion-athena-cortex-exec-background \
         orion-athena-cortex-exec-chat orion-athena-cortex-exec-spark orion-athena-thought \
         orion-athena-harness-governor orion-athena-hub orion-athena-field-digester; do
  printf '%-40s ' "$c"; docker exec "$c" sh -c 'grep -rl --include=*.py "gpu_lease\|kind = '"'"'hold'"'"'" /app 2>/dev/null | wc -l'
done
# every row >= 1. Stronger per-service proofs:
docker exec orion-llm-gateway          sh -c 'grep -rn "hold=" /app --include=pool_placement.py | head -2'
docker exec orion-athena-hub           sh -c 'grep -rn "validate_hold_ref" /app --include=curiosity_investigation.py | head -2'
docker exec orion-athena-field-digester sh -c "grep -rn \"kind = 'hold'\" /app --include=store.py | head -2"
```

## Step 1 — stop new legacy grants, then wait for zero active legacy leases (LOAD-BEARING)

The pool cannot see `durable_resource_leases`. A legacy lease still running when the pool starts
deciding means two schedulers on one card. Waiting alone does not converge: the old broker grants
the next pending demand whenever a lease ends. So first stop it granting, then wait.

**1a [GO] pause every waiting (not running) run** -- a paused run's demand is suspended and the old
broker never grants it. Record the list; step 5 resumes exactly these.

```bash
# every live run that does not hold a legacy lease right now (pending, suspended-for-retry, or new)
PSQL "SELECT r.run_id FROM durable_admission_runs r WHERE r.terminal IS NULL AND r.control IS NULL
      AND NOT EXISTS (SELECT 1 FROM durable_resource_leases l WHERE l.run_id=r.run_id AND l.status='active')" \
  | tee /tmp/gpu-pool-stage4-cutover-paused.txt
while read run; do curl -fsS -X POST "localhost:8124/runs/$run/pause" >/dev/null && echo "paused $run"; done \
  < /tmp/gpu-pool-stage4-cutover-paused.txt
```

**1b wait for the running ones to finish** (their turns complete normally):

```bash
PSQL "SELECT count(*) FROM durable_resource_leases WHERE status='active' AND expires_at > now()"
# repeat until 0.  (2026-09-25 read: 2 active.)  A run accepted after 1a may still be granted --
# pause it the same way and keep waiting. Re-run this count right before steps 2, 4 and 5.
```

Then record the baseline (for acceptance check 1). Take START only now, after the count is 0:

```bash
PSQL "SELECT now()" | tee /tmp/gpu-pool-stage4-cutover-start.txt
PSQL "SELECT (SELECT count(*) FROM durable_resource_demands), (SELECT count(*) FROM durable_resource_leases)"
```

## Step 2 — freeze the old elastic decider (on the CURRENT durable-runs build)

gpu2 stays in whatever state it is in. (2026-09-25 read: `durable_elastic_slot` generation 9,
`state=ready`, `desired_target=agent-burst` — the 27B is loaded on gpu2 by the old path.)

Also set `DURABLE_RUNS_ADMISSION_SHADOW=true`: the old broker then grants nothing at all (a run
submitted during the window only registers its demand) and drives nothing. Do this only with the
step-1b count at 0 -- shadow stops renewing a running lease.

```bash
# [GO] in the PRIMARY checkout's live env file (not committed):
ENVF=/mnt/scripts/Orion-Sapienform/services/orion-durable-runs/.env
sed -i -e 's/^DURABLE_RUNS_ELASTIC_SHADOW=.*/DURABLE_RUNS_ELASTIC_SHADOW=true/' \
       -e 's/^DURABLE_RUNS_ADMISSION_SHADOW=.*/DURABLE_RUNS_ADMISSION_SHADOW=true/' $ENVF
cd $WT_CURRENT_DURABLE_RUNS_BUILD   # a worktree at the commit durable-runs runs today, env files linked
scripts/safe_docker_build.sh orion-durable-runs up -d --force-recreate durable-runs
docker exec orion-athena-durable-runs env | grep -E 'DURABLE_RUNS_(ELASTIC|ADMISSION)_SHADOW'   # both =true
PSQL "SELECT slot, generation, state, desired_target FROM durable_elastic_slot"   # unchanged from before
PSQL "SELECT count(*) FROM durable_resource_leases WHERE status='active' AND expires_at > now()"   # still 0
```

## Step 3 — circe controller trusts the pool (`GPU2_AUTHORITY=pool`)

Only after P2 (the pool's guards and recall are live): under `pool` the controller no longer
checks thermal, the visual baseline or durable leases itself.

```bash
ssh circe@circe
cd /mnt/scripts/Orion-Sapienform
# [GO]
sed -i 's/^GPU2_AUTHORITY=.*/GPU2_AUTHORITY=pool/' services/orion-gpu-lane-controller/.env
docker compose --env-file .env --env-file services/orion-gpu-lane-controller/.env \
  -f services/orion-gpu-lane-controller/docker-compose.yml up -d --force-recreate gpu-lane-controller
curl -fsS http://localhost:8090/health
docker logs --tail=50 orion-circe-gpu-lane-controller 2>&1 | grep "gpu_pool_actuator_started authority=pool"
docker exec orion-circe-gpu-lane-controller env | grep '^GPU2_AUTHORITY='   # =pool
```

## Step 4 — the pool arms actuation for `agent-gpu2`

```bash
PSQL "SELECT count(*) FROM durable_resource_leases WHERE status='active' AND expires_at > now()"   # must be 0
# [GO]
sed -i 's/^GPU_POOL_ACTUATE_ROLES=.*/GPU_POOL_ACTUATE_ROLES=agent-gpu2/' /mnt/scripts/Orion-Sapienform/services/orion-gpu-pool/.env
cd $WT && scripts/safe_docker_build.sh orion-gpu-pool up -d --force-recreate gpu-pool
docker exec orion-athena-gpu-pool env | grep '^GPU_POOL_ACTUATE_ROLES='   # =agent-gpu2
curl -fsS localhost:8127/v1/pool | python3 -c "import json,sys; d=json.load(sys.stdin); print({c['card']: (c.get('swap_state'), c.get('swapped_in')) for c in d['cards']})"
# gpu2 must show the seat ADOPTED (swapped_in contains agent-gpu2 if the old path left it loaded),
# swap_state idle -- never a second transition. Controller access log on circe: no new activate call.
```

The window between steps 2 and 4 is one pool restart: nobody opens gpu2 in it (no load happens).

**Before step 5, confirm the pool knows gpu2's context size.** Until a 4.3 pool has seen agent-gpu2 loaded at least once, a hold that states `min_ctx_tokens` can't trigger the load. It shows as `swap_requested reason=ctx_unknown` instead. Adopting the seat the old path loaded fills it in. Check:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select card, seen_ctx from gpu_pool_cards where card='gpu2'"
# expect {"agent-gpu2": <n>}; if empty, wait for discovery to see the loaded seat before continuing
```

## Step 5 — deploy durable-runs 4.5

```bash
cd $WT   # worktree at merged main containing 4.5, env files linked
PSQL "SELECT count(*) FROM durable_resource_leases WHERE status='active' AND expires_at > now()"   # must be 0
# [GO]
python3 scripts/sync_local_env_from_example.py orion-durable-runs   # adds HOLD_STATUS_POLL_SEC, OUTREACH_HOLD_MAX_SEC
scripts/safe_docker_build.sh orion-durable-runs up -d --build durable-runs
curl -fsS localhost:8124/health
docker exec orion-athena-durable-runs env | grep -E 'DURABLE_RUNS_HOLD_STATUS_POLL_SEC|DURABLE_RUNS_OUTREACH_HOLD_MAX_SEC'
# [GO] resume the runs paused in step 1a -- paused runs never resume on their own:
while read run; do curl -fsS -X POST "localhost:8124/runs/$run/resume" >/dev/null && echo "resumed $run"; done \
  < /tmp/gpu-pool-stage4-cutover-paused.txt
docker logs --since 5m orion-athena-durable-runs 2>&1 | grep -E "durable_hold_|durable_admission_reconcile_failed|Traceback" | tail -20
```

The deleted keys (`DURABLE_RUNS_ELASTIC_*`, `DURABLE_RUNS_ADMISSION_SHADOW`, `DURABLE_RUNS_WIDENING_*`,
`DURABLE_RUNS_LANE_POLICY_JSON`, `DURABLE_RUNS_GATEWAY_URL`) are ignored by the 4.5 build; remove
them from the live `.env` at leisure (they are needed until step 2 only).

On boot every non-terminal run (13 on 2026-09-25) asks the pool for a hold and queues. Verify:

```bash
curl -fsS localhost:8127/v1/pool | python3 -c "import json,sys; d=json.load(sys.stdin); print([(l['holder'], l['status'], l.get('role')) for l in d.get('leases', []) if l['holder'].startswith('durable-runs:')])"
PSQL "SELECT holder, status, role, queued_since FROM gpu_pool_leases WHERE kind='hold' AND holder LIKE 'durable-runs:%' AND status IN ('queued','granted','recalling','backlogged') ORDER BY created_at"
```

## Step 6 — withdraw the frozen pending demands (PRODUCTION WRITE: only with Juniper's explicit go)

Their runs re-registered through the pool in step 5; the rows only keep `durable_demand_pending`'s
legacy half and the (now disabled) capacity reservation pointed at nothing. 11 pending on the
2026-09-25 read (the spec said 13).

```bash
mkdir -p /tmp/gpu-pool-stage4-migrate
# snapshot first (read-only)
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c \
  "COPY (SELECT * FROM durable_resource_demands WHERE status='pending') TO STDOUT WITH CSV HEADER" \
  > /tmp/gpu-pool-stage4-migrate/pending_demands.csv
wc -l /tmp/gpu-pool-stage4-migrate/pending_demands.csv          # header + N rows
# every pending demand's run must already hold (or wait for) a pool hold -- expect 0 rows:
PSQL "SELECT d.run_id FROM durable_resource_demands d JOIN durable_admission_runs r USING(run_id)
      WHERE d.status='pending' AND r.terminal IS NULL AND NOT EXISTS (SELECT 1 FROM gpu_pool_leases l
      WHERE l.kind='hold' AND l.holder='durable-runs:'||d.run_id)"
```

Then, ONLY after Juniper says go. The update is limited to the snapshot's own ids, and it aborts
itself (rolls back) unless it touched exactly that many rows:

```bash
# [GO]
IDS=$(tail -n +2 /tmp/gpu-pool-stage4-migrate/pending_demands.csv | cut -d, -f1 | sed "s/.*/'&'/" | paste -sd,)
N=$(tail -n +2 /tmp/gpu-pool-stage4-migrate/pending_demands.csv | wc -l)
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 <<SQL
BEGIN;
WITH w AS (
  UPDATE durable_resource_demands
     SET status = 'withdrawn',
         decision = decision || '{"withdrawn_reason": "migrated_to_gpu_pool"}'::jsonb
   WHERE status = 'pending' AND demand_id IN ($IDS)
  RETURNING demand_id)
-- Aborts the transaction on a mismatch. The division must depend on the rows (count(*) - count(*)
-- is 0 only at run time): a literal `ELSE 1/0` is constant-folded when the query is planned, so it
-- raised on EVERY run -- verified live 2026-09-26, it rolled back a correct 11-row update.
SELECT CASE WHEN count(*) = $N THEN 'ok:' || count(*) ELSE (1 / (count(*) - count(*)))::text END FROM w;
COMMIT;
SQL
```

Expected output: `ok:<N>` then `COMMIT`. A mismatch prints `ERROR: division by zero` and nothing is
changed (ON_ERROR_STOP stops before COMMIT; the transaction is rolled back).

**What the leftovers were on the 2026-09-26 cutover:** 11 rows -- 9 `pending` plus 2 `suspended`
demands, all belonging to runs already `terminal='failed'`. The `WHERE status = 'pending'` above
does not touch `suspended` rows; to retire those too, snapshot them the same way
(`WHERE status IN ('pending','suspended')` in the COPY) and use the same `status IN (...)` in the
UPDATE, so `$N` still equals the snapshot's row count.

(`demand_id` is the csv's first column; ids are `<run_id>:harness_turn:<resource>` and contain no
commas or quotes.)

Undo (from the snapshot): `UPDATE durable_resource_demands SET status='pending', decision = decision - 'withdrawn_reason' WHERE demand_id IN (<ids from the csv>);`

## Verification (spec acceptance checks, live)

```bash
START=$(cat /tmp/gpu-pool-stage4-cutover-start.txt)
# 1. no new legacy rows; every run accepted since then has a pool hold
PSQL "SELECT (SELECT count(*) FROM durable_resource_demands WHERE created_at > '$START'),
             (SELECT count(*) FROM durable_resource_leases  WHERE granted_at > '$START')"          # 0|0
PSQL "SELECT r.run_id, count(l.lease_id) FROM durable_admission_runs r LEFT JOIN gpu_pool_leases l
      ON l.kind='hold' AND l.holder='durable-runs:'||r.run_id WHERE r.created_at > '$START' GROUP BY 1
      HAVING count(l.lease_id) = 0"                                                                 # no rows
# 2. no self-deadlock: no un-attached agent request lease carrying a durable run's turn id (24 h)
PSQL "SELECT count(*) FROM gpu_pool_leases l WHERE l.kind='request' AND l.work_class='agent'
      AND l.hold_lease_id IS NULL AND l.created_at > now() - interval '24 hours'
      AND l.turn_correlation_id IN (SELECT payload->'detail'->>'turn_correlation_id'
        FROM durable_resource_events WHERE event='run.started' AND generated_at > now() - interval '25 hours')"   # 0
PSQL "SELECT count(*) FROM gpu_pool_leases c JOIN gpu_pool_leases h ON c.hold_lease_id=h.lease_id
      WHERE h.holder LIKE 'durable-runs:%' AND c.created_at > '$START'"                            # > 0 once a run worked
# 4. gpu2 via the pool: swap events, controller log, nothing on athena calls :8090 activate
PSQL "SELECT generated_at, event, role, reason FROM gpu_pool_events WHERE event IN
      ('swap_requested','swap_started','swapped','swap_failed','actuate_refused') AND generated_at > '$START' ORDER BY 1"
ssh circe@circe "docker logs --since 24h orion-circe-gpu-lane-controller 2>&1 | grep -E 'gpu2_transition|POST /v1/gpu-slots/activate'"
# 8. field: durable_demand_pending equals queued/backlogged durable holds; gpu_pool_waiting excludes holds
PSQL "SELECT count(*) FROM gpu_pool_leases WHERE kind='hold' AND holder LIKE 'durable-runs:%' AND status IN ('queued','backlogged')"
# 9. Hub: the curiosity run view shows waiting -> granted (lane = the hold's role) -> released
PSQL "SELECT run_id, event, payload->'detail'->>'lane' FROM durable_resource_events
      WHERE event IN ('run.waiting_resource','run.lane_assigned','resource.lease_released')
      AND generated_at > '$START' ORDER BY generated_at LIMIT 30"
```

Check 2 depends on the gateway stamping each call's `turn_correlation_id` with the harness turn
id; if the positive query (children > 0) is zero after a run worked, treat check 2 as UNVERIFIED.

## Rollback

Any step can be reversed in reverse order; the later the step, the more has to be undone.

1. **Before step 5:** reverse steps 4, 3, 2 exactly (`GPU_POOL_ACTUATE_ROLES=`, `GPU2_AUTHORITY=durable`,
   `DURABLE_RUNS_ELASTIC_SHADOW=false`, restarting pool, controller, durable-runs). The pool adopts
   whatever gpu2 holds; the controller fence falls back to durable-runs' `/elastic/status`.
2. **After step 5 [GO]:** in this order -- (a) stop the 4.5 durable-runs container
   (`docker stop orion-athena-durable-runs`); (b) cancel every durable-run hold at the pool (below),
   so no pool hold and no legacy lease are ever live at once; (c) reverse 4 and 3; (d) start the
   previous durable-runs image (the step-2 build) with both shadow flags back to `false`. The old
   build re-registers its demands on resume.

   ```bash
   cd $WT && PYTHONPATH=. /mnt/scripts/Orion-Sapienform/.venv/bin/python - <<'PY'
   import asyncio, os
   from orion.core.bus.async_service import OrionBusAsync
   from orion.gpu_pool.client import release_lease
   import psycopg
   async def main():
       bus = OrionBusAsync(url=os.environ["ORION_BUS_URL"]); await bus.connect()
       rows = psycopg.connect(os.environ["PG_DSN"]).execute(
           "SELECT lease_id FROM gpu_pool_leases WHERE holder LIKE 'durable-runs:%' "
           "AND status IN ('queued','granted','recalling','backlogged','retry_wait')").fetchall()
       for (lease_id,) in rows:
           print(lease_id, (await release_lease(bus, lease_id, source="rollback", outcome="cancelled")).status)
       await bus.close()
   asyncio.run(main())
   PY
   ```
   (`ORION_BUS_URL=redis://100.92.216.81:6379/0`, `PG_DSN` = the conjourney DSN.) Checkpoints written by
   4.5 carry a hold ref in `lease`; the old build's worker-recovery path is EXPECTED to clear it on
   the next drive (UNVERIFIED -- not exercised by a test). An admitted reflect run checkpointed by 4.5
   (graph `llm_call`) cannot be driven by the old build, which has no admitted reflect graph: cancel
   such runs (`POST /runs/<id>/cancel`) before step (d). Rollback after step 6 also needs the step-6
   undo SQL above.
3. Do not roll back 4.4 consumers: they accept both the old token and the hold ref.

## After the cutover (PR 4.6, not this runbook)

Delete the old `ResourceLeaseV1` / `X-Orion-Resource-Lease` / gateway `LeaseGuard` path across its
importers, the controller's `durable` authority branch and the gpu2 activate route, the
field-digester legacy half, the deprecated `ResourceRequirementV1` fields, the dead env keys
(`GPU2_AUTHORITY_URL`, `LLM_GATEWAY_LEASE_VALIDATION_*`, `HUB_CURIOSITY_ELASTIC_ACTIVATION_ENABLED`,
and Hub's `HUB_CURIOSITY_LEASE_VALIDATION_URL` once Door-A's release URL has its own key).
