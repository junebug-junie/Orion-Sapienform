# Runbook: GPU pool stage 5.3 cutover — agent-gpu2 moves from the gpu2 bridge onto the generic launch executor

Spec: `docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md` (PR #2404),
including "Corrections from building 5.1" and "5.2". PRs: 5.1 #2408 (merged), 5.2 #2409 (merged), 5.3 #2415
PR report: `docs/superpowers/pr-reports/2026-09-29-gpu-pool-stage5-3-cutover-pr.md`.

## What this changes, in plain words

When the pool decides to load the second 27B on circe's gpu2, circe's controller used to run one of two
hard-coded gpu2 moves. After this cutover it runs the same steps from the YAML instead: drain and stop
diffusion, start `atlas-agent-burst` with the card number and the model name set by the controller, wait
for it to answer, and put diffusion back if it doesn't. The model is the same one as today (the pool now
names it explicitly: `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex`, which is circe's
`ATLAS_AGENT_PROFILE_NAME`, read 2026-09-29).

What you will notice:

- the 27B gets up to **900 s** to become ready (was 600 s); the pool's first deadline is 1500 s and it
  faults a stuck card at 3000 s;
- controller failure reasons name the role: `burst_upstream_not_idle` is now `upstream_not_idle:agent-gpu2`;
- the controller's `GET /v1/gpu-slots/circe-gpu2/status` no longer shows progress (its `state` stays
  `neither`); progress is in the Hub GPU pool panel (from the bus results);
- `swap_*` events carry `detail.profile`.

The bridge code stays in the controller until 5.6, so the rollback is a YAML revert.

> **Superseded by 5.6 (2026-09-30):** the bridge and the `swap.load`/`swap.unload` keys are deleted, and
> the config validator now refuses them. The YAML-revert rollback below no longer works (every action would
> be refused `config_unloadable`). Roll back 5.6 by redeploying the previous durable-runs and controller
> images (a `git revert` of the 5.6 PR), not by editing `config/gpu_pool.yaml`.

## Conventions

- athena commands run from a **worktree** of merged `main` (`scripts/safe_docker_build.sh` refuses the
  shared checkout); `$WT` is that worktree. circe commands likewise from a circe worktree `$CWT`.
- `.env` files live only in each host's primary checkout (`/mnt/scripts/Orion-Sapienform`). Link them
  into the worktree before any deploy so the container gets the live values.
- Postgres reads: `PSQL() { docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "$1"; }`
- **Every step that changes production is marked [GO] and needs Juniper's explicit go.** Unmarked
  steps are read-only.

## Live state when this was written (2026-09-29 ~22:10 UTC)

- athena pool: restarted 22:06 UTC, `config_digest=e58f5238846b0bb7` (the 5.1/5.2 YAML), mode
  `observe`, `GPU_POOL_ACTUATE_ROLES=agent-gpu2`. gpu2 `swap_state=idle`, `swapped_in=[]`, last action
  `agent-gpu2:unload:g78` (idle unload 19:12 UTC, succeeded).
- circe: checkout at `f720356c1` (stage 4.5), controller up 3 days. So **circe's launch digest for
  agent-gpu2 does not match the athena pool's right now**: the next load the pool asks for will be
  refused `launch_digest_mismatch` (safe: a refusal plus a 600 s cooldown, the card is untouched), and
  agent-gpu2 cannot load until circe is updated. UNVERIFIED that a refusal has happened yet (no load was
  asked for between 22:06 and 22:10).
- circe `services/orion-llamacpp-host/.env` has `ATLAS_AGENT_PROFILE_NAME=qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex`
  and **not** the two 5.1 keys (`ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES`, `ATLAS_AGENT_BURST_PROFILE_NAME`).
  Compose defaults cover both (`:-2`, and the profile falls back to `ATLAS_AGENT_PROFILE_NAME`), and the
  controller sets both on every start anyway; sync them for parity (step 1).
- orion-thought (athena): `ORION_VISUAL_ELASTIC_STATUS_ENABLED=true` — it reads the controller's
  `/v1/gpu-slots/circe-gpu2/status` before each image. The spec said this was off; it is on. It keeps
  working after 5.3 because it defers on `active != "diffusion"`, and `active` is read live from
  `docker compose ps`. See step 6.

## Step 0 — preconditions (read-only; all must hold)

```bash
# athena. gpu2 idle, nothing in flight, and (preferably) diffusion resident:
curl -fsS localhost:8127/v1/pool | python3 -c "
import json,sys; d=json.load(sys.stdin)
c=[c for c in d['cards'] if c['card']=='gpu2'][0]
a=c.get('actuation') or {}
print('swap_state', c.get('swap_state'), 'swapped_in', c.get('swapped_in'), 'cooldown_until', c.get('cooldown_until'))
print('last action', a.get('action_id'), 'outcome', a.get('outcome'))"
# expect: swap_state idle, last action outcome not null (finished). swapped_in [] is the preferred state
# (diffusion resident). If agent-gpu2 is loaded, either wait for its idle unload, or accept that the
# first action after the cutover is a generic UNLOAD of a seat the bridge loaded (same container; tested).

# no hold or call running on the seat right now (else wait -- a restart does not unload it, but a
# load/unload decided during the deploy window would be refused and cool down for 600 s):
PSQL "SELECT lease_id, holder, status FROM gpu_pool_leases WHERE role='agent-gpu2' AND status IN ('granted','recalling')"
# expect no rows

# circe. the controller has nothing in flight:
ssh circe@circe 'docker exec orion-circe-gpu-lane-controller cat /state/gpu2_pool_fence.json' \
  | python3 -c "import json,sys; s=json.load(sys.stdin); print('in_flight', s.get('in_flight'), 'generations', s.get('generations'))"
# expect in_flight None; note the gpu2 generation (the pool's next action must be higher; it is -- the
# pool reads its own last generation from gpu_pool_cards)
ssh circe@circe 'docker ps --format "{{.Names}} {{.Status}}" | grep -E "diffusion-host|agent-burst|lane-controller"'
# expect orion-circe-diffusion-host Up, no running agent-burst (unless the seat is loaded, see above)
```

## The probe used at every step

`scripts/gpu_pool_actuator_probe.py` asks circe's controller two read-only questions over the bus and
prints one verdict line each:

- `status`: controller up, can parse its own checkout's YAML, what it sees on gpu2, anything in flight;
- `digest`: a `load` naming a profile that is in no allow-list. The controller checks the launch digest
  first, so the refusal says which: `profile_not_allowed` = both hosts agree (**OK**),
  `launch_digest_mismatch` = they don't (**MISMATCH**). No generation is spent and no container is
  touched (tested against the real controller in `services/orion-gpu-pool/tests/test_stage5_3_cutover_e2e.py`).

The pool logs both answers as stale and does not change state. [GO] once for the whole runbook (it is
a production bus message, read-only in effect). Run it from an athena checkout at the commit the
**pool** runs:

```bash
probe() { (cd $WT && ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. \
           /mnt/scripts/Orion-Sapienform/.venv/bin/python scripts/gpu_pool_actuator_probe.py "$@"); }
# $WT must contain the script (it ships with 5.3). Before 5.3 is deployed, give it the pool's YAML:
git -C $WT show 38d65a36e:config/gpu_pool.yaml > /tmp/gpu_pool-5.2.yaml
# probe --config /tmp/gpu_pool-5.2.yaml     (step 1)      probe     (steps 4+, pool on $SHA)
```

## Step 1 — bring circe to the pool's commit (5.1 + 5.2), BEFORE the cutover

The athena pool already runs 5.1 + 5.2 (`config_digest=e58f5238846b0bb7`, rebuilt 22:06 UTC); circe is
still on `f720356c1`. Until circe catches up, every load or unload the pool asks for is refused
`launch_digest_mismatch`. This step changes nothing about how gpu2 is driven (agent-gpu2 keeps its
bridge verbs at `38d65a36e`); it only makes the controller agree with the pool again. Juniper may
already have done it -- then run only the checks at the end of this step.

```bash
ssh circe@circe
cd /mnt/scripts/Orion-Sapienform
# [GO] the commit the pool runs (5.2's merge, or later main WITHOUT 5.3)
git fetch origin && git checkout --detach 38d65a36e   # or: git pull --ff-only, if main does not contain 5.3 yet
git rev-parse HEAD
# [GO] the two 5.1 keys (the default sync skips the ATLAS_ prefix: 5.1 correction 4)
python3 scripts/sync_local_env_from_example.py orion-llamacpp-host --all-keys
grep -E '^ATLAS_AGENT_BURST_(CUDA_VISIBLE_DEVICES|PROFILE_NAME)=' services/orion-llamacpp-host/.env
# expect ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES=2 and ATLAS_AGENT_BURST_PROFILE_NAME= (empty)
# [GO] rebuild the controller from a detached worktree at the same commit, env files linked
CWT=/mnt/scripts/Orion-Sapienform-deploy-gpu-pool
git worktree add --detach $CWT HEAD
for f in .env services/orion-gpu-lane-controller/.env; do ln -sf /mnt/scripts/Orion-Sapienform/$f $CWT/$f; done
cd $CWT && scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
curl -fsS http://localhost:8090/health
docker logs --tail=50 orion-circe-gpu-lane-controller 2>&1 | grep -E "gpu_pool_actuator_started|Traceback"
# the /repo mount must be the PRIMARY checkout (GPU_LANE_HOST_REPO_ROOT default), not $CWT:
docker inspect orion-circe-gpu-lane-controller --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}' | grep /repo
# expect /mnt/scripts/Orion-Sapienform -> /repo
```

Verify before going on (athena):

```bash
probe --config /tmp/gpu_pool-5.2.yaml
# expect:
#   status  agent-gpu2 digest=4efbb051879a5c61 -> OK: observed={'agent-gpu2': 'exited', 'diffusion': 'running'} in_flight=False ...
#   digest  agent-gpu2 digest=4efbb051879a5c61 -> OK: launch digests agree
```

Optionally watch one bridge load/unload here (controller log `gpu2_transition`), which separates "new
controller image" from "new code path" before step 3.

## Step 2 — merge 5.3

5.1 (#2408, `ad635377a`) and 5.2 (#2409, `38d65a36e`) are already on `main`.

```bash
# [GO] merge this PR (base main) once CI is green, then record the merge commit
gh pr merge 2415 --merge
SHA=$(git -C /mnt/scripts/Orion-Sapienform ls-remote origin refs/heads/main | cut -f1); echo $SHA
```

Expected at `$SHA`: agent-gpu2 launch digest `3a0317dd8f08532f...` (was `4efbb051879a5c61...`), pool
config digest `08b098deef36cb6a` (was `e58f5238846b0bb7`).

## Step 3 — the cutover: both hosts at `$SHA`, controller first, then the pool

The controller parses its checkout's YAML with the code in its image (5.1 correction 1), and refuses
any action whose digest differs from its checkout's. Between the circe pull and the pool rebuild, a
load/unload the pool asks for is refused `launch_digest_mismatch` (safe: 600 s cooldown, card
untouched), so keep 3a–3b together and do them with the step-0 checks holding.

```bash
# 3a [GO] circe
ssh circe@circe
cd /mnt/scripts/Orion-Sapienform && git fetch origin && git checkout --detach $SHA && git rev-parse HEAD   # == $SHA
git -C $CWT checkout --detach $SHA
cd $CWT && scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
docker exec orion-circe-gpu-lane-controller python -c "from orion.gpu_pool.config import load_pool_config; c=load_pool_config('/repo/config/gpu_pool.yaml'); print(c.load_profile('agent-gpu2'), c.roles['agent-gpu2'].swap.bridged)"
# expect: qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex False   (config_unloadable = image not rebuilt)

# 3b [GO] athena: the pool bakes the YAML into its image, so rebuild it
git -C /mnt/scripts/Orion-Sapienform pull --ff-only && git -C /mnt/scripts/Orion-Sapienform rev-parse HEAD   # == $SHA
cd $WT && git checkout --detach $SHA     # $WT: a worktree, .env + services/orion-gpu-pool/.env linked from the primary checkout
scripts/safe_docker_build.sh orion-gpu-pool up -d --build gpu-pool
curl -fsS localhost:8127/health | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('mode'), d.get('config_digest'))"
# expect: observe 08b098deef36cb6a
docker exec orion-athena-gpu-pool env | grep -E '^GPU_POOL_(MODE|ACTUATE_ROLES)='   # observe / agent-gpu2
```

(Leaving circe's primary checkout on a detached `$SHA` is deliberate: it is the running config.
`git checkout main && git pull --ff-only` later brings it back onto the branch at the same commit.)

## Step 4 — adoption + probe (no transition on boot)

```bash
START=$(PSQL "SELECT now()"); echo $START > /tmp/gpu-pool-stage5-3-start.txt
probe
# expect:
#   status  agent-gpu2 digest=3a0317dd8f08532f -> OK: observed={'agent-gpu2': 'exited', 'diffusion': 'running'} in_flight=False ...
#   digest  agent-gpu2 digest=3a0317dd8f08532f -> OK: launch digests agree
sleep 60
PSQL "SELECT generated_at, event, reason, detail->>'action' FROM gpu_pool_events
      WHERE role='agent-gpu2' AND generated_at > '$START' ORDER BY 1"
# expect no swap_started. (A 'swapped reason=reconciled:status' only if an action was in flight.)
ssh circe@circe "docker logs --since 5m orion-circe-gpu-lane-controller 2>&1 | grep -E 'launch_exec (load|unload)|gpu2_transition|launch_digest_mismatch'"
# expect only the probe's own refusal line (reason=profile_not_allowed), nothing else
```

## Step 5 — thought's pre-generate check still reads a sane gpu2 slot

```bash
curl -fsS http://100.112.254.99:8090/v1/gpu-slots/circe-gpu2/status | python3 -c "
import json,sys; d=json.load(sys.stdin); print(d['enabled'], d['active'], d['state'])"
# expect: True diffusion neither   (state stays 'neither' after the controller restart; the check in
# orion-thought call_diffusion_generate defers only on active != diffusion, draining/activating, or
# failed-not-restored -- none of which a stale 'neither' triggers)
```

## Verification (spec acceptance checks 1–4)

`START=$(cat /tmp/gpu-pool-stage5-3-start.txt)`

### 1. Generic load

Wait for the next natural load (37 in the 4 days before 2026-09-29, so usually within hours): an
`agent` hold queued ≥ 1200 s with `agent` busy. Then:

```bash
PSQL "SELECT generated_at, event, reason, detail->>'profile', detail->>'action_id' FROM gpu_pool_events
      WHERE role='agent-gpu2' AND event IN ('swap_started','swapped','swap_failed','actuate_refused')
      AND generated_at > '$START' ORDER BY 1"
# expect swap_started (reason demand, profile qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex) then
# swapped with the same action_id
ssh circe@circe "docker logs --since 2h orion-circe-gpu-lane-controller 2>&1 | grep -E 'launch_exec (load|up) role=agent-gpu2'"
# expect: launch_exec load role=agent-gpu2 service=atlas-agent-burst env=ATLAS_AGENT_BURST_CUDA_VISIBLE_DEVICES=2 ATLAS_AGENT_BURST_PROFILE_NAME=qwen3.8-27b-... profile=qwen3.8-27b-... evicts=diffusion ...
ssh circe@circe "docker logs --since 2h orion-circe-gpu-lane-controller 2>&1 | grep -cE 'gpu2_transition|BRIDGE'"   # expect 0
ssh circe@circe "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"   # index 2 carries the 27B (~20+ GiB)
ssh circe@circe "docker inspect orion-circe-atlas-llamacpp-agent-burst --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(CUDA_VISIBLE_DEVICES_OVERRIDE|LLM_PROFILE_NAME)='"
# expect CUDA_VISIBLE_DEVICES_OVERRIDE=2 and LLM_PROFILE_NAME=qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex
PSQL "SELECT generated_at, detail->>'profile_name' FROM gpu_pool_events WHERE role='agent-gpu2' AND event='discovery_confirmed' AND generated_at > '$START' ORDER BY 1 LIMIT 1"
```

**Forcing one instead of waiting [GO]** (production leases, released at the end): three background
holds on class `agent` so one queues past the 1200 s seat trigger. The script heartbeats them and
releases all of them on exit.

```bash
cd $WT && ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. /mnt/scripts/Orion-Sapienform/.venv/bin/python - <<'PY'
import asyncio, os, time, json, urllib.request
from orion.core.bus.async_service import OrionBusAsync
from orion.gpu_pool.client import acquire_hold, heartbeat_lease, release_lease
SRC = "operator:stage5-3-check"
def gpu2():
    d = json.load(urllib.request.urlopen("http://localhost:8127/v1/pool"))
    return [c for c in d["cards"] if c["card"] == "gpu2"][0]
async def main():
    bus = OrionBusAsync(url=os.environ["ORION_BUS_URL"]); await bus.connect()
    ids = []
    try:
        for i in range(3):
            r = await acquire_hold(bus, holder=SRC, work_class="agent", request_id=f"stage5-3-check:{i}", source=SRC)
            print(i, r.status, r.lease_id, getattr(r.grant, "role", None)); ids.append(r.lease_id)
        t0 = time.time()
        while time.time() - t0 < 2700:
            for lid in ids:
                await heartbeat_lease(bus, lid, source=SRC)
            c = gpu2()
            print(int(time.time() - t0), c["swap_state"], c["swapped_in"], (c.get("actuation") or {}).get("phase"))
            if "agent-gpu2" in c["swapped_in"] and c["swap_state"] == "idle":
                break
            await asyncio.sleep(30)
    finally:
        for lid in ids:
            print("release", lid, (await release_lease(bus, lid, source=SRC, outcome="cancelled")).status)
        await bus.close()
asyncio.run(main())
PY
```

### 2. Generic unload + owner reclaim

Natural: the next reverie-visual run (≈ 20/day) takes a `diffusion` hold while the 27B is loaded.
(After the forced check above, the seat instead unloads `reason=idle` 300 s after the holds are
released — also a generic unload, but not the reclaim case.)

```bash
PSQL "SELECT generated_at, event, reason, detail->>'action' FROM gpu_pool_events
      WHERE role='agent-gpu2' AND event IN ('swap_started','swapped','swap_failed') AND detail->>'action'='unload'
      AND generated_at > '$START' ORDER BY 1"
# expect swap_started + swapped with reason owner_reclaim (or idle / max_hold)
ssh circe@circe "docker logs --since 6h orion-circe-gpu-lane-controller 2>&1 | grep -E 'launch_exec (unload role=agent-gpu2|up role=diffusion)'"
# expect: launch_exec up role=diffusion service=diffusion-host env=CUDA_VISIBLE_DEVICES=2
ssh circe@circe "docker inspect orion-circe-diffusion-host --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^CUDA_VISIBLE_DEVICES='"   # =2
PSQL "SELECT lease_id, status, granted_at FROM gpu_pool_leases WHERE work_class='diffusion' AND kind='hold' AND created_at > '$START' ORDER BY created_at"
# the reclaiming hold is granted after the unload; then an image is stored:
PSQL "SELECT count(*), max(created_at) FROM durable_admission_runs WHERE run_id LIKE 'reverie-visual-%' AND terminal='succeeded' AND created_at > '$START'"
```

A busy 27B at unload time now shows as `swap_failed reason=upstream_not_idle:agent-gpu2` (was
`burst_upstream_not_idle`), the card goes to `fault`, and discovery clears it back to loaded within a
probe interval, then the unload retries after the 600 s cooldown — the same behaviour as the bridge,
only the reason text changed.

### 3. Failed load (rollback)

Not forced live: forcing a readiness failure on circe means breaking the 27B container on purpose.
Evidence is the end-to-end test (`services/orion-gpu-pool/tests/test_stage5_3_cutover_e2e.py::
test_failed_load_is_rolled_back_and_the_pool_cools_down`: seat never ready -> seat stopped, diffusion
restarted on index 2, `swap_failed reason=model_readiness_timeout:agent-gpu2 restored=true`, card idle,
cooldown). Live status stays **UNVERIFIED** until a natural failure; when one happens:

```bash
PSQL "SELECT generated_at, reason, detail->>'restored', detail->>'phase', detail->>'profile' FROM gpu_pool_events
      WHERE role='agent-gpu2' AND event='swap_failed' AND generated_at > '$START' ORDER BY 1"
# a failed load with restored=true leaves the card idle; restored=false faults it (Hub panel: FAULT)
```

### 4. Profile gate

The probe's `digest` check IS this check: a hand-crafted `GpuActuateV1` naming a profile outside
`launch.profiles`, refused before any generation or container is touched.

```bash
ssh circe@circe 'docker exec orion-circe-gpu-lane-controller cat /state/gpu2_pool_fence.json' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['generations'])" > /tmp/gpu-pool-stage5-3-gen-before.txt
probe --check digest
# expect: digest  agent-gpu2 digest=3a0317dd8f08532f -> OK: launch digests agree   (= reason profile_not_allowed)
ssh circe@circe "docker logs --since 5m orion-circe-gpu-lane-controller 2>&1 | grep 'probe-digest:agent-gpu2' | grep profile_not_allowed"
ssh circe@circe 'docker exec orion-circe-gpu-lane-controller cat /state/gpu2_pool_fence.json' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['generations'])" | diff - /tmp/gpu-pool-stage5-3-gen-before.txt && echo "no generation spent"
```

CI half of check 4: `orion/gpu_pool/tests/test_stage5_config.py` (the old literal
`CUDA_VISIBLE_DEVICES_OVERRIDE=2` on a launch service is refused by `check_gpu_pool_config.py`).

## Rollback

> **Superseded by 5.6** -- see the note at the top: this YAML revert is refused since 5.6.

The bridge is still in the controller (until 5.6), so rolling back is a config revert. Do it with
gpu2 idle (step 0 checks), or accept one refused action + 600 s cooldown during the window.

```bash
# [GO] revert the 5.3 YAML change (a revert PR of the 5.3 merge, or just the config/gpu_pool.yaml hunk):
#   agent-gpu2.swap: {evicts: [diffusion], load: gpu2/agent, unload: gpu2/restore, after_wait_sec: 1200, guards: [...]}
#   agent-gpu2.launch: drop `profiles` (a bridged seat may not list profiles -- the validator refuses it,
#   because the bridge refuses any profile the pool would send)
# merge it; REVERT_SHA=<merge commit>. Then exactly step 3 with $SHA=$REVERT_SHA:
#   3a circe: checkout --detach $REVERT_SHA in the primary checkout and $CWT, then
#       scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
#   3b athena: pull, $WT at $REVERT_SHA, scripts/safe_docker_build.sh orion-gpu-pool up -d --build gpu-pool
# then step 4's probe: both lines OK (digest now matches the bridged shape)
```

Rebuilding circe's controller is strictly needed only if the revert also reverts code; for a YAML-only
revert a pull on circe is enough for the digest to match (the 5.2+ image parses both shapes), but
rebuild anyway so image and checkout stay one commit. The pool must be rebuilt (it bakes the YAML
into its image). After rollback: the controller log shows `gpu2_transition` again on the next load,
and `swap_*` events carry `detail.profile=null`.

The rollback shape is tested: `services/orion-gpu-lane-controller/tests/cutover_config.py` builds it
from the committed YAML, and the bridge tests (`test_actuator_bus.py`, `test_launch_exec.py::
test_rollback_config_still_uses_the_bridge`) run against it.

## After the cutover

- Remove the circe deploy worktree when done: `git -C /mnt/scripts/Orion-Sapienform worktree remove $CWT`.
- 5.4 (world + diffusion leases, and deleting thought's elastic pre-check) can start once checks 1–2
  have been seen live.
