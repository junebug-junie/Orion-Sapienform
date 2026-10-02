# feat(llamacpp-host, gpu-pool): stage 7.2 -- the pool loads Ternary-Bonsai on agent-gpu2, 2 slots x 131K

## Summary

- The next time the GPU pool loads the gpu2 agent seat (`agent-gpu2`), it starts Ternary-Bonsai-2-27B
  with **two 131K conversations at once** instead of the Q4 27B with one. The pool does this through its
  existing load path: the only pool-side change is the order of `agent-gpu2.launch.profiles`.
- The seat's container (`atlas-agent-burst`) gets its own image, `orion-llamacpp-host-prism:0.1.0`
  (`Dockerfile.prism`): the stock image **plus** PrismML's llama.cpp fork at `/app/prism/`. Each
  profile says which binary it runs (`llamacpp.server_build: prism`), so the Q4 profile still runs the
  stock binary in the same image. **Rollback is one line**: swap the two `profiles` entries.
- One durable-run hold per role is **unchanged** (H1). The second slot serves one-off agent calls;
  two Orion runs at once is stage 7.3.
- The llama.cpp #27148 prompt-cache knobs (`cache_ram_mib`, `cache_idle_slots`) are supported in
  profiles but **left unset** (choice below). A post-deploy bleed canary script is committed, not run.
- Gates: `check_gpu_pool_config.py` refuses a fork profile on a service not built from
  `Dockerfile.prism`; the wrapper refuses to boot a fork profile on an image without the fork.

## Outcome moved

Before: an agent call that arrives while a curiosity/self-sense run holds gpu2 waits for that run's
current call to finish (the card has one slot). After: it gets the second slot at once. Measured
baseline below; the 7-day queries say whether it moved. Throughput is not the claim: #2434 measured
two concurrent Bonsai runs at ~23.4 tok/s each (~47 total) vs ~52 alone, so the value is queueing.

## Current architecture

- `config/gpu_pool.yaml` `agent-gpu2.launch.profiles: [qwen3.8-27b-...-agent-flex]`; the pool sends
  `profiles[0]` (`PoolConfig.load_profile`) on every load. The circe controller runs
  `docker compose up -d --no-build --no-deps atlas-agent-burst` with `ATLAS_AGENT_BURST_PROFILE_NAME`,
  after a `launch_digest` check against its own checkout (`pool_fence.resolve`).
- `atlas-agent-burst` ran `orion-llamacpp-host:0.1.0` (stock llama.cpp), shared with chat/metacog/fast/agent.
- Bonsai only ran in `services/orion-llamacpp-bonsai-host` (manual bake-off image, fork copied over `/app`).
- Discovery (`orion/gpu_pool/discovery.py:97-102`) reads `total_slots` and per-slot `n_ctx` from `/props`.
- Live today (2026-10-02 ~04:45 UTC): `agent-gpu2` unloaded; gpu1 `agent` 1 slot x 131072.

## Architecture touched

- orion-llamacpp-host: `Dockerfile.prism` (new), `scripts/build-prism-volta.sh` (new),
  `docker-compose.atlas-workers.yml` (`atlas-agent-burst` image/Dockerfile only), `app/profiles.py`
  (`server_build`, `cache_ram_mib`, `cache_idle_slots`), `app/main.py` (binary choice + fork env,
  cache flags), `scripts/probe_slot_bleed.py` (new), README.
- Config: `config/llm_profiles.yaml` (new `ternary-bonsai2-27b-pq2-v100-32gb-circe-agent`),
  `config/gpu_pool.yaml` (agent-gpu2 profile order).
- Pool core: `orion/gpu_pool/config.py` `check_launch` (prism-image gate). No scheduler change.
- No bus, schema, channel, registry, API, migration, or env change.

## Files changed

- `services/orion-llamacpp-host/Dockerfile.prism`: stock image + fork at `/app/prism` (Prism `88c4bc6`, b10750, sm_70, CUDA 12.8; same pin/checks as the bake-off Dockerfile).
- `services/orion-llamacpp-host/scripts/build-prism-volta.sh`: builds the image on circe, stock half from `.env` `LLAMACPP_IMAGE_TAG` (refuses to guess).
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`: `atlas-agent-burst` builds/runs the prism image.
- `services/orion-llamacpp-host/app/profiles.py`, `app/main.py`: per-profile binary, fork-first `LD_LIBRARY_PATH` (launch and `--help`/`--version` probes), cache knobs (fail closed).
- `services/orion-llamacpp-host/scripts/probe_slot_bleed.py`: #27148 bleed canary for the live seat.
- `services/orion-llamacpp-host/README.md`: "agent-gpu2 on Ternary-Bonsai".
- `config/llm_profiles.yaml`: Bonsai agent profile.
- `config/gpu_pool.yaml`: Bonsai first, Q4 second.
- `orion/gpu_pool/config.py`: `check_launch` prism-image gate.
- Tests: `orion/gpu_pool/tests/test_stage7_2_bonsai_seat.py` (new), `services/orion-llamacpp-host/tests/test_prism_seat.py` (new), `tests/test_probe_slot_bleed.py` (new); default-profile pins updated in `orion/gpu_pool/tests/test_stage5_config.py`, `services/orion-gpu-lane-controller/tests/test_{actuator_bus,launch_exec}.py`, `services/orion-gpu-pool/tests/test_stage5_3_cutover_e2e.py`; `services/orion-llamacpp-bonsai-host/tests/test_bonsai_contract.py` (the fork is now allowed on `atlas-agent-burst` only).
- `.github/workflows/orion-gpu-pool-tests.yml`: runs the two new llamacpp-host test files.

## Choices made

- **One image with both binaries, binary chosen per profile** (not an env-selected image as the design
  doc sketched). The pool can only send a profile name and a card index, so an image chosen by env
  would make rollback two edits on two hosts. Here rollback is the profile order alone.
- **#27148 knobs: supported, unset.** Juniper's rule (spec, "Juniper's answers" 2) is to turn the idle-slot
  RAM cache off only if the leak reproduces; the 2026-10-01 probe did not reproduce on metacog/fast.
  The spec also says that probe does not clear Bonsai, so `probe_slot_bleed.py` runs against the live
  seat after deploy. On `LEAK`, set `cache_idle_slots: false` on every multi-slot profile.
- **Profile values:** `reasoning: auto` (Prism: `--reasoning on` overrides a client's thinking-off),
  `reasoning_effort: xhigh` (spec D7, parity), `preserve_thinking: true` (design question 3, parity),
  `n_predict 16384`, FA on, same sampling as agent-flex.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: `agent-gpu2` loads Bonsai 2 x 131K on its next load; `LlamaCppConfig` gains three
  optional fields (default `None` = no change for any existing profile).
- Compatibility: older wrappers ignore the new profile fields (pydantic default `extra="ignore"`), so the
  bake-off image keeps working. **Changing `launch.profiles` moves agent-gpu2's `launch_digest`**:
  athena's pool and circe's checkout must be on the same commit, or the controller refuses
  `agent-gpu2` loads/unloads with `launch_digest_mismatch` (safe: gpu2 stays with diffusion/world).

## Env/config changes

- Added/removed/renamed keys: none. `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. python scripts/check_gpu_pool_config.py      -> ok (2 launch blocks, digest e068488e9859e9ce)
python scripts/check_env_template_parity.py                -> PASS (94 services)
python scripts/check_circe_worker_refs.py                  -> PASS
git diff --check                                           -> clean
pytest orion/gpu_pool/tests services/orion-gpu-lane-controller/tests -> 434 passed
services/orion-gpu-pool: pytest tests                      -> 147 passed, 15 skipped (Postgres-only)
services/orion-llamacpp-host: pytest tests                 -> 50 passed, 1 failed (pre-existing on
    origin/main: test_qwen3_8b_atlas_metacog_profile_q5km_single_lane_16k expects --parallel 1, the
    profile is 4; not in CI)
services/orion-llamacpp-bonsai-host: pytest tests          -> 7 passed
```

What the new tests pin: acceptance check 3 config side (discovery of 2 x 131072 confirmed; a Bonsai
announce on the Q4 file is a mismatch), H1 unchanged (a third hold is not granted) while the second slot
grants a one-off call that waits at 1 slot (control), 131K per slot keeps a 100K request placeable
(control: 4 x 65K does not), the profile half of acceptance check 10 (reorder -> next load is Q4, gate
clean, digest moves for agent-gpu2 only), the prism-image gate (with a Q4-only control), wrapper argv/env
for both profiles, fail-closed paths, and the probe's verdicts against a fake llama.cpp.

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py -> VERDICT: PASS
```

Gap (owned by 7.3 per the spec): the pool-day eval models agent roles at 1 slot with occupancy-blind
service time, so it is not evidence for 2 slots. The 7.2 evidence is live: the bleed canary and the
7-day queries below.

## Deploy gate and order (Juniper)

**Gate: stage 6.2's 48 h checkpoint first** (Juniper's answer 4). 6.2's per-role telemetry started
2026-10-01 01:27 UTC, so the window closes ~2026-10-03 01:30 UTC; run its Q0-Q5
(`docs/superpowers/pr-reports/2026-09-30-gpu-pool-stage6-2-inference-clocks-pr.md`). **Merging is the
deploy trigger**: athena's post-merge auto-rebuild rebuilds `orion-gpu-pool` on the next `git pull`
(this PR touches `orion/gpu_pool/config.py`), so do not merge before the checkpoint passes.

Order after merge (one line each):

```bash
# 1. circe: build the image from a fresh worktree of main (touches no running container; ~CUDA compile)
cd /mnt/scripts/Orion-Sapienform && git fetch origin && git worktree add --detach ../Orion-Sapienform-prism-build origin/main && ../Orion-Sapienform-prism-build/services/orion-llamacpp-host/scripts/build-prism-volta.sh
# 2. circe: move the controller's checkout (compose + gpu_pool.yaml) to main
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
# 3. athena: pull main (post-merge auto-rebuild redeploys orion-gpu-pool with the new config)
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

No llamacpp-host container restarts: the pool starts `atlas-agent-burst` on its next gpu2 load (demand
waiting >= 1200 s). chat/metacog/fast/agent are untouched. If gpu2 is loaded with the Q4 at deploy time,
it keeps serving Q4 until its idle unload; the following load is Bonsai.

Between steps 2 and 3, agent-gpu2 loads are refused (`launch_digest_mismatch`) -- by design, gpu2 stays
with diffusion. Running step 3 before step 1 would make the first load fail (`startup_failed`, image
missing under `--no-build`), roll back to diffusion, and retry after the 600 s cooldown.

## Rollback

Swap the two entries in `config/gpu_pool.yaml` `agent-gpu2.launch.profiles` (one-line PR), then the same
steps 2 and 3. The image stays; the Q4 profile runs its stock binary.

## Post-deploy checks (an agent runs these from athena; nothing for Juniper)

1. Discovery (acceptance 3): `curl -s localhost:8127/v1/pool` shows agent-gpu2 `confirmed`, `slots: 2`,
   `ctx_per_slot: 131072`, profile `ternary-bonsai2-27b-pq2-v100-32gb-circe-agent`, after its first load.
2. Bleed canary, once that seat is loaded and idle:
   `python3 services/orion-llamacpp-host/scripts/probe_slot_bleed.py --url http://100.112.254.99:8016`
   (exit 0 PASS / 1 LEAK / 2 REFUSED / 3 INCONCLUSIVE; evidence under `/tmp/slot-bleed-probe/`). It goes
   straight to the worker (athena is allowed through circe's port gate), refuses unless `/props` shows
   Bonsai with >= 2 slots and `/slots` all idle, re-checks idleness before every phase, and stops if
   real work arrives.
3. Thermals in the soak (spec downstream 6): gpu2 temperature/power from Hub biometrics.

## 7-day measurement (stage 6.2 two-clock telemetry + pool events)

Run in `docker exec -i orion-athena-sql-db psql -U postgres -d conjourney`. First find the switch time:

```sql
select min(created_at) from gpu_pool_events
where event = 'discovery_confirmed' and role = 'agent-gpu2'
  and detail->>'profile_name' = 'ternary-bonsai2-27b-pq2-v100-32gb-circe-agent';
-- then: \set t7 '<that timestamp>'
```

**M1 -- wait p90 per role, one-off calls vs durable-run holds, 7 days before vs after (pool's own clock).**

```sql
select role,
       case when holder like 'durable-runs:%' then 'hold' else 'call' end as kind, priority,
       case when created_at >= :'t7'::timestamptz then 'after' else 'before' end as period,
       coalesce(detail->'grant'->>'profile_name', '?') as profile,
       count(*) as grants,
       round(percentile_cont(0.5) within group (order by waited_ms)::numeric) as wait_p50_ms,
       round(percentile_cont(0.9) within group (order by waited_ms)::numeric) as wait_p90_ms
from gpu_pool_events
where event = 'granted' and role in ('agent', 'agent-gpu2')
  and created_at >= :'t7'::timestamptz - interval '7 days'
  and created_at <  :'t7'::timestamptz + interval '7 days'
group by 1, 2, 3, 4, 5 order by 1, 2, 3, 4, 5;
```

Baseline (7 days to 2026-10-02 05:00 UTC, all Q4): one-off `system` calls p90 **3,292 ms on agent**
(2,040 grants) and **438 ms on agent-gpu2** (909); holds p90 18,226 s on agent and 9,496 s on agent-gpu2.
Expect agent-gpu2 `call` p90 to fall; holds should not change (H1 unchanged). Spec acceptance 6: the
one-off p90 must stay within 2x baseline.

**M2 -- decode speed solo vs shared, busy-at-grant, per-slot cache reuse (6.2 projection).**

```sql
with src as (
  select created_at, event_json->'atom'->>'summary' as s
  from grammar_events
  where trace_id like 'llm_gateway.inference:%'
    and event_json->'atom'->>'semantic_role' = 'llm_inference_window_observed'
    and created_at >= :'t7'::timestamptz - interval '7 days'
    and created_at <  :'t7'::timestamptz + interval '7 days'
),
roles as (
  select src.created_at, m[1] as role, m[2] as body
  from src, regexp_matches(substring(src.s from 'roles=(\S+)'), '([a-z0-9_.-]+)\[([^\]]*)\]', 'g') as m
),
kv as (
  select created_at, role,
    case when created_at >= :'t7'::timestamptz then 'after' else 'before' end as period,
    (regexp_match(body, '(?:^|\|)calls:(\d+)'))[1]::int                        as calls,
    (regexp_match(body, '(?:^|\|)wait_p95_ms:(\d+)'))[1]::int                  as wait_p95_ms,
    (regexp_match(body, '(?:^|\|)decode_tps_solo_p50:([0-9.]+)'))[1]::float    as tps_solo,
    (regexp_match(body, '(?:^|\|)decode_tps_solo_n:(\d+)'))[1]::int            as tps_solo_n,
    (regexp_match(body, '(?:^|\|)decode_tps_shared_p50:([0-9.]+)'))[1]::float  as tps_shared,
    (regexp_match(body, '(?:^|\|)decode_tps_shared_n:(\d+)'))[1]::int          as tps_shared_n,
    (regexp_match(body, '(?:^|\|)busy_max:(\d+)'))[1]::int                     as busy_max,
    (regexp_match(body, '(?:^|\|)slots:(\d+)'))[1]::int                        as slots,
    (regexp_match(body, '(?:^|\|)prompt_n:(\d+)'))[1]::int                     as prompt_n,
    (regexp_match(body, '(?:^|\|)cache_n:(\d+)'))[1]::int                      as cache_n
  from roles where role in ('agent', 'agent-gpu2')
)
select role, period, array_agg(distinct slots) as slots, sum(calls) as calls,
  round(percentile_cont(0.5) within group (order by tps_solo)::numeric, 1)   as tps_solo_p50,
  sum(tps_solo_n) as solo_n,
  round(percentile_cont(0.5) within group (order by tps_shared)::numeric, 1) as tps_shared_p50,
  sum(tps_shared_n) as shared_n,
  count(*) filter (where busy_max >= 2) as windows_busy_2plus,
  round(percentile_cont(0.9) within group (order by wait_p95_ms)::numeric) as wait_p95_window_p90_ms,
  round(sum(cache_n)::numeric / nullif(sum(cache_n) + sum(prompt_n), 0), 3) as cache_reuse
from kv group by 1, 2 order by 1, 2;
```

Run against live data on 2026-10-02 with a trial `t7` (both queries parse and return rows). Today's
`after` rows: agent solo p50 28.5 tok/s (83 samples), agent-gpu2 28.0 (17), no shared samples (1 slot).
Read: agent-gpu2 `slots` = `{2}` after; `windows_busy_2plus` > 0 shows the second slot in use;
`tps_shared_p50` is the real 2 x 131K number (#2434's 4 x 65K measured 23.4); `tps_solo_p50` should sit
near Bonsai's ~52 short / ~32 at 61K, above Q4's ~28. 6.2's window has no `before` rows for the 7 days
ahead of 2026-10-01 01:27 UTC; its baseline is the 48 h checkpoint itself.

## Review findings fixed

(filled after the review subagent)

## Restart required

```text
No manual container restart. circe: build image + pull (controller reads the new checkout per request).
athena: pull (auto-rebuilds orion-gpu-pool). The pool restarts the gpu2 seat itself on its next load.
```

## Risks / concerns

- Severity: high (privacy). Concern: llama.cpp #27148 could restore one conversation into another slot on
  Bonsai/Prism; untested on this model. Mitigation: the canary runs right after the first load; on LEAK,
  `cache_idle_slots: false` on every multi-slot profile, or roll back the profile order.
- Severity: medium. Concern: quality -- Bonsai 19/20 vs Q4 20/20, one fabricated-looking story in the
  bake-off. Mitigation: the design doc's A/B (`turn_ok` within 5 points over >= 1 day / 30 turns) and a
  hand-read sample of curiosity findings; rollback is one line.
- Severity: medium. Concern: per-call speed when both slots decode (23.4 tok/s each at 4 x 65K, below
  Q4's 25-33). The 2 x 131K two-active-run number has never been measured; M2's `tps_shared_p50` measures it.
- Severity: low. Concern: deploy order (image before the pool's next load). Mitigation: a wrong order
  fails safe (load refused or rolled back to diffusion) and the order is above.
- Severity: low. Concern: gpu2 thermals under a ~100% duty cycle. Mitigation: the existing thermal guard
  gates loads; `cooling_incident` shed covers running work; check in the soak.

UNVERIFIED (each becomes true or false on first load):
- `Dockerfile.prism` building on circe (not built here: no CUDA/Volta toolchain on athena).
- Prism b10750's `/props` reporting per-slot `n_ctx` (131072) the way b10398 does.
- VRAM at 2 x 131K with world co-resident (expected ~24.1 GB + ~1 GB, from the 4 x 65K measurement).
- The FCC/Claude Code passthrough (`/v1/messages`) on Prism with the Bonsai template.
- `id_slot` being honored on `/v1/chat/completions` (the canary records it either way).

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
