# feat(gpu-pool): Orion's sight on `perception`, never shed by heat, with a load alert (A0b)

**Merge and deploy #2856 (A0a) everywhere first, including circe's lane controller.**

## Summary

- vision-council's scene reads move from `metacog_background` to a new route, `metacog_perception`, at pool priority `perception`. That priority ranks below Juniper's turn, above Orion's own work, and is never shed by heat.
- The route is registered in `orion/llm/routes.py`:
  - accepted;
  - pinned to the metacog profile;
  - system-only: hidden from Juniper's Compute picker and refused as an override, the same as `metacog_turn`.
- The pool now watches perception's own GPU use instead of shedding it. A Hub warning card fires when the last hour's held slot-seconds pass a ceiling, and an info card when they fall back.
- vision-council's settings, `.env_example` and compose default all agree on the new route. The compose default previously said `metacog`, which disagreed with the others.

## Outcome moved

- **Failure mode removed:** Orion going blind whenever the cabinet runs hot.
- **The evidence:** in the 7 days to 10-23, `gpu_pool_events` shows vision-council refused **25,430 times** with `shed:cabinet_hot`. Over the same 7 days it received 19,587 grants and held 33.1 GPU-hours.
- **After deploy:** vision-council keeps getting grants while `cabinet_hot` is active.
- **Design:** Patch A0 of `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md` (PR #2855). Juniper decided vision is Tier 0 and gets the perception priority.

## Current architecture

- vision-council calls the gateway on `COUNCIL_LLM_ROUTE=metacog_background` (pool class metacog, priority background).
- The hardware-watch reflex asserts `cabinet_hot` at ≥34 °C. The shed board then blocks new `background` and `system` grants.
- The council's transition gate already limits LLM calls to scene changes plus a 10-minute refresh.

## Architecture touched

- **Pool config:** a new route, and `priorities:` now names `perception` explicitly.
- **Gateway route registry:** `orion/llm/routes.py`.
- **gpu-pool runtime:** a perception load tracker and an alert.
- **vision-council:** its route default.

## Files changed

- `config/gpu_pool.yaml`: `priorities:` names perception; the new `metacog_perception: {class: metacog, priority: perception}` route.
- `orion/llm/routes.py`: `metacog_perception` added to ACCEPTED, DISPLAY_ORDER, METACOG (profile pin) and SYSTEM (hidden from the human picker).
- `orion/gpu_pool/perception_load.py` (new): a pure 1-hour rolling sum of perception held slot-seconds, with high/recovered transitions.
- `services/orion-gpu-pool/app/runtime.py`: records a perception hold **once per transition**, only when the lease leaves a held slot. Checks the load every tick and fires the alert.
- `services/orion-gpu-pool/app/controller_alert.py`: `build_perception_request` (a warning card).
- `services/orion-gpu-pool/app/{main,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`: three new env keys and their documentation.
- `services/orion-vision-council/app/settings.py`, `.env_example`, `docker-compose.yml`: route default `metacog_perception`.
- Tests:
  - `orion/gpu_pool/tests/test_perception_load.py`
  - `services/orion-gpu-pool/tests/test_perception_runtime.py`
  - a pinned route test in `test_route_view.py`
  - the updated council default test and gateway models-list test

## Schema / bus / API changes

- Added: route `metacog_perception`. Pool priority `perception` is now used by a real caller.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - vision-council leases are never heat-shed.
  - They rank above `system` work on metacog, including Mind's non-turn reads. Juniper's turn (`metacog_turn`, interactive) still outranks them.
- Compatibility notes:
  - Requires A0a's schema and config code on every loader.
  - Until llm-gateway is rebuilt with this PR, the gateway does not know the route. Hence the deploy order below.

## Env/config changes

- Added keys (orion-gpu-pool):
  - `GPU_POOL_PERCEPTION_ALERT_ENABLED=true`
  - `GPU_POOL_PERCEPTION_ALERT_HELD_SEC_PER_HOUR=3360`
  - `GPU_POOL_PERCEPTION_REALERT_SEC=21600`
- Changed default (orion-vision-council): `COUNCIL_LLM_ROUTE` from `metacog_background` to `metacog_perception`.
- `.env_example` updated: yes (both services).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The three gpu-pool keys were added.
- skipped keys requiring operator action:
  - **`COUNCIL_LLM_ROUTE` in `services/orion-vision-council/.env`.** The live value is explicitly `metacog_background`. The sync script never overwrites a diverged value, so it must be flipped by hand.
  - Do this **after** the gateway runs this PR. If it is flipped earlier, a council restart would get `RouteNotInPool`. See the restart steps.

## Tests run

```text
orion/gpu_pool/tests + orion/llm                    524 passed
services/orion-gpu-pool/tests                       197 passed, 15 skipped
services/orion-llm-gateway/tests                    459 passed
services/orion-vision-council/tests                 127 passed
services/orion-mind/tests/test_mind_turn_route.py   10 passed
tests/test_regulation_arousal.py                    68 passed
services/orion-sql-writer/tests/test_gpu_pool_state_history.py  8 passed
scripts: check_gpu_pool_config, check_chat_route_poachers, check_circe_worker_refs,
         check_service_hostname_refs, check_env_template_parity, check_env_key_single_source  all PASS
check_service_env_compose_parity orion-gpu-pool: FAILS on GPU_POOL_LEASE_RETENTION_HOURS, pre-existing on main (not this PR)
launch_digest(agent-gpu2), launch_digest(diffusion): byte-identical to main
```

## Evals run

```text
None new. Ceiling basis, live query (gpu_pool_events, vision-council released rows, 337 hours to 10-23):
hourly held slot-seconds p50 177, p90 1,679, p99 2,120, max 2,811 -> ceiling 3,360 (2x p90).
```

## Docker/build/smoke checks

```text
Not run in this session. Live proof after deploy (below).
```

## Review findings fixed

- Finding (MUST): the live vision-council `.env` pins `metacog_background`, so the new default would never take effect.
  - Fix: named as an explicit operator step, ordered after the gateway rebuild.
  - Evidence: `services/orion-vision-council/.env:13` on athena.
- Finding (SHOULD): one transition can emit `expired`+`released` (or `aborted`+`released`), and `granted_at` is never cleared. Per-event recording double-counted, and also re-counted a `retry_wait` lease's old grant.
  - Fix: record once in `_transition`, only when the prior status is granted/recalling and the new one is not.
  - Evidence: the new `test_expired_perception_lease_counts_its_hold_once`. The old code recorded 70 s against a 35 s cap.
- Finding (SHOULD): the 3,360 ceiling was measured while vision-council was being heat-shed, so its normal unshed load may be higher. An `error` card would email Juniper after 60 min unacked.
  - Fix: severity is `warning` (a Hub card, no email) until a week of perception-priority data re-baselines it.
  - Evidence: `controller_alert.py`, plus the test asserting `warning`.
- Finding (SHOULD): the runtime heat test could pass because slots were full.
  - Fix: it now asserts the system lease is refused with `shed:cabinet_hot`.

## Restart required

Order matters. Run on athena, from the clean primary `main` checkout, after A0a (#2856) is deployed everywhere.

1. Rebuild the gpu-pool:
   ```bash
   scripts/safe_docker_build.sh orion-gpu-pool up -d --build
   ```
2. Rebuild the gateway:
   ```bash
   scripts/safe_docker_build.sh orion-llm-gateway up -d --build
   ```
3. Flip the live council route:
   ```bash
   sed -i 's/^COUNCIL_LLM_ROUTE=metacog_background$/COUNCIL_LLM_ROUTE=metacog_perception/' services/orion-vision-council/.env
   ```
4. Rebuild vision-council:
   ```bash
   scripts/safe_docker_build.sh orion-vision-council up -d --build
   ```

Live proof:

```sql
select priority, count(*) filter (where event='granted') granted,
       count(*) filter (where reason='shed:cabinet_hot') shed
from gpu_pool_events where holder='vision-council' and created_at > now() - interval '1 hour' group by 1;
```

Expected: `perception` rows with grants, and 0 `shed:cabinet_hot`, including during any hot hour.

## Risks / concerns

- Severity: medium.
  - Concern: perception now outranks Mind's non-turn reads and the daily metacog draft on gpu3. Under heavy vision load those wait longer.
  - Mitigation: this is the intended Tier 0 over Tier 1 order. The load card fires if vision use doubles its normal p90.
- Severity: low.
  - Concern: perception GPU heat is no longer shed at 34 °C.
  - Mitigation:
    - The council is ~33 GPU-h a week on one V100.
    - The 14-day correlation of cabinet temperature with circe watts is weak (r 0.10–0.22, `UNVERIFIED` causally).
    - The load alert watches for growth.
- Severity: low.
  - Concern: the ceiling may need retuning after a week.
  - Mitigation: severity is `warning`. Re-run the ceiling query against `priority='perception'` rows after 7 days.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2857

🤖 Generated with [Claude Code](https://claude.com/claude-code)
