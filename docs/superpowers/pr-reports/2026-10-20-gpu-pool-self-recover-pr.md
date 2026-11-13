# fix(gpu-pool): gpu2 heals itself from "seat and diffusion both running", and a stuck card reaches Juniper

## Summary

- **The probe lied in two places, not one.** diffusion-host's `/health` said 200, and so did its `/ready`. With CPU offload a "loaded" FLUX pipeline sits in CPU RAM, so on 10-16 diffusion loaded fine (log: `diffusion model loaded` at 04:03:00) beside the agent-gpu2 seat and answered ready for 4 days with ~7.6 GiB free. Fix: `/ready` now also needs room on the card (`DIFFUSION_READY_MIN_GPU_GB=20`), and the pool's discovery probes launch-managed services on `launch.ready` with the controller's own body contract (`ready_body_ok`).
- **The pool now clears the fault by itself.** With the honest probe, `_clear_faults` sees "seat up, diffusion not ready" and marks gpu2 idle with the seat loaded. It no longer clears it the wrong way at pool boot before the seat has announced.
- **The lane controller resolves "both running" instead of refusing.** It stops the side that holds at most `GPU_LANE_RESOLVE_EMPTY_MAX_MIB` (2048) of GPU memory, is not ready, and has no work in flight. It acts only when the other side provably holds the card. Then it runs the requested load/unload as usual. Evidence: `docker top` host PIDs joined to `nvidia-smi --query-compute-apps`, run through `docker exec` in every running role's own container. Flag `GPU_LANE_RESOLVE_BOTH_RUNNING_ENABLED=true`.
- **A stuck card now reaches Juniper.** The pool raises a *critical* card (emailed at once) after 30 min in fault, repeats every 6 h, and sends an info card on recovery. orion-thought raises a *critical* card after 6 painting attempts in a row produce nothing.
- **End-to-end tests** wire the real pool runtime to the real controller code over a simulated circe: a rebuild restarts diffusion (with or without a pool restart) and gpu2 heals and paints; the seat restarts mid-lend; the seat starts beside a diffusion-owned card.

## Outcome moved

The 10-16 state ("seat holds gpu2, diffusion up beside it, card faulted") previously needed a human `docker stop`, and nobody was told for 4 days (03:02 10-16 to 05:08 10-20: 0 images, 35 attempts in a row abandoned). Now the card returns to idle with no human action. The next painting unloads the seat and starts diffusion fresh. If something still leaves a card faulted, Juniper gets an immediate email within 30 min.

## Current architecture (before)

- Pool discovery probed every `kind: service` role on `health` (`up` = HTTP below 500). Only the lane controller used `launch.ready`.
- `_clear_faults` cleared a fault only when exactly one side looked up. With diffusion's `/health` always 200 and the seat alive, both looked up, so it never fired.
- The controller raised `seat_and_evicted_both_running` on both load and unload.
- Alerts: the painting-gap check fired once at severity `error` (pending card, one email after 60 min unacked). The guardian's GPU watch probes *controllers* (refusals). A card fault with a healthy controller was neither.

## Why nobody was told (live evidence)

- `visual_painting_gap` **did fire**: `notify_requests` 2026-10-16 15:10:43, severity `error`, `drop_reason=immediate_critical_only_severity_error`, escalated to email 16:11:03, never acked. Edge-triggered, so it said nothing more for 3.5 days. Whether that one email arrived is **UNVERIFIED**: orion-notify restarted 10-16 18:09 and its earlier logs are gone. It was one of 51 `error` cards since 10-01.
- The **mesh guardian was blind first, then watching the wrong thing**. Until its rebuild at 10-16 18:08 it could not parse `gpu_pool.yaml` (`gpu_watch_config_unloadable`, every 6 h). After that it logged `gpu probe cycle {'agent-gpu2/digest': 'ok', 'diffusion/digest': 'ok'} alerts=0` hourly. That was correct: the controller was healthy. Card fault state is not something it watches.
- **The pool never alerted on a fault.** Its only alert was the controller-health card (refusals). `seat_and_evicted_both_running` is a *failed* action, not a refusal.

## Architecture touched

- `orion/gpu_pool/config.py`: `ready_body_ok` (one readiness contract), `RoleSpec.discovery_path` / `discovery_strict`.
- orion-gpu-pool (athena): strict probe + `make_prober`; card-fault alert in `runtime._check_fault_alerts`; `controller_alert.build_fault_request`.
- orion-gpu-lane-controller (circe): `launch_exec.resolve_both_running` + GPU memory evidence.
- orion-diffusion-host (circe): `/ready` room check.
- orion-thought (athena): `visual_attempt_failure_streak` check + `store.visual_attempt_failure_streak()`.

## Files changed

- `orion/gpu_pool/config.py`: shared ready contract, discovery path per role.
- `services/orion-gpu-pool/app/{main,runtime,controller_alert,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`: honest probe, fault alert.
- `services/orion-gpu-pool/tests/test_gpu2_both_running_incident.py`: incident replay through the real prober; fault alert.
- `services/orion-gpu-pool/tests/test_gpu2_self_heal_e2e.py`: pool + real controller end to end.
- `services/orion-gpu-lane-controller/app/{launch_exec,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`, `tests/test_launch_exec.py`: resolver.
- `services/orion-diffusion-host/app/{main,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`, `tests/test_health.py`, `tests/test_generate.py`: `/ready` room check.
- `services/orion-thought/app/{visual_chain_health_monitor,store,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`, `tests/test_visual_attempt_streak_monitor.py`, `evals/test_attempt_streak_replay.py`, `evals/fixtures/visual_attempts_2026-09-14_to_10-19.json`: streak check + replay eval.

## Schema / bus / API changes

- Added: none on the bus or schema registry. New notify `event_kind`s ride the existing `/attention/request` HTTP API: `orion.gpu_pool.card_fault.v1`, `orion.reverie.visual_attempt_failure_streak.health.attention.v1`.
- Behavior changed: diffusion-host `GET /ready` returns 503 when the card lacks room. Its body adds `gpu_available_gb`, `gpu_min_gb`, `gpu_check_error`. Controller failure reasons may carry `seat_and_evicted_both_running:<why>` and `...:resolved=<role>`.
- Compatibility: `launch_digest` is unchanged (no `gpu_pool.yaml` edit), so pool and controller need no lockstep for the digest. The controller and diffusion images must be rebuilt for the new code.

## Env/config changes

- Added keys: `GPU_LANE_RESOLVE_BOTH_RUNNING_ENABLED=true`, `GPU_LANE_RESOLVE_EMPTY_MAX_MIB=2048` (lane controller); `DIFFUSION_READY_MIN_GPU_GB=20` (diffusion-host); `GPU_POOL_FAULT_ALERT_ENABLED=true`, `GPU_POOL_FAULT_ALERT_AFTER_SEC=1800`, `GPU_POOL_FAULT_REALERT_SEC=21600` (gpu-pool); `ORION_VISUAL_ATTEMPT_STREAK_CHECK_ENABLED=true`, `ORION_VISUAL_ATTEMPT_FAILURE_STREAK_THRESHOLD=6` (thought).
- Removed / renamed keys: none.
- `.env_example` updated: yes (4 services); compose fallbacks match.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys <services>`: yes, all 8 keys added on athena's checkout.
- Skipped keys requiring operator action: none. **circe's own `.env` files are not synced from here.** The compose fallbacks carry the same defaults, so no hand edit is needed.

## Metric quality gate (new signals)

1. **diffusion `/ready` room** = `torch.cuda.mem_get_info` free + `memory_reserved`, the driver's count (`app/main.py::_gpu_available_gb`). Live: gpu2 with the seat had 7791 MiB free; diffusion's own idle CUDA context adds ~0.3-0.5 GiB. An empty card is ~31 GiB. Rest state is "ready" whenever the card is free. Theory: offloaded FLUX needs ~24 GB at peak, so a generation with ~7.6 GiB would run out of memory. Independent of `model_loaded`.
2. **Resolver VRAM per role** = host PIDs (`docker top`) joined to nvidia-smi compute apps. Checked live from inside the real controller container on circe: llama-server PID 3846941 showed 23828 MiB, in the format the parser reads.
3. **Failed-attempt streak** = `reverie_visual_attempt` rows since the newest `produced` one. Live read 2026-10-20: 35. History (405 attempts): ordinary streaks max out at 5. It resets to 0 on any produced attempt.
4. **Card fault age** = pool in-memory time since `swap_state` became `fault`. No new instrument.
- Reversibility: each is behind its own flag; no schema, manifest or training default depends on them.

## Tests run

```text
services/orion-gpu-pool:            pytest tests -q            -> 194 passed, 15 skipped (Postgres; run in CI)
services/orion-gpu-lane-controller: pytest tests -q            -> 101 passed
services/orion-diffusion-host:      pytest tests -q            -> 61 passed
orion/gpu_pool/tests                                            -> 449 passed
services/orion-thought:             pytest tests evals -q      -> 1 failed (test_settings_mind_enrichment, fails identically on origin/main), rest pass
Old-code check: the new incident, e2e and controller tests fail on origin/main code
  (pool: gpu2 stays "fault"; controller: plain seat_and_evicted_both_running; diffusion: loaded == ready).
  test_rebuild_restarts_the_seat_mid_lend passes on old code too, by design: it guards a path that already worked.
```

## Evals run

```text
services/orion-gpu-pool/evals/run_pool_day_eval.py         -> VERDICT: PASS
services/orion-gpu-pool/evals/run_controller_stale_eval.py -> pass
services/orion-thought/evals/test_attempt_streak_replay.py -> threshold 6 flags exactly the 4 outage streaks (09-25, 10-08, 10-09, 10-16)
services/orion-thought/evals/test_painting_gap_replay.py   -> unchanged, pass
```

## Docker/build/smoke checks

```text
No image built and nothing deployed (task: do not deploy).
Read-only live checks:
- circe: docker exec orion-circe-gpu-lane-controller docker top orion-circe-atlas-llamacpp-agent-burst -eo pid -> PIDs 3846498, 3846941
- circe: docker exec orion-circe-gpu-lane-controller docker exec orion-circe-atlas-llamacpp-agent-burst nvidia-smi --query-compute-apps=... -> "3846941, 23828" (+ world-model 872 on gpu2)
- athena Postgres: gpu_pool_events 10-16..10-20 (reconcile_ambiguous:boot, 2x seat_and_evicted_both_running, fault_cleared 10-20 05:08), notify_requests, reverie_visual_attempt.
UNVERIFIED: whether nvidia-smi inside diffusion-host's own container answers (it was stopped). If it does not, the resolver refuses (gpu_memory_unknown) and the 30-min fault alert fires instead.
```

## Review findings fixed

- Finding: the "never stops a side holding memory" promise rested on one container's nvidia-smi seeing the other's processes.
  - Fix: nvidia-smi runs in every role's own container and the views are merged; any container that cannot answer means refuse.
  - Evidence: `test_each_side_is_judged_by_its_own_containers_view`, `test_a_hidden_holder_is_still_seen_by_its_own_view`, `test_no_nvidia_smi_in_the_empty_sides_container_refuses`.
- Finding: diffusion `/ready` called CUDA on the event loop (first call may start CUDA).
  - Fix: `asyncio.to_thread` + 5 s timeout, fail closed.
  - Evidence: `test_ready_times_out_closed_on_a_hung_gpu_read`.
- Finding: a failed action hid what the resolver had stopped.
  - Fix: `...:resolved=<role>` on load and unload failures; the unload-before-idle-check order is documented as harmless.
  - Evidence: `test_mirror_load_failure_rolls_back_to_diffusion_and_names_the_resolver_stop`, `test_unload_on_a_busy_seat_after_the_resolver_says_so`.
- Finding: mirror load case, its rollback, and multi-container roles were untested.
  - Fix: tests added (`test_mirror_load_*`, `test_pids_from_every_container_of_a_role_are_summed`).
- Finding: memory alone does not prove idle.
  - Fix: a role with a drain spec whose status is not `in_flight:false` is never stopped (`:empty_side_in_flight`).
  - Evidence: `test_empty_side_with_work_in_flight_is_not_stopped`.
- Finding (nit): GB vs GiB; 20 is below the 24 GB peak.
  - Fix: docs say GiB, and that `/ready` means "card not taken", not "a generation fits". `_gpu_available_gb` formula test added.

Second review (commit 2: alerts, e2e):

- Finding (must-fix): the streak SQL counted attempts still in progress (`active`, `unknown`) as failures, so 5 ordinary failures + 1 running painting would page.
  - Fix: `outcome NOT IN ('active', 'unknown')`.
  - Evidence: run read-only on live Postgres with a CTE (produced, 5 failures, 1 active) -> 5; on the real table -> 35. Plus `test_streak_sql_never_counts_an_attempt_still_in_progress`.
- Finding: an operator `clear_fault` sent "recovered" while the card was still reconciling, and reset the 30-min clock.
  - Fix: `loading`/`unloading` after a fault counts as still faulted until it settles idle.
  - Evidence: `test_an_operator_clear_is_not_recovery_until_the_card_settles`.
- Finding: a fault that survived a pool restart lost its reason and start time in the alert.
  - Fix: seeded from the stored `swap_action` at start.
  - Evidence: `test_a_fault_that_survives_a_pool_restart_keeps_its_reason_and_start`.
- Finding: the resolver checked "in flight" and then stopped without latching a drain.
  - Fix: a drain-spec role is drained (latched) before the stop. If the drain fails, it is undrained and the resolver refuses.
- Finding: nvidia-smi availability in diffusion-host's container was unchecked.
  - Fix/evidence: its image declares `NVIDIA_DRIVER_CAPABILITIES=compute,utility` (the capability that injects nvidia-smi). A live exec is still UNVERIFIED (container stopped).
- Nits fixed: the streak check runs in the gap loop (documented); recovery wording no longer claims "an image was produced"; dead fallback removed; first `/ready` after boot may time out once while CUDA starts (documented).

## Restart required

Deploy order: circe first (controller + diffusion), then athena. Each command is one line, run from the clean primary `main` checkout after merge and `git pull`.

```bash
# circe
scripts/safe_docker_build.sh orion-gpu-lane-controller up -d --build
scripts/safe_docker_build.sh orion-diffusion-host up -d --build
# athena
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: medium. Concern: the resolver stops containers on its own. Mitigation: it stops only the side with at most 2 GiB that is not ready and has nothing in flight (drained first when it has a drain), and only when the other side holds more than 2 GiB. The pool's generation fences every stop. Flag off = old refusal.
- Severity: low. Concern: an idle offloaded diffusion holds ~0 GiB. If the *seat* is the broken side and diffusion is idle, neither holds memory, so the resolver refuses. Mitigation: refusal is safe, and the 30-min critical card fires.
- Severity: low. Concern: the 6-attempt streak would also have paged on 09-25 and 10-08 (9 in a row, causes unknown). Mitigation: both were real ~14 h outages that the gap check also flagged.
- Severity: low. Concern: the fault-alert timer is memory-only. Mitigation: a fault stored before a restart is re-seeded from its `swap_action.finished_at`. A fault made by the boot reconcile itself starts its clock at boot.
- Severity: low. Concern: pre-existing, not from this PR: `check_service_env_compose_parity.py orion-gpu-pool` reports `GPU_POOL_LEASE_RETENTION_HOURS` missing from compose on main (this gate is not in CI). `orion-thought` `test_settings_mind_enrichment` fails identically on origin/main.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2785

🤖 Generated with [Claude Code](https://claude.com/claude-code)
