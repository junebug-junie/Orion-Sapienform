# Arousal: a GPU queue jammed by a broken lane controller is not strain

## Summary

- Orion's arousal reading no longer calls a GPU queue "strained" when the queue is stuck because the
  GPU lane controller is refusing to act. During such a fault the GPU input reads like a missing
  input: the level is `unknown` (readers then behave exactly as before arousal existed) and the
  reading says why: `gpu_fault:<kind>:<seat>`, e.g. `gpu_fault:config_unreadable:agent-gpu2`.
- The fault fact is one the pool already records: `actuate_refused` rows in `gpu_pool_events`, judged
  by the pool's own "a retry cannot fix this" classifier (`orion/gpu_pool/controller_health.py`),
  ended by the seat's next `swapped`. No GPU pool change.
- Real load still strains: the 2026-10-12 14:08 span (no refusal, 199 grants in 40 min) still reads
  strained for the same 26 min.
- Hub outreach now re-reads arousal right before it delivers. This is the cause of the 4
  `arousal_view reader=outreach level=unknown` lines (a reader-side stale cache, not a producer gap).
- Replay scripts take the fault input, with `--no-gpu-fault` for a before/after on the same rows.

## Outcome moved

Reader replay, 2026-10-09 06:50 -> 2026-10-13 17:00 UTC, same export, shipped reader rules:

| | before | after |
|---|---|---|
| 10-09 strained minutes | 1028 | 0 (1030 unknown, all `gpu_fault:config_unreadable:agent-gpu2`) |
| 10-10 strained minutes | 310 | 88 (204 unknown while the fault lasted) |
| 10-12 14:08-14:34 (real load) | strained 26 min | strained 26 min |
| 10-09 decisions held (curiosity / self-inquiry / outreach) | 5/5, 6/6, 4/4 | 0, 0, 0 |
| whole window held (curiosity / self-inquiry / outreach) | 5 / 7 / 4 | 0 / 1 / 0 |

The remaining self-inquiry hold (10-10) and the 10-10 06:18, 09:56, 10:28 strained spans are after the
controller swapped at 06:01: the backlog it left was real work being drained.

Reducer replay totals (minutes): before engaged 242.5 / idle 4743.5 / strained 1384 / unknown 0;
after 242.5 / 4759.5 / 134 / 1234. Every unknown step's reason is `gpu_fault:config_unreadable:agent-gpu2`.
Evidence: `docs/superpowers/evidence/2026-10-13-arousal-gpu-fault-vs-load/reader_replay_before_after.json`.

## Metric gate (new input: `gpu_fault`)

1. **Provenance.** `services/orion-gpu-pool/app/runtime.py` `_finish(..., event="actuate_refused", reason=res.reason)`
   when the lane controller answers `refused` to the pool's own load/unload; sql-writer lands it in
   `gpu_pool_events` (`role` = seat). A refusal on a status probe lands as `swap_failed` /
   `status_refused:<reason>`. The clear is `swapped` (controller did what was asked).
2. **Independence.** S2 depth comes from `gpu_pool_state_history` (the 5 s state broadcast). The fault
   comes from the controller's reply to a swap the pool asked for. A jam *causes* depth, but depth
   never causes a refusal; and the 10-12 load span had depth >= 2 with zero refusals. Not a transform
   of depth. It is not a new arousal signal either: it only removes depth's vote when depth can't be
   attributed to load.
3. **Theory anchor.** Queue depth measures load only if the pool can serve the queue. A controller
   that refuses every actuation (config it cannot read, digest mismatch, ...) means the pool cannot
   change what is loaded, so the queue measures that fault. The pool already encodes this exact
   rule for its own alerting (`controller_health.classify`: unfixable-by-retry refusal kinds;
   `on_answered` clears). Arousal reuses it instead of inventing a second taxonomy.
4. **Live data.**
   - 10-09 04:07 -> 10-10 05:51: 155 `actuate_refused` `config_unloadable:ValidationError`, all on
     `agent-gpu2`, median 601 s apart, max 10 min 04 s. Agent queue 9-18 all day; memory_distill and
     diffusion got **zero** grants until 10-10 02:00. Ended by `swapped` at 10-10 06:01:51.
   - 10-12 14:00-14:40: **0** refusals; 199 grants (110 chat->agent, 10 agent, 64 metacog ...).
   - Rest state: no refusal anywhere in the table after 10-10 05:51 -> `gpu_fault` reads None every
     step since (not degenerate: it was on for 26 h and off for 3+ days, cleanly).
   - **Progress-based detection was checked and rejected.** Minutes with depth >= 2 and zero
     grants/releases pool-wide in the prior 5 min: 10 of 1030 on 10-09, 13 of 274 on 10-10, 0 on
     10-11/10-12. The jammed day still granted agent work at 15-34/h, so "nothing moving" would have
     missed it. Per-class "no grant while queued" would misfire on the load span: memory_distill
     waited 1,191 s on 10-12 14:07 under ordinary contention.
5. **Existing mechanism.** `orion/gpu_pool/controller_health.py` (the pool's in-memory degraded-seat
   tracker). Reused for classification. Its state is memory-only and not published, so arousal
   re-derives from `gpu_pool_events` every step, which also survives restarts of either service.
6. **Reversibility.** One optional field on a producer-internal model (`ArousalInputsV1`; a checkpoint
   that no longer validates is dropped, costing one step of cabinet hysteresis) and a reason string.
   No reading-schema change, no migration, no reader change for the fault. Revert = revert the commit.

**Not covered (named, not built):** a card left in `fault` after a non-refusal `swap_failed`
(`actuator_unreachable`, failed unload). Only 7 such events exist, all before
`gpu_pool_state_history` began (10-09 06:46), so there is no live queue data to validate a detector.
`backlog_depth` stays excluded (empty in every snapshot).

## Why outreach read `unknown` 4 times in 40 h

Not a producer gap. The regulate step's checkpoints are on time every 120 s across all four moments
(e.g. 10-13 15:33:52, 15:35:52, 15:37:52, 15:39:52). Each `unknown` is `reason=stale age_sec=365-391`
logged within 10 ms of `endogenous_outreach_sent`, right after `agent_lane_attempt_failed_fallback_to_chat`
(210 s agent timeout + chat fallback). Outreach refreshed the arousal reading once at the top of the
tick, composed for 6+ minutes, then re-gated before delivery on that cached copy; past its 360 s max
age it read stale -> unknown, which never blocks. So strain that began mid-compose could not hold
the send (the 14:06 line is the same shape). Fix: refresh the reader before the pre-delivery re-gate.
Regression test fails without it (`(True, 'sent')`).

## Current architecture

Durable-runs `temporal_self.update` regulate node, every 120 s: reads E1 Juniper turns, S1 cabinet
reflex, S2 GPU queue depth sustained (`gpu_pool_state_history`), folds through the pure
`classify_arousal`, writes `RegulationStateV1` to Redis `orion:regulation:latest` (TTL 360 s).
Readers: orion-dream sleep, Hub curiosity investigation / self-inquiry, Hub background outreach.

## Architecture touched

- `orion/regulation/arousal.py` (pure reducer), `orion/schemas/regulation.py` (inputs model).
- orion-durable-runs regulate read (`app/regulation_store.py`): one extra indexed read of
  `gpu_pool_events` (0.2 ms live, `idx_gpu_pool_events_generated`).
- orion-hub outreach pre-delivery re-gate.

## Files changed

- `orion/regulation/arousal.py`: `gpu_fault_from_events`, fault handling in `classify_arousal`.
- `orion/schemas/regulation.py`: `ArousalInputsV1.gpu_fault`; docs.
- `services/orion-durable-runs/app/regulation_store.py`: read pool events, pass `gpu_fault`.
- `services/orion-hub/scripts/endogenous_outreach.py`: refresh arousal before the pre-delivery re-gate.
- `orion/inner_state_registry.py`: S2 note.
- `scripts/analysis/measure_arousal_reducer_replay.py`, `measure_arousal_reader_replay.py`: fault
  input, `--no-gpu-fault`, `unknown_spans`.
- Tests: `tests/test_regulation_arousal.py`, `services/orion-durable-runs/tests/test_temporal_self_regulate.py`,
  `services/orion-hub/tests/test_regulation_arousal_readers.py`.
- Evidence JSON and this report.

## Schema / bus / API changes

- Added: `ArousalInputsV1.gpu_fault: Optional[str] = None` (producer-internal; only durable-runs
  builds and reads it).
- Removed / Renamed: none.
- Behavior changed: `ArousalReadingV1.reasons` may contain `gpu_fault:<kind>:<seat>`; with a queue at/above
  the floor the level is `unknown` (it is the first reason unless another input is also stale). Readers' `arousal_view` reports it as
  `producer_unknown:gpu_fault:...`.
- Compatibility: `ArousalReadingV1` / `RegulationStateV1` unchanged, so readers need no rollout.
  No bus channel, no registry change.

## Env/config changes

- Added / Removed / Renamed keys: none. `.env_example` updated: no.
- `python scripts/sync_local_env_from_example.py` run from the primary root: no key changes.
- Skipped keys: none.

## Tests run

```text
tests/test_regulation_arousal.py + test_regulation_arousal_view.py            63 passed
tests/test_gpu_pool_actuator_probe.py, test_inner_state_*, regulation tests   89 passed
services/orion-durable-runs/tests (ORION_ADMISSION_TEST_DSN on a throwaway
  postgres:16-alpine container, so the 2 Postgres tests ran)                 475 passed, 1 skipped
services/orion-gpu-pool/tests -k controller                                  16 passed
services/orion-hub tests/test_endogenous_outreach.py, test_regulation_arousal_readers.py,
  test_regulation_must_not_readers.py, test_curiosity_investigation.py      465 passed
Mutation checks: s2_faulted forced False -> 4 reducer tests fail; hub refresh removed ->
  the pre-delivery test fails with (True, 'sent').
Static: check_definition_drift --gate PASS, check_inner_state_registry OK, check_metric_lineage
  --gate PASS, check_sentience_instruments --static-only OK, check_settings_defaults orion-hub
  --example-drift OK (orion-durable-runs is not in that check's list; no env keys changed),
  check_migration_required_by OK (no migration), git diff --check clean.
```

## Evals run

```text
scripts/analysis/measure_arousal_reader_replay.py rows.jsonl --start 2026-10-09T06:50:00+00:00 --end 2026-10-13T17:00:00+00:00 [--no-gpu-fault]
scripts/analysis/measure_arousal_reducer_replay.py  (same rows, same window) [--no-gpu-fault]
Results in "Outcome moved"; JSON in docs/superpowers/evidence/2026-10-13-arousal-gpu-fault-vs-load/.
```

## Docker/build/smoke checks

```text
Not deployed (per instruction). No Dockerfile or compose change: durable-runs already copies orion/
(which includes orion/gpu_pool/controller_health.py) and reads gpu_pool_events (as postgres) for the
chronicle's gpu_wait source. Live path UNVERIFIED until deployed.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`.

- Finding (MUST-FIX as filed): a fault on one seat voids the *summed* queue, so real load in an
  unrelated class during a fault reads `unknown`, not `strained`.
  - Decision: kept by design, with evidence, and pinned by `test_fault_voids_the_whole_summed_queue_by_design`.
    On 10-09 the jam spread past agent-gpu2's own configured classes: memory_distill (roles `[agent]`
    only) and diffusion (roles `[diffusion]`, waiting on gpu2's unload) sat at 1-2 and 1 with zero
    grants all day. Scoping the void to the seat's classes (`classes[c].roles`) would still have
    summed 2-3 >= floor and read strained all day, which is the bug being fixed. `unknown` is the
    readers' documented "behave as before arousal existed", never looser than that baseline.
    Listed under Risks.
- Finding: `swapped` with `fault_cleared:discovery` / `fault_cleared:operator` cleared the fault though
  the controller never answered (flap risk).
  - Fix: those are neutral. Evidence: `test_a_card_fault_clear_is_not_a_controller_answer`.
- Finding: same-timestamp rows were decided by row order.
  - Fix: a refusal wins a tie; SQL orders by `generated_at, event_id`. Evidence:
    `test_same_instant_refusal_and_swap_is_a_fault_in_either_row_order`.
- Finding: the replay script's event sort could raise on NULL reason/role ties.
  - Fix: sort by timestamp only (stable). Replay output re-run: identical.
- NIT: operator clear has `role` NULL. Moot after the `fault_cleared:` fix (it is that reason).
- NIT: a failed own-action `swap_failed` proves the controller read its config but stays neutral here
  (reason alone can't tell it from an ack timeout), so a fault can outlive the fix by up to 30 min.
  Documented, not changed.
- NIT: the replay window starts 10-09 06:50 because `gpu_pool_state_history` starts 06:46; the fault
  began 04:07, so "before" strain is cut at the window edge. The export holds 154 of the 155 refusals
  (one is before the window's lookback).
- NIT: hub test pins the re-read, not the stale-cache form; docstring now says so.

## Restart required

Order does not matter (no reading-schema change). Production runs from the primary checkout on
main, after merge (`git -C /mnt/scripts/Orion-Sapienform pull --ff-only`), one line each:

```bash
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-durable-runs up -d --build
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

Verify: `docker exec orion-athena-durable-runs python -c "import orion.regulation.arousal as a; print(a.GPU_FAULT_LOOKBACK_SEC)"`
prints 1800.0, and `curl -s localhost:8124/regulation/state` shows a reading with no
`gpu_fault` reason (no refusals since 10-10).

## Risks / concerns

- Severity: medium. Concern: during a controller fault, real load in any class reads `unknown`, not
  `strained` (the fault voids the whole summed queue; see Review findings for why). Mitigation:
  `unknown` is the readers' pre-arousal behaviour; the pool raises its own controller alert; a fault
  is ended by the seat's next `swapped`. Follow-up if it bites: scope by observed starvation, not
  by configured classes.
- Severity: low. Concern: during a controller fault an `engaged` reading also becomes `unknown`
  (any stale input does), so outreach / curiosity lose their "Juniper is here, back off" stretch for
  the fault's duration. Mitigation: that is the pre-arousal behaviour; the fault is surfaced by the
  pool's own controller alert.
- Severity: low. Concern: strain can start at fault onset before the first refusal lands (the pool
  only asks the controller when it wants to swap). Mitigation: bounded by one swap cooldown (~10 min);
  history before 10-09 06:46 cannot show it.
- Severity: low. Concern: a non-refusal card fault is not detected (see "Not covered").

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
