# feat(transport): per-hop EWMA baseline gate (log-only) + kill the metacog self-loop

## Summary

- Orion's transport health now learns each connection's own normal speed. The unit is one hop: a bus channel, a verb, or an HTTP route. It reports when a hop is suddenly slow, has been slow for a long time, or has settled into a new normal. It also reports when a hop is timing out or has stopped answering. Each of these is an episode with three kinds of row: one when it starts, one each time it gets twice as bad, and one when it ends. It is no longer one row per 30-second window.
- The baseline cannot quietly learn "busy" as normal. Windows that look like incidents never teach it. Slow drift is measured against a floor that barely rises and does not rise at all while the hop is saturated. That floor only moves after a `regime_shift` row has said the normal changed.
- The old "p95 above 5 seconds" branch is removed outright, along with its env key. The slow call it kept catching was metacog's own background LLM write-up, so each write-up triggered the next one: about 2,000 junk rows a day.
- The new gate ships **log-only** (`EQUILIBRIUM_TRANSPORT_BASELINE_ENABLE=true`, `EMIT=false`). Metacog's own dispatch (`log_orion_metacognition`) is measured but can never trigger.
- Pure reducer `orion/metacog/transport_baseline.py`. Service glue in `services/orion-equilibrium-service/app/transport_baseline_gate.py`. State is kept in Redis with a config fingerprint.

## Outcome moved

- Removes the transport self-loop, the single largest source of `orion_metacog` rows (about 1,900 a day from `cortex-orch:success_latency_ms_p95`).
- In a synthetic 24-hour mesh replay with six hops and five kinds of incident, the gate would publish **11 triggers**. The legacy pooled-p95 rule fired on 8,640 windows over the same data. Every incident produced exactly its expected episode, and calm hops produced none.
- Calm hops return to rest: `z` median between -0.05 and 0.11, `saturation_ratio` median between 1.01 and 1.05.

## Current architecture

Equilibrium consumed `orion:rpc_health:snapshot` with two gate conditions. The first fired on `timeout_count > 0`. The second fired on `success_latency_ms_p95 >= 5000`. That p95 is pooled across every channel of a service, and a window holds about 2 calls, so in practice it measured "whichever LLM call happened". It was also cortex-orch measuring metacog's own draft call.

## Architecture touched

- New pure reducer: `orion/metacog/transport_baseline.py`.
- Equilibrium: a new gate module, `_handle_rpc_health_snapshot` in `service.py`, and a `bypass_cooldown` option on `_publish_metacog_trigger`.
- Input: `channel_latency` from the snapshot payload, read as a plain dict. It is being added in a parallel patch (A0). Until that lands, every snapshot is skipped, logged as `transport_baseline_skip reason=no_channel_latency`.
- Output: `MetacogTriggerV1(trigger_kind="transport")`, with upstream `evidence_source="transport_baseline"`, per the agreed contract.

## Files changed

- `orion/metacog/transport_baseline.py`: the reducer (A1–A4).
- `orion/metacog/tests/test_transport_baseline.py`: 34 tests.
- `services/orion-equilibrium-service/app/transport_baseline_gate.py`:
  - builds the reducer config from settings
  - saves and loads its state
  - maps reducer events to triggers
  - enforces an hourly publish budget
  - writes structured log lines
- `services/orion-equilibrium-service/app/service.py`: snapshot handler, state load at boot and save after each window, cooldown bypass.
- `services/orion-equilibrium-service/app/transport_metacog_gate.py`: latency branch removed; the timeout branch stays.
- `services/orion-equilibrium-service/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: keys added and removed.
- `services/orion-equilibrium-service/tests/test_transport_baseline_gate.py` (new) and `tests/test_transport_metacog_gate.py` (latency tests turned into a self-loop regression test).
- `services/orion-equilibrium-service/evals/run_transport_baseline_mesh_eval.py` and `evals/test_transport_baseline_mesh_eval.py`: the 24-hour mesh replay.

## Schema / bus / API changes

- Added: none to schemas or channels. The transport `upstream` dict gains these fields; `upstream` is free-form, so this is not a schema change:
  - `evidence_source="transport_baseline"`
  - `condition`, `phase`, `service`, `instance`, `key`
  - `z`, `saturation_ratio`, `baseline_ms`, `floor_ms`, `window_mean_ms`
  - `calls_per_min`, `calls_per_min_usual`
  - `duration_s`, `peak_ms`, `timeout_count`
- Removed: the legacy rpc_health trigger no longer carries `latency_p95_threshold_ms`, and no longer fires on latency.
- Behavior changed:
  - **Metric-definition change.** Transport latency is now judged per hop, against that hop's own learned baseline, not against a flat 5-second pooled p95.
  - `upstream.instance` carries the snapshot's `instance`, or falls back to `node` when that is missing.
  - When `EMIT` is effective, the legacy snapshot timeout branch is not called. The `rpc_timeout` grammar source and the `bus_synaptic` source are unchanged.
- Compatibility: snapshots from old producers, without `channel_latency`, are skipped and logged, never guessed at.

## Env/config changes

- Added keys:
  - `EQUILIBRIUM_TRANSPORT_BASELINE_ENABLE=true`
  - `EQUILIBRIUM_TRANSPORT_BASELINE_EMIT=false`
  - `EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS=log_orion_metacognition`
  - `EQUILIBRIUM_TRANSPORT_BASELINE_STATE_KEY=equilibrium:transport_baseline_state:v1`
  - `EQUILIBRIUM_TRANSPORT_BASELINE_MIN_CALLS=5`
  - `EQUILIBRIUM_TRANSPORT_BASELINE_N_WARM=10`
  - `EQUILIBRIUM_TRANSPORT_BASELINE_SPIKE_Z=3.0`
  - `EQUILIBRIUM_TRANSPORT_BASELINE_SATURATION_RATIO=2.0`
  - `EQUILIBRIUM_TRANSPORT_BASELINE_REGIME_AFTER_SEC=21600`
  - `EQUILIBRIUM_TRANSPORT_BASELINE_MAX_TRIGGERS_PER_HOUR=30`
- Removed keys: `EQUILIBRIUM_METACOG_TRANSPORT_LATENCY_P95_THRESHOLD_MS`.
- `.env_example` updated: yes. Compose and README too.
- Local `.env` synced: yes, with `python scripts/sync_local_env_from_example.py --all-keys orion-equilibrium-service`. The default mode skips `EQUILIBRIUM_*`, because that prefix is not in `SYNC_PREFIXES`.
  - Two older template keys that were also missing were added: `CHANNEL_EQUILIBRIUM_TRANSITION` and `EQUILIBRIUM_TRANSITION_PUBLISH_ENABLE`.
  - **The removed key was still in the local `.env`** (line 99, `=5000`). I deleted it by hand; a backup is at `/tmp/claude-1000/equilibrium.env.bak`.
- Skipped keys requiring operator action: none.

## Metric quality gate (A1/A2 per-hop log-latency EWMA)

1. **Provenance.** `orion/core/bus/rpc_health.py` records the elapsed time of each `rpc_request`, and its hop-keyed `channel_latency` sums are being added by the parallel A0 patch. This consumer folds `log_ms_sum / success_count`, the geometric mean of the window.
2. **Independence.**
   - Timeouts are the censored tail of the same sensor. `zero_success` now subsumes `timeout` for a single failure, so one outage produces one row.
   - Load is evidence only and never fires (queueing makes it causally upstream of latency).
   - `bus_synaptic` is a different sensor and stays separate.
   - `fast` (guarded, used for z) and `level`/`floor` (used for the ratio) come from the same series. They are one condition family, not independent signals.
3. **Theory anchor.**
   - EWMA control chart on log service times (Roberts 1959; Lucas & Saccucci 1990).
   - Phase I/II separation, so the baseline never learns from out-of-control windows.
   - An asymmetric reference floor as a slow-drift detector.
4. **Live-data sanity.** UNVERIFIED on live data. A live sample on 2026-09-24 saw 6 snapshots in 35 seconds from cortex-exec and cortex-orch; **0 carried `channel_latency`**. The A0 producer is not deployed.
   - What the synthetic replay verified about rest state:
     - calm hops rest at `z` about 0 and `ratio` about 1.0
     - there is no `mean(|z|)` aggregate, so there is no `sqrt(2/pi)` floor
   - Two rest-state defects were **found and fixed** by this check:
     - A floor that followed raw window means tracked the lower edge of the noise, so calm hops read a ratio of 1.22–1.41. The floor now follows the smoothed level.
     - One unlucky first window set a floor that the slow-up half-life then kept, so one hop was stuck at 1.31. Warm-up now uses a running mean, and the floor equals the level until warm.
5. **Existing mechanism.** Reuses:
   - `orion/bus/ewma.py::compute_ewma_update`, with `min_variance=0.01` on the log-ms scale
   - the Redis state pattern from `repair_pressure_trend`
   - the fingerprint closes the caveat `trend_reducer` disclosed
6. **Reversibility.** Everything is behind flags, and state is one Redis key. Rollback: set `ENABLE=false` and delete `equilibrium:transport_baseline_state:v1`. Nothing reaches training defaults.

**Deviations from the spec, and why:**
- **Saturation uses an unguarded `level`, not `fast`.** Guard 1 freezes `fast` during a step change, so `exp(fast - floor)` would never see the step.
- **The floor follows `level`, with time-based half-lives: 3 days up, 90 seconds down.** The spec's `alpha_up = 0.002` per window is a half-life of about 2.9 hours, not the 3 days it states. At that rate the floor would absorb a 2.5x plateau before the 6-hour regime check.
- **A spike held hot for `T_regime` also states a `regime_shift`.** This is review finding 1: a 1.5x step reaches z of 3 or more but never reaches a 2x ratio.
- **Episodes close after 15 minutes quiet.**

## Tests run

```text
pytest services/orion-equilibrium-service/tests services/orion-equilibrium-service/evals/test_transport_baseline_mesh_eval.py orion/metacog/tests -q
266 passed
```

Mutation checks: each guard's test fails when its guard is removed. That covers guard 1, the clip, the sustain rule, the ratio source, floor-follows-level, floor warm-up, the floor freeze, the re-seed, and review fixes 1, 3, 4, 5, 6, 8 and 10. Fix 2 is pinned through the aging path. The eviction test originally passed for the wrong reason; it was fixed and re-verified.

## Evals run

```text
python services/orion-equilibrium-service/evals/run_transport_baseline_mesh_eval.py -> passed=true
triggers 11 vs legacy pooled-p95 window fires 8640
self-loop never triggers; calm hops never trigger; chat spike open+close;
chat timeout 1 episode; creep -> saturation + exactly 1 regime_shift, no close;
state outage -> 1 zero_success episode; quiet-hours z/ratio at rest; nothing left open
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-equilibrium-service build -> Image Built
docker run --rm (import check) -> fingerprint d62168825fc3834f, budget 30/h, close_quiet_s 900
No up/deploy performed.
```

## Review findings fixed

The code review ran in a subagent against `origin/main...HEAD`. There were 11 findings, and all 11 are fixed.

1. **Blocker.** A 1.5x step left a spike open forever, and the floor crept up with nothing announced.
   - Fix: a spike held hot for `T_regime` states one `regime_shift` and re-seeds the baseline.
   - Evidence: `test_moderate_step_below_saturation_ratio_still_becomes_regime_shift`.
2. A regime shift could be declared across hours of silence.
   - Fix: regime is judged on hot time, and latency episodes age out when no evidence arrives.
   - Evidence: `test_silence_does_not_count_toward_regime_shift_and_closes_saturation`.
3. A duplicate snapshot was folded twice.
   - Fix: a window with `window_end <=` the last folded window is ignored.
   - Evidence: `test_duplicate_snapshot_is_not_folded_twice`.
4. A timeout-only window inside a sparse pool bypassed guard 1.
   - Fix: that window now taints the pool.
   - Evidence: `test_timeout_only_window_taints_an_in_progress_pool`.
5. Two processes of one service collapsed onto one key.
   - Fix: identity falls back to `node` when `instance` is missing.
   - Evidence: `test_node_is_the_identity_fallback_when_instance_is_missing`.
6. One outage produced two rows, and nothing capped the burst.
   - Fix: `zero_success` subsumes `timeout`; `close_quiet_s` is now 900; there is an hourly budget.
   - Evidence: `test_outage_is_one_zero_success_row_not_two`, `test_mesh_wide_outage_is_capped_by_budget`.
7. Corrupt saved state could crash the service at boot, or wedge the gate.
   - Fix: every field is type-checked at load and loading never raises; a fold exception cold-starts the gate, and the legacy branch still runs.
   - Evidence: `test_corrupt_state_values_cold_start_instead_of_crashing`, `test_fold_exception_cold_starts_and_keeps_legacy_branch`.
8. Eviction dropped open episodes with no close row.
   - Fix: eviction now emits close rows.
   - Evidence: `test_eviction_closes_open_episodes`.
9. Skipped snapshots were invisible, so "no data" looked like "calm".
   - Fix: skips are logged with a rate limit.
   - Evidence: `test_skipped_snapshots_are_logged_rate_limited`.
10. The spike sustain count bridged long gaps.
    - Fix: the count resets after a gap.
    - Evidence: `test_spike_sustain_does_not_bridge_a_long_gap`.
11. The acceptance check 7 test did not check "never nominal".
    - Fix: it now asserts the ratio stays at or above the close ratio throughout the plateau.

## Restart required

Rebuild and restart equilibrium when Juniper chooses to deploy. It stays log-only; nothing publishes until `EMIT=true`:

```bash
cd /mnt/scripts/Orion-Sapienform   # after merge + pull
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
docker logs --tail=200 orion-equilibrium-service 2>&1 | grep transport_baseline
```

Expect `transport_baseline_skip reason=no_channel_latency` until the A0 producer patch is deployed.

## Risks / concerns

- **Severity: medium.** None of this is verified live yet (UNVERIFIED). The gate is inert until the A0 producer ships `channel_latency`.
  - Mitigation: skip lines make that state visible; spec acceptance check 1 (the log-only week) gates `EMIT`.
- **Severity: medium.** The thresholds are proposals: `min_calls`, `n_warm`, `spike_z`, `R`, `T_regime`, `close_quiet_s` and the budget.
  - Mitigation: they are env-tunable, and a change cold-starts the state with a logged reason.
- **Severity: low.** A saturation episode on a sparse hop can close, and a regime's hot time can reset, if latency evidence stops for more than 15 minutes.
  - This is deliberate: silence is not evidence of a sustained regime. The cost is that very bursty hops may take longer to state a regime shift.
- **Severity: low.** The hourly budget drops triggers past 30 an hour; each drop is logged as `transport_baseline_suppressed`. A dropped close leaves the metacog table with an unmatched open row.
  - Accepted as a safety cap. It only matters after `EMIT` is turned on.

## PR link

(filled on creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
