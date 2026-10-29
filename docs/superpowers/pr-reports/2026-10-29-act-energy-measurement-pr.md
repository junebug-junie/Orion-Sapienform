# Measured energy, money and heat per act

## Summary

- Every act that holds a GPU now gets a measured cost when it closes: watt-hours per host, dollars, and cabinet degrees per host. Before this, only reverie images had an energy figure, and only through the power-intent loop.
- The biometrics service now reports each GPU card's own watts and memory (`gpuN_watts`, `gpuN_mem_used_mb`, `gpuN_mem_total_mb`). Before, it reported one watt total per host.
- The biometrics service now reports each docker container's CPU watts (`container_cpu_watts.<name>`). Before, CPU work was invisible per process.
- The arithmetic that subtracts a baseline from watts is shared with the power-intent loop (`incremental_joules`). It is not copied.
- A new eval compares the per-act estimate with the 1 Hz power-intent meter on the same reverie windows.

## Outcome moved

The body economy can now see what an act cost in energy, money and heat. Live before this patch: 0 of all acts in `grammar_acts` carried `wh_by_host` (body economy spec, Patch C1: "No act carries wh_by_host"). After deploy, every act whose correlation id matches a GPU pool lease carries it.

## Current architecture

- `orion-biometrics` read nvidia-smi every 30 s. It kept per-card temperature, but only a host total for watts (`gpu_watts_total`). It read CPU package watts (RAPL) as one host total (`cpu_watts_total`).
- `orion-sql-writer`'s act ledger (`orion/grammar/act_ledger.py fill_close_cost`) filled `seconds` and `gpu_slot_s_by_lane` on close. `wh_by_host` was always null.
- The power-intent loop measured reverie images at 1 Hz on one card and priced them in `energy_run_cost` (`incremental_over_baseline`).

## Architecture touched

- `orion-biometrics`: collector reads cgroup v2 `cpu.stat` per container through the host `/sys` mount it already had; names come from the docker root it already mounted. The pipeline adds per-card and per-container keys to `measurements`.
- `orion-sql-writer`: `fill_close_cost` calls a new energy fill in its own savepoint. The image now contains `config/gpu_pool.yaml` and `config/physiology/heat_sensitivity.yaml`.
- Contract: `ActCostV2` gets three optional fields.

## How the numbers are made (attribution theory)

**GPU, per act.** For each GPU pool lease of the act, for each card in the lease:
1. Read the card's watts (`gpuN_watts`) for the lease window. Interpolate them per second.
2. Subtract the card's idle floor. The idle floor is the 5th percentile of the card's own watts over the 24 hours before the lease. Only watts above idle are charged, so the always-on floor is not charged to Orion's choices.
3. At each second, split the above-idle watts equally among all leases on the card at that second (`max_holds: 2` cards are shared). The shares of all leases on a card add up to the card's above-idle energy, so no watt is charged twice.

**Money.** Wh / 1000 x the house tariff's marginal $/kWh (latest `energy_cost_accrued` row, at most 7 days old).

**Heat.** `config/physiology/heat_sensitivity.yaml` fits degrees C per 100 W of hourly mean compute watts. An act that draws E Wh raises its hour's mean by E W, so its heat is sensitivity x E / 100. This is the fitted step response applied linearly. It is an estimate, not a thermometer reading.

**CPU, per container.** Package watts = idle + slope x busy cores (the linear utilisation power model: Fan, Weber and Barroso 2007, "Power provisioning for a warehouse-sized computer"). Fitted on this fleet's history (`scripts/analysis/fit_cpu_idle_watts.py`, 2026-10-15..29): athena 99 W idle + 4.5 W per core (R² 0.91, n 36,881), circe 100 W + 4.8 W per core (R² 0.86, n 36,842), hecate 80 W (R² 0.80, n 343). A container's watts = (package watts - idle) x its share of busy cores. Work outside containers goes to `cpu_watts_host_other`.

**CPU is not charged per act.** An act does not say which containers did its work. So `basis.cpu` says so in words, instead of a 0.

## Persisted fields (for the physiology report agent)

In `grammar_acts.cost_json` (ActCostV2):

- `wh_by_host`: `{host: Wh above idle}`, GPU only.
- `usd`: dollars for the total Wh.
- `heat_c_by_host`: `{host: degrees C}`. A host with no usable heat fit is absent (hecate today).
- `basis`: `{"wh": ..., "usd": ..., "heat": ..., "cpu": ...}` in words: cards, samples in the window, share, idle floor, rate, fit date, or the reason a field is missing.

In `orion_biometrics_summary.measurements` (per node, every 30 s):

- `gpuN_watts`, `gpuN_mem_used_mb`, `gpuN_mem_total_mb`
- `cpu_busy_cores`, `cpu_idle_watts`, `container_cpu_watts.<container>`, `container_cpu_watts_small`, `cpu_watts_host_other`

## Metric quality gate

**`gpuN_watts`**
1. Provenance: `orion/sensors/gpu_host_stats.sh` -> nvidia-smi `power.draw` -> `orion/telemetry/biometrics_pipeline.py extract_measurements` (same row as `gpuN_temp_c`).
2. Independence: not new signal. It is a split of `gpu_watts_total` (live circe row: 40.17 + 42.26 + 45.12 + 77.18 = 204.73 = `gpu_watts_total`). Not in `FLEET_SUM_KEYS`, so it is never added on top of the total.
3. Theory: the card's own power sensor; the per-card split is what lets a lease be charged for its own card.
4. Live data: circe per-card watts over the last 24 h (raw `orion_biometrics`): gpu0 min 36.3 / p05 36.6 / median 38.4 / max 161.4; gpu1 37.4 / 39.0 / 43.7 / 261.1; gpu2 40.3 / 40.8 / 43.1 / 256.1; gpu3 68.8 / 71.7 / 107.3 / 286.2. Not flat, returns to a rest floor every day. athena nvidia-smi now: P4 24.66 W, T10 104.38 W.
5. Existing mechanism: the raw blob in `orion_biometrics.gpu` already had per-card watts, but no summary key. The power-intent loop samples one card at 1 Hz only while a reverie image runs.
6. Reversibility: a new key in a jsonb dict. Remove the three lines; nothing is baked into a schema.

**`gpuN_mem_used_mb`**: nvidia-smi `memory.used`, same row. Live circe: 27962 / 24888 / 1158 / 19483 MiB. Varies with model loads (gpu2 near empty while diffusion is unloaded).
**`gpuN_mem_total_mb`**: nvidia-smi `memory.total`, same row. Constant per card by nature (32768 MiB on circe); it is the denominator for headroom, not a signal.

**`container_cpu_watts.<name>`**
1. Provenance: cgroup v2 `cpu.stat usage_usec` per container (`services/orion-biometrics/app/metrics.py _collect_containers`) and RAPL `cpu_watts_total`.
2. Independence: a split of `cpu_watts_total` above idle. Not independent of it; it says what the total is made of.
3. Theory: linear utilisation power model (above), fitted R² 0.86-0.91.
4. Live data (circe caveat from review: below its 100 W floor, ~10% of circe samples, no split is published): run inside the live biometrics container on athena: 132 containers, 52.3 cores in containers against 55.8 busy host cores. Top: orion-athena-heartbeat 8.49 cores, orion-athena-vector-host 5.42, orion-athena-vision-host 1.15, each zelda-dojo worker 1.0. A container that does nothing reads 0 cores; athena's host never reads idle (minimum 5.8 busy cores in 14 days), so `cpu_watts_host_other` never reaches 0 there, by fact, not by artifact.
5. Existing mechanism: none. `docker stats` exists but is not persisted.
6. Reversibility: jsonb keys plus one env key. Cheap.

**`wh_by_host` per act**
1. Provenance: `orion/grammar/act_energy_fill.py act_energy_cost` <- `gpu_pool_events` leases (same join as `gpu_slot_s_by_lane`) + `gpuN_watts`.
2. Independence: derived from `gpu_slot_s_by_lane` (same leases) and `gpuN_watts`. It adds the watts dimension; it is not a new sensor.
3. Theory: energy is power integrated over time; shared-card energy is split by concurrent lease count.
4. Live data: replay over 90 reverie images (last 7 days) against the 1 Hz power-intent meter (`services/orion-sql-writer/evals/act_energy_vs_power_intent.py`): family total 35.99 Wh vs 29.94 Wh, ratio 1.20 (gate |ratio - 1| <= 0.35: PASS). Mean 0.40 Wh per image vs 0.33 Wh. The floor choice explains the whole gap: the 24 h card floor sits 5.5 W below the power-intent loop's pre-window sample (the card is already warm just before an image), which adds about 0.09 Wh per 60 s image. With the same floor as the meter, the estimate would read about 0.31 Wh against 0.33 Wh (ratio about 0.93). Per image the estimate scatters widely (median ratio 0.33, p10 0.16, p90 5.3), because one 30 s sample can land on or miss the peak. One known image (lease 163c99b3, 2026-10-29 16:40:56): 1.4 Wh from 30 s samples against 0.27 Wh from the 1 Hz meter, because one sample caught the 216 W peak. **Read per-act Wh as an estimate; read family totals as the measurement.**
5. Existing mechanism: `energy_run_cost` (power-intent) covers reverie images only, and only with a declared intent. Reused its subtraction (`incremental_joules`).
6. Reversibility: kill switch `SQL_WRITER_ACT_ENERGY_COST_ENABLED=false`; fields are optional in `cost_json`.

## Files changed

- `orion/telemetry/biometrics_pipeline.py`: per-card watts/VRAM keys; `container_cpu_measurements` (CPU watt split).
- `services/orion-biometrics/app/metrics.py`: `_collect_containers` (cgroup v2 read, container names, busy cores).
- `services/orion-biometrics/app/main.py`: pass the container block to the pipeline after validation (the sample model drops unknown keys).
- `services/orion-biometrics/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `CPU_IDLE_WATTS_BY_NODE`, `DOCKER_ROOT_PATH`.
- `orion/energy/act_energy.py`: pure arithmetic (idle floor, interpolation, lease-share integration, heat, money).
- `orion/energy/run_cost.py`: uses the shared `incremental_joules`.
- `orion/grammar/act_energy_fill.py`: reads leases, card watts, rate, heat fit; returns the cost fields.
- `orion/grammar/act_ledger.py`: calls the fill on close, in its own savepoint; kill switch.
- `orion/schemas/grammar.py`: `ActCostV2.usd`, `heat_c_by_host`, `basis`.
- `services/orion-sql-writer/Dockerfile`: copies `config/gpu_pool.yaml` and `config/physiology/heat_sensitivity.yaml`.
- `services/orion-sql-writer/app/main.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `SQL_WRITER_ACT_ENERGY_COST_ENABLED`.
- `config/physiology/contract.yaml`: registers the fill as a display reader of the heat fit.
- `scripts/analysis/fit_cpu_idle_watts.py`: the CPU idle fit.
- `scripts/sync_local_env_from_example.py`: visits orion-biometrics and syncs the two new keys by default.
- `services/orion-sql-writer/evals/act_energy_vs_power_intent.py`: the live family eval.
- Tests: `orion/energy/tests/test_act_energy.py`, `tests/test_act_energy_measurements.py`, `services/orion-biometrics/tests/test_container_cpu.py`, `services/orion-sql-writer/tests/test_act_energy_fill.py`, `services/orion-sql-writer/tests/test_act_ledger_integration.py` (new Postgres test), `tests/test_act_ledger_contract.py`.
- `.github/workflows/biometrics-loop-tests.yml`: runs the new biometrics tests.

## Schema / bus / API changes

- Added: `ActCostV2.usd`, `ActCostV2.heat_c_by_host`, `ActCostV2.basis` (all optional, default None).
- Removed: none.
- Renamed: none.
- Behavior changed: the writer fills `wh_by_host`, `usd`, `heat_c_by_host`, `basis` on close.
- Compatibility notes: consumer-first holds. The new fields are written only by the writer into `grammar_acts.cost_json`; they never cross the bus. Producers still send `ActCostV1`. `WantV1.cost` (an `ActCostV2`) is None at every producer, so the new keys never reach an older durable-runs reader (pinned by `test_a_want_with_no_cost_carries_no_new_cost_keys_on_the_wire`). `cost_json` readers (durable-runs physiology market, analysis scripts) read it as a dict. Biometrics `measurements` is `Dict[str, float]`; hardware-watch filters its own keys by `^gpu[0-9]+_temp_c$`, so `gpuN_watts` does not enter its heat rule. No bus channel or schema registry change.

## Env/config changes

- Added keys: `CPU_IDLE_WATTS_BY_NODE`, `DOCKER_ROOT_PATH` (orion-biometrics); `SQL_WRITER_ACT_ENERGY_COST_ENABLED` (orion-sql-writer).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, both services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (athena). circe and hecate get the code default when their `.env` lacks the key.
- skipped keys requiring operator action: none.

## Tests run

```text
orion/energy/tests + orion/physiology/tests                     336 passed
tests/test_act_energy_measurements.py                            12 passed
services/orion-biometrics/tests/test_container_cpu.py + rapl     27 passed
services/orion-sql-writer/tests (whole folder) + ledger/contract 1172 passed, 176 skipped
act ledger Postgres integration (throwaway postgres:16)           31 passed (incl. new energy test)
tests/test_act_ledger_contract.py                                 20 passed
orion-static-gates.yml steps (all, run locally)                  all OK after contract.yaml fix
biometrics-loop-tests.yml steps                                   OK
```

Pre-existing failures on main, not touched: `tests/test_biometrics_measurements.py::test_review_disk_and_net_rates_are_not_promoted_to_physical_quantities`, `services/orion-biometrics/tests/test_biometrics_grammar_emit.py::test_circe_node_availability_reflects_expected_offline` (neither is in CI).

## Evals run

```text
python3 services/orion-sql-writer/evals/act_energy_vs_power_intent.py --days 7 --limit 150
n=90  meter 29.94 Wh  act estimate 35.99 Wh  family ratio 1.20  -> PASS (gate 0.35)
per-run ratio median 0.33, p10 0.16, p90 5.29; idle floor 5.5 W under the pre-window baseline
python3 scripts/analysis/fit_cpu_idle_watts.py --since 2026-10-15
athena 98.8 W + 4.51 W/core R2 0.91; circe 99.7 W + 4.79 W/core R2 0.86; hecate 80.2 W R2 0.80
```

## Docker/build/smoke checks

```text
Live collector check (new _collect_containers code, run inside the running orion-athena-biometrics):
busy 55.8 cores, 132 containers, 52.3 cores in containers; names resolved from the docker root.
Live extraction check: extract_measurements on circe's latest raw row -> gpu0..3 watts/mem keys;
per-card watts sum to gpu_watts_total exactly.
```

No image was built or deployed (the parent session deploys after merge).

## Review findings fixed

- Finding: circe's CPU idle floor (100 W fit intercept) is above its real draw in ~10% of samples, so busy containers published 0 W (a fake zero; metric gate step 4 failed for circe).
  - Fix: below the floor, no container split is published (busy cores and the floor still are).
  - Evidence: `test_below_the_fitted_floor_there_is_no_split_not_a_split_of_zero`.
- Finding: per-act Wh carries a +20% family bias and wide per-act scatter, but basis read like a plain measurement.
  - Fix: `basis.wh` names "interpolated: no sample inside the lease" when a lease has no sample; the bias and its cause are documented at `IDLE_QUANTILE` and in the module docstring.
  - Evidence: `test_interpolated_window_is_named_in_basis`; eval numbers above.
- Finding: the energy tests read the committed heat fit, so they break on any refit and the integration test would go red after the fit turns 30 days old (~2026-11-28).
  - Fix: both tests use a fixture fit dated by the test clock; one separate test only checks the real file parses.
  - Evidence: `test_the_committed_heat_fit_parses`; integration test monkeypatches `_heat_config`.
- Finding: several queries per lease x card inside the close transaction.
  - Fix: one watts read and one other-leases read per card over the union window. Live read-only EXPLAIN of the card query: 176 ms.
  - Evidence: `test_one_card_is_read_once_for_many_leases`.
  - Side catch: the restructured SQL had a UNION/ORDER BY error and a NULL type mismatch. The Postgres integration test caught both; fixed.
- Finding: a grant with no terminal event shared the card until "now" (live: ~8 a day); duplicate grant rows could count twice.
  - Fix: capped at the 9000 s longest hold; `DISTINCT ON (lease_id)` on the grant branch.
  - Evidence: `test_a_grant_that_never_ends_stops_sharing_after_the_longest_hold`; integration test now includes a never-ended grant with two grant rows.
- Finding: `basis` alone made a cost row count as measured; the writer's basis text could describe a producer's value.
  - Fix: `basis` is excluded from the "measured" check; the writer drops its basis line when the producer's field wins.
  - Evidence: sql-writer suite 1172 passed.
- Finding: container busy cores used the tick's dt, which includes nvidia-smi latency jitter.
  - Fix: the cgroup read keeps its own monotonic clock.
  - Evidence: `test_rate_uses_the_cgroup_reads_own_clock_not_the_tick_dt`.
- Finding (nits, documented not changed): GPU work with no lease lands on lease holders; acts sharing a correlation id would each be charged in full (live: 1103 acts, 1103 ids); a lease's cards come from one terminal row. All three are written in the `act_energy_fill.py` docstring.
- Finding: the per-container CPU split has no per-act consumer yet. Accepted: it is the per-process measurement Juniper asked for, read by the coming physiology report; `basis.cpu` says CPU is not charged per act.

## Restart required

After merge, from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-biometrics up -d --build
```

circe and hecate run their own orion-biometrics: rebuild it on each host after it pulls (same command there).

## Risks / concerns

- Severity: medium. Concern: one act's Wh comes from ~30 s samples, so a short act can read several times high or low (live: per-image ratio p10 0.16, p90 5.3). Mitigation: `basis.wh` names the sample count; family totals track the 1 Hz meter (ratio 1.20 over 90 images).
- Severity: low. Concern: a durable-run hold and the child call it wraps are two leases on one card, so each is charged half of that card's energy. Reverie images do this (the reverie-visual hold plus the orion-thought diffusion lease). Mitigation: the total is still conserved; the split between the two acts is wrong until leases name their parent.
- Severity: low. Concern: `thought.painting` acts do not share a correlation id with their diffusion lease (live: act c133fa20 vs lease f56c8cc9), so painting acts still get no Wh. Mitigation: follow-up in the painting producer; not changed here.
- Severity: low. Concern: Wh is UNVERIFIED on live acts until after deploy plus 30 minutes of `gpuN_watts` history (the idle floor needs 60 samples). Mitigation: check `select cost_json->'wh_by_host', cost_json->'basis' from grammar_acts where closed_at > now() - interval '1 hour'` after deploy.
- Severity: low. Concern: `orion/physiology/choice.py` still lists heat and energy as `unpriced` (its docstring says wh_by_host is null). Mitigation: pricing them is the body economy's next patch, not this one.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2977

🤖 Generated with [Claude Code](https://claude.com/claude-code)
