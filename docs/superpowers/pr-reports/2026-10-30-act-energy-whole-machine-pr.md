# Charge each act its share of the whole machine, above a pinned rest

## Summary

- An act's energy is now its share of the whole machine's power at the wall, above the machine's resting power. Before, the meter counted only GPU watts above each card's idle floor. On circe that saw about 40% of what work costs: fans, power-supply loss and memory were missing.
- Each host's resting power is now a fixed, dated number in `config/physiology/rest_body.yaml` (circe 541 W, athena 340 W, hecate 96 W). It does not roll. A rolling rest would call long-running work "rest": the Zelda dojo raised athena's daily floor by about 350 W. Juniper has NOT approved these values yet (`approved_at: null`). They are the defaults from the energy allowance proposal (PR #2989).
- The loader refuses a host whose `rest_w` is above its `approved_rest_w`. Raising rest hides Orion's activity, so only a reviewed change can raise it.
- A new hourly table, `orion_energy_attribution_hourly`, splits each host's energy above rest into four parts: sponsored work (the dojo), work charged to acts, work on GPU leases that have no act, and the rest. The unattributed part is never clipped. A negative value is logged as over-attribution.
- `GET /physiology/report` now shows "work not yet metered on <host>" per UTC day, with coverage. It sets the energy that is attributable to acts beside what acts were actually charged (`cost_json.wh_by_host`), and flags OVER-CHARGED when acts were charged more than the machine drew above rest.
- The GPU-only number stays in a new field, `wh_gpu_by_host`. Heat stays priced on the GPU part, because the heat fit was made against compute watts.

## Outcome moved

The per-act energy number now follows the real bill. Live circe, 2026-10-30 00:00 to 03:00 UTC, the same 58 acts:

| | before (GPU only) | after (whole machine) |
|---|---|---|
| Wh charged to acts | 5.35 | 13.93 (2.6x) |

The bigger finding is coverage. Orion used 1,748 Wh above rest on circe in those 3 hours. Acts were charged 14 Wh of it: **0.8%**. The fix does not change that much, because 1,488 Wh (85%) went to GPU leases whose turn has no act at all (the chat, fast and agent lanes). That is now a measured number in the hourly table, not a guess. The allowance proposal's baseline was "62 acts, 4.9 Wh, under 1% coverage". After the fix: 58 acts, 14 Wh, 0.8% coverage. The meter is now right per act; most of Orion's work still does not open an act.

## Current architecture

- `orion/grammar/act_energy_fill.py` read each leased card's `gpuN_watts`, subtracted the card's idle floor (p05 of its last 24 h), and split the rest among the leases on the card. That was `wh_by_host`.
- `basis.cpu` said CPU is not charged. Nothing looked at `chassis_watts`.
- No table said how much energy above rest no act was charged for.

## Architecture touched

- `orion/energy/act_energy.py`: the split. At each second, the host's chassis watts above rest are shared over everything that draws above its own floor: every card above idle, plus the CPU above its fitted idle. The act gets the part that follows its leased cards. `attribute_host_window` does the same split for a whole host and hour, and names every part.
- `orion/grammar/act_energy_fill.py`: reads the host once per act (chassis, CPU, CPU idle, every card), the pinned rest, and the day-keyed card floors. `attribute_host_hour` serves the hourly job and the eval.
- `orion/physiology/rest_body.py` + `config/physiology/rest_body.yaml`: pinned rest, approval guard, sponsored containers.
- `services/orion-sql-writer`: new loop `app/orion_energy_attribution.py`, new model `orion_energy_attribution_hourly` (created by `create_all` at boot, same as `orion_energy_hourly`; no migration). The image now contains `rest_body.yaml`.
- `services/orion-durable-runs/app/physiology_report.py`: reads the new table into `unit_costs`.

## How the split works (theory)

At each second `t` on one host:

```text
above(t) = max(0, chassis_watts(t) - rest_w)                       rest_w pinned, rest_body.yaml
load(t)  = sum over cards max(0, card_w - card_idle) + max(0, cpu_watts_total - cpu_idle_watts)
act(t)   = sum over the act's cards  (card above idle / leases on the card) x above(t) / load(t)
```

The ratio `above / load` is capped at 8 (about 3x the fitted slope). Near zero load the per-sample ratio reached 78 live, which would hand the whole surplus to a card 1 W above idle. Watts over the cap go to nobody: they stay unattributed in both paths.

Why: fans, power-supply loss, memory and voltage regulators draw in proportion to the component load that makes the heat. Power-supply loss is close to a fixed fraction of the load. Fans follow heat.

Closure is exact by construction: at each second, the shares of every lease, the CPU part and the cards nobody leased add up to `above(t)`.

The CPU floor is the row's own `cpu_idle_watts`. That is the fitted intercept (circe 100 W, athena 99 W) that the per-container CPU split already uses. So the sponsored dojo watts and the CPU part of the split are the same watts.

## Metric gate (CLAUDE.md, all six steps)

New metrics: per-act whole-machine Wh, and the hourly `rest / above / sponsored / metered / unattributed` split.

1. **Provenance.** `chassis_watts`: `orion/telemetry/biometrics_pipeline.py extract_measurements` (iLO, or the PDU outlet when there is no BMC; circe is PDU-proxied by the hub). `cpu_watts_total` (RAPL) and `cpu_idle_watts` (setting `CPU_IDLE_WATTS_BY_NODE`, fit in `scripts/analysis/fit_cpu_idle_watts.py`): `container_cpu_measurements` in the same file. `gpuN_watts`: nvidia-smi `power.draw`, same file. `container_cpu_watts.<name>`: same function.
2. **Independence.** `chassis_watts` contains `gpu_watts_total` and `cpu_watts_total` (the pipeline docstring says so). It is not used beside them as a separate signal. The cards and CPU are only the weights that divide it. The chain is short on purpose: this is a decomposition of one measured total, not a new sensor.
3. **Theory.** Linear utilisation power model (Fan, Weber & Barroso 2007) for the CPU floor. Overhead proportional to component load for the rest. Live check, circe 2026-10-16..30, 339 hourly means, rest 541 W, CPU idle 100 W: `chassis above rest = 2.69 x (GPU above idle + CPU above idle)`, R^2 0.79. The residual does not follow the fans (r = -0.05). Load and fan speed move together (r = 0.82). Daily residual is mostly within +-8%; two outliers (10-23 -23%, 10-27 +33%) are days after an all-day busy day, where the rolling card floor read high. The allowance proposal's decile table agrees: busiest vs quietest circe is about +800 W, of which GPU ~330, CPU ~35, everything else ~430.
4. **Live sanity.** Not flat, and it can read calm: in the 2026-10-29 20-21Z quiet hour circe read ~600 W against a 541 W rest, so above-rest was ~60 W, not a floor of hundreds. On a 30 s sample basis circe reads at or below 541 W part of the time (daily minimum 516 W on 10-29; the 7-day p01 was 552-563 W), so the clamp at 0 is reachable. No `chassis_watts` sample under 450 W in 41.6k (no glitch floor). Split-unknown hours are counted (`split_unknown_sec`), not hidden: every hour of 10-29 is split-unknown, because per-card watts and `cpu_idle_watts` started 2026-10-29 ~23:00Z.
5. **Existing mechanism.** `orion_energy_hourly` has per-host kWh but no rest and no split. `orion/introspect/power_attribution.py` is a correlational tool, not a split. `heat_sensitivity.yaml` carries a `baseline_watts_p10` per host, but it is a fit input, rolling over the fit window, and not approved. The proposal (PR #2989) asked for `rest_body.yaml`; it is created here.
6. **Reversibility.** Cheap. The table is new and has one writer (kill: `ORION_ENERGY_ATTRIBUTION_INTERVAL_SEC=0`). The per-act field keeps its name and gains a basis line that names the method per host; the old number is still stored as `wh_gpu_by_host`. Act energy fill has its own kill switch (`SQL_WRITER_ACT_ENERGY_COST_ENABLED=false`).

## Closure check, one real day

The first full day with a split is 2026-10-30 (per-card watts and CPU idle start 10-29 ~23:00Z, and card floors need the day before). Numbers are for 10-30 00:00-03:00 UTC, from `python3 services/orion-sql-writer/evals/act_energy_closure.py --day 2026-10-30` (read-only, live DB):

| host | above rest | sponsored | attributable to acts (`metered_wh`) | leases with no act | unattributed | coverage | acts replayed | closes |
|---|---|---|---|---|---|---|---|---|
| circe | 1,748 Wh | 0 | 13.9 Wh | 1,488 Wh | 1,734 Wh | 0.8% | 13.93 Wh (58 acts) | yes |
| athena | 1,229 Wh | 550 Wh (dojo) | 0 | 0 | 679 Wh | 0% | 0 | yes |

- `above = sponsored + charged + unattributed` holds exactly on both hosts. No hour reads negative.
- The two code paths agree: the per-act fill summed over 58 acts (13.93 Wh) vs the hourly `metered_wh` (13.87 Wh).
- athena has no pool cards, so no act can be charged there today. Its unattributed 679 Wh is the CPU work of everything else (vector-host, meta-tags, sql-writer, heartbeat).
- Full day 10-29 (no split possible yet): circe 9.69 kWh above rest, athena 9.44 kWh above rest, all split-unknown.

## Files changed

- `config/physiology/rest_body.yaml` (new): pinned rest per host, approval value, sponsored containers.
- `orion/physiology/rest_body.py` (new): loader with the `rest_w > approved_rest_w` guard.
- `orion/energy/act_energy.py`: whole-machine split (`machine_energy`), hourly split (`attribute_host_window`, with a time-ordered sweep over leases).
- `orion/grammar/act_energy_fill.py`: reads chassis/CPU/cards per host, pinned rest, day-keyed card floors; `attribute_host_hour`.
- `orion/schemas/grammar.py`: `ActCostV2.wh_gpu_by_host`; docstring for the new meaning of `wh_by_host`.
- `services/orion-sql-writer/app/orion_energy_attribution.py`, `app/models/orion_energy_attribution_hourly.py`, `app/models/__init__.py`, `app/main.py`, `app/settings.py`, `.env_example`, `Dockerfile`, `README.md`: the hourly job.
- `services/orion-durable-runs/app/physiology_report.py`, `README.md`: "work not yet metered" rows.
- `services/orion-sql-writer/evals/act_energy_closure.py` (new), `evals/act_energy_vs_power_intent.py`: closure eval; the power-intent eval now says it checks the GPU part only.
- `config/physiology/contract.yaml`: the act energy reader's reason now names `rest_body.yaml`.
- Tests: `orion/energy/tests/test_act_energy.py`, `orion/physiology/tests/test_rest_body.py`, `services/orion-sql-writer/tests/test_act_energy_fill.py`, `test_act_ledger_integration.py`, `test_orion_energy_attribution_pg.py` (new, added to the CI Postgres step), `services/orion-durable-runs/tests/test_physiology_report.py`.

## Schema / bus / API changes

- Added: `ActCostV2.wh_gpu_by_host` (optional). Table `orion_energy_attribution_hourly`. Report `unit_costs` rows "work not yet metered on <host>, <day> UTC".
- Removed: none.
- Renamed: none.
- Behavior changed: `ActCostV2.wh_by_host` is whole machine above the pinned rest (GPU-only fallback per host, named in `basis.wh`). `usd` follows it. `heat_c_by_host` is priced on `wh_gpu_by_host`. Card idle floors are now keyed to the UTC day (the 24 h before it), so every act on a card that day and the hourly job read the same floor.
- Compatibility: `wh_gpu_by_host` only lands in `grammar_acts.cost_json` (writer-side, never on the bus). Readers of `cost_json` read dicts. Acts filled before deploy keep their GPU-only numbers; the hourly table only counts the split it computes itself, so old and new acts never mix in coverage.

## Env/config changes

- Added keys: `ORION_ENERGY_ATTRIBUTION_INTERVAL_SEC=600` (orion-sql-writer, 0 disables).
- Removed / renamed keys: none.
- `.env_example` updated: yes (`services/orion-sql-writer/.env_example`; also the comment of `SQL_WRITER_ACT_ENERGY_COST_ENABLED`).
- Local `.env` synced with `python scripts/sync_local_env_from_example.py orion-sql-writer --all-keys`: yes, `+ORION_ENERGY_ATTRIBUTION_INTERVAL_SEC=600`.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/energy/tests orion/physiology/tests/test_rest_body.py services/orion-sql-writer/tests/test_act_energy_fill.py  -> passed
pytest services/orion-sql-writer/tests -q                                         -> 1158 passed, 179 skipped
pytest services/orion-durable-runs/tests orion/physiology/tests orion/energy/tests -> 935 passed, 115 skipped
pytest orion/grammar -q                                                            -> 30 passed
Postgres (throwaway postgres:16-alpine container):
  test_act_ledger_integration.py (ORION_SQL_WRITER_TEST_DATABASE_URI)             -> 22 passed
  test_orion_energy_attribution_pg.py (ORION_TEST_PG_URI)                         -> 3 passed
  test_physiology_report.py::test_report_sql_on_postgres (ORION_ADMISSION_TEST_DSN) -> passed
Every run step of .github/workflows/orion-static-gates.yml                        -> 40 OK, 0 FAIL
```

Regression test from the circe numbers: `test_regression_gpu_only_missed_about_half_of_circe_busy_hour` (rest 600 W, busy chassis 1150 W, GPU 210 -> 430 W, CPU 52 -> 125 W: the old meter charged 220 W of a 550 W rise; the act now gets 413 W, and 413 + the CPU's 137 W = 550 W).

## Evals run

```text
python3 services/orion-sql-writer/evals/act_energy_closure.py --day 2026-10-30   -> PASS (table above)
python3 services/orion-sql-writer/evals/act_energy_closure.py --day 2026-10-29   -> PASS (all split-unknown)
python3 services/orion-sql-writer/evals/act_energy_vs_power_intent.py --days 7   -> PASS, family ratio 1.19 (91 reverie images; GPU part)
```

## Docker/build/smoke checks

```text
Not built or deployed here (task says do not deploy). UNVERIFIED live: the hourly loop writing rows, and
a newly closed act carrying the whole-machine basis. The eval replays the same functions read-only
against the live DB.
```

## Review findings fixed

Code review ran in a subagent on commit 3109cc3ef.

- Finding: the hourly `metered_wh` is what is attributable to acts, not what acts were charged. It cannot go negative, so the over-attribution alarm could never fire for real double counting (for example, two acts on one correlation_id).
  - Fix: the report now reads what acts were actually charged per host per day (`ATTRIBUTION_CHARGED_SQL`) and flags OVER-CHARGED. The wording is now "attributable to acts". The model and producer docstrings say the difference.
  - Evidence: `test_attribution_says_absent_table_and_flags_over_attribution`, `test_report_sql_on_postgres`.
- Finding: the per-act fill left a silent pool card out of the load. That made the ratio larger and overcharged the act, while the hourly path marked the same seconds as unknown.
  - Fix: the fill now includes every pool card on the host, the same as the hourly path.
  - Evidence: `test_a_pool_card_that_stops_reporting_is_a_hole_not_a_smaller_load`. The integration test now reports all four circe cards.
- Finding: the per-second ratio had no upper limit (live hourly max 78).
  - Fix: `RATIO_CAP = 8` in `_split`, used by both paths.
  - Evidence: `test_near_zero_load_cannot_take_more_than_the_ratio_cap`.
- Finding: the backfilled hours before 10-29 23Z read as 100% "not metered", but the split was unknown for them.
  - Fix: the report row adds "PARTIAL: for N h ... could not be split".
- Finding: a host with no readings (hecate) redid the full 48 h backfill on every tick.
  - Fix: a cheap check for any chassis reading first, so the host is skipped.
  - Evidence: `test_a_host_with_no_chassis_reading_is_skipped_and_one_failing_host_does_not_stop_others`.
- Finding: one failing host stopped the whole tick.
  - Fix: each host now has its own try/except. Same test.
- Finding: across midnight, `wh_gpu_by_host` used the start-day floor while the machine path used the per-day floor.
  - Fix: `wh_gpu_by_host` now takes the machine path's own GPU part when that path ran.
- Finding: the cache test threw away its second connection, so it asserted nothing.
  - Fix: the test now asserts that the second act makes no floor reads. The stale docstring is fixed.
- Finding: a row with no container read was stored as 0 sponsored watts.
  - Fix: that row is now skipped (no point).
- Not fixed (nit): `machine_energy` counts the other leases on every step. That is O(steps x overlapping leases). Overlapping leases are few per act, so it is left as is.

## Restart required

Juniper, from the clean primary `main` checkout after merge (one service each):

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Then verify:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "select host, hour_start, round(above_rest_wh) above, round(sponsored_wh) sp, round(metered_wh::numeric,1) metered, round(unattributed_wh) un from orion_energy_attribution_hourly order by hour_start desc limit 6"
```

## Risks / concerns

- Severity: high. Concern: coverage is 0.8% on circe. 86% of above-rest energy sits on GPU leases whose turn has no act (chat, fast, agent lanes). Mitigation: now measured per hour (`leased_unmetered_wh`). Follow-up: open acts for those turns, or charge leases to their holder family.
- Severity: medium. Concern: the rest values are not approved by Juniper. Mitigation: the loader guard; `approved_at: null` is visible in every hourly row's `rest_source`.
- Severity: medium. Concern: card idle floors still roll (p05 of the day before). A card busy all day reads a high floor the next day, which moves the split toward the CPU and other cards (not the total). Seen in the fit's 10-27 outlier. Follow-up: pin per-card idle in `rest_body.yaml`.
- Severity: low. Concern: hecate has no `chassis_watts` in `orion_biometrics_summary`, so its acts stay GPU-only and it gets no hourly rows. The basis says so per act.
- Severity: medium. Concern: `wh_by_host` changes meaning on 2026-10-30. Family means over 7 d or 30 d mix GPU-only acts (before) and whole-machine acts (after), and read about 2-2.7x higher as the new acts arrive. Mitigation: `basis.wh` says the method; only new acts carry `wh_gpu_by_host`, so a reader can filter on it.
- Severity: low. Concern: rolling back is lossy. Older code validates stored `cost_json` with `extra="forbid"` and rejects `wh_gpu_by_host`. Then it logs `act_cost_stored_invalid` and keeps only the close's own cost for that act. Mitigation: deploy forward only.
- Severity: low. Concern: the hourly job's first run backfills 48 h (about 3 host-days of per-second work in a thread). Most of that is split-unknown (before 10-29 23Z).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2992

🤖 Generated with [Claude Code](https://claude.com/claude-code)
