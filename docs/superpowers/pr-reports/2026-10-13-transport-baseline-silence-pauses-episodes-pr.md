## Summary

- When a hop goes quiet (nights, or the long gaps between a sparse hop's latency checks), an open slowness episode is now **paused** instead of closed. Its accumulated "hot time" is kept, the floor stays frozen, and only a real calm reading can close it. Before this, quiet aged every episode out after 15 minutes, so the gate never reached its 6-hour "the normal has moved" statement (`regime_shift`) and the floor crept up between episodes: drift absorbed, never stated (spec guard 3 broken live).
- Hot and calm time are now measured from the hop's own traffic (gaps between calls longer than 15 minutes are silence and count as neither), not from the gap between latency checks. Without this second change the pause alone would not have helped: live, the background lane was checked only every 25-47 minutes.
- An episode with no latency check for 24 hours closes as `went_silent` — explicitly not a recovery. Every close now says why (`quiet` / `calm` / `went_silent` / `evicted`) in the trigger, in the metacog evidence text, and in the durable hourly row (`conditions_closed`).
- The learned state migrates in place (checked against the live Redis checkpoint: 60 hops kept), instead of cold-starting on the config change.
- Grader: a floor rise is a FAIL only if it is also material in absolute ms (the gate's own 250 ms line); went_silent closes are noted. New CI workflow: no workflow ran any transport baseline test before this.

## Outcome moved

Replayed on the real stored hourly rows for the two failing hops (2026-09-30 → 2026-10-13):

| | live (stored) | pre-fix reducer, replayed | this branch, replayed |
|---|---|---|---|
| saturation opens since 10-08 18:00 | 44 per hop | 37-55 per hop | **1 per hop** |
| regime_shift stated | 0 | 0 | **1 per hop, 2026-10-09 08:15-13:00 UTC** |
| floor 10-08 20:00 → 10-13 15:00 | 1.26 s → 3.99 s | 1.53 s → 4.2-4.4 s | 1.53 s → frozen, re-seeded to ~6.6-7.5 s at the shift |
| grader on the replayed rows | FAIL | — | **PASS** (both hops, 13.5 days and last 4 days) |

The pre-fix column is the validation that the reconstruction is faithful: the old reducer, fed the reconstruction, reproduces the live failure (same order of open/close pairs, zero shifts, same floor creep).

What the regime_shift says, read honestly (thanks to the parallel investigation): this was **not a real slowdown**. On 2026-10-08 the cabinet overheated, the GPU pool shed, and metacog drafts fell back to the canned template and "succeeded" in 0.3-1 s. The floor follows improvements with a 90 s half-life, so it learned ~1.3 s. Real calls came back at 6-9 s (faster than the ~14-17 s before 10-08). The fixed gate states "normal for this hop moved from 1.55 s to ~9.4 s" once, ~7-12 hours of observed-hot time after the episode opened, and re-seeds. The old gate never said anything and spent five days creeping back.

## Current architecture

Pure reducer `orion/metacog/transport_baseline.py` folds each `RpcHealthSnapshotV1.channel_latency` per (service, instance, hop): guarded `fast` EWMA (z), unguarded `level`, asymmetric `floor` (90 s down / 3 day up, frozen upward while saturated), episodes open → escalate → close. `regime_shift` fires when a spike/saturation episode's `hot_s` reaches 6 h. Wiring: `services/orion-equilibrium-service/app/transport_baseline_gate.py` (triggers, live with `EMIT=true`), `transport_baseline_hourly.py` (durable hourly rows → sql-writer → `transport_baseline_hourly`), grader `scripts/analysis/grade_transport_baseline.py`.

Before: latency episodes aged out after `close_quiet_s` (900 s) with no evaluated window; `hot_s` only counted gaps ≤ 900 s between evaluations.

## Architecture touched

- Reducer: pause semantics, traffic-time crediting (`KeyState.active_s`), `max_silence_s` (24 h), floor freeze while open/paused/about-to-open, `close_reason`, state schema v3 with in-place v2 migration.
- Equilibrium: trigger upstream `close_reason`, migration log line, hourly `conditions_closed`.
- Contract: `TransportBaselineHourlyV1.conditions_closed` (additive). sql-writer column + boot-time `ADD COLUMN IF NOT EXISTS`.
- `orion/metacog/evidence_map.py`: a went_silent close reads "closed because the hop went silent, not because it recovered".
- Semantic layer: `transport_baseline_hourly.v1` registered (`orion/inner_state_registry.py`) with rest/absent semantics; metric lock re-locked (8 added).
- CI: `.github/workflows/transport-baseline-tests.yml`.

## Files changed

- `orion/metacog/transport_baseline.py`: the fix (pause, traffic-time credit, went_silent, floor freeze, close_reason, v3 migration).
- `orion/metacog/tests/test_transport_baseline.py`: 17 new tests; the old "silence closes saturation" test now asserts the pause.
- `orion/metacog/evals/run_transport_baseline_pause_mutations.py`: mutation eval, 8 mutants.
- `services/orion-equilibrium-service/evals/run_transport_baseline_sparse_hop_replay.py` + `fixtures/transport_baseline_hourly_sparse_hops_2026-10-13.jsonl` (668 real hourly rows, trimmed): replay eval; wired into `evals/test_transport_baseline_mesh_eval.py`.
- `services/orion-equilibrium-service/app/transport_baseline_gate.py`, `app/transport_baseline_hourly.py`, their tests, `README.md`.
- `orion/schemas/telemetry/transport_baseline_hourly.py`: `conditions_closed`.
- `services/orion-sql-writer/app/models/transport_baseline_hourly.py`, `app/main.py`, `tests/test_transport_baseline_hourly_sql_shape.py`.
- `orion/metacog/evidence_map.py`, `orion/metacog/tests/test_evidence_map.py`.
- `scripts/analysis/grade_transport_baseline.py` + tests.
- `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`.
- `.github/workflows/transport-baseline-tests.yml`.
- `docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md`: addendum under guard 3.

## Schema / bus / API changes

- Added: `TransportBaselineHourlyV1.conditions_closed: Dict[str, int]` (`"condition:close_reason" -> n`, default `{}`); trigger `upstream.close_reason`; trigger reason suffix `:closed=<reason>` on close rows.
- Removed / renamed: none.
- Behavior changed: latency episodes pause on silence instead of closing; closes carry a reason; reducer state schema 2 → 3 (migrated in place).
- Compatibility: `TransportBaselineHourlyV1` is `extra="forbid"` and sql-writer validates with it, so **deploy orion-sql-writer before orion-equilibrium-service** (its boot adds the column). Equilibrium first would have the old sql-writer reject every hourly row until it is rebuilt. No new bus channel.

## Env/config changes

- Added / removed / renamed keys: none (`max_silence_s` is a reducer constant beside `close_quiet_s`, not an env key).
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys: none.

## Metric quality gate (saturation_ratio / floor, per CLAUDE.md)

1. Provenance: `fold_snapshot` in `orion/metacog/transport_baseline.py` (ratio = `exp(level - floor)`); hourly p50 in `app/transport_baseline_hourly.py:_row`.
2. Independence: not a new metric; same producer, semantics fixed.
3. Theory anchor: spec guard 3 ("drift is stated, never absorbed"); "absence of traffic is not evidence of calm".
4. Live data: pulled `transport_baseline_hourly` (21,329 rows, 09-30 → 10-13) and 42 h of per-window `transport_baseline_obs`/`event` logs. Rest: calm hops read ratio ~1.0-1.2 (grader PASS on 19 hops). The failing hops were not degenerate but *structurally unable to state a shift*: 44 open/close pairs each, every close with `ratio=None` (a silence close, confirmed in the event logs).
5. Existing mechanism: semantic layer had no entry (`check_metric_lineage.py --metric saturation_ratio` → UNREGISTERED; same for `floor_ms`, `transport_baseline`); glossary/registry grep empty. Now registered.
6. Reversibility: constants and a state-schema bump; the v3 → v2 direction would cold-start once.

## Silence-gap distribution (what sets max_silence_s)

From `transport_baseline_hourly`, runs of whole hours with no evaluated window between evaluated hours (true gap is up to ~2 h longer):

- cortex-orch → `orion:cortex:exec:request:background`: 56 runs, p50 2 h, p75 4 h, p90 7 h, p95 8 h, p99 13 h, max 21 h.
- actions → `orion:cortex:request`: 57 runs, p50 2 h, p75 4 h, p90 7 h, p95 9 h, p99 11 h, max 13 h.
- all hops (933 runs): p50 2 h, p75 5 h, p90 14 h, p95 23 h, p99 61 h, max 215 h.
- Per-window logs (42 h): the background lane's evaluations were 20 min - 22 h apart (median 47 min), while it carried calls every few minutes.

24 h bridges every silence observed on the two hops and 95% across all hops, and still ends an episode on a hop that stopped for good within a day.

## Tests run

```text
orion/metacog/tests/test_transport_baseline.py + scripts/analysis/tests/test_grade_transport_baseline.py   81 passed
services/orion-equilibrium-service tests + evals                                                         190 passed
orion/metacog (all)                                                                                      231 passed
services/orion-sql-writer tests/test_transport_baseline_hourly_sql_shape.py                                6 passed
services/orion-sql-writer tests (all): 14 failed / 736 passed -- the same 14 fail on origin/main (diffed on a detached origin/main worktree)
python scripts/check_metric_lineage.py --gate             PASS
python scripts/check_metric_lineage.py --prompt-semantics PASS
python scripts/check_definition_drift.py --gate           PASS (after --update; 8 added)
python scripts/check_inner_state_registry.py              OK (22 entries)
tests/test_metric_prompt_semantics.py + inner_state_registry tests  51 passed
```

## Evals run

```text
PYTHONPATH=. python orion/metacog/evals/run_transport_baseline_pause_mutations.py
KILLED  silence closes latency episodes again (pre-fix aging)
KILLED  silence between calls credited as active time
KILLED  pre-fix hot credit (gap between evaluations <= close_quiet_s)
KILLED  silence resets hot_s
KILLED  silent gap credited as calm time
KILLED  floor not frozen while saturation is open or paused
KILLED  opening window may creep the floor
KILLED  max-silence bound removed
8/8 mutants killed   (the first run had "silent gap credited as calm" SURVIVE and "floor not frozen" SURVIVE; tests were strengthened until both died)

python services/orion-equilibrium-service/evals/run_transport_baseline_sparse_hop_replay.py
background lane, even bursts:  saturation open 1, regime_shift 2026-10-09 08:15 UTC (floor 1550 ms -> window 9421 ms, episode 7.1 h)
background lane, packed bursts: saturation open 1, regime_shift 2026-10-09 13:00 UTC (floor 1557 ms -> window 9432 ms, episode 12.0 h)
actions hop, even:   regime_shift 2026-10-09 08:15 UTC (1564 -> 9548 ms)
actions hop, packed: regime_shift 2026-10-09 13:00 UTC (1572 -> 9553 ms)
all checks pass; mesh eval passes unchanged
```

Grader on live rows (`--days 4`) still FAILs the two hops: those rows were written by the old reducer and record the absorption that happened; this branch cannot rewrite history. Grader on hourly rows produced by this branch's replay: PASS for both hops.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-equilibrium-service build   Built (temporary .env symlinks, removed after)
scripts/safe_docker_build.sh orion-sql-writer build            Built
in-image: orion.metacog.transport_baseline STATE_SCHEMA_VERSION=3, max_silence_s=86400; sql-writer model has conditions_closed
live checkpoint (read-only GET equilibrium:transport_baseline_state:v1 on the bus Redis): schema 2, fingerprint 697b47aac18690e3
  == new config's v2_fingerprint -> migrates in place, 60 hops kept, no cold start
NOT deployed.
```

## Review findings fixed

(filled in after review)

## Restart required

Deploy from the primary checkout on main after merge, sql-writer first:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-sql-writer up -d --build && scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
```

Expect `transport_baseline migrated from=schema_v2:697b47aac18690e3 keys=…` in the equilibrium log, not `cold_start`.

## Risks / concerns

- Severity: medium. Concern: the 10-08 absorption is already done. Live, both hops now read floor ~4.0 s vs level ~6.4-7.1 s (ratio ~1.6-1.8, under the 2.0 saturation line). After deploy the gate will not retroactively state that shift; the floor keeps creeping toward the level at the 3-day half-life. Mitigation: stated here; the grader's floor-rise check keeps it visible. A cold start would have silently reset it instead, which is worse.
- Severity: medium. Concern: **the root cause is not fixed here** — failed/degraded calls (canned-fallback drafts, lease refusals) are recorded as fast successes and can drag a floor down at its 90 s half-life. This branch makes the gate state and re-seed such a move within hours instead of absorbing it over days; it does not stop the poisoning. Follow-up design below.
- Severity: low. Concern: traffic-time credit trusts two agreeing verdicts across thin, un-judged traffic between them (in the replay, single credits reached ~2.4 h). Mitigation: a gap between calls > 15 min still breaks it; a calm verdict in between resets the agreement.
- Severity: low. Concern: replay is a reconstruction from hourly medians (every call in an hour reads the hour's median level; burst spacing assumed, both extremes run). Validated by the old reducer reproducing the stored floor and pair counts.

## Follow-up (not in this PR): keep failed/degraded calls out of latency

Exact design, consumer-first because `RpcChannelLatencyV1` is `extra="forbid"` and every bus client produces it:

1. Contract: `RpcChannelLatencyV1.degraded_count: int = 0` (`orion/schemas/telemetry/rpc_health.py`). Ship the equilibrium consumer first.
2. Reducer: degraded calls never enter `log_ms_sum`; a window with `degraded_count > 0` taints the pool (no `fast` fold, as timeouts do) and **the floor may not move down on it**.
3. Producer seam: `orion/core/bus/async_service.py` `rpc_request()` records success today on any reply. Add an optional reply classifier (`degraded=True` when the reply payload carries a generic marker, e.g. `ok is False` or `metadata.degraded/fallback_reason`) → `rpc_health.record_degraded()` instead of `record_success()`.
4. Producer marker: orion-cortex-exec sets that marker when a step used `_fallback_metacog_draft` (`services/orion-cortex-exec/app/executor.py:971`) or a GPU lease was refused.
5. Regression eval: replay the 10-08 shed shape (a burst of 0.3-1 s fallback "successes" on a ~14 s hop) and assert the floor does not drop.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
