# dev-economics: live ledger producer

## Summary

- Built the live `dev_economics` producer in `orion-cocreation-signals`, implementing `docs/superpowers/specs/2026-07-30-dev-economics-signal-design.md` — the last unshipped domain from the original PR #1491 four-signal backlog (structural_mass, affective_state, doc_semantic_drift already shipped and live).
- Real token/word/$-cost ledger over Juniper's local Claude Code transcript tree (same mount as `affective_state`), priced via `orion/dev_economics/pricing.py`'s explicitly versioned rate table.
- New `orion:substrate:dev_economics_ledger` bus channel — shadow-write (`consumer_services: []`), default `COCREATION_SIGNALS_DEV_ECONOMICS_ENABLED=false`.
- Code review caught a material correctness bug in the first draft and a second round verified the fix — see below.
- 21 new tests (14 producer + 7 pure aggregate), 151 total passing across the touched suites.

## Outcome moved

Codebase-mass/co-creation instrumentation now has a fourth live producer domain (alongside structural_mass, affective_state, doc_semantic_drift) — real dev-economics data flowing to the bus. Currently shadow-write only, pending a live-stream sanity pass.

## Current architecture

Before this patch: `dev_economics` had only offline library code (`orion/dev_economics/claude_code_ingest.py`, `pricing.py`) and a replay script (`scripts/replay_dev_economics_ledger.py`). No live producer, no schema, no channel.

## Architecture touched

- `orion-cocreation-signals` service: new producer loop, settings, main.py wiring, docker-compose/.env_example.
- `orion/bus/channels.yaml` + `orion/schemas/registry.py`: new contract.
- `orion/dev_economics/` and `orion/schemas/`: new pure-logic module + schema.

## Files changed

- `orion/dev_economics/ledger_aggregate.py` (new): `aggregate_session_records` (whole-history aggregation, used by the existing replay script's use case — unchanged in behavior), `SessionUsageDelta`, `diff_session_record`, `has_real_delta`, `aggregate_session_deltas` (the live producer's real code path — real incremental growth since a session was last observed, not window-membership filtering).
- `orion/dev_economics/tests/test_ledger_aggregate.py` (new): 14 tests including an explicit regression test for the multi-tick undercounting bug (`test_diff_captures_real_growth_across_a_multi_tick_session`).
- `orion/schemas/dev_economics.py` (new): `DevEconomicsLedgerV1` — `total_estimated_cost_usd: float | None` (never a fabricated $0.00), `unpriced_session_count` discloses exactly how much of a tick's real activity a partial cost total excludes.
- `orion/bus/channels.yaml`: new `orion:substrate:dev_economics_ledger` entry, shadow-write, `stability: experimental`.
- `orion/schemas/registry.py`: registered `DevEconomicsLedgerV1`.
- `services/orion-cocreation-signals/app/producers/dev_economics.py` (new): `_scan_totals` (full real transcript-tree scan, keyed by `session_id`), `_score_tick` (diffs current scan against an in-process baseline), `dev_economics_loop` (cold-start-scan pattern — tick 1 seeds the baseline without publishing, matching `git_delta_loop`'s own convention; every subsequent tick publishes the real delta, including a real zero-delta "checked, no growth" tick).
- `services/orion-cocreation-signals/app/settings.py`: `COCREATION_SIGNALS_DEV_ECONOMICS_ENABLED`, `_POLL_INTERVAL_SEC`, `CHANNEL_DEV_ECONOMICS_LEDGER`.
- `services/orion-cocreation-signals/app/main.py`: wired `dev_economics_loop` into `run_producers()`.
- `services/orion-cocreation-signals/docker-compose.yml`, `.env_example`: new env var passthrough.
- `services/orion-cocreation-signals/tests/test_dev_economics_producer.py` (new): 14 tests — cold start (no publish), real-delta publish, zero-delta publish, failed-publish non-advancement, missing-mount fail-loud, tick-exception survives, real end-to-end scoring against a real transcript file.

## Schema / bus / API changes

- Added: `orion:substrate:dev_economics_ledger` channel, `DevEconomicsLedgerV1` schema.
- Removed: none.
- Renamed: none.
- Behavior changed: none (new signal only).
- Compatibility notes: none — no dependency on other services beyond the existing `claude_projects` mount `affective_state` already uses.

## Env/config changes

- Added keys: `COCREATION_SIGNALS_DEV_ECONOMICS_ENABLED` (default `false`), `COCREATION_SIGNALS_DEV_ECONOMICS_POLL_INTERVAL_SEC` (900.0), `CHANNEL_DEV_ECONOMICS_LEDGER`.
- Removed keys: `COCREATION_SIGNALS_DEV_ECONOMICS_COLD_START_LOOKBACK_SEC` — added in the first draft (mirroring `affective_state`'s window-based restart-loss reasoning), then removed once the design moved to delta-tracking, where that reasoning no longer applies (a delta-tracking cold start just re-seeds the baseline, same as `git_delta_loop`'s own restart story — no separate lookback knob needed). Never shipped to main; only ever existed within this branch's own commits.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: yes — manually appended/corrected in both the primary checkout's and this worktree's `services/orion-cocreation-signals/.env`.
- skipped keys requiring operator action: none.

## Tests run

```text
.venv/bin/python -m pytest services/orion-cocreation-signals/tests/ orion/dev_economics/tests/ orion/structural_mass/tests/ -q
151 passed, 15 warnings in 5.97s
```

## Evals run

No dedicated eval harness for this producer yet. The pure-logic aggregate module has direct unit coverage against the same pricing table `scripts/replay_dev_economics_ledger.py`'s offline replay already validated against real transcript history.

## Docker/build/smoke checks

Not deployed live this cycle — `COCREATION_SIGNALS_DEV_ECONOMICS_ENABLED` ships `false` by default (same "shadow write, flip on deliberately" convention as every other producer in this program), so no live Docker smoke was required before merge. Import/wiring-checked directly:

```text
PYTHONPATH=. .venv/bin/python -c "from app.producers.dev_economics import dev_economics_loop; from app import main; from app.settings import Settings; Settings()" -> ok
```

## Review findings fixed

- Finding (round 1, material/correctness): `_score_window` filtered `SessionUsageRecord`s by `started_at` falling inside a poll window (mirroring `affective_state_loop`'s per-message-event pattern). But `SessionUsageRecord` is a cumulative snapshot of a whole transcript file re-parsed every tick, not a discrete event — so any session outliving one poll tick (the normal case at a 900s cadence) got its `started_at` captured once, on the tick that first observed it, and then was permanently excluded from every later tick even as the file kept growing. Real ongoing spend/token usage for the dominant case (multi-tick sessions) was silently dropped with no disclosure field.
  - Fix: replaced window-membership filtering with delta-tracking — `diff_session_record` computes real growth in a session's cumulative totals since it was last observed (in-process `last_totals` baseline dict), mirroring `git_delta_loop`'s own cold-start-diff shape instead of `affective_state_loop`'s per-message-window shape. Cold start (tick 1) seeds the baseline without publishing; every later tick publishes the real incremental delta. Removed the now-meaningless `COLD_START_LOOKBACK_SEC` setting since the restart-loss reasoning it existed for no longer applies to a diff-based design.
  - Evidence: new regression test `test_diff_captures_real_growth_across_a_multi_tick_session` (reproduces the exact bug scenario — a session growing across two ticks — and confirms the second tick's real growth is reported, not dropped or double-counted) and `test_real_growth_publishes_the_delta_not_the_cumulative_total` at the producer level. A second review pass (round 2) independently traced a multi-tick + restart + appear-then-vanish scenario by hand against the fixed code and confirmed the fix correct end-to-end ("FIX VERIFIED CORRECT"), re-ran all tests (151 passed), and confirmed zero dangling references to the removed setting across the whole worktree.

## Restart required

```text
No restart required — COCREATION_SIGNALS_DEV_ECONOMICS_ENABLED ships false; no live container is running this code path yet.
```

## Risks / concerns

- Severity: low
- Concern: `session_id` is assumed stable per real transcript file; a file being rotated/reused under the same `session_id` for a genuinely different session (not observed in the real corpus, purely theoretical) would cause its growth to be under-attributed (clamped by the `max(0, ...)` guard) rather than mis-attributed — a conservative failure mode noted by the round-2 review, not a blocking issue.
- Mitigation: same "never fabricate, prefer undercounting to corruption" principle already applied elsewhere in this signal; revisit only if real corpus evidence ever shows this happening.

- Severity: low
- Concern: in-process `last_totals` state is lost on a container restart (same restart-loss shape as `git_delta_loop`'s own accepted gap) — activity between the last successful tick and a restart is folded into the next cold-start baseline rather than reported as a delta.
- Mitigation: disclosed explicitly in the producer's own module docstring; shipped default-off pending a live sanity pass, same convention as every other signal in this program.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/dev-economics-producer
