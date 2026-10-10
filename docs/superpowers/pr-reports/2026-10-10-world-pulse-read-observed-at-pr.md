## Summary

- The reading pipeline asked the model to write its own `created_at` timestamp, and kept whatever it wrote. Models write round, invented times (e.g. `2026-10-11T02:30Z` stamped at `2026-10-10 22:36Z`).
- That invented time became `observed_at` on wp-read concept nodes in the substrate and `created_at` on reading journal rows.
- Both prompts (Stage 1 handoff, Stage 2 result) no longer ask for `created_at`; the pipeline now overwrites it with the server clock when it receives the model output. A model-supplied value is discarded and logged (`world_pulse_read_model_created_at_dropped`), not kept as a second field.
- Regression tests feed a handoff/result whose `created_at` is 4 hours in the future.

## Outcome moved

New wp-read nodes and journal rows carry the time the pipeline actually received the read. Recency/activation (`orion/substrate/activation.py:recency_score`) and curiosity staleness decay (`orion/substrate/endogenous_curiosity.py:_age_seconds`) clamp a future `observed_at` to age 0, so a future-dated node read as maximally fresh for hours; a past-dated one decayed early. Both stop for new rows.

## Current architecture

`services/orion-hub/scripts/world_pulse_read_pipeline.py:_stage1_read` parsed the model JSON and did `parsed.setdefault("created_at", now)`, so the model's value won. `_stage1_json_contract` listed `"created_at": "ISO-8601 UTC"` as a required key. Stage 2 (`world_pulse_read_stage2.py:_stage2_pass`, `_build_stage2_prompt`) had the same shape. `orion/substrate/adapters/world_pulse_read.py:25` uses `handoff.created_at` as `observed_at`; `orion/world_pulse_read/journal.py:31` uses `result.created_at`/`handoff.created_at` as the journal time.

## Architecture touched

Hub reading pipeline only (Stage 1 + Stage 2). No schema, bus, env, or compose changes.

## Files changed

- `orion/world_pulse_read/timestamps.py`: new `stamp_server_created_at` helper (pop model value, log it, write server UTC).
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`: Stage 1 prompt drops `created_at`; parse stamps server time at receipt.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`: same for Stage 2.
- `services/orion-hub/tests/test_world_pulse_read_pipeline.py`, `services/orion-hub/tests/test_world_pulse_read_stage2.py`: regression tests.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `WorldPulseReadHandoffV1.created_at` / `WorldPulseReadStage2ResultV1.created_at` are now always server receipt time (previously model-authored when present, else turn start).
- Compatibility notes: field shape unchanged; stored `handoff_json` rows keep their old values.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-hub: pytest tests -k "world_pulse_read or reading"  -> 283 passed, 65 skipped
root: pytest tests/test_world_pulse_read_{adapter,schemas,seeds,url_filters}.py -> 36 passed
root: pytest tests/test_world_pulse_read_timestamps.py -> 2 passed
Stage 1 regression test run against the pre-fix pipeline: FAILS (stored created_at 2026-10-11T03:30Z > now), passes after.
```

## Evals run

```text
services/orion-hub/evals/test_reading_handoff_eval.py -> 11 passed
```

## Docker/build/smoke checks

```text
Not deployed (per instructions). Live data check below is read-only.
```

### Live data check (read-only, 2026-10-10 ~23:05Z)

Falkor `orion_substrate` holds 251 wp-read concept nodes. Falkor stores no write time, so each node was joined (via `provenance_trace_id`) to its seed row's `world_pulse_read_seed.handoff_at` (Postgres write time of the handoff): 244 joined.

- 52 nodes have `observed_at` later than their handoff write time (worst: 13 days ahead).
- 114 nodes have `observed_at` more than 1 hour before the write time (model wrote stale times too).
- 4 nodes are still in the future relative to now (max `2026-10-11T02:30Z`).
- Postgres side: of 66 handoffs with concepts, 12 have `created_at` > `handoff_at`, 29 more than 1h earlier.

Repair verdict: worth a small, optional one-off (set `observed_at` = `handoff_at` for joined wp-read nodes), not urgent. Future-dated nodes self-correct once the clock passes them (4 left, all within ~3.5h); the cost is mis-weighted recency on ~166 old nodes. Journal rows from these reads carry the same model times. No rows were rewritten.

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`. No blocking defects.

- Finding: `_as_stage2_result` still did `setdefault("created_at", now)`, harmless today (production passes an already-validated model) but a leak if a future caller passed raw model JSON.
  - Fix: it now calls `stamp_server_created_at` too.
  - Evidence: hub reading suite 283 passed.
- Finding: old data stays poisoned; the fix only protects new writes.
  - Fix: documented in Risks and the live check. Checked the one path that re-maps stored handoffs (`orion/world_pulse_read/assertions.py:118`, Stage 2): it only resolves node ids, never writes nodes, so backlog handoffs cannot create new future-dated nodes. Repair left as an optional follow-up (no rows rewritten, per instructions).
- Finding (nit): journal assertion only checked `<= after`.
  - Fix: now `before <= created_at <= after`.
- Finding (nit): `now=` parameter untested.
  - Fix: `tests/test_world_pulse_read_timestamps.py` uses it to pin replace-and-log and stamp-without-log.
  - Evidence: 2 passed.

## Restart required

```bash
# after merge, from the primary checkout on main:
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium
- Concern: new writes only. 52 existing wp-read nodes (and their journal rows) keep model-written future times, 114 keep stale past times. Recency/activation mis-weights them until repaired or aged out.
- Mitigation: optional one-off repair setting `observed_at` from `world_pulse_read_seed.handoff_at` (join on trace id), under the section 14 backfill protocol. Not done here.

- Severity: low
- Concern: `created_at` moves from turn start to receipt (a few minutes later). Nothing found that orders on turn start.
- Mitigation: queue ordering uses Postgres `created_at`/`handoff_at` columns, not this field.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
