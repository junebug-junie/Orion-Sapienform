## Summary

- Orion can now ask "what did node X's prediction error do over the last N hours?" through a new orion-introspect tool, `node_history`.
- The answer is one compact record: how many readings, min / median / max, first and last reading, up to 48 time buckets (mean/min/max each), the five biggest jumps between consecutive readings with when they happened, and the earliest reading still stored.
- Read-only, parameterized SQL, aggregated in Postgres, window capped at 168 h (the writer's prune horizon, pinned by a test). Works for any node in `substrate_node_prediction_error_history`; with no `node_id` it lists which nodes have readings.
- Discoverable where Orion looks: the tool's own description and one line in the introspect brief that every chat and curiosity turn gets: "Substrate node history exists ... call it before saying a node has no history."

## Outcome moved

The 2026-10-09 Orion's Day letter said the synaptic bus node "holds only a current value, no series", so Orion could not tell whether an elevation lined up with the dream gap. The series was in Postgres all along (about 14k rows a week for `node:substrate.bus_synaptic`); no tool Orion had could read it. Now one exists.

## Current architecture

orion-introspect is a per-turn MCP server (`orion/introspect/mcp_server.py`) whose tools RPC over the bus to the service that owns the data. Hub already answers `curiosity`, `orion_day` and `record_correction` from its memory Postgres pool, in READ ONLY transactions. `substrate_node_prediction_error_history` lives in the same `conjourney` database, written by orion-substrate-runtime.

## Architecture touched

- New bus RPC `orion:introspect:node_history:request` (governor -> Hub), reply on the existing `orion:introspect:result:*`.
- New Hub responder, same shape and trust rules as the `orion_day` one, same pool.
- No new service, no new env key, no migration.

## Files changed

- `orion/schemas/introspect.py`: `NodeHistoryArguments` (node_id pattern, hours 1 min..168 h, optional `until`; future and past-retention windows refused), `node_history` operation.
- `orion/schemas/registry.py`, `orion/schema_skew_discovery.py`: register the new arguments model.
- `orion/introspect/node_history.py`: the SQL and the summary item.
- `orion/introspect/tools.py`, `transport.py`, `brief.py`: tool spec, channel, brief line.
- `orion/bus/channels.yaml`: the request channel.
- `services/orion-hub/scripts/node_history_introspect_listener.py`, `scripts/main.py`: responder and start/stop wiring.
- `services/orion-hub/README.md`, `services/orion-harness-governor/README.md`: responder section and overview row.
- `orion/introspect/tests/test_node_history.py`, `orion/introspect/tests/node_history_fake.py`, `services/orion-hub/tests/test_node_history_introspect_listener.py`, existing tool-list tests: tests.
- `services/orion-hub/evals/run_node_history_live_eval.py`: live eval (real SQL vs raw rows).
- `.github/workflows/orion-day-letter-hub-tests.yml`: runs the Hub listener test.
- `config/metrics/metric_definitions.lock.json`: relock for the new channel.

## Schema / bus / API changes

- Added: `NodeHistoryArguments`; operation `node_history` in `IntrospectBusOperation` / `IntrospectOperation`; channel `orion:introspect:node_history:request`.
- Removed: none. Renamed: none.
- Behavior changed: introspect brief lists one more tool and one more guide line.
- Compatibility notes: additive Literal value. Until Hub is redeployed, a call times out and the tool honestly says "answer unknown".

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
PYTHONPATH=. python -m pytest -q orion/introspect/tests services/orion-hub/tests/test_node_history_introspect_listener.py services/orion-hub/tests/test_orion_day_introspect_listener.py
276 passed, 2 skipped
tests/test_single_consumer_channels_gate.py etc. + orion/harness/tests + orion/fcc/tests/test_mcp_tool_names.py: 518 passed, 1 failed
  (test_every_request_kind_single_consumer_channel_is_annotated fails identically on main: gpu_pool + reading:tool channels, not this patch)
scripts/check_metric_lineage.py --gate: PASS
scripts/check_definition_drift.py --gate: PASS (after relock)
check_bus_reply_channels / check_async_routes_not_blocking / check_scripts_dir_no_stdlib_shadow / check_service_hostname_refs: clean
Mutation check: dropping the last-bucket clamp or the recent-first tie-break fails test_node_history.py
```

## Evals run

```text
POSTGRES_URI=... python services/orion-hub/evals/run_node_history_live_eval.py --node node:substrate.bus_synaptic --hours 72
# result_chars=4483 raw_rows=5914
PASS   (every summary field recomputed from the raw rows in Python and matched)
--node node:substrate.biometrics --hours 168: PASS (13,827 rows, 4,291 chars)
--node node:substrate.nope --hours 168: PASS (no item, no raw rows)
Live query cost: ~0.5 s for the 168 h median/first/last aggregate on the busiest node (PK index scan).
```

Live sample, `node:substrate.bus_synaptic`, last 72 h (series trimmed for this report; the tool returns all 48 buckets):

```json
{
 "id": "node:substrate.bus_synaptic",
 "occurred_at": "2026-10-13T18:59:41.873818Z",
 "kind": "node_prediction_error_history",
 "epistemic_status": "record",
 "text": "node:substrate.bus_synaptic prediction error over the 72h ending 2026-10-13T19:00:27Z: 5914 readings, min 0.0, median 0.03125, max 0.1818; first 0.04657 at 2026-10-10T19:01:05Z, last 0.02353 at 2026-10-13T18:59:41Z. Largest change between consecutive readings in the window -0.1173 at 2026-10-12T01:00:39Z.",
 "truncated": false,
 "sensitivity": null,
 "extra": {
  "window_start": "2026-10-10T19:00:27Z",
  "window_end": "2026-10-13T19:00:27Z",
  "count": 5914,
  "min": 0.0,
  "median": 0.03125,
  "max": 0.1818,
  "first": {
   "at": "2026-10-10T19:01:05Z",
   "value": 0.04657
  },
  "last": {
   "at": "2026-10-13T18:59:41Z",
   "value": 0.02353
  },
  "bucket_sec": 5400.0,
  "series_columns": [
   "bucket_start",
   "n",
   "mean",
   "min",
   "max"
  ],
  "series": [
   [
    "2026-10-10T19:00:27Z",
    120,
    0.04103,
    0.0106,
    0.08914
   ],
   [
    "2026-10-10T20:30:27Z",
    121,
    0.03323,
    0.006116,
    0.1074
   ],
   [
    "2026-10-10T22:00:27Z",
    124,
    0.02748,
    0.003236,
    0.09783
   ],
   [
    "2026-10-10T23:30:27Z",
    113,
    0.03479,
    0.003049,
    0.1095
   ],
   [
    "... 40 more buckets ..."
   ],
   [
    "2026-10-13T13:00:27Z",
    120,
    0.03576,
    0.009317,
    0.1319
   ],
   [
    "2026-10-13T14:30:27Z",
    121,
    0.03889,
    0.005814,
    0.1047
   ],
   [
    "2026-10-13T16:00:27Z",
    123,
    0.03979,
    0.009646,
    0.1205
   ],
   [
    "2026-10-13T17:30:27Z",
    122,
    0.03228,
    0.006079,
    0.1346
   ]
  ],
  "empty_buckets": 0,
  "largest_changes": [
   {
    "at": "2026-10-12T01:00:39Z",
    "from": 0.1348,
    "to": 0.01754,
    "delta": -0.1173,
    "since_previous_sec": 90.7
   },
   {
    "at": "2026-10-13T04:31:29Z",
    "from": 0.0431,
    "to": 0.1595,
    "delta": 0.1164,
    "since_previous_sec": 60.2
   },
   {
    "at": "2026-10-12T03:54:01Z",
    "from": 0.1369,
    "to": 0.02075,
    "delta": -0.1162,
    "since_previous_sec": 60.3
   },
   {
    "at": "2026-10-12T03:53:01Z",
    "from": 0.02905,
    "to": 0.1369,
    "delta": 0.1079,
    "since_previous_sec": 30.5
   },
   {
    "at": "2026-10-13T03:29:23Z",
    "from": 0.07438,
    "to": 0.1818,
    "delta": 0.1074,
    "since_previous_sec": 61.3
   }
  ],
  "largest_changes_scope": "between consecutive readings inside the window; ties go to the most recent",
  "earliest_stored_at": "2026-10-06T19:00:01Z"
 }
}
```

## Docker/build/smoke checks

```text
Not run: no Docker change. Live bus path (governor -> Hub responder) is UNVERIFIED until orion-hub is redeployed;
the SQL and summary were verified against live Postgres by the eval above.
```

## Review findings fixed

- Finding: 168 h cap hard-coded, could drift from the writer's retention and make pruned stretches look quiet.
  - Fix: test pins it to `SUBSTRATE_PE_HISTORY_RETENTION_HOURS` in .env_example, settings default and compose; the item now carries `earliest_stored_at` and its text says when the window starts before it.
  - Evidence: `test_max_hours_equals_the_writers_retention`, `test_pruned_window_start_is_said_plainly`.
- Finding: CI tests compared against a Python copy of the SQL, not the SQL.
  - Fix: SQL text pinned to the rules the fake implements; reading-at-window-end lands in the last bucket.
  - Evidence: `test_sql_text_carries_the_rules_the_fake_implements`; mutation check fails as expected.
- Finding: future `until` silently answered a different window.
  - Fix: refused beyond 5 min of clock skew. Evidence: `test_future_until_is_refused`.
- Finding: ties in largest changes favoured the oldest readings.
  - Fix: most recent first. Evidence: `test_tied_changes_favour_the_most_recent`.
- Finding: mistyped node_id looked the same as a quiet node (`items=[]`).
  - Fix: no stored reading at all is a "not found" tool error pointing at the listing. Evidence: `test_node_with_no_stored_reading_is_not_found_not_empty`.
- Findings (nits): tiny `hours` (floor now 1 min), retention-edge double-validation (60 s slack), "within the window" wording, cross-tree test import (moved to `node_history_fake.py`), README said Live before deploy (now says live once Hub is redeployed).

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

The orion-introspect MCP server is spawned per turn by orion-harness-governor from the repo's `orion/` package; if the governor image bakes that package in rather than mounting it, also rebuild it:

```bash
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: low. Concern: live bus path UNVERIFIED until deploy. Mitigation: until then the tool times out and says "answer unknown", never "no history".
- Severity: low. Concern: the busiest node's 168 h query takes ~0.5 s x 4 queries, inside the 15 s RPC timeout. Mitigation: all on the primary key; no table growth beyond 168 h retention.

## PR link

(see below)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
