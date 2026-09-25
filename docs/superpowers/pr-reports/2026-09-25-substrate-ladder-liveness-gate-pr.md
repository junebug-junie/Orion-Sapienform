## Summary

- A new check tells you when part of Orion's cognition ladder has stopped writing, even though every container still says "Up". It looks at the newest row in each step: raw input lanes, steady reducers, field state, attention, proposal, policy, dispatch, feedback, and consolidation. A step fails when that row is older than its limit.
- It also fails when consolidation's last 3 hourly frames are empty. Consolidation keeps running during an outage and just writes nothing.
- A second check finds containers that read a strict schema (`FieldStateV1`) but carry an older copy of it than the service that writes it. That mismatch is what caused the 2026-09-20 outage. The list of reading services comes from scanning the code, not from a hand-kept list.
- `make substrate-ladder-check` runs both checks read-only. `make substrate-ladder-watch` does the same and also raises a debounced Hub Pending Attention card through orion-notify.
- **It caught a live problem on its first run.** `orion-feedback-runtime` still runs the pre-09-20 schema. It logged about 96k `extra_forbidden` errors in 2 hours and writes every feedback frame with `field_before=None`. The feedback rung still looks fresh, so only the schema check sees this.

## Outcome moved

Failure mode: "every container Up, a step of the ladder silently dead." On 09-20 this went unnoticed for about 48h. Replaying that incident now fails the check (exit 1). The attention step is already past its 15-minute limit 22 minutes in, and the schema check flags the reading containers before any rows go missing.

## Current architecture

- `services/orion-hub/scripts/substrate_lattice_routes.py` computes fresh/stale per layer, but only for the transport lane, only when someone opens the Hub lattice tab, and it has no alerting. It also uses unbounded `ORDER BY ... LIMIT 1` queries.
- `check_substrate_projection_schema_drift.py` validates seven singleton rows against the current schema. It does not look at the running containers.
- Nothing compared the schema copies inside running containers, and nothing paged on a stale step.

## Architecture touched

- New stdlib-only module `orion/substrate_ladder_liveness.py` with the pure logic. It lives at the top level because `orion/substrate/__init__` pulls in `requests` and other service dependencies, and CI's static-gates venv does not have them.
- New CLI `scripts/check_substrate_ladder_liveness.py` does the I/O: psycopg2 read-only with a 20s statement timeout, `docker ps/inspect/exec`, `git log --first-parent`, and orion-notify.
- Alert surface: an orion-notify `/attention/request` card, i.e. Hub Pending Attention. Chosen because it is the path host cron already uses for disk-threshold-watchdog and postgres-headroom-watch, and a human acks those cards.

## Files changed

- `orion/substrate_ladder_liveness.py`: the list of steps and their limits, freshness and empty-consolidation rules, import-scan consumer discovery, skew rule, and the report.
- `scripts/check_substrate_ladder_liveness.py`: CLI, live reads, debounced notify, exit codes (0 green, 1 red, 2 could not check).
- `tests/scripts/test_substrate_ladder_liveness.py`: 40 gate tests.
- `tests/fixtures/substrate_ladder_2026-09-20_incident.json`: the incident replay. Timestamps were recovered from live Postgres, and its `_source` field says which values are recovered and which are representative.
- `Makefile`: `substrate-ladder-check`, `substrate-ladder-watch`.
- `.github/workflows/orion-static-gates.yml`: runs the logic tests in CI.
- `scripts/README.md`: usage and the cron line.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: none. This only reads.
- Compatibility notes: n/a.

## Env/config changes

- Added keys: none. Removed/renamed: none. `.env_example` not touched, so no local `.env` sync needed.
- Reads the optional existing `POSTGRES_URI` / `ORION_PG_*`, `NOTIFY_BASE_URL`, `NOTIFY_API_TOKEN`, `TELEMETRY_ROOT`, and `PROJECT`, with the same defaults as `check_postgres_connection_headroom.py`.

## Metric quality gate (for the per-step freshness signals)

1. **Provenance:** each value is `max(generated_at|created_at)` of the table that step's service writes. The consolidation-empty signal reads `consolidation_frame_json->'motif_observations'`.
2. **Independence:** the steps are a causal chain on purpose. A dead step turns everything above it red too, and that is what points at the lowest dead step. The schema check is independent of freshness: today it is red while feedback's freshness is green.
3. **Theory anchor:** a step that writes every tick and has no row for N times its worst normal gap is not running. Consolidation writing only empty frames means its inputs are gone. Both are observations, not inferred states.
4. **Live data:** worst normal gaps from 7-14 days of live rows. Field/attention: 8.6s, limit 15m. Proposal→feedback: 13-20m outside outages, limit 45m. Bus/biometrics grammar lanes: under 1.5m, limit 15m. cortex-exec lane: 13.5m, limit 60m. Steady reducers: 23-4400 receipts per hour, and receipts are pruned after 30m, limit 20m. Consolidation: hourly, limit 2h. Empty consolidation over 60 days: the longest run of empty frames outside the incident was 1 frame, the incident was 52, and the threshold is 3. The limits can return to rest: every step is green in today's live run.
   Reducers that only fire on activity (execution_trajectory, node_pressure, route_arbitration) are left out, because quiet is normal for them.
5. **Existing mechanism:** the Hub lattice routes compute stale for the transport lane only, on demand, with no alert. This does not replace them. It covers every step and alerts.
6. **Reversibility:** a script, two make targets, and one CI step. Nothing is persisted except a local debounce state file.

## Tests run

```text
.venv/bin/python -m pytest tests/scripts/test_substrate_ladder_liveness.py -q     -> 40 passed in 5.4s
(CI-equivalent venv: pydantic pydantic-settings PyYAML pytest only)               -> 40 passed in 5.2s
Mutation check (flip skew direction; compare to main instead of producer;
  treat missing rungs as present; any-empty instead of all-empty)                 -> each caught (5/2/1/2 failures)
Static gates from orion-static-gates.yml (13 scripts)                              -> all exit 0
```

## Evals run

```text
No eval harness applies: this is a deterministic detector. The replay of the real
09-20 incident fixture is the behavioural check: it is red on the hash path, red
on the timestamp fallback, and red on freshness 22 minutes into the incident.
```

## Docker/build/smoke checks

```text
Live read-only run 2026-09-25 ~00:32Z (make substrate-ladder-check):
  16 steps: all fresh (field/attention/proposal/policy/dispatch/feedback 0-2s old,
    consolidation 32m, motifs present)
  FieldStateV1 schema check: field-digester (producer) matches main; attention,
    proposal, policy, hub match the producer
  RED orion-athena-feedback-runtime: schema file differs from the producer's; image built
    2026-09-14T02:18:42Z, producer image 2026-09-24T03:46:58Z
  exit 1
Confirmation: docker logs orion-athena-feedback-runtime shows FieldStateV1
  extra_forbidden in store.py load_field_for_tick/load_latest_field_after (26k in 30 min).
Notify health: GET localhost:7140/health -> ok. Card delivery itself UNVERIFIED:
  I did not send a real card without Juniper's approval.
```

## Review findings fixed

- Finding (must): the schema check compared containers against origin/main instead of what the producer actually writes. A producer deployed ahead of main (the 09-20 shape) stayed green, and a change merged but not yet deployed raised false alarms.
  - Fix: the reference is now the running producer's schema bytes. Image age decides which side is newer. Producer-vs-main drift is reported but is not red.
  - Evidence: `test_producer_deployed_ahead_of_main_still_catches_old_consumers`, `test_merged_but_undeployed_schema_change_is_not_red`.
- Finding (should): the schema commit time came from the side-branch commit date, not the time it landed on main.
  - Fix: `git log --first-parent`, which resolves to 3c1e65a16 at 21:56:46Z instead of 21:15:13Z.
  - Evidence: live `--verbose` output.
- Finding (should): the debounce forgot a key whenever its check errored, so the card was re-sent after a flaky read.
  - Fix: a key is forgotten only when its check ran and came back green (`green_keys`).
  - Evidence: `test_flaky_read_does_not_rearm_a_delivered_card`.
- Finding (should): the "warning" severity could never be chosen.
  - Fix: `LadderReport.severity()` returns warning only when the sole finding is empty consolidation.
  - Evidence: `test_severity_is_warning_only_for_empty_consolidation_alone`.
- Finding (should): no tests covered exit codes 1 and 2, or a failed step query.
  - Fix: added `test_cli_exit_codes` (red wins over could-not-check) and `test_a_failed_rung_query_is_cannot_check_not_fresh`.
  - Evidence: tests pass, and the mutation check shows they fail when that logic is broken.
- Finding (should): the incident replay only exercised the timestamp fallback.
  - Fix: the fixture now carries hashes and includes feedback-runtime. A separate test still checks the fallback path.
  - Evidence: `test_replay_of_2026_09_20_incident_is_red`, `test_replay_is_red_on_the_timestamp_fallback_too`.
- Nits fixed: word-boundary match for the bus schema name (plus a test), no repo code executed inside containers to hash the file, a jsonb type guard, not-running consumers printed as a warning, and the scan's known limits written into its docstring.
- Not fixed: the real-repo consumer scan takes about 5s, above the section 11 "usually under 2s" guide. Kept, because that test is what proves consumer discovery works on the real tree.

## Restart required

```text
No restart required. Nothing deployed.
```

To act on what the check found (Juniper's call, not done here):

```bash
scripts/safe_docker_build.sh orion-feedback-runtime up -d --build   # from an up-to-date main worktree
```

To start alerting (a host crontab change, so not done here):

```cron
*/10 * * * * cd /mnt/scripts/Orion-Sapienform && PATH=/mnt/scripts/Orion-Sapienform/venv/bin:$PATH make substrate-ladder-watch >> /mnt/scripts/Orion-Sapienform/logs/orion-substrate-ladder-liveness.log 2>&1
```

With the current live state, the first cron tick would raise one critical card for feedback-runtime.

## Risks / concerns

- Severity: medium. Concern: only `FieldStateV1` is covered. Other `extra="forbid"` models that are persisted by one service and read by another can break the same way. Mitigation: `STRICT_SCHEMAS` is a one-line addition per schema (path, symbol, producer).
- Severity: low. Concern: the consumer scan can over-include, because a service that imports a module which merely imports the schema counts as a reader. Mitigation: a flagged container still needs a real byte mismatch and an older image to go red.
- Severity: low. Concern: steps are cut off at the 3-day lookback, so anything dead for more than 3 days shows as "no row in window", not its exact age. Mitigation: it is still red.
- Severity: low. Concern: a consumer brought up from a compose file outside `services/<dir>/` shows as not_running (a warning, not red).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2325

🤖 Generated with [Claude Code](https://claude.com/claude-code)
