## Summary

- The shared bus listener (`Hunter` in `orion/core/bus/bus_service_chassis.py`) no longer logs every received message at INFO. Per-message line is now DEBUG.
- INFO keeps: the first message after each (re)subscribe (proves the pipe is flowing again after a reconnect), and one summary line per 60s with counts by channel and message kind. Decode failures, handler errors and reconnect warnings are unchanged.
- orion-signal-gateway now applies its existing `LOG_LEVEL` to loguru too. The chassis logs through loguru, whose default sink is DEBUG, so demoting the line alone would not have silenced it.
- The gateway's OTEL collector sidecar debug exporter goes from `verbosity: normal` (one line per span) to `basic` (one line per batch).

## Outcome moved

Live, 10-minute window (2026-10-11 ~00:45 UTC, containers up since 00:32):

| container | before (lines/h) | before (MB/h) | expected after |
|---|---|---|---|
| orion-signal-gateway-orion-signal-gateway-1 | ~214,000 | ~69 | ~550 (60 summaries + ~480 existing WARNINGs) |
| orion-signal-gateway-otel-collector-1 | ~323,000 | ~32 | a few hundred (one per batch) — UNVERIFIED |

99.8% of the gateway's lines were the single `Hunter intake` line (39,898 of ~40,000). Together these two containers were ~100 MB/h into the fleet-wide journald cap from PR #2602.

## Current architecture

Every service using `Hunter` (19 under `services/`) emitted one INFO line per received envelope. Signal-gateway subscribes to high-rate vision channels (frames, edge health, scene state ~14/s each), so it dominated.

## Architecture touched

- `orion/core/bus/bus_service_chassis.py` (shared chassis, all Hunter users). **The volume drop applies to orion-signal-gateway only.** The other 18 Hunter services never set a loguru level, and loguru's default sink passes DEBUG, so their per-message line still prints (plus one summary line per minute). Biggest remaining one measured live: orion-athena-sql-writer, ~23k intake lines/hour. Follow-up: apply `LOG_LEVEL` to loguru in a shared place so every chassis owner gets it.
- orion-signal-gateway entrypoint and collector config.

## Files changed

- `orion/core/bus/bus_service_chassis.py`: `Hunter._log_intake` — DEBUG per message, INFO first-after-subscribe, 60s rolled-up summary.
- `services/orion-signal-gateway/app/main.py`: `configure_loguru(LOG_LEVEL)`.
- `services/orion-signal-gateway/otel/collector-config.yaml`: debug exporter `basic`.
- `services/orion-signal-gateway/app/tests/test_intake_log_volume.py`: regression tests.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: log levels only.
- Compatibility notes: in orion-signal-gateway, per-message `Hunter intake` lines (trace_id, channel) now need `LOG_LEVEL=DEBUG`; at INFO, use the first-after-subscribe line or the summary (counts by channel+kind). Runbooks that grep `Hunter intake` in this container (e.g. `2026-08-14-biometrics-hub-mode-flip-pr.md:149`) will see only those. Other services unchanged.

## Env/config changes

- Added keys: none (reuses existing `LOG_LEVEL`)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-signal-gateway: pytest -q            -> 37 passed
tests/test_hunter_reconnect.py                      -> 2 passed
Mutation check: per-message line back to INFO       -> test_normal_messages_do_not_log_at_info FAILS (as intended)
```

## Evals run

```text
No eval harness for log volume. Live before-volume measured from journald; after-volume UNVERIFIED until restart.
```

## Docker/build/smoke checks

```text
Not deployed (per instruction). After restart, verify:
journalctl CONTAINER_NAME=orion-signal-gateway-orion-signal-gateway-1 --since -10min | wc -l
```

## Review findings fixed

- Finding: fix is in the shared chassis, but only the gateway sets a loguru level, so the other 18 services are unchanged.
  - Fix: report scoped to the gateway; sql-writer (~23k lines/h) named as follow-up.
  - Evidence: Architecture touched section.
- Finding: PR report was not committed.
  - Fix: committed.
  - Evidence: this file is in the diff.
- Finding: summary counted by kind only, so a kind arriving on an unexpected channel would be invisible.
  - Fix: counts keyed by `channel:kind`.
  - Evidence: `test_intake_summary_rolls_up_counts`.
- Finding: an invalid `LOG_LEVEL` (e.g. `WARN`) made loguru raise at import and would crash-loop the gateway.
  - Fix: fall back to INFO with a warning.
  - Evidence: `test_bad_log_level_falls_back_instead_of_crashing`.
- Not fixed (minor): counts not flushed on stop; silent channel emits no summary. Heartbeat covers absence.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-signal-gateway up -d --build --force-recreate orion-signal-gateway otel-collector
```

`--force-recreate` is needed for the collector: its config is a single-file bind mount, which keeps the old file after `git pull`.

## Risks / concerns

- Severity: low
- Concern: the summary is emitted on message arrival, so a channel that goes silent produces no summary until the next message.
- Mitigation: silence is still visible via the absence of summaries and the existing heartbeat/health path.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
