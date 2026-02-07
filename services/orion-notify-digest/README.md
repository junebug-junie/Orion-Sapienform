# Orion Notify Digest Service

The notify digest service builds a daily summary of notification activity and sends it via `orion-notify`.
It runs on a schedule and can also be triggered on demand for testing.

## Environment

See `.env_example` for full configuration. Key variables:

- `DIGEST_ENABLED` — enable/disable the scheduler
- `DIGEST_TIME_LOCAL` — local time (America/Denver) to run daily digest (HH:MM)
- `DIGEST_WINDOW_HOURS` — lookback window for digest data
- `DIGEST_RECIPIENT_GROUP` — notify recipient group
- `DIGEST_RUN_ON_START` — run once at startup (useful for smoke tests)
- `NOTIFY_SERVICE_URL` / `NOTIFY_API_TOKEN` — notify host connection
- `POSTGRES_URI` — database shared with orion-notify
- `LANDING_PAD_URL` — base URL for Landing Pad (or Hub proxy) topics endpoints
- `TOPICS_WINDOW_MINUTES` — lookback window for topics (default: `DIGEST_WINDOW_HOURS * 60`)
- `TOPICS_MAX_TOPICS` — max topics to include in digest
- `TOPICS_DRIFT_MIN_TURNS` / `TOPICS_DRIFT_MAX_SESSIONS` — drift query filters
- `DRIFT_ALERTS_ENABLED` — enable periodic drift alerts
- `DRIFT_CHECK_INTERVAL_SECONDS` — drift check interval (seconds)
- `DRIFT_ALERT_THRESHOLD` — drift score threshold (default: 0.5)
- `DRIFT_ALERT_MAX_ITEMS` — max drift items included in alert/digest
- `DRIFT_ALERT_SEVERITY` — `warning` or `error`
- `DRIFT_ALERT_EVENT_KIND` — event kind for drift alerts (default: `orion.topics.drift`)
- `DRIFT_ALERT_DEDUPE_WINDOW_SECONDS` — dedupe window for drift alerts

## Scheduling

The scheduler runs daily at `DIGEST_TIME_LOCAL` in America/Denver and ensures only one digest
per window is sent (tracked in the `notify_digest_runs` table).

The digest service reads from the same database as `orion-notify`. Ensure `POSTGRES_URI`
matches the notify service configuration (for SQLite, use a shared volume with
`sqlite:////data/notify.db`).

## Topics Integration

If `LANDING_PAD_URL` is set, the digest fetches topics from:

- `GET {LANDING_PAD_URL}/api/topics/summary?window_minutes=...&max_topics=...`
- `GET {LANDING_PAD_URL}/api/topics/drift?window_minutes=...&min_turns=...&max_sessions=...`

`LANDING_PAD_URL` can point directly to Landing Pad or to a Hub proxy that forwards these routes. If it is not set or the request fails, the digest includes a “Topics unavailable” note.

Topic summary values use the first numeric field found in each item (`count`, `weight`, `score`, or `value`). Drift scores use the first numeric drift field (`drift_score`, `score`, `delta`, `drift`, `magnitude`, `change`).

## Drift Alerts

When `DRIFT_ALERTS_ENABLED=true`, the digest service runs a lightweight periodic loop (default: every 900s) that checks the drift endpoint. If any drift score meets or exceeds `DRIFT_ALERT_THRESHOLD`, the service sends an in-app notification with:

- `event_kind`: `DRIFT_ALERT_EVENT_KIND` (default `orion.topics.drift`)
- `severity`: `DRIFT_ALERT_SEVERITY`
- `channels_requested`: `["in_app"]`
- dedupe key: `topic_drift:<YYYY-MM-DD>:<window_minutes>`

Quiet hours and preferences are enforced by `orion-notify`.

## On-demand run

```bash
python -m app.run_digest --window-hours 1
```

## Local run

```bash
cd services/orion-notify-digest
cp .env_example .env
# edit .env

docker compose up --build
```

## Smoke test

See `scripts/smoke_digest.sh` for a full run that seeds notifications and triggers a digest.
For topic-specific checks, use:

```bash
./scripts/smoke_topic_digest.sh
./scripts/smoke_topic_drift_alert.sh
```
