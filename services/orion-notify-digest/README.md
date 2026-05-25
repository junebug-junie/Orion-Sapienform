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
- `TOPIC_FOUNDRY_URL` — base URL for Topic Foundry (direct or Hub `/api/topic-foundry` proxy)
- `TOPIC_FOUNDRY_MODEL_NAME` — optional model name filter for runs and drift
- `TOPICS_WINDOW_MINUTES` — lookback window label for digest context (default: `DIGEST_WINDOW_HOURS * 60`)
- `TOPICS_MAX_TOPICS` — max topics to include in digest
- `TOPICS_DRIFT_MAX_RECORDS` — max drift records to fetch from Foundry
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

If `TOPIC_FOUNDRY_URL` is set, the digest fetches topics from:

- `GET {TOPIC_FOUNDRY_URL}/runs?format=wrapped&status=complete&limit=1` (resolve latest run)
- `GET {TOPIC_FOUNDRY_URL}/topics?run_id=...&limit=...`
- `GET {TOPIC_FOUNDRY_URL}/drift?model_name=...&limit=...`

`TOPIC_FOUNDRY_URL` can point directly to Topic Foundry or to a Hub proxy (`http://host:8080/api/topic-foundry`). If it is not set or a request fails, the digest includes a “Topics unavailable” note without crashing.

## Drift Alerts

When `DRIFT_ALERTS_ENABLED=true`, the digest service runs a lightweight periodic loop (default: every 900s) that checks Topic Foundry drift records. If any JS divergence score meets or exceeds `DRIFT_ALERT_THRESHOLD`, the service sends an in-app notification with:

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
