# Orion Notify Service

A minimal notification host that centralizes email delivery for Orion services.

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of `app.state.bus` (Notify's in-app/persistence
publish connection).

## Endpoints

- `GET /health` — basic service health
- `POST /notify` — send an email notification
- `POST /attention/request` — create a chat attention request (in-app + ack)
- `POST /attention/{attention_id}/ack` — acknowledge / dismiss / snooze attention request
- `GET /attention?status=pending|acked&limit=50` — list attention requests
- `POST /chat/message` — create a chat message notification
- `POST /chat/message/{message_id}/receipt` — send read receipt for a chat message
- `GET /chat/messages?status=unread|seen&limit=50` — list chat messages
- `GET /recipients` — list recipient profiles
- `GET /recipients/{recipient_group}` — fetch recipient profile
- `PUT /recipients/{recipient_group}` — update recipient profile (quiet hours, timezone)
- `GET /recipients/{recipient_group}/preferences` — list preferences
- `PUT /recipients/{recipient_group}/preferences` — bulk upsert preferences
- `DELETE /recipients/{recipient_group}/preferences/{pref_id}` — delete preference
- `POST /preferences/resolve` — preview preference resolution
- `GET /notifications` — list recent notifications (admin)
- `GET /notifications/{notification_id}` — fetch a specific notification
- `GET /notifications/{notification_id}/attempts` — delivery attempts for a notification

### Request Model (POST /notify)

```json
{
  "source_service": "security-watcher",
  "event_kind": "security.alert",
  "severity": "error",
  "title": "[Orion] Security alert",
  "body_text": "Alert details...",
  "recipient_group": "juniper_primary",
  "dedupe_key": "security-alert-office-cam",
  "dedupe_window_seconds": 60
}
```

### Request Model (POST /attention/request)

```json
{
  "source_service": "orion-cortex",
  "reason": "I want to talk",
  "severity": "info",
  "message": "Can you check the latest summary?",
  "require_ack": true,
  "context": {"session_id": "abc-123"}
}
```

### Request Model (POST /attention/{attention_id}/ack)

```json
{
  "attention_id": "8c0c1a20-9a9f-45b1-9aa0-1af6d1f7a6f3",
  "ack_type": "dismissed",
  "note": "Handled"
}
```

### Request Model (POST /chat/message)

```json
{
  "source_service": "orion-cortex",
  "session_id": "session-123",
  "preview_text": "Short message preview",
  "full_text": "Optional full content",
  "severity": "info",
  "require_read_receipt": true
}
```

### Request Model (POST /chat/message/{message_id}/receipt)

```json
{
  "message_id": "5d7a4f3b-8e7b-4ef9-9ee5-2c1a8f2b9a0f",
  "session_id": "session-123",
  "receipt_type": "opened"
}
```

## Environment

See `.env_example` for required SMTP + DB configuration and optional API token settings.

### Auth

If `API_TOKEN` is set, clients must send `X-Orion-Notify-Token: <token>` with all requests.

### In-app delivery

Enable in-app delivery to the Hub by setting:

- `NOTIFY_IN_APP_ENABLED=true`
- `NOTIFY_IN_APP_CHANNEL=orion:notify:in_app`
- `ORION_BUS_URL` / `ORION_BUS_ENABLED` / `ORION_BUS_ENFORCE_CATALOG`

### Policy

The notify service loads policy rules from `POLICY_RULES_PATH` (default: `/app/app/policy/rules.yaml`).
Rules control delivery channels, recipient groups, dedupe, throttling, and quiet hours.

Attention rules for `orion.chat.attention` can also set:

- `require_ack` (bool)
- `ack_deadline_minutes` (int)
- `escalation_channels` (list, e.g. `["email"]`)

Escalation checks run on a background loop configured by:

- `NOTIFY_ESCALATION_POLL_SECONDS` (default: 60; set `0` to disable)

When SMTP is configured and the poll interval is > 0, notify:

1. Sends **immediate email** for `POST /attention/request` with `severity=critical` (e.g. mesh-health observe-only / max-attempts).
2. Polls sql-writer for pending `severity=error` attention past `ack_deadline_minutes` and sends escalation email (`orion.chat.attention.escalation`).

Mesh-guardian unhealable failures use `severity=critical`; transient unhealthy-confirmed uses `error` and escalates if unacked.

Chat message rules for `orion.chat.message` can set:

- `require_read_receipt` (bool)
- `read_receipt_deadline_minutes` (int)
- `escalation_channels` (list, e.g. `["email"]`)

Presence checks before chat message delivery can be configured with:

- `NOTIFY_PRESENCE_URL` (Hub `/api/presence` endpoint)
- `NOTIFY_PRESENCE_TIMEOUT_SECONDS`
- `NOTIFY_HUB_URL` (optional; used for escalation links)

## Persistence & Schema

Notify is stateless and delegates persistence to `orion-sql-writer` via the Orion Bus.
Read operations are proxied to `orion-sql-writer`'s read API.

## Recipient Profiles & Preferences

Recipient profiles store quiet hours and timezone. Preferences override policy defaults based on event kind or severity:

- `scope_type="event_kind"` (e.g. `orion.chat.message`)
- `scope_type="severity"` (e.g. `warning`)

Preferences can override:

- `channels_enabled` (e.g. `["in_app","email"]`)
- `escalation_enabled` and `escalation_delay_minutes`
- `dedupe_window_seconds`, `throttle_max_per_window`, `throttle_window_seconds`

Guard rails enforced on update:

- `throttle_window_seconds` ∈ [10, 86400]
- `throttle_max_per_window` ∈ [1, 1000]
- `dedupe_window_seconds` ∈ [0, 86400]

Quiet hours default to America/Denver and are stored per recipient profile. During quiet hours, severities below `error` suppress in-app toasts by default (notifications are still stored). Preferences can explicitly re-enable `in_app` during quiet hours by including it in `channels_enabled`.

The resolver applies preferences in the following order:

1. `event_kind` preference
2. `severity` preference
3. policy defaults

The `/preferences/resolve` endpoint returns `channels_final` and a `source_breakdown` explaining each decision.

On startup, notify ensures `juniper_primary` exists and seeds default preferences for chat attention/message plus error/warning/info severities if no preferences are present.

## Local Run

```bash
cd services/orion-notify
cp .env_example .env
# edit .env with SMTP values

docker compose up --build
```

## Smoke Test (Preferences)

```bash
./scripts/smoke_notify_prefs.sh
```

## Smoke Test (Chat Message)

```bash
./scripts/smoke_chat_message.sh
```

This script uses `services/orion-notify/.env` (and `services/orion-hub/.env`) for configuration. If `API_TOKEN` is set in notify, the script will include the `X-Orion-Notify-Token` header automatically.

## Smoke Test (All Notifications)

Run the full notification smoke harness (notify + hub + digest + bus):

```bash
./scripts/smoke_all_notifications.sh
```

Options:

- `--keep-up` / `--no-down`: leave containers running after tests
- `--only <regex>`: run a subset (matches test names like `notify`, `chat_message`, `digest`)
- `--list`: list discovered smoke tests

Requirements: `bash`, `curl`, `jq`, `docker compose`. The harness will pass `X-Orion-Notify-Token` automatically if `API_TOKEN` is set in `services/orion-notify/.env`.

## Calling From Services

Use `orion.notify.client.NotifyClient` to send requests to the notify service.
```python
from orion.notify.client import NotifyClient
from orion.schemas.notify import NotificationRequest

client = NotifyClient(base_url="http://orion-notify:7140", api_token="your-token")
req = NotificationRequest(
    source_service="demo",
    event_kind="demo.event",
    severity="info",
    title="Hello",
    body_text="World",
)
client.send(req)
```

## `notify_requests.status` -- what it means

**Changed 2026-08-30 (PR #1991).** Before that date this column was written once
as `"pending"` at every call site and never updated by anything: 10,900 rows
since 2026-07-24, 100% pending, including emails that were definitely
delivered. Rows created before the cutover carry the old, meaningless `pending`
and cannot be told apart from a genuine post-cutover `pending` except by
`created_at`. Nothing was backfilled -- there is no record anywhere of what
happened to those, so a backfill would be invention.

| value | meaning |
|---|---|
| `sent` | the SMTP transport accepted the message without raising. **Not proof it reached an inbox** -- nothing downstream of SMTP reports back. Do not read it as "Juniper saw it". |
| `failed` | the send raised, including a partial recipient refusal |
| `no_email` | no email was attempted and none will be: policy declined, no SMTP transport configured, or an endpoint that never emails (`/chat/message`) |
| `pending` | nothing attempted yet, outcome unknown. Written when email is **deferred**: `/attention/request` skips immediate email for `severity != "critical"`, but `attention_escalation.py` emails `severity == "error"` attentions past their ack deadline, so an email may still follow. Also the loud fallback for an unmapped outcome (logged as `unmapped_email_outcome`). |

`drop_reason` carries *why*, which the status alone collapses -- policy decline,
unconfigured transport, and deferral are three materially different facts.

**Not the same column as attention status.** `GET /attention?status=pending|acked`
derives its value from `attention_require_ack` / `attention_acked_at`
(`orion-sql-writer/app/api_notify.py::_attention_to_schema`), not from this
column. `health_monitor._has_open_alert` depends on that derived view for alert
dedup and is unaffected by anything here.

**One consumer does read this column:** `orion-notify-digest/app/digest.py`
filters `status == "throttled"` and `== "deduped"`. Neither value was ever
written before this change and neither is written by it, so those counters are
unaffected -- but the column is not unread, and a future value must not collide
with those two.

**Known gap:** an escalation email sent by `attention_escalation.py` does not
update the parent row, which stays `pending`. The escalation
`NotificationRequest` is never persisted as its own record either. Closing that
needs a write-back path or a new persisted record, and is deliberately out of
scope for PR #1991.

