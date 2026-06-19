# PR: Mesh critical failure email notifications

**Branch:** `feat/mesh-critical-email-notifications`  
**Base:** `main`

## Summary

Adds email alerts for mesh-health failures that will not self-heal, using the existing SMTP stack and notify infrastructure (same path as digests/journals).

- **Critical / unhealable** (`observe_only`, `max_attempts`, persistent tier-1 failure) → **immediate email** via `POST /attention/request`
- **Error / may still remediate** (`unhealthy_confirmed`) → in-app Pending Attention first; **escalation email after 60m** if unacked
- Background escalation poller in `orion-notify` (digest-style loop, `NOTIFY_ESCALATION_POLL_SECONDS`)
- sql-writer endpoint to mark `attention_escalated_at` and prevent duplicate escalation spam

## Problem

Mesh-guardian attention requests only reached Hub Pending Attention. `/attention/request` never called `EmailTransport`, and the documented escalation loop was disabled.

## Changes by service

### orion-mesh-guardian
- Unhealable state-machine events emit `severity=critical`
- Transient `unhealthy_confirmed` stays `severity=error`
- Richer attention body: service, heartbeat, event metadata; subject `[Orion mesh] {service} — {event}`

### orion-notify
- `email_delivery.py` — shared send policy + policy enrich
- `attention_escalation.py` — polls sql-writer for stale unacked error attention
- `/attention/request` sends immediate email for critical; enriches context with `ack_deadline_minutes` + `escalation_channels`
- `/health` reports `smtp_configured` and escalation poll interval

### orion-sql-writer
- `POST /api/notify-read/attention/{attention_id}/escalate` — idempotent mark of `attention_escalated_at`

## Test plan

- [x] `pytest services/orion-mesh-guardian/tests/test_state_machine.py` — 14 passed
- [x] `pytest services/orion-notify/tests/test_attention_email_delivery.py services/orion-notify/tests/test_attention_escalation_loop.py services/orion-notify/tests/test_notify_email_delivery.py` — 11 passed
- [x] `pytest services/orion-sql-writer/tests/test_notify_attention_escalate.py` — 2 passed
- [x] `python -m compileall services/orion-notify services/orion-mesh-guardian services/orion-sql-writer`
- [ ] Rebuild `notify`, `mesh-guardian`, `sql-writer` after merge
- [ ] Simulate critical attention → inbox within seconds
- [ ] Dismiss stale pre-fix Hub attention items

## Deploy

```bash
docker compose --env-file .env --env-file services/orion-notify/.env \
  -f services/orion-notify/docker-compose.yml up -d --build notify

docker compose --env-file .env --env-file services/orion-mesh-guardian/.env \
  -f services/orion-mesh-guardian/docker-compose.yml up -d --build mesh-guardian

docker compose --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml up -d --build orion-sql-writer
```

Ensure SMTP vars are set in `services/orion-notify/.env` (`NOTIFY_EMAIL_*`). Escalation runs when `NOTIFY_ESCALATION_POLL_SECONDS > 0` (default 60).

## Non-goals

- No new SMTP stack or microservice
- No Hub UI changes
- No bus/schema registry changes (uses existing notify persistence)

## Code review notes

- Escalation marks DB before SMTP send to avoid duplicate emails on partial failure
- Escalation email includes `Attention ID` in body for Hub lookup
- Multi-notify-replica deploy may still race; acceptable for single-replica Athena stack
- SMTP failure after escalate mark does not retry (anti-spam tradeoff)
