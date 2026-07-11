# Mesh critical failure email notifications — design spec

**Date:** 2026-06-19  
**Status:** Approved  
**Scope:** Email alerts for mesh-health failures that will not self-heal, plus delayed escalation for errors that may still remediate.

---

## Problem

Mesh-guardian raises `attention_request` events for unhealthy services. These appear in Hub **Pending Attention** but do **not** reach email because:

1. Mesh-guardian uses `POST /attention/request`, which only publishes in-app + persistence — it never calls `EmailTransport`.
2. The documented attention **escalation loop** (`NOTIFY_ESCALATION_POLL_SECONDS`) is not running; notify startup explicitly skips it.

SMTP is already configured (`NOTIFY_EMAIL_*`) and used by digests/journals via `NotifyClient.send(NotificationRequest(..., channels_requested=["email"]))` → `POST /notify`.

## Goals

- **Critical, unhealable mesh failures** → immediate email.
- **Error-severity attention** (may still remediate) → in-app first; email if still unacked after policy deadline (default 60 minutes).
- Reuse existing `EmailTransport`, policy rules, and sql-writer escalation columns — no new SMTP stack or service.

## Non-goals

- Email on every transient probe blip or `unhealthy_confirmed` that auto-remediation may fix.
- Hub UI changes.
- Chat-message escalation (can reuse same loop later; not required for mesh MVP).
- Replacing daily digests/journals.

---

## Severity mapping (mesh-guardian)

| State machine event | Phase after transition | Severity | Email timing |
|---------------------|------------------------|----------|--------------|
| `observe_only` | `attention_only` | **critical** | Immediate |
| `max_attempts` | `attention_only` | **critical** | Immediate |
| `attention_only` (persistent failure after tier-1) | `attention_only` | **critical** | Immediate |
| `unhealthy_confirmed` | `unhealthy_confirmed` | **error** | Escalate after deadline if unacked |
| `recovery` | `healthy` | info | None |

Context continues to carry `event_kind: orion.mesh.health.attention.v1`, `service_id`, `event`, `correlation_id`.

## Architecture

### Approach: extend `orion-notify` (not a new service)

Lift patterns from:

- `services/orion-notify-digest/app/main.py` — background scheduler loop on startup.
- `services/orion-world-pulse/app/services/publish_email.py` — `channels_requested=["email"]` via `POST /notify`.
- `orion/notify/transport.py` — shared SMTP sender.

### Data flow

```
mesh-guardian (attention_only / unhealthy_confirmed)
  → POST /attention/request (orion-notify)
      → in-app bus (Hub Pending Attention)
      → persistence bus → sql-writer (notify_requests row)
      → if severity=critical: EmailTransport.send() immediately
      → enrich context with ack_deadline_minutes + escalation_channels from policy

escalation loop (orion-notify, every NOTIFY_ESCALATION_POLL_SECONDS)
  → GET sql-writer /api/notify-read/attention?status=pending
  → filter: severity=error, require_ack, !acked_at, !escalated_at, past deadline
  → POST /notify (event_kind=orion.chat.attention.escalation, channels_requested=email)
  → POST sql-writer /api/notify-read/attention/{id}/escalate (mark escalated_at)
```

### Policy

Existing `chat_attention_default` rule applies to `orion.chat.attention`:

- `ack_deadline_minutes: 60`
- `escalation_channels: ["email"]`

`critical_default` already sets `channels: ["email", "in_app"]`.

On `/attention/request`, load `Policy` from `POLICY_RULES_PATH` and merge `ack_deadline_minutes` + `escalation_channels` into notification context before persistence (sql-writer normalization already maps these to DB columns).

### Email body

Subject: attention title / reason (e.g. `attention_request` or enriched `[Orion] mesh health: landing-pad`).

Body: message text + service metadata from context (`service_id`, `event`, `node`, Hub link if `NOTIFY_HUB_URL` set).

Dedupe:

- Immediate critical: `dedupe_key=mesh-health:{service_id}:{event}` with `dedupe_window_seconds=300` on companion `/notify` if sent separately; attention path uses attention_id uniqueness.
- Escalation: `dedupe_key=attention-escalation:{attention_id}`.

## Components to change

| Component | Change |
|-----------|--------|
| `orion-mesh-guardian` | Map unhealable events to `severity=critical`; keep transient `unhealthy_confirmed` as `error` |
| `orion-notify` | Shared `_maybe_send_email()`; call from `/attention/request`; policy enrich on attention; escalation loop module |
| `orion-sql-writer` | `POST /api/notify-read/attention/{attention_id}/escalate` to set `attention_escalated_at` |
| Tests | notify email-on-attention, escalation loop, guardian severity mapping |
| `.env_example` | Document escalation env vars (already present; verify comments) |

## Error handling

- SMTP failure must not block in-app delivery or persistence (same as `/notify` today).
- Escalation loop errors log and continue; next poll retries.
- If SMTP not configured, log `smtp_transport_not_configured` and skip email (no crash).

## Verification

1. Unit: critical attention → `EmailTransport.send` called once.
2. Unit: error attention → no immediate email.
3. Unit: escalation loop with mocked pending row past deadline → escalation notify + escalate mark.
4. Unit: guardian `observe_only` → severity `critical`.
5. Integration smoke: trigger mesh observe-only → email received within seconds.
6. Integration: error attention unacked past deadline (use 1-minute override in test) → escalation email.

## Acceptance

- [ ] Unhealable mesh failure (observe-only / max-attempts) sends email immediately when SMTP configured.
- [ ] Transient unhealthy-confirmed sends in-app only; email only after 60m unacked.
- [ ] No duplicate escalation emails for same attention_id.
- [ ] Hub Pending Attention unchanged; dismiss still works.
- [ ] Existing digest/journal email path unaffected.
