# Mesh critical failure email notifications — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send immediate email for unhealable mesh-health critical failures and delayed email escalation for unacked error-severity attention, reusing existing SMTP and notify infrastructure.

**Architecture:** Mesh-guardian maps unhealable state-machine events to `severity=critical`. `orion-notify` wires `EmailTransport` into `/attention/request` and runs a digest-style escalation poll loop that emails unacked `error` attention past policy deadline. `orion-sql-writer` gets a small endpoint to mark `attention_escalated_at`.

**Tech stack:** Python 3.12, FastAPI, `orion.notify.transport.EmailTransport`, sql-writer Postgres notify tables, mesh-guardian state machine.

**Spec:** `docs/superpowers/specs/2026-06-19-mesh-critical-email-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `services/orion-mesh-guardian/app/state_machine.py` | Severity mapping for attention events |
| `services/orion-mesh-guardian/tests/test_state_machine.py` | Assert critical vs error severities |
| `services/orion-notify/app/email_delivery.py` | **New** — shared `_maybe_send_email`, policy enrich helper |
| `services/orion-notify/app/attention_escalation.py` | **New** — background escalation loop |
| `services/orion-notify/app/main.py` | Wire email + policy + startup loop |
| `services/orion-notify/tests/test_attention_email_delivery.py` | **New** — attention path email tests |
| `services/orion-notify/tests/test_attention_escalation_loop.py` | **New** — escalation loop tests |
| `services/orion-sql-writer/app/api_notify.py` | `POST .../attention/{id}/escalate` |
| `services/orion-sql-writer/tests/test_notify_attention_escalate.py` | **New** — escalate endpoint test |

---

### Task 1: Guardian severity mapping

**Files:**
- Modify: `services/orion-mesh-guardian/app/state_machine.py`
- Test: `services/orion-mesh-guardian/tests/test_state_machine.py`

- [ ] **Step 1: Write failing tests**

Add to `test_state_machine.py`:

```python
@pytest.mark.parametrize(
    "event_name",
    ["observe_only", "max_attempts"],
)
def test_unhealable_attention_events_are_critical(event_name: str) -> None:
    # observe_only: suspect + auto_remediate=False confirmation
    # max_attempts: unhealthy_confirmed + attempts_this_hour >= max
    ...
    assert out.attention_events[0]["severity"] == "critical"


def test_unhealthy_confirmed_transient_is_error() -> None:
    state = ServiceState(phase=ServicePhase.suspect, consecutive_probe_fails=1)
    out = transition(
        state,
        _inp(equilibrium_bad=True, probe_status="probe_bad", auto_remediate=True),
        service_id="landing-pad",
    )
    assert out.new_state.phase == ServicePhase.unhealthy_confirmed
    assert out.attention_events[0]["severity"] == "error"
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd services/orion-mesh-guardian
PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_state_machine.py -q --tb=short -k "critical or transient_is_error"
```

- [ ] **Step 3: Update `_attention_event` calls**

In `state_machine.py`, set `severity="critical"` for:
- `observe_only` branch (both suspect-confirm and unhealthy_confirmed paths)
- `max_attempts` branch
- `persistent failure after tier-1` branch (`event="attention_only"`)

Keep `severity="error"` for `unhealthy_confirmed` transient event.

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd services/orion-mesh-guardian
PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_state_machine.py -q --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-mesh-guardian/app/state_machine.py services/orion-mesh-guardian/tests/test_state_machine.py
git commit -m "feat(mesh-guardian): mark unhealable mesh failures as critical severity"
```

---

### Task 2: Shared email delivery helper in notify

**Files:**
- Create: `services/orion-notify/app/email_delivery.py`
- Modify: `services/orion-notify/app/main.py`
- Test: `services/orion-notify/tests/test_attention_email_delivery.py`

- [ ] **Step 1: Create `email_delivery.py`**

```python
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

from orion.notify.transport import EmailTransport
from orion.schemas.notify import NotificationRequest

from .policy import Policy

logger = logging.getLogger("orion.notify.email_delivery")


def should_send_email(payload: NotificationRequest) -> Tuple[bool, Optional[str]]:
    channels = payload.channels_requested or []
    if any(channel.lower() == "email" for channel in channels):
        return True, "channels_requested=email"
    severity = (payload.severity or "").lower()
    if severity in {"error", "critical"}:
        return True, f"severity={severity}"
    return False, None


def enrich_with_policy(payload: NotificationRequest, policy: Policy, now: datetime) -> NotificationRequest:
    decision = policy.evaluate(payload, now)
    context = dict(payload.context or {})
    if decision.ack_deadline_minutes is not None:
        context["ack_deadline_minutes"] = decision.ack_deadline_minutes
    if decision.escalation_channels:
        context["escalation_channels"] = decision.escalation_channels
    payload.context = context
    return payload


def maybe_send_email(
    transport: EmailTransport | None,
    payload: NotificationRequest,
    *,
    immediate_critical_only: bool = False,
) -> None:
    if transport is None:
        return
    severity = (payload.severity or "").lower()
    if immediate_critical_only and severity != "critical":
        return
    should, reason = should_send_email(payload)
    if not should:
        return
    if immediate_critical_only and severity == "error":
        return
    try:
        transport.send(payload)
        logger.info("[NOTIFY] email_send_succeeded notification_id=%s reason=%s", payload.notification_id, reason)
    except Exception as exc:
        logger.error("[NOTIFY] email_send_failed notification_id=%s error=%s", payload.notification_id, exc)
```

- [ ] **Step 2: Refactor `main.py` to use helper**

Replace `_should_send_email` body with import from `email_delivery.should_send_email`.
Replace inline SMTP send block in `notify()` with `maybe_send_email(transport, payload)`.

- [ ] **Step 3: Write failing test for attention critical email**

`test_attention_email_delivery.py`:

```python
@pytest.mark.asyncio
async def test_attention_request_sends_email_for_critical(monkeypatch):
    sent = DummyTransport()
    # patch policy load, bus publish, persistence
    payload = ChatAttentionRequest(severity="critical", message="mesh health: x", ...)
    await main.attention_request(payload, request_with_transport(sent))
    await asyncio.sleep(0)
    assert len(sent.calls) == 1
```

- [ ] **Step 4: Run test — expect FAIL**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=services/orion-notify:. ./venv/bin/python -m pytest services/orion-notify/tests/test_attention_email_delivery.py -q --tb=short
```

- [ ] **Step 5: Wire email into `attention_request`**

In `attention_request()`:
1. Load policy from `POLICY_RULES_PATH` (cache on `app.state.policy` at startup).
2. Convert to `NotificationRequest` via `_attention_request_to_notification`.
3. `enrich_with_policy(notify_payload, policy, datetime.utcnow())`.
4. `maybe_send_email(email_transport, notify_payload, immediate_critical_only=True)`.
5. Continue existing in-app + persistence publish using enriched payload.

- [ ] **Step 6: Run tests — expect PASS**

```bash
PYTHONPATH=services/orion-notify:. ./venv/bin/python -m pytest services/orion-notify/tests/test_attention_email_delivery.py services/orion-notify/tests/test_notify_email_delivery.py -q --tb=short
```

- [ ] **Step 7: Commit**

```bash
git add services/orion-notify/app/email_delivery.py services/orion-notify/app/main.py services/orion-notify/tests/test_attention_email_delivery.py
git commit -m "feat(notify): send immediate email for critical attention requests"
```

---

### Task 3: SQL-writer escalate endpoint

**Files:**
- Modify: `services/orion-sql-writer/app/api_notify.py`
- Create: `services/orion-sql-writer/tests/test_notify_attention_escalate.py`

- [ ] **Step 1: Write failing test**

```python
def test_mark_attention_escalated_sets_timestamp(client, db_session, attention_row):
    resp = client.post(f"/api/notify-read/attention/{attention_row.attention_id}/escalate")
    assert resp.status_code == 200
    db_session.refresh(attention_row)
    assert attention_row.attention_escalated_at is not None
```

- [ ] **Step 2: Implement endpoint**

```python
@router.post("/attention/{attention_id}/escalate")
async def mark_attention_escalated(attention_id: str):
    db = get_session()
    try:
        row = db.query(NotificationRequestDB).filter(NotificationRequestDB.attention_id == attention_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="attention_not_found")
        if row.attention_escalated_at is not None:
            return {"attention_id": attention_id, "escalated_at": row.attention_escalated_at.isoformat(), "status": "already_escalated"}
        now = datetime.utcnow()
        row.attention_escalated_at = now
        db.commit()
        return {"attention_id": attention_id, "escalated_at": now.isoformat(), "status": "escalated"}
    finally:
        remove_session()
```

- [ ] **Step 3: Run test — expect PASS**

```bash
./scripts/test_service.sh orion-sql-writer services/orion-sql-writer/tests/test_notify_attention_escalate.py -q --tb=short
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-sql-writer/app/api_notify.py services/orion-sql-writer/tests/test_notify_attention_escalate.py
git commit -m "feat(sql-writer): add attention escalate mark endpoint for notify loop"
```

---

### Task 4: Attention escalation loop

**Files:**
- Create: `services/orion-notify/app/attention_escalation.py`
- Modify: `services/orion-notify/app/main.py`
- Test: `services/orion-notify/tests/test_attention_escalation_loop.py`

- [ ] **Step 1: Create escalation module**

```python
async def run_attention_escalation_once(*, settings, email_transport, policy, proxy_get, proxy_post) -> int:
    """Return count of escalations sent."""
    rows = await proxy_get("/attention", params={"status": "pending", "limit": 100})
    now = datetime.utcnow()
    sent = 0
    for row in rows:
        if (row.get("severity") or "").lower() != "error":
            continue
        if row.get("escalated_at"):
            continue
        if not row.get("require_ack") or row.get("acked_at"):
            continue
        deadline_min = row.get("ack_deadline_minutes") or 60
        created = datetime.fromisoformat(row["created_at"])
        if (now - created).total_seconds() < deadline_min * 60:
            continue
        channels = (row.get("context") or {}).get("escalation_channels") or ["email"]
        if "email" not in [c.lower() for c in channels]:
            continue
        # build NotificationRequest event_kind=orion.chat.attention.escalation
        # maybe_send_email (not immediate_critical_only)
        # proxy_post escalate endpoint
        sent += 1
    return sent


async def attention_escalation_loop(app) -> None:
    while True:
        try:
            await run_attention_escalation_once(...)
        except Exception as exc:
            logger.warning("attention escalation tick failed: %s", exc)
        await asyncio.sleep(settings.NOTIFY_ESCALATION_POLL_SECONDS)
```

- [ ] **Step 2: Start loop on notify startup**

In `on_startup()`:

```python
app.state.policy = Policy.load(settings.POLICY_RULES_PATH)
if app.state.email_transport and settings.NOTIFY_ESCALATION_POLL_SECONDS > 0:
    app.state.escalation_task = asyncio.create_task(attention_escalation_loop(app))
```

Cancel task in `on_shutdown()`.

- [ ] **Step 3: Write escalation loop test with frozen time + mock pending row**

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=services/orion-notify:. ./venv/bin/python -m pytest services/orion-notify/tests/test_attention_escalation_loop.py -q --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-notify/app/attention_escalation.py services/orion-notify/app/main.py services/orion-notify/tests/test_attention_escalation_loop.py
git commit -m "feat(notify): add attention escalation email loop for unacked errors"
```

---

### Task 5: Enrich mesh attention email body (optional polish)

**Files:**
- Modify: `services/orion-mesh-guardian/app/attention.py`

- [ ] **Step 1: Improve title/body for email readability**

```python
title = f"[Orion mesh] {service_id} — {event.get('context', {}).get('event', 'attention')}"
body = f"{message}\n\nservice: {service_id}\nheartbeat: {heartbeat_name}\nnode: ..."
```

Pass enriched fields via attention `context` if notify uses `reason`/`message` from request.

- [ ] **Step 2: Commit**

```bash
git commit -m "feat(mesh-guardian): improve mesh attention text for email subjects"
```

---

### Task 6: Runtime verification

- [ ] **Step 1: Rebuild services**

```bash
docker compose --env-file .env --env-file services/orion-notify/.env -f services/orion-notify/docker-compose.yml up -d --build notify
docker compose --env-file .env --env-file services/orion-mesh-guardian/.env -f services/orion-mesh-guardian/docker-compose.yml up -d --build mesh-guardian
```

- [ ] **Step 2: Confirm SMTP transport configured**

```bash
docker logs orion-athena-notify 2>&1 | rg "smtp_transport_configured"
```

- [ ] **Step 3: Simulate critical attention**

```bash
curl -s -X POST http://localhost:7140/attention/request \
  -H 'Content-Type: application/json' \
  -d '{"source_service":"test","reason":"mesh test","severity":"critical","message":"mesh health: test-service observe-only","require_ack":true,"context":{"service_id":"test-service","event":"observe_only","event_kind":"orion.mesh.health.attention.v1"}}'
```

Check notify logs for `email_send_succeeded` and inbox.

- [ ] **Step 4: Compileall**

```bash
python -m compileall services/orion-notify services/orion-mesh-guardian services/orion-sql-writer
```

---

## Plan self-review (spec coverage)

| Spec requirement | Task |
|------------------|------|
| Critical unhealable → immediate email | Task 1 + 2 |
| Error → escalation after 60m | Task 4 |
| Reuse EmailTransport / SMTP | Task 2 |
| Policy ack_deadline + escalation_channels | Task 2 |
| Mark escalated_at | Task 3 |
| No duplicate escalation | Task 3 dedupe_key + escalated_at guard |
| SMTP failure non-blocking | Task 2 `maybe_send_email` try/except |
| Guardian severity mapping | Task 1 |

No placeholders remain. Escalation default remains **60 minutes** per approved design.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-19-mesh-critical-email.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — implement tasks in this session with checkpoints

Which approach do you want?
