# orion-actions

A minimal **Actions** service (v1) that listens to bus events and orchestrates deterministic action pipelines.

## v1 behavior

Action catalog contains **one** action:
- `respond_to_juniper_collapse_mirror.v1`

Rule:
- Consume `orion:collapse:triage` events.
- Parse payload as `CollapseMirrorEntryV2`.
- If `entry.observer.lower() == "juniper"`, execute the action.

Pipeline:
1) Call RecallService via bus RPC (`recall.query.v1`) using `RecallQueryV1(fragment=...)`.
2) Call LLMGatewayService via bus RPC (`llm.chat.request`).
3) Deliver the final message via **orion-notify** using `NotifyClient.send(NotificationRequest)` with `event_kind="orion.chat.message"`.
4) Emit optional telemetry to `orion:actions:audit` (`kind="actions.audit.v1"`).

## Why notify
`orion-notify` already handles policy (quiet hours, throttle, escalation, dedupe) and publishes in-app events to the hub via `orion:notify:in_app`. Actions should actuate through notify, not invent a new hub push channel.

## Guardrails
- Dedupe uses a small in-memory TTL cache keyed by `entry.event_id` (fallback `entry.id`).
- Concurrency is capped with an asyncio semaphore.
- Kind drift tolerant: subscription is by channel; handler validates payload schema and does not depend on `env.kind`.

## Environment
See `.env_example` for all knobs.

## Health
- `GET /health`
