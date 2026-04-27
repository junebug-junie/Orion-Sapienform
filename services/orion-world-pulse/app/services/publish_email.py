from __future__ import annotations

import logging

from orion.notify.client import NotifyClient
from orion.schemas.notify import NotificationRequest
from orion.schemas.world_pulse import EmailWorldPulseRenderV1

logger = logging.getLogger("orion-world-pulse.email")


def publish_email_preview(*, notify_client: NotifyClient, email: EmailWorldPulseRenderV1, enabled: bool) -> bool:
    if not enabled:
        logger.info("world_pulse_email_publish_result run_id=%s status=disabled", email.run_id)
        return False
    req = NotificationRequest(
        source_service="orion-world-pulse",
        event_kind="orion.world.pulse.email",
        severity="info",
        title=email.subject,
        body_text=email.plaintext_body,
        body_md=email.plaintext_body,
        recipient_group="juniper_primary",
        channels_requested=["email"],
        dedupe_key=f"world-pulse:email:{email.run_id}",
    )
    accepted = notify_client.send(req)
    logger.info("world_pulse_email_publish_result run_id=%s status=%s", email.run_id, "ok" if accepted.ok else "failed")
    return bool(accepted.ok)
