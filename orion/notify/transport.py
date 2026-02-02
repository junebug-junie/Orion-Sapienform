from __future__ import annotations

import base64
import logging
import smtplib
from email.message import EmailMessage
from typing import Iterable, List, Optional

from orion.schemas.notify import NotificationRequest

logger = logging.getLogger("orion.notify.transport")


class EmailTransport:
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_username: str,
        smtp_password: str,
        use_tls: bool,
        default_from: str,
        default_to: Iterable[str],
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.use_tls = use_tls
        self.default_from = default_from
        self.default_to = list(default_to)

    def send(
        self,
        request: NotificationRequest,
        *,
        to_override: Optional[List[str]] = None,
        from_override: Optional[str] = None,
    ) -> None:
        if not self.smtp_host:
            raise ValueError("SMTP host is not configured")

        from_email = from_override or self.default_from
        if not from_email:
            raise ValueError("No from address configured")

        recipients: List[str] = list(to_override or self.default_to)
        if not recipients:
            raise ValueError("No recipients configured")

        msg = EmailMessage()
        msg["Subject"] = request.title
        msg["From"] = from_email
        msg["To"] = ", ".join(recipients)
        msg.set_content(request.body_text or request.body_md or "")

        for attachment in request.attachments or []:
            try:
                data = base64.b64decode(attachment.content_base64)
            except Exception as exc:
                logger.error("[NOTIFY] Failed to decode attachment %s: %s", attachment.filename, exc)
                continue

            maintype, subtype = _split_mime(attachment.mime_type)
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=attachment.filename)

        if self.use_tls:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)


def _split_mime(mime_type: Optional[str]) -> tuple[str, str]:
    if not mime_type:
        return ("application", "octet-stream")
    parts = mime_type.split("/", 1)
    if len(parts) != 2:
        return ("application", "octet-stream")
    return (parts[0], parts[1])
