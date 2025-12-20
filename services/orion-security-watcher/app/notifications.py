from __future__ import annotations

import logging
import os
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import List
import smtplib
import urllib.request

from .models import AlertPayload, AlertSnapshot
from .utils import redact_url

logger = logging.getLogger("orion-security-watcher.notifications")


class Notifier:
    """
    Handles:
    - Capturing snapshots from the vision edge service
    - Sending email alerts with optional attachments
    """

    def __init__(self, settings):
        self.settings = settings

        # Snapshot config
        self.snapshot_url = getattr(settings, "VISION_SNAPSHOT_URL", None)
        self.snapshot_public_url = getattr(settings, "VISION_SNAPSHOT_PUBLIC_URL", "") or ""

        self.snapshot_dir = Path(
            getattr(settings, "SECURITY_SNAPSHOT_DIR", "/mnt/telemetry/orion-security/alerts")
        )
        self.snapshot_count = int(getattr(settings, "SECURITY_SNAPSHOT_COUNT", 3))

        # Ensure directory exists
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"[NOTIFY] Failed to create snapshot dir {self.snapshot_dir}: {e}")

        # Notification mode
        self.mode = getattr(settings, "NOTIFY_MODE", "off")

        # SMTP config
        self.smtp_host = getattr(settings, "NOTIFY_EMAIL_SMTP_HOST", "")
        self.smtp_port = int(getattr(settings, "NOTIFY_EMAIL_SMTP_PORT", 587))
        self.smtp_user = getattr(settings, "NOTIFY_EMAIL_SMTP_USERNAME", "")
        self.smtp_pass = getattr(settings, "NOTIFY_EMAIL_SMTP_PASSWORD", "")
        self.smtp_use_tls = bool(getattr(settings, "NOTIFY_EMAIL_USE_TLS", True))

        # Email headers
        self.email_from = getattr(settings, "NOTIFY_EMAIL_FROM", "")
        to_raw = getattr(settings, "NOTIFY_EMAIL_TO", "") or ""
        self.email_to = [e.strip() for e in to_raw.split(",") if e.strip()]

    # ─────────────────────────────────────────────
    # Snapshot capture
    # ─────────────────────────────────────────────

    def capture_snapshots(self, alert: AlertPayload) -> List[AlertSnapshot]:
        """
        Pull frames from the vision edge /snapshot.jpg endpoint and save to disk.

        If snapshot URL or directory are misconfigured, returns [].
        """
        if not self.snapshot_url:
            logger.info("[NOTIFY] No VISION_SNAPSHOT_URL configured; skipping snapshots")
            return []

        if self.snapshot_count <= 0:
            logger.info("[NOTIFY] SECURITY_SNAPSHOT_COUNT <= 0; skipping snapshots")
            return []

        snapshots: List[AlertSnapshot] = []
        safe_url = redact_url(self.snapshot_url)

        for i in range(self.snapshot_count):
            try:
                logger.debug(f"[NOTIFY] Fetching snapshot {i+1}/{self.snapshot_count} from {safe_url}")
                with urllib.request.urlopen(self.snapshot_url, timeout=5) as resp:
                    if getattr(resp, "status", 200) != 200:
                        logger.error(f"[NOTIFY] Snapshot HTTP {getattr(resp, 'status', '???')} from {safe_url}")
                        continue
                    data = resp.read()
            except Exception as e:
                logger.error(f"[NOTIFY] Error fetching snapshot from {safe_url}: {e}")
                continue

            ts = datetime.utcnow()
            filename = f"{alert.alert_id}_snap{i+1}_{int(ts.timestamp())}.jpg"
            path = self.snapshot_dir / filename

            try:
                with path.open("wb") as f:
                    f.write(data)
                snapshots.append(AlertSnapshot(ts=ts, path=str(path)))
                logger.info(f"[NOTIFY] Saved snapshot to {path}")
            except Exception as e:
                logger.error(f"[NOTIFY] Failed to write snapshot file {path}: {e}")

        if not snapshots:
            logger.warning("[NOTIFY] No snapshots captured (all attempts failed)")

        return snapshots

    # ─────────────────────────────────────────────
    # Email sending
    # ─────────────────────────────────────────────

    def send_email(self, alert: AlertPayload, snapshots: List[AlertSnapshot]) -> None:
        """
        Send an email with alert details and optional snapshot attachments.

        IMPORTANT: redact URLs to avoid leaking user:pass@ in camera IDs.
        """
        if self.mode != "inline":
            logger.info(f"[NOTIFY] NOTIFY_MODE={self.mode}; skipping email send")
            return

        if not (self.smtp_host and self.email_from and self.email_to):
            logger.error(
                "[NOTIFY] NOTIFY_EMAIL_SMTP_HOST / NOTIFY_EMAIL_FROM / NOTIFY_EMAIL_TO "
                "not fully configured; skipping email send"
            )
            return

        safe_camera = redact_url(str(alert.camera_id))

        subject = f"[Orion] Security alert: {alert.reason} ({alert.severity})"
        body_lines = [
            f"Time (UTC): {alert.ts.isoformat()}",
            f"Camera: {safe_camera}",
            f"Armed: {alert.armed}",
            f"Mode: {alert.mode}",
            f"Humans present: {alert.humans_present}",
            "",
            f"Best identity: {alert.best_identity} (conf={alert.best_identity_conf:.2f})",
            f"Identity votes: {alert.identity_votes}",
            "",
        ]

        # Optional safe/public snapshot URL (no creds)
        if self.snapshot_public_url.strip():
            body_lines.append(f"Snapshot URL: {self.snapshot_public_url.strip()}")
            body_lines.append("")

        if snapshots:
            body_lines.append(f"{len(snapshots)} snapshot(s) captured and attached.")
        else:
            body_lines.append("No snapshots captured for this alert.")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.email_from
        msg["To"] = ", ".join(self.email_to)
        msg.set_content("\n".join(body_lines))

        # Attach JPEGs
        for snap in snapshots:
            try:
                with open(snap.path, "rb") as f:
                    data = f.read()
                filename = os.path.basename(snap.path)
                msg.add_attachment(data, maintype="image", subtype="jpeg", filename=filename)
            except Exception as e:
                logger.error(f"[NOTIFY] Failed to attach snapshot {snap.path}: {e}")

        try:
            if self.smtp_use_tls:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    if self.smtp_user and self.smtp_pass:
                        server.login(self.smtp_user, self.smtp_pass)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    if self.smtp_user and self.smtp_pass:
                        server.login(self.smtp_user, self.smtp_pass)
                    server.send_message(msg)

            logger.info(f"[NOTIFY] Sent alert email to {self.email_to} ({len(snapshots)} attached)")
        except Exception as e:
            logger.error(f"[NOTIFY] Failed to send email: {e}")
