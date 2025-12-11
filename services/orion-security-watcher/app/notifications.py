# app/notifications.py
from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import List

import requests

from .models import AlertPayload, AlertSnapshot
from .settings import Settings


class Notifier:
    """
    Inline notification handler (email). Can be turned off via NOTIFY_MODE / NOTIFY_EMAIL_ENABLED.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_dir = Path(settings.SECURITY_SNAPSHOT_DIR)

    def capture_snapshots(self, alert: AlertPayload) -> List[AlertSnapshot]:
        """
        Pulls snapshots from the vision snapshot endpoint and writes them to disk.
        Returns metadata for inclusion in alert payload.
        """
        snapshots: List[AlertSnapshot] = []
        self.base_dir.mkdir(parents=True, exist_ok=True)

        alert_dir = self.base_dir / alert.alert_id
        alert_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.settings.SECURITY_SNAPSHOT_COUNT):
            ts = datetime.utcnow()
            try:
                resp = requests.get(self.settings.VISION_SNAPSHOT_URL, timeout=5)
                resp.raise_for_status()
                filename = f"{alert.camera_id}_{alert.visit_id}_{i}_{ts.strftime('%Y%m%dT%H%M%SZ')}.jpg"
                full_path = alert_dir / filename
                full_path.write_bytes(resp.content)

                snapshots.append(
                    AlertSnapshot(
                        kind="snapshot",
                        captured_at=ts,
                        filename=str(full_path),
                        url=None,
                        note="local snapshot; no cloud URL yet",
                    )
                )
            except Exception:
                # best effort; continue
                continue

        return snapshots

    def send_email(self, alert: AlertPayload, snapshots: List[AlertSnapshot]) -> None:
        if not (self.settings.NOTIFY_EMAIL_ENABLED and self.settings.NOTIFY_EMAIL_SMTP_HOST and self.settings.NOTIFY_EMAIL_TO):
            return

        msg = EmailMessage()
        subject = f"[ORION SECURITY] Alert on {alert.camera_id} at {alert.ts.isoformat()}"
        msg["Subject"] = subject
        msg["From"] = self.settings.NOTIFY_EMAIL_FROM
        msg["To"] = self.settings.NOTIFY_EMAIL_TO

        body_lines = [
            "Orion Security Alert",
            "",
            f"Camera: {alert.camera_id}",
            f"Alert ID: {alert.alert_id}",
            f"Visit ID: {alert.visit_id}",
            f"Time (UTC): {alert.ts.isoformat()}",
            f"Armed: {alert.armed}",
            f"Mode: {alert.mode}",
            "",
            f"Humans present: {alert.humans_present}",
            f"Best identity: {alert.best_identity} (conf={alert.best_identity_conf:.2f})",
            f"Reason: {alert.reason}",
            "",
            "Identity votes:",
        ]
        for k, v in alert.identity_votes.items():
            body_lines.append(f"  - {k}: {v:.2f}")
        body_lines.append("")
        body_lines.append(f"Snapshots attached: {len(snapshots)}")

        msg.set_content("\n".join(body_lines))

        # Attach snapshots
        for snap in snapshots:
            if not snap.filename:
                continue
            try:
                data = Path(snap.filename).read_bytes()
            except Exception:
                continue
            name = os.path.basename(snap.filename)
            msg.add_attachment(
                data,
                maintype="image",
                subtype="jpeg",
                filename=name,
            )

        try:
            if self.settings.NOTIFY_EMAIL_USE_TLS:
                with smtplib.SMTP(self.settings.NOTIFY_EMAIL_SMTP_HOST, self.settings.NOTIFY_EMAIL_SMTP_PORT) as s:
                    s.starttls()
                    if self.settings.NOTIFY_EMAIL_SMTP_USERNAME:
                        s.login(
                            self.settings.NOTIFY_EMAIL_SMTP_USERNAME,
                            self.settings.NOTIFY_EMAIL_SMTP_PASSWORD,
                        )
                    s.send_message(msg)
            else:
                with smtplib.SMTP(self.settings.NOTIFY_EMAIL_SMTP_HOST, self.settings.NOTIFY_EMAIL_SMTP_PORT) as s:
                    if self.settings.NOTIFY_EMAIL_SMTP_USERNAME:
                        s.login(
                            self.settings.NOTIFY_EMAIL_SMTP_USERNAME,
                            self.settings.NOTIFY_EMAIL_SMTP_PASSWORD,
                        )
                    s.send_message(msg)
        except Exception:
            # Don't crash service on email failure
            pass
