from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import List
import urllib.request

from .models import AlertPayload, AlertSnapshot
from .utils import redact_url
from orion.notify.client import NotifyClient
from orion.schemas.notify import NotificationAttachment, NotificationRequest

logger = logging.getLogger("orion-security-watcher.notifications")


class Notifier:
    """
    Handles:
    - Capturing snapshots from the vision edge service
    - Sending alerts via the notify service
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

        # Notify service config
        self.notify_service_url = getattr(settings, "NOTIFY_SERVICE_URL", "")
        self.notify_api_token = getattr(settings, "NOTIFY_API_TOKEN", "") or None


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

        if not self.notify_service_url:
            logger.error("[NOTIFY] NOTIFY_SERVICE_URL not configured; skipping email send")
            return

        safe_camera = redact_url(str(alert.camera_id))

        title = f"[Orion] Security alert: {alert.reason} ({alert.severity})"
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

        attachments = self._build_attachments(snapshots)

        request = NotificationRequest(
            source_service=self.settings.SERVICE_NAME,
            event_kind="security.alert",
            severity=alert.severity,
            title=title,
            body_text="\n".join(body_lines),
            context={
                "camera_id": safe_camera,
                "armed": alert.armed,
                "mode": alert.mode,
                "humans_present": alert.humans_present,
                "best_identity": alert.best_identity,
                "best_identity_conf": alert.best_identity_conf,
                "identity_votes": alert.identity_votes,
            },
            tags=["security", "alert"],
            recipient_group="juniper_primary",
            attachments=attachments or None,
        )

        client = NotifyClient(base_url=self.notify_service_url, api_token=self.notify_api_token)
        response = client.send(request)

        if response.ok:
            logger.info(f"[NOTIFY] Sent alert email via notify service ({len(snapshots)} attached)")
        else:
            logger.error(f"[NOTIFY] Failed to send email via notify service: {response.detail}")

    def _build_attachments(self, snapshots: List[AlertSnapshot]) -> List[NotificationAttachment]:
        attachments: List[NotificationAttachment] = []
        for snap in snapshots:
            try:
                path = Path(snap.path)
                attachments.append(
                    NotificationAttachment(
                        filename=path.name,
                        content_base64=base64.b64encode(path.read_bytes()).decode("utf-8"),
                        mime_type="image/jpeg",
                    )
                )
            except Exception as exc:
                logger.error(f"[NOTIFY] Failed to attach snapshot {snap.path}: {exc}")
        return attachments
