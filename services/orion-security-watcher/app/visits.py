from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional, Tuple

from .models import AlertPayload, SecurityState, VisionEvent, VisitSummary

logger = logging.getLogger("orion-security-watcher.visits")


class VisitManager:
    """
    v1 Visit + Alert manager with basic debouncing and size thresholds.

    - Treat any 'human-ish' detection as presence (face, presence, yolo:person),
      BUT only if:
        * it is big enough (bbox area threshold), AND
        * it persists across a minimum number of frames.
    - When system is ARMED, the first such event outside the global cooldown
      window triggers an alert.
    - We still emit a VisitSummary for bookkeeping, but keep it simple for now.
    """

    def __init__(self, settings):
        self.settings = settings

        # Last time we emitted an alert (for cooldown)
        self.last_alert_ts: Optional[datetime] = None

        # Simple visit tracking
        self.last_visit_id: Optional[str] = None
        self.last_visit_ts: Optional[datetime] = None

        # Cooldowns (seconds). Fallback defaults if not in settings.
        self.global_cooldown: int = getattr(settings, "SECURITY_GLOBAL_COOLDOWN_SEC", 300)
        self.identity_cooldown: int = getattr(
            settings, "SECURITY_IDENTITY_COOLDOWN_SEC", 600
        )

        # --- Noise resistance knobs ---

        # Minimum bounding-box area (px^2) to treat a FACE as valid human
        # e.g. 50x50 = 2500
        self.min_face_area: int = getattr(settings, "SECURITY_MIN_FACE_AREA", 2500)

        # Minimum area for YOLO 'person' detections
        self.min_person_area: int = getattr(settings, "SECURITY_MIN_PERSON_AREA", 2500)

        # Minimum YOLO confidence for a 'person' detection.
        # IMPORTANT: your YOLO detector runs with conf ~= 0.25 by default,
        # so keeping this at 0.50 would often suppress all YOLO humans.
        self.min_yolo_score: float = float(getattr(settings, "SECURITY_MIN_YOLO_SCORE", 0.30))

        # Require at least this many consecutive "human-ish" events before
        # we say humans_present=True for alert purposes.
        self.min_human_streak: int = getattr(settings, "SECURITY_MIN_HUMAN_STREAK", 3)

        # Max frame gap between human events to keep the streak going
        # (at 15 FPS, 45 frames ≈ 3 seconds).
        self.streak_max_gap: int = getattr(settings, "SECURITY_STREAK_MAX_FRAME_GAP", 45)

        # Internal streak state
        self._human_streak: int = 0
        self._last_human_frame_index: Optional[int] = None

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _raw_human_detection(self, ev: VisionEvent) -> bool:
        """
        Detect whether this single event *suggests* a human is present,
        ignoring streak/debouncing.

        v1 rules:
        - Any 'presence' detection with score >= 0.5
        - Any 'face' detection with score >= 0.5 and bbox area >= min_face_area
        - Any 'yolo' detection whose label is 'person' with score >= min_yolo_score
          and area >= min_person_area
        """
        for d in ev.detections:
            kind = (d.kind or "").lower()
            label = (d.label or "").lower()

            x, y, w, h = d.bbox
            area = max(0, w) * max(0, h)

            # Presence detector is explicit and already smoothed on the edge
            if kind == "presence" and d.score >= 0.5:
                return True

            # Face detection: we know a face is a human, but enforce area
            if kind == "face" and d.score >= 0.5 and area >= self.min_face_area:
                return True

            # YOLO: look for 'person' class with area + confidence gating
            if (
                kind == "yolo"
                and label == "person"
                and d.score >= self.min_yolo_score
                and area >= self.min_person_area
            ):
                return True

        return False

    def _update_human_streak(self, ev: VisionEvent, raw_humans: bool) -> bool:
        """
        Update the internal streak counters and return whether we now
        consider humans_present=True (debounced).
        """
        frame_index = ev.frame_index or 0

        if not raw_humans:
            # No human in this event → reset streak
            self._human_streak = 0
            self._last_human_frame_index = None
            return False

        # Human-ish evidence in this event
        if self._last_human_frame_index is None:
            # Start new streak
            self._human_streak = 1
        else:
            gap = frame_index - self._last_human_frame_index
            if gap <= self.streak_max_gap:
                self._human_streak += 1
            else:
                # Too long since last human frame → restart streak
                self._human_streak = 1

        self._last_human_frame_index = frame_index

        # Only treat as humans_present after enough consecutive-ish events
        return self._human_streak >= self.min_human_streak

    # ─────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────

    def process_event(
        self,
        event: VisionEvent,
        state: SecurityState,
    ) -> Tuple[Optional[VisitSummary], Optional[AlertPayload]]:
        """
        Process one vision event and return:

        - visit_summary: VisitSummary | None
        - alert_payload: AlertPayload | None

        v1 policy (with debouncing):

        - Use _raw_human_detection() to see if *this* event looks human-ish.
        - Feed that into streak logic; only when the streak is long enough
          do we treat humans_present=True.
        - Always build a visit summary when humans_present=True (even if not armed).
        - If ARMED and humans_present, and global cooldown allows,
          emit a high-severity alert with best_identity="unknown".
        """
        now = event.ts

        raw_humans = self._raw_human_detection(event)
        humans_present = self._update_human_streak(event, raw_humans)

        logger.debug(
            f"[VISIT] event ts={now} stream={event.stream_id} "
            f"armed={state.armed} raw_humans={raw_humans} "
            f"streak={self._human_streak} humans_present={humans_present}"
        )

        # Build a minimal visit summary if humans are (debounced) present
        visit_summary: Optional[VisitSummary] = None
        visit_id: Optional[str] = self.last_visit_id
        first_ts: Optional[datetime] = self.last_visit_ts

        if humans_present:
            if visit_id is None:
                visit_id = f"{event.stream_id}-{uuid.uuid4().hex[:8]}"
                first_ts = now

            self.last_visit_id = visit_id
            self.last_visit_ts = now

            visit_summary = VisitSummary(
                visit_id=visit_id,
                first_ts=first_ts or now,
                last_ts=now,
                stream_id=event.stream_id,
                humans_present=True,
                events=1,
            )

        # If not armed or no (debounced) humans, no alert
        if not state.armed or not humans_present:
            return visit_summary, None

        # Check global cooldown
        if self.last_alert_ts:
            delta = (now - self.last_alert_ts).total_seconds()
            if delta < self.global_cooldown:
                logger.info(
                    f"[VISIT] Alert suppressed by global cooldown "
                    f"({delta:.1f}s < {self.global_cooldown}s)"
                )
                return visit_summary, None

        # Build alert payload (identity layer is future; v1 = 'unknown')
        camera_id = event.meta.get("camera", event.stream_id)

        alert = AlertPayload(
            ts=now,
            service=self.settings.SERVICE_NAME,
            version=self.settings.SERVICE_VERSION,
            alert_id=f"{event.stream_id}-{now.strftime('%Y%m%dT%H%M%S')}",
            visit_id=visit_id or f"{event.stream_id}-single-{uuid.uuid4().hex[:6]}",
            camera_id=camera_id,
            armed=state.armed,
            mode=state.mode,
            humans_present=True,
            best_identity="unknown",
            best_identity_conf=0.0,
            identity_votes={"unknown": 1.0},
            reason="armed_human_detected",
            severity="high",
            snapshots=[],
            rate_limit={
                "global_blocked": False,
                "identity_blocked": False,
                "global_cooldown_sec": self.global_cooldown,
                "identity_cooldown_sec": self.identity_cooldown,
            },
        )

        self.last_alert_ts = now

        logger.info(
            f"[VISIT] Generated alert {alert.alert_id} "
            f"stream={event.stream_id} mode={state.mode}"
        )

        return visit_summary, alert
