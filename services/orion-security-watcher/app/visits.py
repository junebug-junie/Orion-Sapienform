# app/visits.py
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, Optional

from .models import AlertPayload, AlertSnapshot, SecurityState, VisionEvent, VisitSummary
from .settings import Settings


class Visit:
    """
    Per-camera visit: a continuous period with human/motion activity.
    """

    def __init__(self, camera_id: str, event_ts: datetime):
        self.visit_id = str(uuid.uuid4())
        self.camera_id = camera_id
        self.started_at = event_ts
        self.last_seen_at = event_ts

        self.frames_seen: int = 0
        self.motion_frames: int = 0
        self.human_frames: int = 0

        self.identity_votes: Dict[str, float] = {}
        self.alert_sent: bool = False

    def observe(self, event: VisionEvent, settings: Settings) -> None:
        self.frames_seen += 1
        self.last_seen_at = event.ts

        # crude area filter if meta has width/height
        width = None
        height = None
        if event.meta:
            width = event.meta.get("width")
            height = event.meta.get("height")

        has_motion = False
        has_human_frame = False

        for det in event.detections:
            if det.kind == "motion":
                has_motion = True

            if det.kind in settings.human_kinds:
                score = det.score or 0.0
                if score < settings.HUMAN_MIN_SCORE:
                    continue

                # Optional bbox area filtering
                if det.bbox and width and height:
                    x, y, w, h = det.bbox
                    area = w * h
                    frac = area / float(width * height)
                    if frac < settings.MIN_BBOX_AREA_FRACTION:
                        continue

                has_human_frame = True

                # Identity labeling:
                # - presence with label: use that
                # - yolo "person" -> unknown
                label = det.label or "unknown"
                if det.kind == "yolo" and label == "person":
                    label = "unknown"

                # votes
                self.identity_votes[label] = self.identity_votes.get(label, 0.0) + score

        if has_motion:
            self.motion_frames += 1
        if has_human_frame:
            self.human_frames += 1

    def is_idle(self, now: datetime, settings: Settings) -> bool:
        return (now - self.last_seen_at).total_seconds() > settings.VISIT_IDLE_TIMEOUT_SEC

    def humans_present(self, settings: Settings) -> bool:
        return self.human_frames >= settings.MIN_PERSON_FRAMES

    def best_identity(self) -> tuple[str, float]:
        if not self.identity_votes:
            return "unknown", 0.0
        label = max(self.identity_votes, key=lambda k: self.identity_votes[k])
        total = sum(self.identity_votes.values()) or 1.0
        return label, self.identity_votes[label] / total


class VisitManager:
    """
    Manages visits per camera + rate limiting + alert/visit payloads.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.active_visits: Dict[str, Visit] = {}

        self.last_alert_ts: Optional[datetime] = None
        self.last_alert_by_identity: Dict[str, datetime] = {}

    def _get_visit(self, camera_id: str, event_ts: datetime) -> Visit:
        visit = self.active_visits.get(camera_id)
        if visit is None:
            visit = Visit(camera_id=camera_id, event_ts=event_ts)
            self.active_visits[camera_id] = visit
            return visit

        # if idle for too long, close and create new
        if visit.is_idle(event_ts, self.settings):
            self.active_visits[camera_id] = Visit(camera_id=camera_id, event_ts=event_ts)
        return self.active_visits[camera_id]

    def process_event(
        self,
        event: VisionEvent,
        state: SecurityState,
    ) -> tuple[Optional[VisitSummary], Optional[AlertPayload]]:
        """
        Process a single VisionEvent.
        Returns (visit_summary, alert_payload) when a visit ends; otherwise (None, None).
        """

        camera_id = event.stream_id
        if self.settings.camera_ids and camera_id not in self.settings.camera_ids:
            return None, None

        visit = self._get_visit(camera_id, event.ts)
        visit.observe(event, self.settings)

        # Decide if we should end the visit now.
        # We end visits when they go idle, but since we only see messages on detection,
        # we approximate by checking duration + last_seen.
        # We'll flush visits when a new one starts OR when explicitly idle.
        # Here we only flush if idle.
        now = event.ts
        if not visit.is_idle(now, self.settings):
            # keep visit open
            return None, None

        # At this point, we consider visit ended.
        visit_ended = visit
        # Remove from active
        self.active_visits.pop(camera_id, None)

        duration = (visit_ended.last_seen_at - visit_ended.started_at).total_seconds()
        humans_present = visit_ended.humans_present(self.settings)
        best_identity, best_conf = visit_ended.best_identity()

        # simple counts
        known_ids = [i for i in visit_ended.identity_votes.keys() if i in self.settings.allowed_identities]
        num_known = len(known_ids)
        num_unknown = len(visit_ended.identity_votes) - num_known
        if num_unknown < 0:
            num_unknown = 0

        decision = {
            "armed": state.armed,
            "mode": state.mode,
            "incident": False,
            "reason": None,
        }

        # Only consider incident if system armed + humans present + duration OK
        incident = False
        reason = None

        if state.armed and state.enabled and self.settings.SECURITY_ENABLED:
            if duration >= self.settings.MIN_VISIT_DURATION_SEC and humans_present:
                reason = "human_detected_while_armed"
                incident = True

        decision["incident"] = incident
        decision["reason"] = reason

        visit_summary = VisitSummary(
            ts=datetime.utcnow(),
            service=self.settings.SERVICE_NAME,
            version=self.settings.SERVICE_VERSION,
            visit_id=visit_ended.visit_id,
            camera_id=visit_ended.camera_id,
            started_at=visit_ended.started_at,
            ended_at=visit_ended.last_seen_at,
            duration_sec=duration,
            frames_seen=visit_ended.frames_seen,
            motion_frames=visit_ended.motion_frames,
            human_frames=visit_ended.human_frames,
            humans_present=humans_present,
            num_tracks=1,
            num_unknown_tracks=num_unknown if humans_present else 0,
            num_known_tracks=num_known if humans_present else 0,
            best_identity=best_identity,
            best_identity_conf=best_conf,
            identity_votes=visit_ended.identity_votes,
            security_decision=decision,
        )

        alert_payload: Optional[AlertPayload] = None

        if incident and not visit_ended.alert_sent:
            if self._can_send_alert(best_identity, datetime.utcnow()):
                rl_info = self._build_rate_limit_info(best_identity, datetime.utcnow(), will_send=True)
                alert_payload = AlertPayload(
                    ts=datetime.utcnow(),
                    service=self.settings.SERVICE_NAME,
                    version=self.settings.SERVICE_VERSION,
                    alert_id=str(uuid.uuid4()),
                    visit_id=visit_ended.visit_id,
                    camera_id=visit_ended.camera_id,
                    armed=state.armed,
                    mode=state.mode,
                    humans_present=humans_present,
                    best_identity=best_identity,
                    best_identity_conf=best_conf,
                    identity_votes=visit_ended.identity_votes,
                    reason=reason or "unknown",
                    severity="high",
                    snapshots=[],  # filled later by notifier
                    rate_limit=rl_info,
                )
                visit_ended.alert_sent = True
                self._record_alert(best_identity, datetime.utcnow())
            else:
                # rate-limited, no alert
                _ = self._build_rate_limit_info(best_identity, datetime.utcnow(), will_send=False)

        return visit_summary, alert_payload

    def _can_send_alert(self, identity: str, now: datetime) -> bool:
        # global
        if self.last_alert_ts is not None:
            if (now - self.last_alert_ts).total_seconds() < self.settings.SECURITY_GLOBAL_COOLDOWN_SEC:
                return False

        key = identity or "unknown"
        last = self.last_alert_by_identity.get(key)
        if last is not None:
            if (now - last).total_seconds() < self.settings.SECURITY_IDENTITY_COOLDOWN_SEC:
                return False

        return True

    def _record_alert(self, identity: str, now: datetime) -> None:
        self.last_alert_ts = now
        key = identity or "unknown"
        self.last_alert_by_identity[key] = now

    def _build_rate_limit_info(self, identity: str, now: datetime, will_send: bool) -> dict:
        info = {
            "global_blocked": False,
            "identity_blocked": False,
            "global_cooldown_sec": self.settings.SECURITY_GLOBAL_COOLDOWN_SEC,
            "identity_cooldown_sec": self.settings.SECURITY_IDENTITY_COOLDOWN_SEC,
        }
        if self.last_alert_ts is not None:
            if (now - self.last_alert_ts).total_seconds() < self.settings.SECURITY_GLOBAL_COOLDOWN_SEC:
                info["global_blocked"] = True
        key = identity or "unknown"
        last = self.last_alert_by_identity.get(key)
        if last is not None:
            if (now - last).total_seconds() < self.settings.SECURITY_IDENTITY_COOLDOWN_SEC:
                info["identity_blocked"] = True
        return info
