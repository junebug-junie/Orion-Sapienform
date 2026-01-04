import time
import logging
from collections import deque
from typing import Dict, List, Optional, Deque

from orion.schemas.vision import (
    VisionEdgeArtifact, VisionGuardSignal, VisionGuardAlert, VisionObject
)
from .settings import get_settings

logger = logging.getLogger("orion-security-watcher.guard")

class CameraBuffer:
    def __init__(self, window_seconds: float):
        self.window_seconds = window_seconds
        self.artifacts: Deque[VisionEdgeArtifact] = deque()
        self.last_signal_ts: float = 0.0

    def add(self, artifact: VisionEdgeArtifact):
        self.artifacts.append(artifact)
        self._prune()

    def _prune(self):
        now = time.time()
        while self.artifacts and (now - self.artifacts[0].timing.get("ts", now) > self.window_seconds + 5): # +5 buffer
            # Actually rely on timing['ts'] if available, else use current time?
            # Artifact doesn't mandate 'ts' in root, check timing.
            # Use received time if not present?
            # I'll rely on artifact timing if available.
            self.artifacts.popleft()

    def get_window(self, start_ts: float) -> List[VisionEdgeArtifact]:
        return [a for a in self.artifacts if a.timing.get("ts", 0) >= start_ts]

class VisionGuard:
    def __init__(self, settings):
        self.settings = settings
        self.buffers: Dict[str, CameraBuffer] = {}
        self.last_alert_ts: Dict[str, float] = {}

    def process_artifact(self, artifact: VisionEdgeArtifact) -> Optional[VisionGuardAlert]:
        """
        Ingest artifact, return Alert if triggered.
        Signal generation is done periodically via emit_signals().
        """
        cam_id = artifact.inputs.get("camera_id") or "unknown"
        if cam_id not in self.buffers:
            self.buffers[cam_id] = CameraBuffer(self.settings.GUARD_WINDOW_SECONDS)

        # Add to buffer
        # Ensure we have a timestamp
        if "ts" not in artifact.timing:
            artifact.timing["ts"] = time.time()

        self.buffers[cam_id].add(artifact)

        # Immediate alert check (optional, or rely on periodic)
        # Check for sustained presence
        return self._check_alert(cam_id)

    def _check_alert(self, cam_id: str) -> Optional[VisionGuardAlert]:
        # Logic:
        # Check last N seconds (GUARD_SUSTAIN_SECONDS)
        # If count of frames with "person" > threshold -> Alert

        now = time.time()
        buffer = self.buffers[cam_id]
        window = buffer.get_window(now - self.settings.GUARD_SUSTAIN_SECONDS)

        if not window:
            return None

        # Check for cooldown
        last = self.last_alert_ts.get(cam_id, 0)
        if now - last < self.settings.GUARD_ALERT_COOLDOWN_SECONDS:
            return None

        # Count positive frames
        positive_frames = 0
        evidence_ids = []

        for art in window:
            has_person = False
            if art.outputs.objects:
                for obj in art.outputs.objects:
                    # Logic: label=person, score > THRESHOLD
                    if obj.label == "person" and obj.score >= self.settings.GUARD_PERSON_MIN_CONF:
                        has_person = True
                        break

            if has_person:
                positive_frames += 1
                evidence_ids.append(art.artifact_id)

        # Alert Condition
        # If we have enough frames with person
        # E.g. 50% of frames in window? Or min count?
        # Prompt: "GUARD_PERSON_MIN_COUNT" -> maybe min count of detections per frame? Or count of frames?
        # Let's assume count of frames for robustness.
        # But wait, "GUARD_PERSON_MIN_COUNT" usually means "at least 1 person".
        # "GUARD_SUSTAIN_SECONDS" implies duration.
        # I'll require detection in > 50% of frames in sustain window, OR just > 1 frame?
        # Let's say: if positive_frames >= (FPS * sustain / 2)?
        # But we don't know FPS.
        # Let's just say if positive_frames >= 2 (to avoid single flicker) and cover > 0.5s range?

        if positive_frames >= self.settings.GUARD_PERSON_MIN_COUNT:
            self.last_alert_ts[cam_id] = now
            return VisionGuardAlert(
                camera_id=cam_id,
                ts=now,
                alert_type="person_presence",
                severity="high",
                summary=f"Person detected in {positive_frames} frames over {self.settings.GUARD_SUSTAIN_SECONDS}s",
                evidence_refs=evidence_ids
            )

        return None

    def generate_signals(self) -> List[VisionGuardSignal]:
        """
        Called periodically to emit status signals for all cameras.
        """
        signals = []
        now = time.time()

        for cam_id, buffer in self.buffers.items():
            # Rate limit signal emission per camera?
            if now - buffer.last_signal_ts < self.settings.GUARD_EMIT_EVERY_SECONDS:
                continue

            buffer.last_signal_ts = now

            # Analyze window (GUARD_WINDOW_SECONDS)
            window = buffer.get_window(now - self.settings.GUARD_WINDOW_SECONDS)
            if not window:
                # Unknown/Absent
                signals.append(VisionGuardSignal(
                    camera_id=cam_id,
                    window_start=now - self.settings.GUARD_WINDOW_SECONDS,
                    window_end=now,
                    decision="absent", # or unknown if no data? "absent" if we are receiving artifacts but empty.
                    # If we received artifacts (window not empty), we can say absent if no people.
                    # If window is empty (no artifacts), implies no connection or no frames?
                    # But get_window checks timestamp.
                    confidence=0.0,
                    summary={"msg": "No data"},
                    evidence_refs=[]
                ))
                continue

            # Check presence in window
            person_confs = []
            evidence_ids = []

            for art in window:
                if art.outputs.objects:
                    max_conf = 0.0
                    for obj in art.outputs.objects:
                        if obj.label == "person":
                            max_conf = max(max_conf, obj.score)
                    if max_conf > 0:
                        person_confs.append(max_conf)
                        if max_conf >= self.settings.GUARD_PERSON_MIN_CONF:
                            evidence_ids.append(art.artifact_id)

            # Decision
            decision = "absent"
            avg_conf = 0.0
            if person_confs:
                avg_conf = sum(person_confs) / len(person_confs)
                # If we have any strong detection?
                if any(c >= self.settings.GUARD_PERSON_MIN_CONF for c in person_confs):
                    decision = "presence"

            signals.append(VisionGuardSignal(
                camera_id=cam_id,
                window_start=now - self.settings.GUARD_WINDOW_SECONDS,
                window_end=now,
                decision=decision,
                confidence=avg_conf,
                summary={"person_detections": len(person_confs), "total_frames": len(window)},
                evidence_refs=evidence_ids,
                salience=avg_conf # simple map
            ))

        return signals
