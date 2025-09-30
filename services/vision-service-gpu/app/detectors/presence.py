import time

class PresenceDetector:
    def __init__(self, timeout=60, label="Juniper"):
        self.timeout = timeout
        self.label = label
        self.present = False
        self.last_seen = 0

    def update(self, detections):
        now = time.time()
        presence_events = []

        # Look for person/face detections
        person_like = [d for d in detections if d["kind"] in ("yolo", "face")]

        if person_like:
            self.last_seen = now
            if not self.present:
                self.present = True
                presence_events.append({
                    "kind": "presence",
                    "state": "present",
                    "label": self.label,
                    "score": 1.0
                })
        else:
            if self.present and now - self.last_seen > self.timeout:
                self.present = False
                presence_events.append({
                    "kind": "presence",
                    "state": "absent",
                    "label": self.label,
                    "score": 1.0
                })

        return presence_events
