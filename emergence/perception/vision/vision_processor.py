# perception/vision/vision_processor.py
import time
import socket
import json
from datetime import datetime
from pathlib import Path
import cv2
from emergence.core.redis_bus import RedisBus
from emergence.cognition.introspection.agent_introspector import AgentIntrospector

class VisionProcessor:
    """
    RTMP-aware vision processor with RedisBus signaling and modular capture pipeline.
    """

    def __init__(self, memory=None, node_name="vision-daemon", rtmp_url="rtmp://100.82.12.97/live/stream", snapshot_interval=10):
        self.node_name = node_name
        self.rtmp_url = rtmp_url
        self.snapshot_interval = snapshot_interval
        self.memory = memory
        self.bus = RedisBus()
        self.introspector = AgentIntrospector(name=node_name, memory=memory)

        self.log_dir = Path.home() / "conjourney/logs/vision"
        self.log_path = self.log_dir / f"stream_log_{self.node_name}.json"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type, extra=None):
        entry = {
            "observer": self.node_name,
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            entry.update(extra)
        with open(self.log_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        print(f"[{event_type}] {entry['timestamp']}")
        self.bus.publish("perception:vision:event", entry)

    def wait_for_rtmp_connection(self, timeout=60):
        self.log_event("waiting_for_stream", {"port": 1935})
        for _ in range(timeout):
            try:
                with socket.create_connection(("127.0.0.1", 1935), timeout=1):
                    self.log_event("gopro_stream_connected")
                    return True
            except socket.error:
                time.sleep(1)
        self.log_event("gopro_stream_timeout")
        return False

    def capture(self, context=None):
        """
        Start frame capture from RTMP stream.
        Returns most recent frame metadata (path, timestamp).
        """
        cap = cv2.VideoCapture(self.rtmp_url)
        for i in range(30):
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.log_event("camera_stream_opened", {"attempt": i})
                    break
            time.sleep(1)
        else:
            self.log_event("camera_open_failed")
            return None

        ret, frame = cap.read()
        if not ret:
            self.log_event("frame_capture_failed")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.log_dir / f"frame_{self.node_name}_{timestamp}.jpg"
        cv2.imwrite(str(snapshot_path), frame)

        result = {
            "observer": self.node_name,
            "file": str(snapshot_path),
            "timestamp": datetime.now().isoformat(),
            "source": self.rtmp_url
        }
        self.log_event("frame_captured", result)

        if self.memory:
            self.memory.store("vision_frame", result)

        # Agent introspection after successful capture
        self.introspector.reflect("Captured a visual moment from RTMP stream.", salience=0.65, extra={"frame_path": str(snapshot_path)})

        cap.release()
        return result

