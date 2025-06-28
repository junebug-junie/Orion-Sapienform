import subprocess
import time
import socket
import json
from datetime import datetime
from pathlib import Path
import cv2

# --- Config ---
NODE_NAME = "carbon-x1"
RTMP_URL = "rtmp://192.168.0.2/live/stream"
LOG_DIR = Path.home() / "conjourney/vision_logs"
STREAM_TIMEOUT_SEC = 60
SNAPSHOT_INTERVAL_SEC = 10
CAMERA_ID = 0  # 0 for default RTMP input

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR / f"stream_log_{NODE_NAME}.json"

# --- Helpers ---
def log_event(event_type, extra=None):
    entry = {
        "observer": NODE_NAME,
        "event": event_type,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        entry.update(extra)
    with open(log_path, "a") as f:
        json.dump(entry, f)
        f.write("\n")
    print(f"[{event_type}] {entry['timestamp']}")


def wait_for_rtmp_connection(timeout=STREAM_TIMEOUT_SEC):
    print("🔍 Waiting for RTMP connection on port 1935...")
    for _ in range(timeout):
        try:
            with socket.create_connection(("127.0.0.1", 1935), timeout=1):
                log_event("gopro_stream_connected")
                return True
        except socket.error:
            time.sleep(1)
    log_event("gopro_stream_timeout")
    return False


def capture_snapshots():
    print("📸 Starting snapshot capture...")
    cap = cv2.VideoCapture(RTMP_URL)
    for i in range(30):
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                log_event("camera_stream_opened", {"attempt": i})
                break
        time.sleep(1)
    else:
        log_event("camera_open_failed")
        return


    log_event("camera_stream_opened")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            log_event("frame_capture_failed")
            break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = LOG_DIR / f"frame_{NODE_NAME}_{timestamp}.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        log_event("frame_captured", {"file": str(snapshot_path)})

        frame_count += 1
        time.sleep(SNAPSHOT_INTERVAL_SEC)

    cap.release()
    log_event("camera_stream_closed")


# --- Main Routine ---
if __name__ == "__main__":
    log_event("vision_node_start", {"stream_url": RTMP_URL})
    if wait_for_rtmp_connection():
        capture_snapshots()
    log_event("vision_node_stop")
