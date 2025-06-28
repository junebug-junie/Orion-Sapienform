# perception/vision_daemon.py
import time
from emergence.perception.vision import VisionProcessor

class VisionDaemon:
    """
    Continuously captures snapshots at a fixed interval.
    Optionally listens for Redis trigger messages.
    """

    def __init__(self, interval=10):
        self.processor = VisionProcessor()
        self.interval = interval

    def run_loop(self):
        print("[VisionDaemon] Starting capture loop...")
        if not self.processor.wait_for_rtmp_connection():
            print("[VisionDaemon] RTMP stream unavailable.")
            return

        while True:
            self.processor.capture()
            time.sleep(self.interval)

if __name__ == "__main__":
    daemon = VisionDaemon(interval=10)
    daemon.run_loop()
