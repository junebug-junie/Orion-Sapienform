# perception/vision_trigger_listener.py
import asyncio
from emergence.core.redis_bus import RedisBus
from emergence.perception.vision.vision_processor import VisionProcessor

class VisionTriggerListener:
    def __init__(self, interval=None):
        self.bus = RedisBus()
        self.vision = VisionProcessor(
            node_name="chrysalis",
            rtmp_url="rtmp://100.82.12.97/live/stream"
        )
        self.bus.subscribe("perception:vision:trigger", self.handle_trigger)

    def handle_trigger(self, message):
        command = message.get("command")
        if command == "snap":
            print("[Trigger] Snap command received.")
            self.vision.capture()
        else:
            print(f"[Trigger] Unknown command: {command}")

    def listen(self):
        print("[TriggerListener] Listening for snap commands...")
        self.bus.listen_forever()

if __name__ == "__main__":
    listener = VisionTriggerListener()
    listener.listen()
