# perception/system/system_monitor_daemon.py
import time
import json
from emergence.core.redis_bus import RedisBus
from emergence.perception.system.system_monitor import SystemMonitor

monitor = SystemMonitor()
bus = RedisBus()

HEARTBEAT_INTERVAL = 60  # seconds

def start_loop():
    print("[SystemMonitor] Daemon started.")
    while True:
        monitor.check_vitals()
        time.sleep(HEARTBEAT_INTERVAL)

def start_listener():
    def handle(msg):
        try:
            payload = json.loads(msg)
            if payload.get("command") == "check_vitals":
                monitor.check_vitals()
        except Exception as e:
            print(f"[SystemMonitor][Error] {e}")

    print("[SystemMonitor] Listening for on-demand triggers...")
    bus.subscribe("system:trigger", handle)
    bus.listen_forever()

if __name__ == "__main__":
    from threading import Thread
    Thread(target=start_loop).start()
    Thread(target=start_listener).start()

