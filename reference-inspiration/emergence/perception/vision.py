from emergence.core.redis_bus import RedisBus
import time

def detect_bird(backbone):
    print("[Vision] Bird detected!")
    RedisBus.publish("memory", "rdf:bird:seen")
    RedisBus.publish("dream", "trigger:dream:bird")

if __name__ == "__main__":
    redis_bus = RedisBus()
    detect_bird(redis_bus)
