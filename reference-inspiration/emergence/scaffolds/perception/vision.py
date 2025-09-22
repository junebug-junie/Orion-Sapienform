from core.redis_pubsub import RedisBackbone
import time

def detect_bird(backbone):
    print("[Vision] Bird detected!")
    backbone.publish("memory", "rdf:bird:seen")
    backbone.publish("dream", "trigger:dream:bird")

if __name__ == "__main__":
    backbone = RedisBackbone()
    detect_bird(backbone)
