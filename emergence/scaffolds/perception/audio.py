from core.redis_pubsub import RedisBackbone
import time

def detect_cat(backbone):
    print("[Audio] Heard the word 'cat'!")
    backbone.publish("memory", "rdf:cat:heard")
    backbone.publish("dream", "trigger:dream:cat")