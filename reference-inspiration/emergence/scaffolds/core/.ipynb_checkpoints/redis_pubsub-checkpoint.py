import redis
import threading

class RedisBackbone:
    def __init__(self, host='localhost', port=6379, channels=None):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self.pubsub = self.redis.pubsub()
        self.channels = channels or []
        self.thread = None

    def subscribe(self, callback):
        if self.channels:
            self.pubsub.subscribe(**{ch: callback for ch in self.channels})
            self.thread = self.pubsub.run_in_thread(sleep_time=0.001)

    def publish(self, channel, message):
        self.redis.publish(channel, message)

    def stop(self):
        if self.thread:
            self.thread.stop()
            self.pubsub.close()

# Example usage:
# backbone = RedisBackbone(channels=["vision", "audio"])
# backbone.publish("vision", "bird_seen")
