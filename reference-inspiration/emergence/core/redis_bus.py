import redis
import json
from typing import Callable, Optional

class RedisBus:
    """
    General-purpose Redis Pub/Sub interface for broadcasting and listening across Conjourney mesh.
    """

    def __init__(self, host="localhost", port=6379, prefix="conjourney"):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self.prefix = prefix
        self.subscriptions = {}

    def _channel(self, topic: str) -> str:
        return f"{self.prefix}:{topic}"

    def publish(self, topic: str, message: dict):
        """
        Publish a JSON-encoded message to a topic.
        """
        channel = self._channel(topic)
        self.redis.publish(channel, json.dumps(message))

    def subscribe(self, topic: str, handler: Callable[[dict], None]):
        """
        Subscribe a handler function to a topic.
        """
        channel = self._channel(topic)
        self.subscriptions[channel] = handler

    def listen_forever(self):
        """
        Listen on all registered subscriptions and dispatch messages.
        """
        pubsub = self.redis.pubsub()
        pubsub.subscribe(*self.subscriptions.keys())

        print(f"[RedisBus] Listening on: {', '.join(self.subscriptions.keys())}")

        for msg in pubsub.listen():
            if msg["type"] != "message":
                continue

            channel = msg["channel"]
            payload = json.loads(msg["data"])

            if handler := self.subscriptions.get(channel):
                try:
                    handler(payload)
                except Exception as e:
                    print(f"[RedisBus] ⚠️ Error in handler for {channel}: {e}")

