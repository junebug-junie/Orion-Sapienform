import asyncio
import redis.asyncio as redis
import os

# Get connection details from environment variables
REDIS_URL = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
LISTEN_CHANNEL = os.getenv("CHANNEL_EVENTS_TRIAGE", "orion:collapse:triage")

async def main():
    print(f"Attempting to connect to Redis at {REDIS_URL}")
    try:
        async with redis.from_url(REDIS_URL, decode_responses=True) as client:
            print("Connection successful. Pinging server...")
            pong = await client.ping()
            print(f"Server responded: {pong}")

            async with client.pubsub() as pubsub:
                await pubsub.subscribe(LISTEN_CHANNEL)
                print(f"Successfully subscribed to '{LISTEN_CHANNEL}'. Waiting for a message...")

                # Wait for one message
                async for message in pubsub.listen():
                    print(f"Received message: {message}")
                    # After receiving one message, we're done.
                    break
        print("Test complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
