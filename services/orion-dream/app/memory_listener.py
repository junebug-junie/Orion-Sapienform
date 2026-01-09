import asyncio
import json
import time
import uuid

from orion.core.bus.async_service import OrionBusAsync

from .settings import settings

PUBSUB_CHANNELS = [
    settings.CHANNEL_COLLAPSE_SQL_PUBLISH,
    settings.CHANNEL_COLLAPSE_TAGS_PUBLISH,
    settings.CHANNEL_TELEMETRY_PUBLISH,
    settings.CHANNEL_CHAT,
]

STREAMS = []  # keep empty unless you have XADD-based publishers
BUFFER = settings.CHANNEL_DREAM_BUFFER

def _now_ts():
    return time.time()

def _norm_payload(raw: dict) -> dict:
    # If it looks like a chat trace (prompt/response), merge into one text fragment
    if "prompt" in raw and "response" in raw:
        trace_id = raw.get("trace_id") or str(uuid.uuid4())
        text = f"User: {raw.get('prompt','').strip()}\nOrion: {raw.get('response','').strip()}"
        return {
            "id": trace_id,
            "kind": "chat",
            "ts": _now_ts(),
            "text": text,
            "tags": ["dialogue"],
        }
    # Otherwise pass through and ensure an id/ts/kind
    out = dict(raw)
    out.setdefault("id", str(uuid.uuid4()))
    out.setdefault("ts", _now_ts())
    out.setdefault("kind", out.get("kind", "event"))
    return out

async def _pubsub_loop(bus: OrionBusAsync) -> None:
    async with bus.subscribe(*PUBSUB_CHANNELS) as pubsub:
        print(f"[listener] Subscribed to Pub/Sub: {PUBSUB_CHANNELS}")
        async for msg in bus.iter_messages(pubsub):
            try:
                decoded = bus.codec.decode(msg.get("data", b""))
                raw = decoded.envelope.payload if decoded.ok else decoded.raw
                frag = _norm_payload(raw if isinstance(raw, dict) else {"raw": raw})
                await bus.redis.xadd(BUFFER, {"payload": json.dumps(frag)}, maxlen=5000)
            except Exception as e:
                print(f"❌ Pub/Sub normalize error: {e}")

async def _stream_loop(bus: OrionBusAsync) -> None:
    if not STREAMS:
        return
    last_ids = {s: "$" for s in STREAMS}
    print(f"[listener] XREAD streams: {STREAMS}")
    while True:
        try:
            results = await bus.redis.xread(last_ids, block=5000, count=10)
            for stream_name, messages in results:
                for msg_id, data in messages:
                    payload = {k.decode(): v.decode() for k, v in data.items()}
                    raw = json.loads(payload.get("payload")) if "payload" in payload else payload
                    frag = _norm_payload(raw)
                    await bus.redis.xadd(BUFFER, {"payload": json.dumps(frag)}, maxlen=5000)
                    last_ids[stream_name.decode()] = msg_id
        except Exception as e:
            print(f"❌ Stream mirror error: {e}")
            await asyncio.sleep(1)

async def mirror_to_buffer():
    bus = OrionBusAsync(settings.ORION_BUS_URL)
    await bus.connect()
    await asyncio.gather(_pubsub_loop(bus), _stream_loop(bus))

if __name__ == "__main__":
    asyncio.run(mirror_to_buffer())
