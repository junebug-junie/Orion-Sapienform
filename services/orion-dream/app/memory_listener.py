import asyncio, json, time
from redis import asyncio as aioredis
from .settings import settings

STREAMS = [
    settings.CHANNEL_COLLAPSE_SQL_PUBLISH

    settings.CHANNEL_RDF_CONFIRM
    settings.RDF_ERROR

    settings.CHANNEL_MEMORY_COLLAPSE,
    settings.CHANNEL_MEMORY_BIOMETRICS,
    settings.CHANNEL_MEMORY_RAG,

]

async def mirror_to_buffer():
    r = aioredis.from_url(settings.REDIS_URL)
    last_ids = {s: "$" for s in STREAMS}
    print("[memory_listener] Mirroring memory events to dream buffer…")
    while True:
        results = await r.xread(last_ids, block=5000, count=10)
        for stream_name, messages in results:
            for msg_id, data in messages:
                payload = {k.decode(): json.loads(v) if v.startswith(b"{") else v.decode() for k,v in data.items()}
                await r.xadd(settings.CHANNEL_DREAM_BUFFER, {"payload": json.dumps(payload)}, maxlen=5000)
                last_ids[stream_name.decode()] = msg_id

asyncio.run(mirror_to_buffer())
