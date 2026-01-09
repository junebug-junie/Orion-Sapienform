import asyncio
import json
import threading
import traceback

from sqlalchemy.orm import Session

from orion.core.bus.async_service import OrionBusAsync
from orion.core.sql_router.db import SessionLocal

def default_writer(model_class, payload: dict, db: Session):
    obj = model_class(**payload)
    db.add(obj)
    db.commit()
    return obj

def start_worker_thread(
    bus_url: str,
    channel: str,
    model_class,
    writer_func=default_writer,
    publish_channel=None,
):
    async def worker_async() -> None:
        bus = OrionBusAsync(bus_url)
        await bus.connect()
        print(f"📡 Subscribed to {channel} (model={model_class.__name__})")
        try:
            async with bus.subscribe(channel) as pubsub:
                async for msg in bus.iter_messages(pubsub):
                    try:
                        data = msg.get("data")
                        if isinstance(data, (bytes, bytearray)):
                            data = data.decode("utf-8", "ignore")
                        if isinstance(data, str):
                            data = json.loads(data)
                        with SessionLocal() as db:
                            row = writer_func(model_class, data, db)
                            print(f"✅ Inserted row: {getattr(row, 'id', '[no id]')}")
                        if publish_channel:
                            await bus.publish(publish_channel, data)
                    except Exception as e:
                        print(f"❌ Error in worker: {e}")
                        traceback.print_exc()
        finally:
            await bus.close()

    def worker() -> None:
        asyncio.run(worker_async())
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread
