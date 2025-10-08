import json, threading, traceback
from sqlalchemy.orm import Session
from orion.core.sql_router.db import SessionLocal
from orion.core.bus import OrionBus

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
    def worker():
        bus = OrionBus(url=bus_url)
        print(f"📡 Subscribed to {channel} (model={model_class.__name__})")
        for msg in bus.subscribe(channel):
            try:
                data = json.loads(msg) if isinstance(msg, (bytes, str)) else msg
                with SessionLocal() as db:
                    row = writer_func(model_class, data, db)
                    print(f"✅ Inserted row: {getattr(row, 'id', '[no id]')}")
                if publish_channel:
                    bus.publish(publish_channel, data)
            except Exception as e:
                print(f"❌ Error in worker: {e}")
                traceback.print_exc()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread
