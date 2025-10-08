from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import threading

from app import routes
from app.db import init_db, get_db
from app.chroma_db import embedder
from app.settings import settings
from orion.core.bus.service import OrionBus
from app.services.collapse_service import log_and_persist
from orion.schemas.collapse_mirror import CollapseMirrorEntry

load_dotenv()

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
)

app.include_router(routes.router, prefix="/api")

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
    }

@app.get("/")
def read_root():
    return {"message": "Conjourney Memory API is alive"}


# 👂 Intake listener
def listen_for_intake(bus: OrionBus):
    print("📡 Intake listener started, subscribing to collapse.intake ...")
    for msg in bus.subscribe("collapse.intake"):
        print("👂 Collapse intake:", msg)
        try:
            entry = CollapseMirrorEntry(**msg).with_defaults()  # ✅ enforce schema

            db = next(get_db())

            try:
                log_and_persist(entry=entry, db=db)
                print(f"Collapse persisted: {entry.summary[:10]}...")
            finally:
                db.close()

        except Exception as e:
            import traceback
            print(f"❌ Intake error: {e}")
            traceback.print_exc()
            print("⚠️ Error processing intake:", e)

# Model warmup
@app.on_event("startup")
async def startup_event():
    # Init DB + embedder warmup
    init_db()
    try:
        _ = embedder.encode("warmup").tolist()
        print("✅ Embedding model warmed up")
    except Exception as e:
        print("⚠️ Embedding warmup failed:", e)

    # Start intake listener in background thread
    bus = OrionBus(url=settings.ORION_BUS_URL)
    if bus.enabled:
        thread = threading.Thread(target=listen_for_intake, args=(bus,), daemon=True)
        thread.start()
        print("📡 Intake listener thread launched")
    else:
        print("⚠️ OrionBus disabled, intake listener not started")


# Allow local run without uvicorn cli
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8087, reload=True)
