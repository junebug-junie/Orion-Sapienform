from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app import routes
from app.db import init_db
from app.chroma_db import embedder  # from your chroma.py
from app.settings import settings

load_dotenv()

app = FastAPI(
    title="Orion Collapse Mirror",
    version=settings.SERVICE_VERSION,
)

# Optional: CORS setup if SDK/frontend will call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount collapse mirror routes
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

# Model warmup
@app.on_event("startup")
async def warmup():
    init_db()
    try:
        _ = embedder.encode("warmup").tolist()
        print("✅ Embedding model warmed up")
    except Exception as e:
        print("⚠️ Embedding warmup failed:", e)

# Allow local run without uvicorn cli
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8087, reload=True)
