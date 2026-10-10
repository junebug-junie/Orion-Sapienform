from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .service import build_chassis
from .settings import settings


chassis = build_chassis()

ROLLUPS_RETIRED_DETAIL = (
    "spark-state rollups retired 2026-10-10: their input channel "
    "orion:spark:state:snapshot has had no producer since orion-spark-introspector "
    "was deleted 2026-07-28. Table spark_state_rollups is frozen history, not live state."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await chassis.start_background()
    try:
        yield
    finally:
        await chassis.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/rollups")
async def get_rollups() -> JSONResponse:
    # Marked absent on purpose: serving the frozen table here would present
    # 2026-07-28-era (or all-zero) values as if they were current state.
    return JSONResponse({"retired": True, "detail": ROLLUPS_RETIRED_DETAIL}, status_code=410)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
