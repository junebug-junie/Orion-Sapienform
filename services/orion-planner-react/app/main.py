from fastapi import FastAPI

from .settings import settings
from .api import router as planner_router


app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": settings.service_version,
    }


app.include_router(planner_router, prefix="")
