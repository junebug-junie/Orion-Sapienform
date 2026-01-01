import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, Clock
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.biometrics import BiometricsPayload
from app.metrics import collect_biometrics
from app.settings import settings

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(settings.SERVICE_NAME)

def chassis_cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        health_channel=settings.HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
        shutdown_timeout_sec=settings.SHUTDOWN_GRACE_SEC,
    )

async def publish_metrics(bus):
    if not settings.ORION_BUS_ENABLED:
        return

    try:
        data = collect_biometrics()
        payload = BiometricsPayload(
            timestamp=data["timestamp"],
            gpu=data["gpu"],
            cpu=data["cpu"],
            node=data["node"],
            service_name=data["service_name"],
            service_version=data["service_version"],
        )

        env = BaseEnvelope(
            kind="telemetry.biometrics",
            source=ServiceRef(
                name=settings.SERVICE_NAME,
                version=settings.SERVICE_VERSION,
                node=settings.NODE_NAME
            ),
            payload=payload.model_dump(mode="json"),
        )

        await bus.publish(settings.TELEMETRY_PUBLISH_CHANNEL, env)
        logger.debug(f"Published biometrics to {settings.TELEMETRY_PUBLISH_CHANNEL}")
    except Exception as e:
        logger.error(f"Failed to publish biometrics: {e}")

class BiometricsWorker(Clock):
    def __init__(self, cfg: ChassisConfig, *, interval_sec: float):
        # We pass a dummy tick because we override _run or tick usage
        super().__init__(cfg, interval_sec=interval_sec, tick=self.do_tick)

    async def do_tick(self):
        # This method is called by Clock._run
        # But Clock._run in base class calls self.tick()
        # So 'self.tick' is this method.
        # Inside here, we can access self.bus!
        await publish_metrics(self.bus)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION} (node={settings.NODE_NAME})")

    metrics_worker = BiometricsWorker(chassis_cfg(), interval_sec=settings.TELEMETRY_INTERVAL)
    await metrics_worker.start_background()

    yield

    await metrics_worker.stop()

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "telemetry_publish_channel": settings.TELEMETRY_PUBLISH_CHANNEL,
        "bus_url": settings.ORION_BUS_URL,
        "node": settings.NODE_NAME,
    }
