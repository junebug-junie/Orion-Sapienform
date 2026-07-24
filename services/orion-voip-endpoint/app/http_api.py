import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Callable, Optional

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .settings import Settings
from .asterisk_control import asterisk_cmd

logger = logging.getLogger("orion-voip-endpoint.http_api")


def build_heartbeat_chassis(settings: Settings) -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from `settings.bus_redis_url`
    (VOIP_BUS_REDIS_URL, this service's own command/status bus in bus_integration.py) --
    uses ORION_BUS_URL, matching this repo's general bus-native heartbeat convention (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md)."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=settings.orion_bus_url,
            bus_enabled=True,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )


def create_app(
    settings: Settings,
    bus_publish: Callable[[str], None],
    do_echo_call: Callable[[], dict],
    do_page_echo: Callable[[], dict],
) -> FastAPI:
    """
    Build FastAPI app with dependencies wired in from main.py.
    """

    app = FastAPI(title="Orion VoIP Endpoint", version="0.1.0")
    app.state.heartbeat_chassis = None

    @app.on_event("startup")
    async def _start_heartbeat() -> None:
        try:
            app.state.heartbeat_chassis = build_heartbeat_chassis(settings)
            await app.state.heartbeat_chassis.start_background()
            logger.info(
                "system_health_heartbeat_started service=%s interval_sec=%s",
                settings.service_name,
                settings.heartbeat_interval_sec,
            )
        except Exception as exc:
            logger.warning("system_health_heartbeat_start_failed error=%s", exc)
            app.state.heartbeat_chassis = None

    @app.on_event("shutdown")
    async def _stop_heartbeat() -> None:
        chassis: Optional[HeartbeatOnly] = app.state.heartbeat_chassis
        if chassis is not None:
            try:
                await chassis.stop()
            except Exception as exc:
                logger.warning("system_health_heartbeat_stop_error error=%s", exc)
            app.state.heartbeat_chassis = None

    @app.get("/health")
    def health():
        """
        Basic health: check that Asterisk responds.
        Also publishes a status ping to the bus.
        """
        try:
            result = asterisk_cmd("core show uptime")
            ok = result.returncode == 0
            bus_publish(
                "health",
                ok=ok,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
            )
            return JSONResponse(
                {
                    "ok": ok,
                    "lan_host_ip": str(settings.lan_host_ip),
                    "tailscale_host_ip": str(settings.tailscale_host_ip),
                    "sip_ext": settings.sip_ext,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                },
                status_code=200 if ok else 500,
            )
        except Exception as e:
            bus_publish("health_error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/echo-call")
    def echo_call():
        """
        Originate an echo test to the registered phone (ext).
        """
        try:
            return do_echo_call()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/page-echo")
    def page_echo():
        """
        Page the phone (auto-answer if configured) and route to echo test.
        """
        try:
            return do_page_echo()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

