from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Callable

from .settings import Settings
from .asterisk_control import asterisk_cmd


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

