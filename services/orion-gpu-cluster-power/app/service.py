# app/service.py

import logging
from typing import Optional

import httpx

from .settings import Settings, settings
from orion.core.bus.service import OrionBus

logger = logging.getLogger("gpu-cluster-power.service")


class PsuService:
    """
    Core PSU proxy behavior:
      - POSTs to the DYDTech PSU board /power endpoint
      - Publishes events to Orion bus (if enabled)
      - Handles commands from bus (on/off/cycle)
    """

    def __init__(self, cfg: Settings):
        self.settings = cfg
        self.bus: Optional[OrionBus] = None

    # ------- Bus state -------

    @property
    def bus_enabled(self) -> bool:
        return bool(self.bus and self.bus.enabled)

    async def init_bus(self) -> None:
        """
        Initialize OrionBus using ORION_BUS_URL / ORION_BUS_ENABLED env vars.
        OrionBus reads ORION_BUS_URL from env if url=None.
        """
        if not self.settings.orion_bus_enabled:
            logger.info("OrionBus disabled via settings; skipping init")
            return

        self.bus = OrionBus(
            url=None,
            enabled=self.settings.orion_bus_enabled,
        )

        if not self.bus.enabled:
            logger.warning("OrionBus failed to enable; running HTTP-only.")
        else:
            logger.info("OrionBus initialized for PSU proxy at %s", self.bus.url)

    # ------- HTTP helpers -------

    def _build_url(self, path: str) -> str:
        base = self.settings.psu_base_url.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return f"{base}{path}"

    async def _emit_event(self, action: str, status_code: int, body: str) -> None:
        """
        Publish a PSU event to the bus, if enabled.
        """
        if not self.bus_enabled:
            return

        payload = {
            "service": self.settings.service_name,
            "node": self.settings.node_name,
            "action": action,
            "status_code": status_code,
            "body": body,
        }

        try:
            # OrionBus.publish(channel: str, message: dict)
            self.bus.publish(self.settings.bus_channel_psu_events, payload)
        except Exception as e:
            logger.warning("Failed to publish PSU event: %s", e)

    async def _post_power(self, path: str, payload: dict, action: str) -> dict:
        """
        Core HTTP call to the PSU board.

        The board expects:
          POST /power
          Content-Type: application/json
          Body: {"power": 0|1, "mode": 0..7 or 255}
        """
        url = self._build_url(path)
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)

        body = r.text
        logger.info("PSU %s → %s (%s)", action, r.status_code, body[:200])
        await self._emit_event(action, r.status_code, body)

        return {"action": action, "status_code": r.status_code, "body": body}

    # ------- Public actions (used by HTTP + bus) -------

    async def on(self) -> dict:
        # Match panel behavior: mode=0, power=1 → ON
        payload = {"power": 1, "mode": 0}
        return await self._post_power(self.settings.psu_on_path, payload, "on")

    async def off(self) -> dict:
        # mode=0, power=0 → OFF
        payload = {"power": 0, "mode": 0}
        return await self._post_power(self.settings.psu_off_path, payload, "off")

    async def cycle(self, mode: int = 0) -> dict:
        """
        Configure the auto-cycle interval.

        mode values (from the HTML):
          0: no set
          1: 2 mins
          2: 30 mins
          3: 1 hour
          4: 3 hour
          5: 6 hour
          6: 12 hour
          7: 24 hour
        """
        payload = {"power": 0, "mode": mode}
        return await self._post_power(
            self.settings.psu_cycle_path,
            payload,
            f"cycle:{mode}",
        )

    async def handle_command(self, action: str, mode: Optional[int] = None) -> None:
        """
        Execute a PSU action coming from the bus.
        If mode is not provided for 'cycle', default to 0 (no set).
        """
        if action == "on":
            await self.on()
        elif action == "off":
            await self.off()
        elif action == "cycle":
            await self.cycle(mode or 0)
        else:
            logger.warning("Unknown PSU action requested: %s", action)
