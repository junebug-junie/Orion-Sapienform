from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests
import urllib3

logger = logging.getLogger("orion.biometrics.ilo")

# iLO/BMC RedFish endpoints use a self-signed cert by default on nearly every
# deployment (HPE iLO, Gigabyte BMC alike) -- verify=False below is intentional,
# not an oversight. Suppress the resulting per-request InsecureRequestWarning
# once at import instead of it spamming every poll tick.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class IloSnapshot:
    fetched_at: float = 0.0
    thermal_c: Dict[str, float] = field(default_factory=dict)
    fan_pct: Dict[str, float] = field(default_factory=dict)
    power_watts: Optional[float] = None
    error: Optional[str] = None


def fetch_ilo_snapshot(
    host: str, username: str, password: str, timeout_sec: float = 8.0
) -> IloSnapshot:
    """One blocking RedFish pull of thermal/fan/power data from an iLO/BMC.

    Standard DMTF RedFish (`/redfish/v1/Chassis/` -> first chassis member's
    `/Thermal/` and `/Power/`) rather than hardcoding an HPE-specific chassis
    path -- confirmed live against athena's iLO 2026-07-25 (real, non-degenerate
    temps/fans/power). Should generalize to other RedFish-compliant BMCs (e.g.
    circe's Gigabyte variant) without code changes, though that's unverified
    until credentials for a second vendor are available.

    Not fatal on any failure -- returns an IloSnapshot with `.error` set so the
    caller (a slow-poll background loop, not the heartbeat's own bounded hook --
    see IloPoller below) can just log and retry next interval.
    """
    if not (host and username and password):
        return IloSnapshot(fetched_at=time.time(), error="not_configured")

    base = host.rstrip("/")
    session = requests.Session()
    session.verify = False
    session.auth = (username, password)
    # Confirmed live against athena's iLO 2026-07-25: requests' default headers
    # (gzip/deflate Accept-Encoding, keep-alive) cause this BMC's embedded HTTP
    # server to abort the connection mid-handshake ("Remote end closed connection
    # without response") even though plain curl and a raw TLS socket both work
    # fine against the same host -- explicit identity encoding + connection-close
    # avoids whatever iLO's web server mishandles.
    session.headers.update(
        {"Accept-Encoding": "identity", "Connection": "close", "OData-Version": "4.0"}
    )
    try:
        chassis_resp = session.get(f"{base}/redfish/v1/Chassis/", timeout=timeout_sec)
        chassis_resp.raise_for_status()
        members = chassis_resp.json().get("Members") or []
        if not members:
            return IloSnapshot(fetched_at=time.time(), error="no_chassis_members")
        chassis_path = members[0]["@odata.id"].rstrip("/")

        thermal_c: Dict[str, float] = {}
        fan_pct: Dict[str, float] = {}
        power_watts: Optional[float] = None

        thermal_resp = session.get(f"{base}{chassis_path}/Thermal/", timeout=timeout_sec)
        if thermal_resp.ok:
            data = thermal_resp.json()
            for t in data.get("Temperatures") or []:
                if (t.get("Status") or {}).get("State") != "Enabled":
                    # "Absent" sensors report ReadingCelsius=0 for an unpopulated
                    # slot -- not real data, skip rather than record a fake zero.
                    continue
                reading = t.get("ReadingCelsius")
                name = t.get("Name")
                if reading is not None and name:
                    thermal_c[name] = float(reading)
            for fan in data.get("Fans") or []:
                if (fan.get("Status") or {}).get("State") != "Enabled":
                    continue
                reading = fan.get("Reading")
                name = fan.get("Name")
                # RedFish fans can report Percent or RPM depending on vendor --
                # confirmed live only against athena's HPE iLO (Percent). Only
                # HPE/athena is verified so far; skip (don't mislabel) rather than
                # record an RPM value under a field named/typed as a percentage
                # once a second vendor (e.g. circe's Gigabyte BMC) is wired in.
                units = fan.get("ReadingUnits")
                if reading is not None and name and units == "Percent":
                    fan_pct[name] = float(reading)

        power_resp = session.get(f"{base}{chassis_path}/Power/", timeout=timeout_sec)
        if power_resp.ok:
            controls = power_resp.json().get("PowerControl") or []
            if controls:
                watts = controls[0].get("PowerConsumedWatts")
                if watts is not None:
                    power_watts = float(watts)

        return IloSnapshot(
            fetched_at=time.time(),
            thermal_c=thermal_c,
            fan_pct=fan_pct,
            power_watts=power_watts,
        )
    except Exception as exc:
        return IloSnapshot(fetched_at=time.time(), error=str(exc))


class IloPoller:
    """Background slow-poll loop, decoupled from the fast SystemHealthV1
    heartbeat cadence.

    The chassis's `heartbeat_details` hook is bounded to roughly half the
    heartbeat interval (commonly ~3s) and runs synchronously in a worker thread
    on every heartbeat tick -- a live network round-trip to an out-of-band BMC
    (which is a much weaker processor than the host CPU, not built for
    high-frequency polling) does not reliably fit that budget, and a slow iLO
    call would risk silently dropping the *other* heartbeat_details producer
    (disk-capacity telemetry) sharing that same bounded hook. Instead this
    poller runs on its own `ILO_POLL_INTERVAL_SEC` cadence (default 60s) and
    caches the last result; `details()` just reads the cache, which is instant.
    """

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        interval_sec: float = 60.0,
        timeout_sec: float = 8.0,
    ) -> None:
        self._host = host
        self._username = username
        self._password = password
        self._interval_sec = interval_sec
        self._timeout_sec = timeout_sec
        self._snapshot = IloSnapshot(
            error=None if self.enabled else "not_configured"
        )
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return bool(self._host and self._username and self._password)

    async def start_background(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop(), name="ilo-poller")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except Exception:
                self._task.cancel()
            self._task = None

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._snapshot = await asyncio.to_thread(
                    fetch_ilo_snapshot,
                    self._host,
                    self._username,
                    self._password,
                    self._timeout_sec,
                )
                if self._snapshot.error:
                    logger.warning("ilo_poll_failed error=%s", self._snapshot.error)
            except Exception as exc:
                logger.warning("ilo_poll_loop_error error=%s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_sec)
            except asyncio.TimeoutError:
                pass

    def details(self) -> Dict[str, Any]:
        """Synchronous, instant read of the last cached poll -- safe to call
        from the heartbeat's bounded heartbeat_details hook."""
        if not self.enabled:
            return {}
        snap = self._snapshot
        result: Dict[str, Any] = {"ilo_fetched_at": snap.fetched_at}
        if snap.thermal_c:
            result["ilo_thermal_c"] = snap.thermal_c
        if snap.fan_pct:
            result["ilo_fan_pct"] = snap.fan_pct
        if snap.power_watts is not None:
            result["ilo_power_watts"] = snap.power_watts
        if snap.error:
            result["ilo_error"] = snap.error
        return result
