"""Per-outlet power from a Panduit PDU over SNMP (ROADMAP B6).

WHY THIS EXISTS
---------------
circe has no reachable BMC, so it is the one machine in the fleet with no `chassis_watts`.
Every fleet power total this arc has produced carries `measurements_missing:
{"chassis_watts": ["circe"]}` -- honest, but a whole machine wide. The PDU it is plugged into
meters every outlet individually, which closes that gap without ever fixing the BMC.

It is also the first measurement in this arc taken AT THE WALL rather than from a machine's own
sensors, which is what makes it a cross-check rather than just another reading. Validated live
2026-08-15 against atlas, which has both: PDU outlets 34+35 read 291 W at the same instant its
iLO read 291 W.

MULTI-PSU MACHINES SPAN OUTLETS
-------------------------------
A server with redundant supplies draws on one outlet per PSU, so a node's chassis draw is the
SUM of its outlets, and the outlet->node map is physical cabling that SNMP cannot tell us.
Verified by hand 2026-08-15:

    circe  = outlets 19, 25, 31   (3 PSUs)   ~420 W
    atlas  = outlets 34, 35       (2 PSUs)   ~291 W

The device's own outlet names are all generic ("OUTLET19", "OUTLET34", ...) and are not
editable on this firmware, so the map cannot be read from the device and lives in
`PDU_OUTLETS` instead. If anything is ever re-cabled, that env value is what must change --
nothing detects it automatically.

STALE as of 2026-08-21 -- re-cabling happened. atlas is decommissioned (its disks now run
athena; see config/biometrics/node_catalog.yaml); outlets 34,35 are athena's own reading now,
not atlas's. circe was physically relocated in the same rack work and its real outlets are
UNVERIFIED -- SNMP to this PDU was timing out entirely (even to the previously-good 34/35)
when this was written, so the 19,25,31 value above could not be re-checked. Do not trust it
until it's confirmed live.

CONTAINMENT
-----------
For a node that has BOTH an iLO and PDU outlets (atlas), these are two independent meters on
THE SAME WATTS, not two contributions. They must never be summed. `chassis_watts` keeps one
value per node; the PDU figure is exposed separately as `pdu_watts` so the pair stays
comparable, which is exactly how iLO got validated in the first place.

OID DECODE, verified against the device's own admin panel 2026-08-15
--------------------------------------------------------------------
Base: 1.3.6.1.4.1.19536.10.1.5.1.1.<col>.1.<outlet>   (enterprise 19536 = Panduit)

    col 2   outlet name (STRING, generic on this firmware)
    col 12  apparent power, VA        panel OUTLET34: 159  <-> SNMP 159
    col 13  ACTIVE POWER, WATTS       panel OUTLET34: 156  <-> SNMP 156
    col 14  energy, Wh                panel OUTLET34: 339.4 kWh <-> SNMP 339274

Column 13 is the one that matters. Confirmed two ways: exact agreement with the admin panel per
outlet, and the sum over all powered outlets (678 W) matching the unit-level total (680 W).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("orion.biometrics.pdu")

# Panduit per-outlet monitor table. Active power in whole watts.
OID_OUTLET_WATTS = "1.3.6.1.4.1.19536.10.1.5.1.1.13.1.{outlet}"
# Cumulative energy in Wh. Kept because it is the only quantity here that integrates -- a
# wattage sample says what the machine costs now, this says what it has cost.
OID_OUTLET_ENERGY_WH = "1.3.6.1.4.1.19536.10.1.5.1.1.14.1.{outlet}"


@dataclass
class PduSnapshot:
    watts: Optional[float] = None
    energy_wh: Optional[float] = None
    per_outlet_watts: Dict[int, float] = field(default_factory=dict)
    fetched_at: Optional[str] = None
    error: Optional[str] = None


def parse_proxy_outlets(raw: str) -> Dict[str, List[int]]:
    """`'{"circe": [19,25,31]}'` -> `{"circe": [19, 25, 31]}`. Empty/garbage -> {}.

    Outlets the HUB polls ON BEHALF OF a node that cannot reach the PDU itself.

    WHY THIS EXISTS. Per-node polling was the right default -- it keeps `chassis_watts` a
    self-report and needs no new trust. It does not work for the one node the whole feature was
    built for: **circe's network card is dead**, so it reaches the bus over Tailscale but has no
    LAN path to the PDU. Confirmed live 2026-08-16, failing every 65 s:

        orion.biometrics.pdu - WARNING - pdu_poll_failed error=5 second timeout exceeded on UDP

    This should have been the design from the start. circe's BMC at 192.168.1.150 was already
    known unreachable, which was direct evidence that circe's LAN path was suspect -- and circe
    was the entire justification for the feature.

    Proxy polling is also strictly MORE informative than self-polling for circe: the outlets
    report its draw whether circe is up or not, so a powered-down circe reads a true ~0 W
    instead of vanishing into `measurements_missing`.

    Raises nothing -- a malformed value disables proxying rather than taking the collector down.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("pdu_proxy_outlets_unparseable value=%r", raw[:120])
        return {}
    if not isinstance(parsed, dict):
        logger.warning("pdu_proxy_outlets_not_an_object type=%s", type(parsed).__name__)
        return {}
    out: Dict[str, List[int]] = {}
    for node, outlets in parsed.items():
        node = str(node).strip().lower()
        if not node:
            continue
        if isinstance(outlets, str):
            values = parse_outlets(outlets)
        elif isinstance(outlets, list):
            values = parse_outlets(",".join(str(v) for v in outlets))
        else:
            logger.warning("pdu_proxy_bad_value node=%s type=%s", node, type(outlets).__name__)
            continue
        if values:
            out[node] = values
    return out


def parse_outlets(raw: str) -> List[int]:
    """`"19,25,31"` -> `[19, 25, 31]`. Empty/garbage -> [] (feature simply off).

    Raises nothing: a malformed env value must not take the whole collector down, and an empty
    outlet list already means "this node has no PDU outlets", which is the correct reading for
    athena.
    """
    out: List[int] = []
    for part in (raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            logger.warning("pdu_outlet_unparseable value=%r", part)
            continue
        if value <= 0:
            logger.warning("pdu_outlet_out_of_range value=%r", value)
            continue
        if value not in out:
            out.append(value)
    return out


def fetch_pdu_snapshot(
    host: str,
    outlets: List[int],
    community: str,
    port: int,
    timeout_sec: float,
) -> PduSnapshot:
    """One synchronous SNMP read of this node's outlets. Read-only GETs, never SET.

    ALL-OR-NOTHING over the outlet set. A machine's chassis draw is the sum of its PSUs, so a
    partial read is not a smaller number -- it is a different quantity. Reporting 2 of circe's
    3 outlets would understate it by a third while looking like a valid reading, which is the
    same error `_sum_of` exists to prevent for GPU power.
    """
    if not host or not outlets:
        return PduSnapshot(error="not_configured")

    try:
        from puresnmp import Client, ObjectIdentifier as OID, V2C
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        return PduSnapshot(error=f"puresnmp_unavailable:{exc}")

    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        client = Client(host, V2C(community), port=port)
        client.configure(timeout=int(max(1, round(timeout_sec))), retries=1)

        per_outlet: Dict[int, float] = {}
        energy_total = 0.0
        energy_complete = True

        for outlet in outlets:
            raw = asyncio.run(
                client.get(OID(OID_OUTLET_WATTS.format(outlet=outlet)))
            )
            value = _as_number(raw)
            if value is None:
                return PduSnapshot(
                    fetched_at=fetched_at,
                    error=f"outlet_{outlet}_unreadable",
                )
            if value < 0:
                # A negative draw is a sensor fault, not a reading -- same rule as chassis_watts.
                return PduSnapshot(fetched_at=fetched_at, error=f"outlet_{outlet}_negative")
            per_outlet[outlet] = float(value)

            try:
                raw_wh = asyncio.run(
                    client.get(OID(OID_OUTLET_ENERGY_WH.format(outlet=outlet)))
                )
                wh = _as_number(raw_wh)
                if wh is None or wh < 0:
                    energy_complete = False
                else:
                    energy_total += float(wh)
            except Exception:
                # Energy is secondary. Losing it must not cost us the wattage, which is the
                # number the fleet total actually needs.
                energy_complete = False

        return PduSnapshot(
            watts=sum(per_outlet.values()),
            energy_wh=energy_total if energy_complete else None,
            per_outlet_watts=per_outlet,
            fetched_at=fetched_at,
        )
    except Exception as exc:
        return PduSnapshot(fetched_at=fetched_at, error=str(exc))


def _as_number(value: Any) -> Optional[float]:
    """Coerce a puresnmp return to a number, rejecting bool and unparseable types."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    for attr in ("value", "val", "data"):
        if hasattr(value, attr):
            try:
                return float(getattr(value, attr))
            except (TypeError, ValueError):
                pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class PduPoller:
    """Background slow-poll loop, same shape and same reasoning as `IloPoller`.

    A PDU's management processor is even weaker than a BMC's and is shared by every node
    plugged into it -- three nodes polling it on a fast cadence is three times the load on one
    small controller. Default cadence is 60 s and `details()` reads a cache.
    """

    def __init__(
        self,
        host: str,
        outlets: List[int],
        *,
        community: str = "public",
        port: int = 161,
        interval_sec: float = 60.0,
        timeout_sec: float = 5.0,
    ) -> None:
        self._host = host
        self._outlets = list(outlets)
        self._community = community
        self._port = port
        self._interval_sec = interval_sec
        self._timeout_sec = timeout_sec
        self._snapshot = PduSnapshot(error=None if self.enabled else "not_configured")
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return bool(self._host and self._outlets)

    async def start_background(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop(), name="pdu-poller")

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
                    fetch_pdu_snapshot,
                    self._host,
                    self._outlets,
                    self._community,
                    self._port,
                    self._timeout_sec,
                )
                if self._snapshot.error:
                    logger.warning("pdu_poll_failed error=%s", self._snapshot.error)
            except Exception as exc:
                logger.warning("pdu_poll_loop_error error=%s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_sec)
            except asyncio.TimeoutError:
                pass

    def details(self) -> Dict[str, Any]:
        """Instant read of the last cached poll.

        Returns {} when unconfigured -- ABSENT, not zero. A node with no PDU outlets (athena)
        and a node whose PDU is unreachable must both leave `chassis_watts` alone rather than
        assert 0 W.
        """
        if not self.enabled:
            return {}
        snap = self._snapshot
        result: Dict[str, Any] = {"pdu_fetched_at": snap.fetched_at}
        if snap.error:
            result["pdu_error"] = snap.error
            return result
        if snap.watts is not None:
            result["pdu_watts"] = snap.watts
        if snap.energy_wh is not None:
            result["pdu_energy_wh"] = snap.energy_wh
        if snap.per_outlet_watts:
            result["pdu_outlet_watts"] = {str(k): v for k, v in snap.per_outlet_watts.items()}
        return result
