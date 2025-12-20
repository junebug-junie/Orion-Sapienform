from __future__ import annotations

import logging
from typing import Any, Optional, Dict

from puresnmp import Client, ObjectIdentifier as OID, V1

from .models import PowerStatus

logger = logging.getLogger("orion-power-guard.ups-snmp")

# ─────────────────────────────────────────────
# APC PowerNet-MIB OIDs
# ─────────────────────────────────────────────
OID_BASIC_OUTPUT_STATUS = OID("1.3.6.1.4.1.318.1.1.1.4.1.1.0")
OID_ADV_BATT_CAPACITY = OID("1.3.6.1.4.1.318.1.1.1.2.2.1.0")

# OID .3.2.1.0 is "upsInputVoltage" (Basic)
# Returns INTEGER RMS Volts (e.g. 120, 124). NOT tenths.
OID_BASIC_INPUT_VOLTAGE = OID("1.3.6.1.4.1.318.1.1.1.3.2.1.0")


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    for attr in ("value", "val", "data"):
        if hasattr(v, attr):
            try:
                return int(getattr(v, attr))
            except Exception:
                pass
    if hasattr(v, "ticks"):
        try:
            return int(getattr(v, "ticks"))
        except Exception:
            pass
    try:
        return int(v)
    except Exception:
        logger.debug("Unable to coerce SNMP value %r to int", v)
        return None


class SNMPUPSClient:
    def __init__(
        self,
        host: str,
        community: str = "public",
        port: int = 161,
        timeout: int = 3,
        retries: int = 1,
        line_voltage_onbatt_threshold: float = 80.0,
    ) -> None:
        self.host = host
        self.community = community
        self.port = port
        self.timeout = timeout
        self.retries = retries
        self.line_voltage_onbatt_threshold = line_voltage_onbatt_threshold

        self._client = Client(host, V1(community), port=port)
        self._client.configure(timeout=timeout, retries=retries)

    async def _fetch_raw(self) -> Dict[str, Optional[int]]:
        oids = [
            OID_BASIC_OUTPUT_STATUS,
            OID_ADV_BATT_CAPACITY,
            OID_BASIC_INPUT_VOLTAGE,  # using basic volts
        ]
        raw_vals = await self._client.multiget(oids)
        # Log raw values for debugging
        logger.info("[SNMP] raw APC values (basic_status, batt_cap, line_volt): %r", raw_vals)

        ints = [_to_int(v) for v in raw_vals]

        return {
            "output_status": ints[0],
            "batt_capacity_pct": ints[1],
            "input_line_voltage": ints[2],
        }

    async def get_status(self) -> PowerStatus:
        vals = await self._fetch_raw()

        output_status = vals["output_status"]
        batt_capacity = vals["batt_capacity_pct"]
        input_voltage_raw = vals["input_line_voltage"]

        # [FIX] Do NOT divide by 10. This OID returns whole volts.
        input_line_voltage: Optional[float] = None
        if input_voltage_raw is not None:
            input_line_voltage = float(input_voltage_raw)

        # Decide on-battery from input line voltage
        on_battery = False
        raw_status = "UNKNOWN"

        if input_line_voltage is None:
            raw_status = "ONBATT_NO_LINE"
            on_battery = True
        else:
            if input_line_voltage < self.line_voltage_onbatt_threshold:
                # e.g. < 80V
                raw_status = "ONBATT"
                on_battery = True
            else:
                # Wall power present
                if output_status is None:
                    raw_status = "ONLINE"
                elif output_status == 2:
                    raw_status = "ONLINE"
                elif output_status == 3:
                    raw_status = "ONBATT"
                elif output_status == 4:
                    raw_status = "BOOST"
                elif output_status == 5:
                    raw_status = "TRIM"
                else:
                    raw_status = f"STATE_{output_status}"

        battery_charge_pct = float(batt_capacity) if batt_capacity is not None else None
        time_left_min = None

        if raw_status == "UNKNOWN":
            logger.warning(
                "Unable to interpret APC status from SNMP; treating as ONLINE. "
                "output_status=%r batt_capacity=%r input_line_voltage=%r",
                output_status,
                batt_capacity,
                input_line_voltage,
            )
            on_battery = False

        return PowerStatus(
            raw_status=raw_status,
            on_battery=on_battery,
            line_voltage=input_line_voltage,
            battery_charge_pct=battery_charge_pct,
            time_left_min=time_left_min,
            raw={
                "output_status": output_status,
                "battery_capacity_pct": batt_capacity,
                "input_line_voltage_raw": input_voltage_raw,
            },
        )
