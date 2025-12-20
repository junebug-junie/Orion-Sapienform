from __future__ import annotations

import logging
import socket
import struct
from typing import Dict, Optional

from .models import PowerStatus

logger = logging.getLogger("orion-power-guard.ups-nis")

class NISUPSClient:
    """
    Reads APC UPS status from a local or remote apcupsd NIS server (TCP 3551).
    """

    def __init__(self, host: str, port: int = 3551, timeout: int = 5) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def _get_raw_data(self) -> str:
        """Connects to apcupsd and requests the 'status' dump."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect((self.host, self.port))
            
            # apcupsd NIS protocol: send 2-byte length, then command
            # "status" is the command to get the full dump
            cmd = b"status"
            s.send(struct.pack("!H", len(cmd)) + cmd)

            response = b""
            while True:
                # Read 2-byte length prefix for each line
                header = s.recv(2)
                if len(header) < 2:
                    break
                length = struct.unpack("!H", header)[0]
                if length == 0:
                    break
                
                line = s.recv(length)
                response += line
            
            return response.decode("ascii", errors="ignore")
            
        except Exception as e:
            logger.error("Failed to connect to apcupsd at %s:%s - %s", self.host, self.port, e)
            return ""
        finally:
            s.close()

    def _parse_key_value(self, raw: str) -> Dict[str, str]:
        data = {}
        for line in raw.splitlines():
            # Lines look like: "STATUS   : ONLINE"
            if ":" in line:
                key, val = line.split(":", 1)
                data[key.strip()] = val.strip()
        return data

    async def get_status(self) -> PowerStatus:
        """
        Async wrapper to fetch and parse status. 
        (We keep it async to match the interface, even though socket is blocking here for simplicity).
        """
        raw_text = self._get_raw_data()
        data = self._parse_key_value(raw_text)

        # 1. Parse Status
        # STATUS can be "ONLINE", "ONBATT", "ONLINE CHARGING", "COMMLOST", etc.
        status_str = data.get("STATUS", "UNKNOWN")
        
        on_battery = "ONBATT" in status_str

        # 2. Parse Voltage
        # LINEV : 124.0 Volts
        line_v_str = data.get("LINEV", "0.0").split()[0]
        try:
            line_voltage = float(line_v_str)
        except ValueError:
            line_voltage = 0.0

        # 3. Parse Battery
        # BCHARGE : 100.0 Percent
        batt_str = data.get("BCHARGE", "0.0").split()[0]
        try:
            battery_pct = float(batt_str)
        except ValueError:
            battery_pct = None

        # 4. Parse Time Left
        # TIMELEFT : 45.0 Minutes
        time_str = data.get("TIMELEFT", "0.0").split()[0]
        try:
            time_left_min = float(time_str)
        except ValueError:
            time_left_min = None

        return PowerStatus(
            raw_status=status_str,
            on_battery=on_battery,
            line_voltage=line_voltage,
            battery_charge_pct=battery_pct,
            time_left_min=time_left_min,
            raw=data
        )
