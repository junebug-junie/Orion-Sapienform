from pathlib import Path
from typing import Optional

from pydantic import Field, IPvAnyAddress, AnyUrl, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Env-driven config for the Orion VoIP endpoint (Athena).

    Reads all VOIP_* variables that we wired in docker-compose.yml / .env. A handful of
    fields below (service_name/service_version/node_name/orion_bus_url/
    heartbeat_interval_sec) use an explicit `alias=` to opt out of the VOIP_ env_prefix
    below, matching this repo's general SERVICE_NAME/SERVICE_VERSION/NODE_NAME/
    ORION_BUS_URL/HEARTBEAT_INTERVAL_SEC convention used by every other service (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md).
    """

    # IPs
    lan_host_ip: IPvAnyAddress              # VOIP_LAN_HOST_IP
    tailscale_host_ip: Optional[IPvAnyAddress] = None  # VOIP_TAILSCALE_HOST_IP

    # Identity (explicit alias -- bypasses the VOIP_ env_prefix below)
    service_name: str = Field("orion-voip-endpoint", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus-native SystemHealthV1 heartbeat (explicit alias -- bypasses VOIP_ prefix)
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Cisco device identity
    phone_mac: str                          # VOIP_PHONE_MAC

    # SIP creds
    sip_ext: str = "1001"                   # VOIP_SIP_EXT
    sip_secret: str = "supersecret"         # VOIP_SIP_SECRET

    # Behavior / codecs
    autoanswer: bool = True                 # VOIP_AUTOANSWER
    codecs: str = "ulaw,alaw,g722"          # VOIP_CODECS

    # HTTP API
    api_port: int = 8085                    # VOIP_API_PORT

    # Orion Bus / Redis
    bus_redis_url: Optional[AnyUrl] = None  # VOIP_BUS_REDIS_URL
    bus_command_channel: str = "orion:voip:command"  # VOIP_BUS_COMMAND_CHANNEL
    bus_status_channel: str = "orion:voip:status"    # VOIP_BUS_STATUS_CHANNEL

    class Config:
        env_prefix = "VOIP_"
        case_sensitive = False

    # --- Normalizers / helpers ---

    @field_validator("phone_mac")
    @classmethod
    def normalize_mac(cls, v: str) -> str:
        """
        Normalize MAC to colon-separated lowercase internally.
        We'll generate Cisco-style SEP filenames separately.
        """
        v = v.strip()
        v = v.replace("-", ":").replace(".", ":")
        parts = [p.zfill(2) for p in v.split(":") if p]
        if len(parts) != 6:
            raise ValueError(f"Invalid MAC format: {v}")
        return ":".join(parts).lower()

    @property
    def cisco_sep_filename(self) -> str:
        """
        Cisco SEP filename: 'SEP' + MAC uppercase, no colons + '.cnf.xml'
        Example: ac:44:f2:1c:7d:d5 -> SEPAC44F21C7DD5.cnf.xml
        """
        hexstr = self.phone_mac.replace(":", "").upper()
        return f"SEP{hexstr}.cnf.xml"

    @property
    def codecs_list(self) -> list[str]:
        return [c.strip() for c in self.codecs.split(",") if c.strip()]

    @property
    def asterisk_etc_dir(self) -> Path:
        return Path("/etc/asterisk")

    @property
    def tftp_root(self) -> Path:
        return Path("/tftpboot")

    def summary(self) -> str:
        return (
            f"LAN={self.lan_host_ip}, "
            f"TS={self.tailscale_host_ip}, "
            f"MAC={self.phone_mac}, "
            f"EXT={self.sip_ext}, "
            f"AUTOANSWER={self.autoanswer}, "
            f"API_PORT={self.api_port}, "
            f"BUS={self.bus_redis_url}, "
            f"CMD_CH={self.bus_command_channel}, "
            f"STATUS_CH={self.bus_status_channel}"
        )
