import json
import os
from typing import Dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    SERVICE_NAME: str = Field(default="orion-biometrics")
    SERVICE_VERSION: str = Field(default="0.1.0")
    NODE_NAME: str = Field(default="unknown")

    # Bus
    ORION_BUS_URL: str = Field(default="redis://orion-redis:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False)

    # Channels
    TELEMETRY_PUBLISH_CHANNEL: str = Field(default="orion:telemetry:biometrics")
    BIOMETRICS_SAMPLE_CHANNEL: str = Field(default="orion:biometrics:sample")
    BIOMETRICS_SUMMARY_CHANNEL: str = Field(default="orion:biometrics:summary")
    BIOMETRICS_INDUCTION_CHANNEL: str = Field(default="orion:biometrics:induction")
    BIOMETRICS_CLUSTER_CHANNEL: str = Field(default="orion:biometrics:cluster")
    SPARK_SIGNAL_CHANNEL: str = Field(default="orion:spark:signal")
    PUBLISH_BIOMETRICS_GRAMMAR: bool = Field(default=True)
    GRAMMAR_EVENT_CHANNEL: str = Field(default="orion:grammar:event")
    NODE_CATALOG_PATH: str = Field(default="/app/config/biometrics/node_catalog.yaml")

    # Behavior
    TELEMETRY_INTERVAL: int = Field(default=30)
    LOG_LEVEL: str = Field(default="INFO")
    BIOMETRICS_MODE: str = Field(default="agent")
    CLUSTER_PUBLISH_INTERVAL: int = Field(default=15)
    role_weights: Dict[str, float] = Field(
        default_factory=lambda: {"atlas": 0.7, "athena": 0.3, "other": 0.5},
        alias="CLUSTER_ROLE_WEIGHTS",
    )
    SPARK_SIGNAL_TTL_MS: int = Field(default=15000)

    THERMAL_MIN_C: float = Field(default=50.0)
    THERMAL_MAX_C: float = Field(default=85.0)
    # Denominator for disk_pressure. Still a per-node constant because the kernel does not
    # report block-device throughput the way it reports link speed -- there is nothing to
    # measure it from without benchmarking. Treat it as an order-of-magnitude anchor, not a
    # precise ceiling: athena spans ten devices from a 10k SAS spinner to a Samsung 990 PRO,
    # so no single scalar is right for all of them. The raw byte rate is now published in
    # `measurements.disk_bytes_per_sec`; prefer that when you need the real number.
    DISK_BW_MBPS: float = Field(default=200.0)
    # FALLBACK ONLY. The live denominator for net_pressure is the summed link speed of the
    # node's up physical NICs, read from the kernel via HOST_SYS_PATH (see
    # BiometricsCollector._link_speed_mbps). This constant applies only when that read fails,
    # e.g. the host sysfs mount is absent. It was previously the sole source: one value,
    # 125 MB/s, for three heterogeneous hosts -- right for athena's 1 GbE by coincidence and
    # never verified on atlas or circe.
    NET_BW_MBPS: float = Field(default=125.0)

    # Read-only bind mounts of the HOST /proc and /sys, so the collector can measure the node
    # rather than its own container. Network counters and sysfs are namespaced; /proc/diskstats
    # is not, which is why disk needed a different fix. Empty disables the host read and makes
    # the collector fall back to its own namespace, reporting `network.scope="container"` so a
    # consumer can tell the difference instead of reading veth traffic as node traffic.
    HOST_PROC_PATH: str = Field(default="/host_proc")
    HOST_SYS_PATH: str = Field(default="/host_sys")
    POWER_BAND_ALPHA: float = Field(default=0.1)

    TABLE_NAME: str = Field(default="biometrics_raw")

    # Chassis Defaults
    HEARTBEAT_INTERVAL_SEC: float = 10.0
    ORION_HEALTH_CHANNEL: str = "orion:system:health"
    ERROR_CHANNEL: str = "orion:system:error"
    SHUTDOWN_GRACE_SEC: float = 10.0

    # Disk-capacity telemetry (piggybacked onto the SystemHealthV1 heartbeat's
    # `details` dict -- see README.md "Disk capacity telemetry" section). Maps a
    # short mount name to the read-only in-container bind-mount path configured in
    # docker-compose.yml. Node-level, not per-service: every node running this
    # service reports its own 5 mounts independently once redeployed there.
    DISK_CAPACITY_MOUNTS: Dict[str, str] = Field(
        default_factory=lambda: {
            "docker": "/host_mnt/docker",
            "scripts": "/host_mnt/scripts",
            "postgres": "/host_mnt/postgres",
            "graphdb": "/host_mnt/graphdb",
            "telemetry": "/host_mnt/telemetry",
        }
    )

    # iLO/BMC out-of-band hardware telemetry (piggybacked onto the same heartbeat
    # `details` dict as disk capacity, above). Node-level secret: real values live
    # only in this node's local .env, never in .env_example. Empty ILO_HOST means
    # iLO collection is disabled for this node -- not every node has one.
    ILO_HOST: str = Field(default="")
    ILO_USERNAME: str = Field(default="")
    ILO_PASSWORD: str = Field(default="")
    ILO_POLL_INTERVAL_SEC: float = Field(default=60.0)
    ILO_REQUEST_TIMEOUT_SEC: float = Field(default=8.0)

    # Rack PDU per-outlet power (SNMP, read-only GETs). PER-NODE, exactly like ILO_HOST above.
    #
    # PDU_OUTLETS is THIS node's own outlets, and a multi-PSU server spans several -- its
    # chassis draw is their sum. The mapping is physical cabling that SNMP cannot report (the
    # device's outlet names are generic and not editable on this firmware), so it lives in
    # config and re-cabling means editing it. Traced by hand 2026-08-15:
    #     circe   PDU_OUTLETS=19,25,31   (3 PSUs)
    #     atlas   PDU_OUTLETS=34,35      (2 PSUs)
    # athena is not on this PDU and leaves it empty, which disables the poller entirely.
    #
    # STALE as of 2026-08-21. atlas is decommissioned (see node_catalog.yaml); its old
    # outlets 34,35 are now athena's own reading -- athena's disks moved into that chassis
    # and it inherited the iLO/PDU position. circe was separately relocated in the same rack
    # work and its real outlet numbers are UNVERIFIED (SNMP to this PDU was fully timing out,
    # even to the previously-good 34/35, when this was written) -- do not trust the 19,25,31
    # value above until it's re-checked live.
    #
    # This is the only source of chassis power for a node with no BMC, which is the whole
    # reason it exists: circe has never reported watts, and every fleet total to date has
    # carried `measurements_missing: {"chassis_watts": ["circe"]}`.
    PDU_HOST: str = Field(default="")
    PDU_OUTLETS: str = Field(default="")
    PDU_SNMP_COMMUNITY: str = Field(default="public")
    PDU_SNMP_PORT: int = Field(default=161)
    # A PDU's controller is weaker than a BMC's AND shared by every node plugged into it, so
    # several nodes polling fast is several times the load on one small processor.
    PDU_POLL_INTERVAL_SEC: float = Field(default=60.0)
    PDU_REQUEST_TIMEOUT_SEC: float = Field(default=5.0)

    # Outlets this node polls ON BEHALF OF another node, as JSON: {"circe": [19,25,31]}
    #
    # HUB ONLY, and only for nodes that cannot reach the PDU themselves. circe's NIC is dead --
    # it reaches the bus over Tailscale but has no LAN path to 192.168.1.39, so its own poller
    # fails every 65 s. athena can reach it.
    #
    # Also strictly better than self-polling for circe: the outlets report its draw whether
    # circe is powered or not, so a shut-down circe reads a true ~0 W instead of disappearing
    # into measurements_missing.
    #
    # A proxied reading is published with provenance (`measurements_proxied`) and never
    # overwrites a node's own reading.
    PDU_PROXY_OUTLETS: str = Field(default="")

    @field_validator("role_weights", mode="before")
    @classmethod
    def _parse_role_weights(cls, value: object) -> Dict[str, float]:
        if isinstance(value, dict):
            return {str(k): float(v) for k, v in value.items()}
        if isinstance(value, str):
            try:
                data = json.loads(value)
            except json.JSONDecodeError:
                return {"atlas": 0.7, "athena": 0.3, "other": 0.5}
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        return {"atlas": 0.7, "athena": 0.3, "other": 0.5}

    @field_validator("DISK_CAPACITY_MOUNTS", mode="before")
    @classmethod
    def _parse_disk_capacity_mounts(cls, value: object) -> Dict[str, str]:
        _default = {
            "docker": "/host_mnt/docker",
            "scripts": "/host_mnt/scripts",
            "postgres": "/host_mnt/postgres",
            "graphdb": "/host_mnt/graphdb",
            "telemetry": "/host_mnt/telemetry",
        }
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        if isinstance(value, str):
            if not value.strip():
                return _default
            try:
                data = json.loads(value)
            except json.JSONDecodeError:
                return _default
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        return _default

# pydantic-settings decodes Dict-typed env vars as JSON at its own source layer,
# before _parse_disk_capacity_mounts (which already handles "" gracefully) ever
# runs -- an empty string (e.g. docker-compose interpolating an unset host env
# var) throws SettingsError there instead of reaching our validator's fallback.
# Confirmed live 2026-07-24: orion-biometrics crash-looped on exactly this after
# .env_example gained DISK_CAPACITY_MOUNTS but a live .env hadn't been synced yet.
if not os.environ.get("DISK_CAPACITY_MOUNTS", "").strip():
    os.environ["DISK_CAPACITY_MOUNTS"] = "{}"

settings = Settings()

try:
    settings.role_weights = json.loads(settings.CLUSTER_ROLE_WEIGHTS)
except Exception:
    settings.role_weights = {"atlas": 0.7, "athena": 0.3, "other": 0.5}
