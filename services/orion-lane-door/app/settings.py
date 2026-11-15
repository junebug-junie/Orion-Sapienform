"""Lane door settings, from the environment (stdlib only: the door has no dependencies)."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

MODES = ("off", "shadow", "enforce")


def key_hash(key: str) -> str | None:
    """What the door and the gateway compare to see they hold the same key, without sending it.
    services/orion-llm-gateway/app/lane_door.py computes the same value (a test pins that)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16] if key else None


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    lane: str                     # the pool role this door fronts (chat, agent, agent-gpu2, fast, agent-deep)
    upstream_host: str            # the engine on the docker network
    upstream_port: int
    mode: str = "shadow"
    key: str = ""                 # LANE_DOOR_KEY: the same secret the gateway holds
    listen_host: str = "0.0.0.0"
    listen_port: int = 8080
    log_dir: str = "/var/log/lane-door"
    log_retention_days: int = 30
    connect_timeout_sec: float = 5.0
    summary_interval_sec: float = 600.0
    upstream_host_header: str = ""   # "" = pass the caller's Host through unchanged (what Strata sees today)

    @property
    def key_hash(self) -> str | None:
        return key_hash(self.key)


def load() -> Settings:
    upstream = os.environ.get("LANE_DOOR_UPSTREAM", "").strip()
    if ":" not in upstream:
        raise SystemExit("LANE_DOOR_UPSTREAM must be host:port (the engine on the docker network)")
    host, port = upstream.rsplit(":", 1)
    lane = os.environ.get("LANE_DOOR_LANE", "").strip()
    if not lane:
        raise SystemExit("LANE_DOOR_LANE must name the pool role this door fronts")
    mode = os.environ.get("LANE_DOOR_MODE", "shadow").strip().lower() or "shadow"
    if mode not in MODES:
        raise SystemExit(f"LANE_DOOR_MODE={mode!r}: one of {MODES}")
    if mode == "enforce" and not os.environ.get("LANE_DOOR_KEY", "").strip():
        # Enforcing with no key would refuse every call, the gateway's included (Orion mute on this lane).
        # Refusing to start is louder and leaves the seat visibly down on the pool panel instead.
        raise SystemExit("LANE_DOOR_MODE=enforce needs LANE_DOOR_KEY: without it every call would be refused")
    return Settings(
        lane=lane, upstream_host=host, upstream_port=int(port), mode=mode,
        key=os.environ.get("LANE_DOOR_KEY", "").strip(),
        listen_host=os.environ.get("LANE_DOOR_LISTEN_HOST", "0.0.0.0").strip() or "0.0.0.0",
        listen_port=_int("LANE_DOOR_LISTEN_PORT", 8080),
        log_dir=os.environ.get("LANE_DOOR_LOG_DIR", "/var/log/lane-door").strip() or "/var/log/lane-door",
        log_retention_days=_int("LANE_DOOR_LOG_RETENTION_DAYS", 30),
        connect_timeout_sec=_float("LANE_DOOR_CONNECT_TIMEOUT_SEC", 5.0),
        summary_interval_sec=_float("LANE_DOOR_SUMMARY_INTERVAL_SEC", 600.0),
        upstream_host_header=os.environ.get("LANE_DOOR_UPSTREAM_HOST_HEADER", "").strip(),
    )
