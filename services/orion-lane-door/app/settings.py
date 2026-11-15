"""Lane door settings, from the environment (stdlib only: the door has no dependencies)."""
from __future__ import annotations

import os
from dataclasses import dataclass

MODES = ("off", "shadow", "enforce")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    lane: str                     # the pool role this door fronts (chat, agent, agent-gpu2, fast, agent-deep)
    upstream_host: str            # the engine, on the host's loopback (127.0.0.1)
    upstream_port: int            # the engine's loopback port (180xx)
    mode: str = "shadow"
    listen_host: str = ""         # "" = every interface, IPv4 and IPv6
    listen_port: int = 8080       # the lane port (8011...)
    log_dir: str = "/var/log/lane-door"
    log_retention_days: int = 30
    connect_timeout_sec: float = 5.0
    summary_interval_sec: float = 600.0
    upstream_host_header: str = ""   # "" = pass the caller's Host through unchanged (what Strata sees today)
    bind_poll_sec: float = 2.0       # how often a waiting door re-checks its engine and the lane port
    drain_sec: float = 100.0         # on stop: how long calls in flight may finish


def load() -> Settings:
    upstream = os.environ.get("LANE_DOOR_UPSTREAM", "").strip()
    if ":" not in upstream:
        raise SystemExit("LANE_DOOR_UPSTREAM must be host:port (the engine on the host's loopback)")
    host, port = upstream.rsplit(":", 1)
    lane = os.environ.get("LANE_DOOR_LANE", "").strip()
    if not lane:
        raise SystemExit("LANE_DOOR_LANE must name the pool role this door fronts")
    mode = os.environ.get("LANE_DOOR_MODE", "shadow").strip().lower() or "shadow"
    if mode not in MODES:
        raise SystemExit(f"LANE_DOOR_MODE={mode!r}: one of {MODES}")
    return Settings(
        lane=lane, upstream_host=host, upstream_port=int(port), mode=mode,
        listen_host=os.environ.get("LANE_DOOR_LISTEN_HOST", "").strip(),
        listen_port=_int("LANE_DOOR_LISTEN_PORT", 8080),
        log_dir=os.environ.get("LANE_DOOR_LOG_DIR", "/var/log/lane-door").strip() or "/var/log/lane-door",
        log_retention_days=_int("LANE_DOOR_LOG_RETENTION_DAYS", 30),
        connect_timeout_sec=_float("LANE_DOOR_CONNECT_TIMEOUT_SEC", 5.0),
        summary_interval_sec=_float("LANE_DOOR_SUMMARY_INTERVAL_SEC", 600.0),
        upstream_host_header=os.environ.get("LANE_DOOR_UPSTREAM_HOST_HEADER", "").strip(),
        bind_poll_sec=_float("LANE_DOOR_BIND_POLL_SEC", 2.0),
        drain_sec=_float("LANE_DOOR_DRAIN_SEC", 100.0),
    )
