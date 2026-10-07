"""Readers advertise which substrate graph shapes they can read; writers of new shapes wait for them.

Old reader code cannot hydrate a graph that contains an Assertion node or role-bearing
edges, and code that is already running cannot be fixed by a deploy order someone
forgets. So the rollout is mechanical:

1. every substrate reader service calls ``advertise_at_startup()`` once at process start
   (a daemon thread, off the request path, never blocking boot). It writes one Redis key
   per reader, on the same server as the graph (``FALKORDB_URI``), with a TTL, and
   re-writes it every ``ADVERTISE_INTERVAL_S``. Store construction also advertises
   (``falkor_store.bootstrap_substrate_reader``), but that alone is not enough: many
   readers construct their store lazily or never, so the gate never opened (2026-10-06);
2. a writer of the new shapes (``AssertionProjector``, the memory referent projector)
   calls ``readiness`` before every write pass and writes NOTHING until every required
   reader has advertised. While waiting it reports exactly which readers are missing.

Readers built from this code are also forward-tolerant (falkor_codec.node_row_is_known /
edge_row_is_known): rows of a kind they do not know are skipped and counted, never fatal.

Reader identity: ``SUBSTRATE_READER_NAME``, else ``SERVICE_NAME``, else the hostname.
The required set: ``SUBSTRATE_ASSERTION_REQUIRED_READERS`` (comma list) on the writer,
default ``DEFAULT_REQUIRED_READERS``. A reader rolled back to code without the refresher,
or simply dead, drops out of the gate when its key expires (``CAPABILITY_TTL_S``); delete
``orion:substrate:reader_capability:<name>`` to drop it out immediately.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CAPABILITY = "assertion_core_v1"
KEY_PREFIX = "orion:substrate:reader_capability:"
# Names are what the services advertise (their SERVICE_NAME); must match
# services/orion-memory-consolidation/.env_example (test_required_reader_names_match_services).
DEFAULT_REQUIRED_READERS: tuple[str, ...] = (
    "orion-substrate-runtime",
    "hub",
    "recall",
    "cortex-exec",
    "cortex-orch",
    "spark-concept-induction",
)
# A key outlives two missed refreshes, so one slow Redis round trip never closes the gate,
# but a dead or rolled-back reader drops out within half an hour.
ADVERTISE_INTERVAL_S = 600.0
CAPABILITY_TTL_S = 1800


def reader_name() -> str:
    for env in ("SUBSTRATE_READER_NAME", "SERVICE_NAME"):
        value = str(os.getenv(env, "") or "").strip()
        if value:
            return value
    return socket.gethostname()


def required_readers_from_env() -> tuple[str, ...]:
    raw = str(os.getenv("SUBSTRATE_ASSERTION_REQUIRED_READERS", "") or "").strip()
    if not raw:
        return DEFAULT_REQUIRED_READERS
    return tuple(sorted({part.strip() for part in raw.split(",") if part.strip()}))


def _redis(uri: str) -> Any:
    import redis

    return redis.Redis.from_url(uri, socket_timeout=2.0, socket_connect_timeout=2.0, decode_responses=True)


def advertise(uri: str, *, name: Optional[str] = None, client: Any = None,
              ttl_s: Optional[int] = CAPABILITY_TTL_S) -> bool:
    """Best effort, never raises: a reader must not fail to start because of this.
    The key expires after ``ttl_s`` (None = never; only for tests)."""
    reader = name or reader_name()
    try:
        conn = client if client is not None else _redis(uri)
        conn.set(KEY_PREFIX + reader, json.dumps({
            "capabilities": [CAPABILITY], "at": datetime.now(timezone.utc).isoformat()}),
            ex=int(ttl_s) if ttl_s else None)
        logger.info("substrate_reader_capability_advertised reader=%s capability=%s", reader, CAPABILITY)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("substrate_reader_capability_advertise_failed reader=%s error=%s", reader, type(exc).__name__)
        return False


_started: dict[tuple[str, str], threading.Thread] = {}
_started_lock = threading.Lock()


def advertise_at_startup(uri: Optional[str] = None, *, name: Optional[str] = None,
                         interval_s: float = ADVERTISE_INTERVAL_S, ttl_s: int = CAPABILITY_TTL_S,
                         stop: Optional[threading.Event] = None,
                         client: Any = None) -> Optional[threading.Thread]:
    """Call once at process start in every substrate reader service. Returns immediately.

    Starts a daemon thread that advertises now and again every ``interval_s`` with a key TTL
    of ``ttl_s``. Each attempt is bounded by the Redis client's 2s timeouts and never raises,
    so a down FalkorDB only means "not advertised yet", retried on the next tick. Idempotent
    per (uri, reader): a second call returns the running thread. No ``FALKORDB_URI`` = no-op.
    """
    try:
        target = str(uri if uri is not None else os.getenv("FALKORDB_URI", "") or "").strip()
        reader = name or reader_name()
        if not target:
            logger.warning("substrate_reader_capability_startup_skipped reader=%s reason=no_FALKORDB_URI", reader)
            return None
        if ttl_s and ttl_s <= interval_s:
            logger.warning("substrate_reader_capability_ttl_not_above_interval ttl_s=%s interval_s=%s",
                           ttl_s, interval_s)
        key = (target, reader)
        with _started_lock:
            running = _started.get(key)
            if running is not None and running.is_alive():
                return running
            halt = stop or threading.Event()

            def _loop() -> None:
                while not halt.is_set():
                    try:
                        advertise(target, name=reader, client=client, ttl_s=ttl_s)
                    except Exception:  # noqa: BLE001 -- advertise never raises; belt and braces
                        pass
                    halt.wait(interval_s)

            thread = threading.Thread(target=_loop, name=f"substrate-reader-advertise-{reader}", daemon=True)
            thread.start()
            _started[key] = thread
        logger.info("substrate_reader_capability_refresher_started reader=%s interval_s=%s ttl_s=%s",
                    reader, interval_s, ttl_s)
        return thread
    except Exception as exc:  # noqa: BLE001 -- must never break a service's boot
        logger.warning("substrate_reader_capability_startup_failed error=%s", type(exc).__name__)
        return None


@dataclass(frozen=True)
class ReadinessV1:
    ready: bool
    missing: tuple[str, ...] = ()
    present: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "missing": list(self.missing), "present": list(self.present),
                "reason": self.reason}


def readiness(uri: str, required: tuple[str, ...], *, client: Any = None) -> ReadinessV1:
    """Ready only when every required reader advertises CAPABILITY. Unreachable = not ready."""
    try:
        conn = client if client is not None else _redis(uri)
        present = []
        for reader in required:
            raw = conn.get(KEY_PREFIX + reader)
            try:
                caps = json.loads(raw).get("capabilities", []) if raw else []
            except (TypeError, ValueError):
                caps = []
            if CAPABILITY in caps:
                present.append(reader)
    except Exception as exc:  # noqa: BLE001
        return ReadinessV1(ready=False, missing=tuple(required), reason=f"unavailable:{type(exc).__name__}")
    missing = tuple(r for r in required if r not in present)
    return ReadinessV1(ready=not missing, missing=missing, present=tuple(present),
                       reason="" if not missing else "readers_not_ready")


ReadinessCheck = Callable[[], ReadinessV1]


@dataclass
class _Always:
    """For tests and in-process stores only: a graph nobody else reads."""

    result: ReadinessV1 = field(default_factory=lambda: ReadinessV1(ready=True, reason="in_process"))

    def __call__(self) -> ReadinessV1:
        return self.result


ALWAYS_READY = _Always()
