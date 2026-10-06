"""Readers advertise which substrate graph shapes they can read; writers of new shapes wait for them.

Old reader code cannot hydrate a graph that contains an Assertion node or role-bearing
edges, and code that is already running cannot be fixed by a deploy order someone
forgets. So the rollout is mechanical:

1. every Falkor substrate reader built from this code advertises ``CAPABILITY`` when its
   store is constructed (``FalkorSubstrateStore``, ``build_falkor_direct_concept_store_from_env``):
   one Redis key per reader, on the same server as the graph;
2. a writer of the new shapes (``AssertionProjector``, the memory referent projector)
   calls ``readiness`` before every write pass and writes NOTHING until every required
   reader has advertised. While waiting it reports exactly which readers are missing.

Readers built from this code are also forward-tolerant (falkor_codec.node_row_is_known /
edge_row_is_known): rows of a kind they do not know are skipped and counted, never fatal.

Reader identity: ``SUBSTRATE_READER_NAME``, else ``SERVICE_NAME``, else the hostname.
The required set: ``SUBSTRATE_ASSERTION_REQUIRED_READERS`` (comma list) on the writer,
default ``DEFAULT_REQUIRED_READERS``. Rolling a reader back to old code leaves its key in
place: delete ``orion:substrate:reader_capability:<name>`` when you do that.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CAPABILITY = "assertion_core_v1"
KEY_PREFIX = "orion:substrate:reader_capability:"
DEFAULT_REQUIRED_READERS: tuple[str, ...] = (
    "orion-substrate-runtime",
    "orion-hub",
    "orion-recall",
    "orion-cortex-exec",
    "orion-cortex-orch",
    "orion-spark-concept-induction",
    "orion-field-digester",
    "orion-world-pulse",
    "orion-meta-tags",
)


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


def advertise(uri: str, *, name: Optional[str] = None, client: Any = None) -> bool:
    """Best effort, never raises: a reader must not fail to start because of this."""
    reader = name or reader_name()
    try:
        conn = client if client is not None else _redis(uri)
        conn.set(KEY_PREFIX + reader, json.dumps({
            "capabilities": [CAPABILITY], "at": datetime.now(timezone.utc).isoformat()}))
        logger.info("substrate_reader_capability_advertised reader=%s capability=%s", reader, CAPABILITY)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("substrate_reader_capability_advertise_failed reader=%s error=%s", reader, type(exc).__name__)
        return False


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
