from __future__ import annotations

from collections import deque
from typing import List, Dict, Any
import logging

from orion.schemas.telemetry.cognition_trace import CognitionTracePayload

logger = logging.getLogger("orion.cortex.exec.trace_cache")

class TraceCache:
    _instance = None

    def __init__(self, maxlen: int = 10):
        self.buffer: deque[CognitionTracePayload] = deque(maxlen=maxlen)

    @classmethod
    def get_instance(cls) -> TraceCache:
        if cls._instance is None:
            cls._instance = TraceCache()
        return cls._instance

    def append(self, trace: CognitionTracePayload) -> None:
        self.buffer.append(trace)

    def get_recent(self, k: int = 5) -> List[CognitionTracePayload]:
        # Return last k traces reversed (newest first)
        return list(reversed(self.buffer))[:k]

def get_trace_cache() -> TraceCache:
    return TraceCache.get_instance()
