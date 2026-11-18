"""Juniper first: her live turns through run_mind, shared by every loop instance in the process.

``services/orion-mind/app/main.py`` wraps a positively marked Juniper turn in ``JUNIPER_TURNS.turn()``. A loop
does not bid while a turn runs, and a thought whose model call overlapped one is recorded ``preempted`` and
discarded (``orion/mindloop/runtime.py``). One tracker per process, so every instance yields to the same turn.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator


class JuniperTurns:
    """Juniper's live turns through run_mind. Thread-safe: run_mind runs on the request path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._last_started = 0.0

    @contextmanager
    def turn(self) -> Iterator[None]:
        with self._lock:
            self._active += 1
            self._last_started = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1

    def active(self) -> bool:
        with self._lock:
            return self._active > 0

    def started_since(self, mono: float) -> bool:
        with self._lock:
            return self._active > 0 or self._last_started >= mono


JUNIPER_TURNS = JuniperTurns()
