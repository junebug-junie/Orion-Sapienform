"""No test in this package may open a network connection: a replay test that reached a live worker,
the pool, Postgres or FalkorDB would be the exact failure the package exists to prevent."""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    real_connect = socket.socket.connect

    def guarded(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError(f"network disabled in model_replay tests (tried {address!r})")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, address: guarded(self, address))
