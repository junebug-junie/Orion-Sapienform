"""The shared readiness gate (#2515 review item 2) and forward-tolerant decoding."""

from __future__ import annotations

import json

from orion.substrate.falkor_codec import edge_row_is_known, node_row_is_known
from orion.substrate.reader_capability import (
    CAPABILITY, KEY_PREFIX, advertise, readiness, reader_name, required_readers_from_env,
)


class FakeRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value, ex=None):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)


def test_writers_wait_until_every_required_reader_advertises():
    r = FakeRedis()
    required = ("orion-hub", "orion-recall")
    assert readiness("redis://x", required, client=r).missing == required
    advertise("redis://x", name="orion-hub", client=r)
    state = readiness("redis://x", required, client=r)
    assert (state.ready, state.missing, state.reason) == (False, ("orion-recall",), "readers_not_ready")
    advertise("redis://x", name="orion-recall", client=r)
    assert readiness("redis://x", required, client=r).ready


def test_an_old_capability_or_an_unreachable_server_is_not_ready():
    r = FakeRedis()
    r.set(KEY_PREFIX + "orion-hub", json.dumps({"capabilities": ["something_older"]}))
    assert not readiness("redis://x", ("orion-hub",), client=r).ready

    class Down:
        def get(self, _key):
            raise ConnectionError("down")

    down = readiness("redis://x", ("orion-hub",), client=Down())
    assert not down.ready and down.reason == "unavailable:ConnectionError"


def test_advertising_never_raises():
    class Down:
        def set(self, *_a, **_k):
            raise ConnectionError("down")

    assert advertise("redis://x", name="r", client=Down()) is False


def test_reader_identity_and_required_set_come_from_env(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_READER_NAME", "orion-cortex-exec-background")
    assert reader_name() == "orion-cortex-exec-background"
    monkeypatch.setenv("SUBSTRATE_ASSERTION_REQUIRED_READERS", "b, a ,a")
    assert required_readers_from_env() == ("a", "b")
    assert CAPABILITY == "assertion_core_v1"


def test_an_old_shape_reader_simulated_by_the_decoder_skips_new_rows():
    """What forward tolerance means for the NEXT shape: a row this code does not know is
    reported unknown (and skipped by hydration), never decoded into an exception."""
    assert not node_row_is_known({"node_kind": "future_kind"})
    assert node_row_is_known({"node_kind": "assertion"})
    base = {"predicate": "co_occurs_with", "source_kind": "entity", "target_kind": "entity"}
    assert edge_row_is_known(base)
    assert not edge_row_is_known({**base, "predicate": "future_predicate"})
    assert not edge_row_is_known({**base, "target_kind": "future_kind"})
    assert not edge_row_is_known({**base, "edge_role": "future_role"})
