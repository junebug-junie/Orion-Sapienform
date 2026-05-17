from __future__ import annotations

import importlib
import sys
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
if str(HUB_ROOT) not in sys.path:
    sys.path.insert(0, str(HUB_ROOT))

presence_session = importlib.import_module("scripts.presence_session")


class _Store:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)


def test_inject_session_presence_uses_store_when_payload_missing():
    store = _Store()
    store.values["sid-1"] = {"audience_mode": "kid_present"}
    out = presence_session.inject_session_presence({}, "sid-1", store)
    assert out["presence_context"]["audience_mode"] == "kid_present"


def test_inject_session_presence_fills_empty_client_dict_from_store():
    store = _Store()
    store.values["sid-1"] = {"audience_mode": "kid_present"}
    out = presence_session.inject_session_presence({"presence_context": {}}, "sid-1", store)
    assert out["presence_context"]["audience_mode"] == "kid_present"


def test_inject_session_presence_preserves_client_payload():
    store = _Store()
    store.values["sid-1"] = {"audience_mode": "solo"}
    out = presence_session.inject_session_presence(
        {"presence_context": {"audience_mode": "family"}},
        "sid-1",
        store,
    )
    assert out["presence_context"]["audience_mode"] == "family"
