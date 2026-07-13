from __future__ import annotations

import json

from app.chat_stance import _project_reverie_glimpse


def _fresh_payload(**overrides):
    payload = {
        "interpretation": "The coalition is fixated on unresolved transport anomalies.",
        "hollow": False,
        "evidence_refs": ["evt:1", "evt:2"],
        "coalition": ["node:a", "node:b"],
        "chain_id": "chain:123",
    }
    payload.update(overrides)
    return payload


def test_none_when_ctx_key_absent():
    assert _project_reverie_glimpse({}) is None


def test_none_when_hollow_true():
    ctx = {"latest_reverie_thought": _fresh_payload(hollow=True)}
    assert _project_reverie_glimpse(ctx) is None


def test_none_when_interpretation_empty_or_whitespace():
    ctx = {"latest_reverie_thought": _fresh_payload(interpretation="   ")}
    assert _project_reverie_glimpse(ctx) is None

    ctx2 = {"latest_reverie_thought": _fresh_payload(interpretation="")}
    assert _project_reverie_glimpse(ctx2) is None


def test_returns_interpretation_verbatim_for_dict_payload():
    payload = _fresh_payload()
    ctx = {"latest_reverie_thought": payload}
    result = _project_reverie_glimpse(ctx)
    assert result == payload["interpretation"]
    assert isinstance(result, str)
    # Only the interpretation string comes back -- the other fields' concrete
    # values never leak into the projected result (note: the word "coalition"
    # itself legitimately appears in this example's prose, so we assert on the
    # distinguishing evidence/chain values instead of the field names).
    assert "evt:1" not in result
    assert "evt:2" not in result
    assert "node:a" not in result
    assert "node:b" not in result
    assert "chain:123" not in result


def test_returns_interpretation_verbatim_for_json_string_payload():
    payload = _fresh_payload()
    ctx = {"latest_reverie_thought": json.dumps(payload)}
    result = _project_reverie_glimpse(ctx)
    assert result == payload["interpretation"]


def test_none_when_payload_is_malformed_json_string():
    ctx = {"latest_reverie_thought": "{not valid json"}
    assert _project_reverie_glimpse(ctx) is None


def test_none_when_payload_is_unexpected_type():
    ctx = {"latest_reverie_thought": 12345}
    assert _project_reverie_glimpse(ctx) is None


def test_ctx_key_wiring_only_sets_no_other_fields(monkeypatch):
    """Guard the call-site contract: chat_reverie_glimpse, when set, is exactly
    the interpretation string and nothing else (dict, tuple, evidence, etc.)."""
    import app.chat_stance as chat_stance

    payload = _fresh_payload()
    ctx = {"latest_reverie_thought": payload}
    glimpse = chat_stance._project_reverie_glimpse(ctx)
    if glimpse:
        ctx["chat_reverie_glimpse"] = glimpse
    assert ctx["chat_reverie_glimpse"] == payload["interpretation"]
    assert isinstance(ctx["chat_reverie_glimpse"], str)
