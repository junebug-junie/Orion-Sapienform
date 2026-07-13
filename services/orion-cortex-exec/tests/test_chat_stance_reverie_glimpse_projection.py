from __future__ import annotations

import json

from app.chat_stance import _project_reverie_glimpse

# Real, valid SpontaneousThoughtV1 shape: coalition must validate as a real
# CoalitionSnapshotV1 (not a bare list) and evidence_refs must be a genuine
# subset of its grounding_ids(), or is_hollow()'s real guard rejects the
# thought -- this is the whole point of re-deriving hollowness from the
# schema instead of trusting a bare stored `hollow` bool.
_INTERPRETATION = "The coalition is fixated on unresolved transport anomalies."


def _fresh_payload(**overrides):
    payload = {
        "thought_id": "th-1",
        "correlation_id": "corr-1",
        "interpretation": _INTERPRETATION,
        "hollow": False,
        "evidence_refs": ["node:a", "loop:open-1"],
        "coalition": {
            "attended_node_ids": ["node:a", "node:b"],
            "selected_open_loop_id": "loop:open-1",
            "open_loop_ids": ["loop:open-1", "loop:open-2"],
            "generated_at": "2026-07-12T00:00:00+00:00",
            "broadcast_stale": False,
        },
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
    # Empty/whitespace interpretation is independently rejected by is_hollow()
    # itself (interpretation_too_short), re-deriving the same guard the
    # producer already applied -- not merely trusting the stored bool.
    ctx = {"latest_reverie_thought": _fresh_payload(interpretation="   ")}
    assert _project_reverie_glimpse(ctx) is None

    ctx2 = {"latest_reverie_thought": _fresh_payload(interpretation="")}
    assert _project_reverie_glimpse(ctx2) is None


def test_none_when_coalition_absent():
    """is_hollow() rejects absent_coalition even if the stored `hollow` bool
    says False -- this is exactly the defense-in-depth case a shallow
    `payload.get("hollow") is True` check would have missed."""
    ctx = {"latest_reverie_thought": _fresh_payload(coalition=None, hollow=False)}
    assert _project_reverie_glimpse(ctx) is None


def test_none_when_evidence_refs_unanchored():
    """is_hollow() rejects evidence outside the coalition's grounding ids, even
    if the stored `hollow` bool says False."""
    ctx = {
        "latest_reverie_thought": _fresh_payload(
            evidence_refs=["node:not-in-coalition"], hollow=False
        )
    }
    assert _project_reverie_glimpse(ctx) is None


def test_none_when_payload_fails_schema_validation():
    """A payload that no longer validates as SpontaneousThoughtV1 at all
    (e.g. schema drift) must fail closed, not partially parse."""
    ctx = {"latest_reverie_thought": {"interpretation": "missing required fields"}}
    assert _project_reverie_glimpse(ctx) is None


def test_returns_interpretation_verbatim_for_dict_payload():
    payload = _fresh_payload()
    ctx = {"latest_reverie_thought": payload}
    result = _project_reverie_glimpse(ctx)
    assert result == _INTERPRETATION
    assert isinstance(result, str)
    # Only the interpretation string comes back -- no other field's concrete
    # values leak into the projected result.
    assert "evt:1" not in result
    assert "node:a" not in result
    assert "node:b" not in result
    assert "loop:open-1" not in result
    assert "chain:123" not in result


def test_returns_interpretation_verbatim_for_json_string_payload():
    payload = _fresh_payload()
    ctx = {"latest_reverie_thought": json.dumps(payload)}
    result = _project_reverie_glimpse(ctx)
    assert result == _INTERPRETATION


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
    assert ctx["chat_reverie_glimpse"] == _INTERPRETATION
    assert isinstance(ctx["chat_reverie_glimpse"], str)
