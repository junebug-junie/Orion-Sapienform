"""Cockpit turn sighting timeline read API (join/gap logic only).

Persistence shape is covered by
services/orion-sql-writer/tests/test_cockpit_turn_sighting_sql_shape.py.
These tests cover ordering, canonical stage gaps, and empty-timeline honesty.
"""

from __future__ import annotations

import os

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

import pytest

from scripts import chat_cockpit_routes as mod


@pytest.mark.asyncio
async def test_empty_timeline_reports_all_canonical_gaps(monkeypatch):
    monkeypatch.setattr(mod, "_load_hops", lambda cid: [])
    body = await mod.get_cockpit_timeline("corr-1")
    assert body["correlation_id"] == "corr-1"
    assert body["hops"] == []
    assert body["complete"] is False
    assert "stance_decision" in body["gaps"]
    assert "motor_hop" in body["gaps"]
    assert "pre_turn_appraisal" in body["gaps"]
    assert "thought_rpc" in body["gaps"]
    assert "harness_dispatch" in body["gaps"]


@pytest.mark.asyncio
async def test_timeline_orders_by_seq_and_shrinks_gaps(monkeypatch):
    monkeypatch.setattr(
        mod,
        "_load_hops",
        lambda cid: [
            {"correlation_id": cid, "seq": 1, "stage": "motor_hop", "status": "ok"},
            {"correlation_id": cid, "seq": 0, "stage": "stance_decision", "status": "ok"},
        ],
    )
    body = await mod.get_cockpit_timeline("corr-1")
    assert [h["seq"] for h in body["hops"]] == [0, 1]
    assert "stance_decision" not in body["gaps"]
    assert "motor_hop" not in body["gaps"]
    assert "motor_boot" in body["gaps"]


@pytest.mark.asyncio
async def test_ingress_ok_hop_removes_ingress_from_gaps(monkeypatch):
    monkeypatch.setattr(
        mod,
        "_load_hops",
        lambda cid: [
            {
                "correlation_id": cid,
                "seq": 0,
                "stage": "ingress",
                "status": "ok",
                "visor_line": "ingress · 5 chars",
                "raw": {"user_message": "hello", "observation_published": False},
            },
        ],
    )
    body = await mod.get_cockpit_timeline("corr-ingress")
    assert "ingress" not in body["gaps"]
    assert body["hops"][0]["status"] == "ok"
    assert body["hops"][0]["stage"] == "ingress"
