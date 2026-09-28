"""Replay of corr beab81a3 through probe -> attention frame -> stance_react.j2.

Live turn: "so far so good. I'll be pretty busy the next few days with work
travel, so won't have much time to do dev on you." Orion replied "Safe travels
... Catch you on the other side." with no curiosity. On that turn the probe
returned [], the only open loop was Orion's own "Biometrics prediction error"
thread, and the stance emitted avoid_open_ended_questions.

The probe's raw output is faked here (the live model read is covered by
evals/run_current_turn_disclosure_live_eval.py); everything downstream of it
runs the real code path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jinja2 import Environment

import app.current_turn_llm_signals as signals_module
from app.attention_frame import build_attention_frame
from app.current_turn_llm_signals import populate_current_turn_llm_signals
from orion.schemas.attention_frame import AttentionSignalV1

_USER_TEXT = (
    "so far so good. I'll be pretty busy the next few days with work travel, "
    "so won't have much time to do dev on you."
)
_PROMPT = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "stance_react.j2"


class _BiometricsThread:
    """Strength 0.58*0.58~=0.34 and 2 refs (breadth 0.5) reproduce the live
    attention_salience_trace row for this corr: salience 0.5,
    evidence_strength 0.34, evidence_breadth 0.5."""

    detector_id = "concept_induction_v1"

    def detect(self, ctx, inputs, belief_lineage):
        return [
            AttentionSignalV1(
                signal_id="biometrics-pe",
                source=self.detector_id,
                target_text="Biometrics prediction error",
                target_type_hint="anomaly",
                signal_kind="concept_pressure",
                salience=0.58,
                confidence=0.58,
                evidence_refs=["concept:biometrics", "substrate:biometrics"],
            )
        ]


def _inputs():
    return {
        "identity": {"orion": [], "juniper": [], "response_policy": []},
        "concept_induction": {"self": [], "relationship": [], "growth": [], "tension": []},
        "social": {"social_posture": [], "relationship_facets": [], "hazards": []},
        "reflective": {"themes": [], "tensions": [], "dream_motifs": []},
        "autonomy": {"summary": {"top_drives": [], "active_tensions": []}, "debug": {}},
        "reasoning_summary": {"hazards": [], "tensions": [], "fallback_recommended": False},
        "situation": {},
    }


async def _run_turn(monkeypatch, raw_probe_output: str) -> dict:
    from orion.substrate.attention.detectors import CurrentTurnSignalDetector

    async def _fake_llm_call(bus, *, prompt):
        return raw_probe_output

    monkeypatch.setattr(signals_module, "_llm_call", _fake_llm_call)
    signals_module.bind_current_turn_llm_signals_bus(object())
    ctx: dict = {"user_message": _USER_TEXT}
    await populate_current_turn_llm_signals(ctx)
    frame = build_attention_frame(
        ctx=ctx,
        inputs=_inputs(),
        detectors=[CurrentTurnSignalDetector(), _BiometricsThread()],
    )
    ctx["chat_attention_frame"] = frame.model_dump(mode="json")
    return ctx


def _render_stance(ctx: dict) -> str:
    return Environment().from_string(_PROMPT.read_text()).render(
        user_message=ctx["user_message"],
        stance_inputs={"user_message": ctx["user_message"]},
        association={},
        repair_bundle=None,
        coalition_projection=None,
        chat_attention_frame=ctx["chat_attention_frame"],
    )


@pytest.mark.asyncio
async def test_travel_disclosure_reaches_the_stance_prompt_as_orions_own_question(monkeypatch) -> None:
    ctx = await _run_turn(
        monkeypatch,
        json.dumps(
            {
                "wants_direct_answer": False,
                "items": [
                    {"phrase": "work travel the next few days", "type": "plan", "question": "Where are you headed?"}
                ],
            }
        ),
    )
    selected = ctx["chat_attention_frame"]["selected_action"]
    assert selected["action_type"] == "ask"
    assert selected["question_text"] == "Where are you headed?"
    assert "Biometrics" not in (selected["question_text"] or "")
    assert not any(s["reason"] == "user_needs_direct_answer" for s in ctx["chat_attention_frame"]["suppressions"])

    rendered = _render_stance(ctx)
    frame_block = rendered.split("ATTENTION FRAME", 1)[1].split("POSTURE ASSESSMENT", 1)[0]
    assert "Where are you headed?" in frame_block
    assert "Oríon's own curiosity" in frame_block
    assert "do not emit avoid_open_ended_questions" in frame_block
    assert "even when connection_seek is low" in frame_block
    relational = rendered.split("RELATIONAL VS INSTRUMENTAL DISCIPLINE", 1)[1]
    assert "unless ATTENTION FRAME selected an ask" in relational


@pytest.mark.asyncio
async def test_same_turn_with_the_old_empty_probe_read_asks_nothing(monkeypatch) -> None:
    """The live failure: an empty read leaves only Orion's background thread at
    its recorded strength, which stays below the ask bar -- no question at all."""
    ctx = await _run_turn(monkeypatch, json.dumps({"wants_direct_answer": False, "items": []}))
    selected = ctx["chat_attention_frame"]["selected_action"]
    assert (selected or {}).get("action_type") != "ask"


@pytest.mark.asyncio
async def test_same_turn_with_probe_down_fails_closed(monkeypatch) -> None:
    ctx = await _run_turn(monkeypatch, "not json at all")
    frame = ctx["chat_attention_frame"]
    assert any(s["target_ref"] == "turn_read_unavailable" for s in frame["suppressions"])
    assert (frame["selected_action"] or {}).get("action_type") != "ask"
