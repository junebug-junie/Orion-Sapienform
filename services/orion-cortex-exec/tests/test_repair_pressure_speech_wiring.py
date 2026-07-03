from __future__ import annotations

from app.chat_stance import compile_speech_contract
from orion.schemas.chat_stance import ChatStanceBrief
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY


def _resolve_speech_contract(metadata: dict | None, *, enabled: bool = True) -> str:
    """Mirror executor merge logic under test."""
    brief = ChatStanceBrief(
        conversation_frame="mixed",
        task_mode="direct_response",
        user_intent="fix this",
        self_relevance="operational",
        juniper_relevance="practical",
        answer_strategy="direct",
        stance_summary="repair",
    )
    repair_contract = None
    if enabled and isinstance(metadata, dict):
        raw = metadata.get(REPAIR_PRESSURE_CONTRACT_METADATA_KEY)
        if isinstance(raw, dict):
            repair_contract = raw
    return compile_speech_contract(brief, repair_contract=repair_contract)


def test_metadata_repair_concrete_reaches_speech_contract() -> None:
    md = {
        REPAIR_PRESSURE_CONTRACT_METADATA_KEY: {
            "mode": "repair_concrete",
            "rules": ["include file/module boundaries"],
        }
    }
    text = _resolve_speech_contract(md)
    assert "include file/module boundaries" in text
    assert "Repair turn" in text


def test_metadata_absent_keeps_instrumental_contract() -> None:
    text = _resolve_speech_contract({})
    assert text == "Answer directly."


def test_metadata_disabled_keeps_instrumental_contract() -> None:
    md = {
        REPAIR_PRESSURE_CONTRACT_METADATA_KEY: {
            "mode": "repair_concrete",
            "rules": ["include file/module boundaries"],
        }
    }
    text = _resolve_speech_contract(md, enabled=False)
    assert text == "Answer directly."
