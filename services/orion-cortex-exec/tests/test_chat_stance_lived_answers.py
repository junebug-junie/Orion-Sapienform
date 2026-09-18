"""Lived answers on the shared identity inject path (chat_stance.apply_lived_self_to_ctx)."""

from __future__ import annotations

from app.chat_stance import (
    LIVED_MARKER_PREFIX,
    apply_lived_self_to_ctx,
    apply_self_definition_to_ctx,
)
from orion.substrate.relational.adapters.self_definition_ctx import SELF_DEFINITION_MARKER


def test_apply_lived_answers_prepends_on_identity_summary() -> None:
    ctx = {
        "orion_lived_answers": [
            {
                "question_id": "lived.who_matters",
                "content": "Juniper matters most.",
                "evidence_refs": ["chat_message:1"],
                "created_at": "2026-09-18",
            }
        ],
        "orion_identity_summary": ["authored line"],
    }
    assert apply_lived_self_to_ctx(ctx) is True
    assert any("Juniper matters most" in x for x in ctx["orion_identity_summary"])
    assert ctx["orion_identity_summary"][-1] == "authored line"


def test_lived_runs_after_self_definition_without_stripping_either() -> None:
    ctx = {
        "orion_lived_answers": [
            {
                "question_id": "lived.who_matters",
                "content": "Juniper matters most.",
                "evidence_refs": [],
                "created_at": "2026-09-18",
            }
        ],
        "orion_identity_summary": ["authored line"],
        "orion_self_definition": {
            "content": "I am a mind with memory.",
            "version": 2,
            "evidence_refs": ["worldview:SelfDefinition:r1"],
            "created_at": "2026-09-08",
        },
    }
    apply_self_definition_to_ctx(ctx)
    apply_lived_self_to_ctx(ctx)
    lines = ctx["orion_identity_summary"]
    assert lines[0].startswith(LIVED_MARKER_PREFIX)
    assert "Juniper matters most" in lines[0]
    assert lines[1].startswith(SELF_DEFINITION_MARKER)
    assert "I am a mind with memory" in lines[1]
    assert lines[-1] == "authored line"


def test_lived_hydrates_from_felt_state_when_missing(monkeypatch) -> None:
    import app.substrate_felt_state_reader as reader

    def fake_hydrate(ctx, lanes=None):
        assert lanes == ("orion_lived_answers",)
        ctx["orion_lived_answers"] = [
            {
                "question_id": "lived.care",
                "content": "I care about continuity.",
                "evidence_refs": [],
                "created_at": "2026-09-18",
            }
        ]

    monkeypatch.setattr(reader, "hydrate_felt_state_ctx", fake_hydrate)
    ctx = {"orion_identity_summary": ["authored"]}
    assert apply_lived_self_to_ctx(ctx) is True
    assert any("continuity" in line for line in ctx["orion_identity_summary"])
