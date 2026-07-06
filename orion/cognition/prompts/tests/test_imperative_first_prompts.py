from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_voice_finalize_uses_grammar_not_contract_flags() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/orion_voice_finalize.j2").read_text(encoding="utf-8")
    assert "grammar_receipts" in text
    assert "requires_repo_grounding" not in text
    assert text.count("alignment_verdict is misaligned") == 1


def test_reflect_prompt_includes_grammar_receipts() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/harness_finalize_reflect.j2").read_text(encoding="utf-8")
    assert "grammar_receipts" in text
