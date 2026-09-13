from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def test_response_repair_prompt_is_minimal() -> None:
    text = (REPO / "orion/cognition/prompts/orion_response_repair.j2").read_text(encoding="utf-8")
    assert "smallest necessary correction" in text.lower() or "smallest necessary" in text
    for banned in (
        "refine voice and rhythm",
        "STYLE RULES",
        "WHO YOU ARE",
        "VOICE CONTRACT",
        "STANCE HARNESS",
        "companion presence",
    ):
        assert banned not in text
    assert "orion_voice_finalize" not in text
    assert not (REPO / "orion/cognition/prompts/orion_voice_finalize.j2").exists()
    assert not (REPO / "orion/cognition/verbs/orion_voice_finalize.yaml").exists()
    assert (REPO / "orion/cognition/verbs/orion_response_repair.yaml").exists()
