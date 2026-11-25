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


def test_judge_and_repair_never_treat_a_cut_receipt_as_absence() -> None:
    """corr 8297d0c3: facts from the unseen part of a cut receipt were called unsupported, and the
    repair wrote "not in my records". Both prompts must read the cut mark the motor writes."""
    from orion.harness.fcc_motor import TOOL_RESULT_CUT_MARK

    mark = TOOL_RESULT_CUT_MARK.split(":", 1)[0]  # "[receipt cut"
    reflect = " ".join((REPO / "orion/cognition/prompts/harness_finalize_reflect.j2").read_text(encoding="utf-8").split())
    repair = " ".join((REPO / "orion/cognition/prompts/orion_response_repair.j2").read_text(encoding="utf-8").split())
    assert f'"{mark}: ...]"' in reflect and "same subject as that cut result" in reflect
    # Scoped: a cut receipt excuses only claims on its own subject, never an unrelated invention.
    assert "A claim on a subject no receipt was about, or one the visible text contradicts, is still unsupported" in reflect
    assert f'"{mark}: ...]"' in repair and "Never write that something is not in Orion's records" in repair
