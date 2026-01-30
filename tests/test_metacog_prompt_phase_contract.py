from pathlib import Path


def _read_template(name: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "orion" / "cognition" / "prompts" / name).read_text()


def _extract_example_json(text: str) -> str:
    start_tag = "<example_json>"
    end_tag = "</example_json>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    assert start != -1, "example_json start tag missing"
    assert end != -1, "example_json end tag missing"
    return text[start + len(start_tag):end].strip()


def test_draft_prompt_requires_patch_only():
    text = _read_template("log_orion_metacognition_draft.j2")
    example = _extract_example_json(text)
    assert "MetacogDraftTextPatchV1" in text
    assert "FULL CollapseMirrorEntryV2" not in text
    for forbidden_key in ("event_id", "state_snapshot", "tags", "tag_scores", "change_type_scores"):
        assert f"\"{forbidden_key}\"" not in example


def test_enrich_prompt_requires_patch_only():
    text = _read_template("log_orion_metacognition_enrich.j2")
    example = _extract_example_json(text)
    assert "MetacogEnrichScorePatchV1" in text
    assert "FULL CollapseMirrorEntryV2" not in text
    for forbidden_key in ("summary", "mantra", "what_changed", "event_id", "state_snapshot", "tags"):
        assert f"\"{forbidden_key}\"" not in example
