from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILES_YAML = REPO_ROOT / "config" / "vision_profiles.yaml"

# runner.py's last-resort list, reached only when neither the request nor the
# profile supplies prompts. Duplicated rather than imported because importing
# the runner pulls in torch/PIL, which are container-only.
RUNNER_FALLBACK = {"person", "face", "phone", "screen", "door", "package"}


def _detector_params() -> dict:
    cfg = yaml.safe_load(PROFILES_YAML.read_text())
    profile = next(
        p for p in cfg["profiles"] if p["name"] == "retina_detect_open_vocab"
    )
    return profile["params"]


def test_detector_ships_a_default_vocabulary() -> None:
    """
    Regression: with `default_prompts` unset, the runner fell back to a
    six-word list, so an open-vocabulary detector could only ever report
    doors, screens and packages. Live cam0 narratives over 19 days were
    exactly that and nothing else.
    """
    prompts = _detector_params().get("default_prompts") or []
    assert prompts, "retina_detect_open_vocab must ship default_prompts"
    assert len(prompts) > len(RUNNER_FALLBACK)


def test_vocabulary_reaches_past_the_hardcoded_fallback() -> None:
    prompts = set(_detector_params()["default_prompts"])
    assert prompts - RUNNER_FALLBACK, "vocabulary adds nothing over the fallback"


def test_no_identity_terms_in_vocabulary() -> None:
    """Identity / face / re-ID is a stated non-goal of the perception design."""
    banned = {"face", "person's face", "identity", "id card", "license plate"}
    prompts = {p.lower() for p in _detector_params()["default_prompts"]}
    assert not (prompts & banned)


def test_vocabulary_entries_are_unique_and_clean() -> None:
    prompts = _detector_params()["default_prompts"]
    assert len(prompts) == len(set(prompts)), "duplicate prompt terms"
    for p in prompts:
        assert isinstance(p, str) and p.strip() == p and p, f"bad prompt: {p!r}"
        # ' . ' is the caption separator the runner joins on.
        assert "." not in p, f"prompt must not contain the separator: {p!r}"


def test_vocabulary_is_single_words_only() -> None:
    """
    GroundingDINO resolves detections to token spans in the dot-joined caption,
    and adjacent multi-word phrases bleed together. Live on 2026-08-12,
    `cardboard box . storage bin` produced the believed labels
    `cardboard box storage` and `storage`, and never `storage bin`. A garbled
    label propagates into the narrative and churns the habituation gate.
    """
    for prompt in _detector_params()["default_prompts"]:
        assert " " not in prompt, f"multi-word prompt risks span bleed: {prompt!r}"


def test_vocabulary_fits_the_text_encoder_budget() -> None:
    """
    GroundingDINO's BERT text encoder caps at 256 tokens; an overlong caption
    silently truncates and the tail terms stop being detectable.
    """
    caption = " . ".join(_detector_params()["default_prompts"])
    assert len(caption) < 900, f"caption likely to truncate: {len(caption)} chars"
