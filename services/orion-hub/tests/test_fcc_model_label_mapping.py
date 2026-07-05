from __future__ import annotations

import pytest

from scripts.fcc_model_mapping import (
    DEFAULT_FCC_MODEL_LABEL,
    label_to_claude_model_id,
)


def test_label_to_tier_model_ids() -> None:
    assert label_to_claude_model_id("MODEL") == "claude-sonnet-4-20250514"
    assert label_to_claude_model_id("MODEL_OPUS") == "claude-opus-4-20250514"
    assert label_to_claude_model_id("MODEL_SONNET") == "claude-sonnet-4-20250514"
    assert label_to_claude_model_id("MODEL_HAIKU") == "claude-haiku-4-20250514"


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="unknown fcc model label"):
        label_to_claude_model_id("MODEL_GHOST")


def test_default_label_is_model() -> None:
    assert DEFAULT_FCC_MODEL_LABEL == "MODEL"
