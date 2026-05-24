"""Unit tests for OpenAI-shaped logprob summary extraction."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.llm_uncertainty import (  # noqa: E402
    extract_llm_uncertainty_from_openai_response,
    summarize_logprob_content,
)


def _sample_content():
    return [
        {"token": "The", "logprob": -0.1, "top_logprobs": [{"token": "The", "logprob": -0.1}, {"token": "A", "logprob": -2.5}]},
        {"token": " cat", "logprob": -0.3, "top_logprobs": [{"token": " cat", "logprob": -0.3}, {"token": " dog", "logprob": -1.8}]},
        {"token": " sat", "logprob": -3.5, "top_logprobs": [{"token": " sat", "logprob": -3.5}, {"token": " ran", "logprob": -3.2}]},
    ]


def test_summarize_logprob_content_computes_means_and_counts() -> None:
    summary = summarize_logprob_content(_sample_content())
    assert summary["available"] is True
    assert summary["token_count_observed"] == 3
    assert summary["mean_logprob"] == pytest.approx((-0.1 + -0.3 + -3.5) / 3, rel=1e-3)
    assert summary["min_logprob"] == pytest.approx(-3.5, rel=1e-3)
    assert summary["low_logprob_token_count"] >= 1
    assert summary["schema_version"] == "v1"


def test_extract_from_openai_response_reads_choices_logprobs() -> None:
    raw = {"choices": [{"logprobs": {"content": _sample_content()}}]}
    out = extract_llm_uncertainty_from_openai_response(raw, source="llamacpp_openai_chat")
    assert out is not None
    assert out["source"] == "llamacpp_openai_chat"
    assert out["available"] is True
    assert out["diagnostic_only"] is True


def test_extract_returns_none_when_no_logprobs() -> None:
    assert extract_llm_uncertainty_from_openai_response({"choices": [{}]}, source="x") is None
