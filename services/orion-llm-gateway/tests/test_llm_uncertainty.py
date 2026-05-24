"""Unit tests for OpenAI-shaped logprob summary extraction."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.llm_uncertainty import (  # noqa: E402
    extract_llm_uncertainty_from_native_completion,
    extract_llm_uncertainty_from_openai_response,
    native_completion_probs_to_logprob_content,
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


def test_summarize_includes_confidence_semantics() -> None:
    summary = summarize_logprob_content(_sample_content())
    assert summary["confidence_semantics"] == "language_surface_stability_not_truth"


def test_summarize_unstable_span_counts_one_run_at_min_len() -> None:
    """Three consecutive low-margin tokens (min_len=3) produce one unstable span."""
    low_margin = {
        "token": "x",
        "logprob": -0.1,
        "top_logprobs": [
            {"token": "x", "logprob": -0.1},
            {"token": "y", "logprob": -0.15},
        ],
    }
    high_margin = {
        "token": "z",
        "logprob": -0.1,
        "top_logprobs": [
            {"token": "z", "logprob": -0.1},
            {"token": "w", "logprob": -2.0},
        ],
    }
    content = [low_margin, low_margin, low_margin, high_margin]
    summary = summarize_logprob_content(content)
    assert summary["unstable_span_count"] == 1


def _native_prob_token(token: str, logprob: float, alt_token: str, alt_logprob: float) -> dict:
    return {
        "token": token,
        "logprob": logprob,
        "top_logprobs": [
            {"token": token, "logprob": logprob},
            {"token": alt_token, "logprob": alt_logprob},
        ],
    }


def test_native_completion_probs_to_logprob_content_reads_probs_array() -> None:
    raw = {
        "content": "The cat sat",
        "probs": [
            _native_prob_token("The", -0.1, "A", -2.5),
            _native_prob_token(" cat", -0.3, " dog", -1.8),
        ],
    }
    content = native_completion_probs_to_logprob_content(raw)
    assert len(content) == 2
    assert content[0]["token"] == "The"


def test_native_completion_probs_to_logprob_content_reads_completion_probabilities() -> None:
    raw = {
        "completion_probabilities": [
            {
                "content": "Hi",
                "probs": [_native_prob_token("Hi", -0.2, "Hey", -1.5)],
            }
        ]
    }
    content = native_completion_probs_to_logprob_content(raw)
    assert len(content) == 1
    assert content[0]["logprob"] == pytest.approx(-0.2, rel=1e-3)


def test_extract_from_native_completion_sets_source() -> None:
    raw = {
        "content": "OK",
        "probs": [_native_prob_token("OK", -0.2, "NO", -2.0)],
    }
    out = extract_llm_uncertainty_from_native_completion(raw)
    assert out is not None
    assert out["source"] == "llamacpp_native_completion"
    assert out["available"] is True
    assert out["confidence_semantics"] == "language_surface_stability_not_truth"


def test_extract_from_native_completion_returns_none_without_probs() -> None:
    assert extract_llm_uncertainty_from_native_completion({"content": "x"}) is None
