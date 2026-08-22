"""Unit tests for OpenAI-shaped logprob summary extraction."""
import math
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


def test_native_completion_probs_to_logprob_content_reads_legacy_completion_probabilities() -> None:
    """Older, real llama.cpp /completion shape: a flat list of per-token entries, each
    carrying its own top-k alternatives as {"content": tok, "probs": [{"tok_str", "prob"}]}
    -- plain probabilities, not logprobs.

    A prior version of this function instead speculatively supported a single-element
    "wrapper holding a nested probs array" shape that never matched any real server
    response. Because that wrapper check ran first and only tested `first.get("probs")
    is a list`, it silently intercepted *this* real shape too (every per-token entry
    here also has a list under "probs"), took only the first token's alternatives, and
    dropped every other token -- plus those alternatives use "tok_str"/"prob", not the
    "token"/"logprob" keys the wrapper branch assumed, so extraction produced nothing.
    Caught by code review 2026-08-19; reproduced directly against this exact shape
    before the fix (returned `[]`)."""
    raw = {
        "completion_probabilities": [
            {"content": "Hi", "probs": [{"tok_str": "Hi", "prob": 0.8}, {"tok_str": "Hey", "prob": 0.15}]},
            {"content": " there", "probs": [{"tok_str": " there", "prob": 0.6}, {"tok_str": " friend", "prob": 0.3}]},
        ]
    }
    content = native_completion_probs_to_logprob_content(raw)
    assert len(content) == 2
    assert content[0]["token"] == "Hi"
    assert content[0]["logprob"] == pytest.approx(math.log(0.8), rel=1e-6)
    assert content[1]["token"] == " there"
    assert content[1]["logprob"] == pytest.approx(math.log(0.6), rel=1e-6)


def test_legacy_completion_probabilities_falls_back_when_sampled_token_missing_from_alternatives() -> None:
    """If top-k truncation drops the actually-sampled token from its own alternative
    list, fall back to the best listed alternative instead of dropping the token."""
    raw = {
        "completion_probabilities": [
            {"content": "Hi", "probs": [{"tok_str": "Hey", "prob": 0.5}, {"tok_str": "Yo", "prob": 0.3}]},
        ]
    }
    content = native_completion_probs_to_logprob_content(raw)
    assert len(content) == 1
    assert content[0]["logprob"] == pytest.approx(math.log(0.5), rel=1e-6)


def test_native_completion_probs_to_logprob_content_reads_flat_completion_probabilities() -> None:
    """The current real shape: live-verified 2026-08-19 against atlas-worker-fast-1's
    llama.cpp /completion endpoint with n_probs set. `completion_probabilities` is a
    flat list of per-token entries, each already carrying logprob/top_logprobs
    directly. Before this fix, the extractor only recognized a single-element
    "wrapper" shape that no real server actually sends, so this real shape produced
    an empty content list on every request, and `extract_llm_uncertainty_from_native_completion`
    silently returned None for every orion-mind semantic_synthesis call -- the
    llm_surface_instability metacog trigger never fired."""
    raw = {
        "content": '{"name": "Orion", "mood": "curious"}',
        "completion_probabilities": [
            {
                "id": 4913,
                "token": '{"',
                "logprob": 0.0,
                "top_logprobs": [
                    {"id": 4913, "token": '{"', "logprob": 0.0},
                    {"id": 5212, "token": ' {"', "logprob": -22.04},
                ],
            },
            {
                "id": 606,
                "token": "name",
                "logprob": -0.2,
                "top_logprobs": [
                    {"id": 606, "token": "name", "logprob": -0.2},
                    {"id": 829, "token": " name", "logprob": -3.1},
                ],
            },
        ],
    }
    content = native_completion_probs_to_logprob_content(raw)
    assert len(content) == 2
    assert content[0]["token"] == '{"'
    assert content[0]["logprob"] == pytest.approx(0.0, abs=1e-9)
    assert content[1]["logprob"] == pytest.approx(-0.2, rel=1e-3)

    summary = extract_llm_uncertainty_from_native_completion(raw)
    assert summary is not None
    assert summary["available"] is True
    assert summary["token_count_observed"] == 2


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
