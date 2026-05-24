"""Mind llm_uncertainty telemetry for semantic synthesis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _mind_prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


@pytest.fixture(autouse=True)
def _prep() -> None:
    _mind_prep()


_UNC = {
    "schema_version": "v1",
    "available": True,
    "mean_logprob": -0.4,
    "mean_top1_margin": 0.9,
    "token_count_observed": 10,
    "low_logprob_token_count": 1,
    "unstable_span_count": 0,
}


def _semantic_payload() -> dict[str, Any]:
    return {
        "schema_version": "mind.semantic_synthesis.v1",
        "model_id": "quick",
        "extraction_mode": "llm",
        "claims": [
            {
                "claim_id": "c1",
                "label": "evening plan",
                "summary": "User mentions watching a show.",
                "claim_kind": "relationship_claim",
                "evidence_refs": ["current_turn:0"],
                "source_kinds": ["current_turn"],
                "anchor": "relationship",
                "confidence": 0.9,
                "salience_hint": 0.8,
                "recommended_effect": "receive_warmly",
            }
        ],
        "suppressed": [],
        "diagnostics": {"evidence_item_count": 1, "llm_ok": True},
    }


def test_semantic_synthesis_telemetry_carries_llm_uncertainty(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.evidence import build_evidence_pack
    from app.llm_context import MindLLMRequestContext
    from app.settings import settings
    from app.synthesis import run_semantic_synthesis

    monkeypatch.setattr(settings, "MIND_LLM_RETURN_LOGPROBS_SEMANTIC", True)
    monkeypatch.setattr(settings, "MIND_LLM_LOGPROB_PROBE_MODE", "native_completion")

    class FakeClient:
        def request_json(self, **kwargs: Any) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
            assert kwargs.get("extra_options") == {
                "return_logprobs": True,
                "logprobs_top_k": 5,
                "logprob_summary_only": True,
                "logprob_probe_mode": "native_completion",
            }
            return _semantic_payload(), None, {"model_used": "quick", "llm_uncertainty": _UNC}

    pack = build_evidence_pack(
        {
            "user_text": "watching a show with Amanda tonight",
            "messages_tail": [{"role": "user", "content": "watching a show with Amanda tonight"}],
        }
    )
    ctx = MindLLMRequestContext(
        correlation_id="00000000-0000-4000-8000-000000000001",
        mind_run_id="run-1",
        phase_name="semantic_synthesis",
    )
    result, err, telemetry = run_semantic_synthesis(
        pack,
        client=FakeClient(),
        route="quick",
        model_id="quick",
        max_tokens=512,
        context=ctx,
    )
    assert err is None
    assert result is not None
    assert telemetry.llm_uncertainty == _UNC
    assert telemetry.to_dict().get("llm_uncertainty") == _UNC
