"""Issue #3274: held seats invent claim_kind / anchor / recommended_effect values.

Fixtures are real semantic_synthesis replies captured 2026-11-13 from the bus (agent seat, agent-gpu2
seat, and a metacog probe with the same request). On origin/main every one failed
``semantic_schema_invalid`` and Mind gave the held turn nothing (``fallback_contract_only``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "held_semantic_replies_2026-11-13.json"


def _replies() -> list[dict[str, Any]]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["replies"]


def _run(raw: Any):  # type: ignore[no-untyped-def]
    from app.evidence import build_evidence_pack
    from app.synthesis import run_semantic_synthesis

    pack = build_evidence_pack({"user_text": "held background turn"})

    class _Client:
        def request_json(self, **kwargs):  # type: ignore[no-untyped-def]
            return raw, None, {"model_used": "agent"}

    return run_semantic_synthesis(
        pack, client=_Client(), route="agent", model_id="agent", max_tokens=2048  # type: ignore[arg-type]
    )


def test_fixture_covers_both_held_seats_and_metacog() -> None:
    seats = {r["seat"] for r in _replies()}
    assert {"agent", "agent-gpu2", "metacog"} <= seats


@pytest.mark.parametrize("reply", _replies(), ids=lambda r: r["seat"])
def test_captured_reply_is_off_contract_as_captured(reply: dict[str, Any]) -> None:
    """The bug is real: the raw reply does not validate against the contract by itself."""
    from app.llm_client import extract_json
    from orion.mind.synthesis_v1 import SemanticSynthesisV1

    raw = extract_json(reply["content"])
    assert isinstance(raw, dict)
    with pytest.raises(ValidationError):
        SemanticSynthesisV1.model_validate(raw)


@pytest.mark.parametrize("reply", _replies(), ids=lambda r: r["seat"])
def test_captured_held_reply_passes_with_real_content(reply: dict[str, Any]) -> None:
    from app.llm_client import extract_json
    from app.synthesis import OFF_VOCAB_METADATA_KEY, OFF_VOCAB_NOTE

    raw = extract_json(reply["content"])
    raw_claims = raw["claims"]
    result, err, telemetry = _run(raw)

    assert err is None
    assert telemetry.status == "ok"
    assert telemetry.validation_ok is True
    assert telemetry.raw_claim_count == len(raw_claims)
    assert telemetry.retained_claim_count and telemetry.retained_claim_count > 0
    assert result is not None and result.claims
    assert OFF_VOCAB_NOTE in result.diagnostics.notes

    by_id = {c["claim_id"]: c for c in raw_claims}
    saw_off_vocab = False
    for claim in result.claims:
        original = by_id[claim.claim_id]
        # Real content passes through untouched.
        assert claim.summary == original["summary"] and claim.summary.strip()
        assert claim.label == original["label"]
        assert claim.evidence_refs
        # Every coerced label keeps its original value on the record.
        off_vocab = claim.metadata.get(OFF_VOCAB_METADATA_KEY) or {}
        for field, value in off_vocab.items():
            saw_off_vocab = True
            assert original[field] == value
        for field in ("claim_kind", "anchor", "recommended_effect"):
            if field not in off_vocab:
                assert getattr(claim, field) == original[field]
    assert saw_off_vocab


def test_gpu2_claims_citing_refs_outside_the_pack_are_still_filtered() -> None:
    from app.llm_client import extract_json

    reply = next(r for r in _replies() if r["seat"] == "agent-gpu2")
    _, err, telemetry = _run(extract_json(reply["content"]))
    assert err is None
    # c9/c10 cite situation_compact:1, which this pack does not carry: the guardrail drops them.
    assert telemetry.raw_claim_count == 10
    assert telemetry.retained_claim_count == 8
    assert telemetry.filter_reasons_by_count.get("unsupported_or_weak") == 2


def test_off_vocab_claim_without_content_still_fails() -> None:
    """Kill means kill: a claim with no summary is not rescued by the enum coercion."""
    raw = {
        "schema_version": "mind.semantic_synthesis.v1",
        "claims": [
            {
                "claim_id": "c1",
                "label": "event",
                "summary": "",
                "claim_kind": "event_claim",
                "evidence_refs": ["current_turn:0"],
                "source_kinds": ["current_turn"],
                "anchor": "event",
                "recommended_effect": "acknowledge",
            }
        ],
    }
    result, err, telemetry = _run(raw)
    assert result is None
    assert err == "semantic_schema_invalid"
    assert telemetry.status == "schema_invalid"


def test_off_vocab_claim_without_evidence_is_filtered_not_ok() -> None:
    raw = {
        "schema_version": "mind.semantic_synthesis.v1",
        "claims": [
            {
                "claim_id": "c1",
                "label": "made up",
                "summary": "A claim that cites nothing.",
                "claim_kind": "state_claim",
                "evidence_refs": [],
                "source_kinds": [],
                "anchor": "state",
                "recommended_effect": "monitor",
            }
        ],
    }
    result, err, telemetry = _run(raw)
    assert err is None
    assert telemetry.status == "filtered"
    assert telemetry.error == "semantic_synthesis_empty"
    assert result is not None and not result.claims


def test_out_of_range_confidence_still_fails() -> None:
    raw = {
        "schema_version": "mind.semantic_synthesis.v1",
        "claims": [
            {
                "claim_id": "c1",
                "label": "event",
                "summary": "Something happened.",
                "claim_kind": "event_claim",
                "evidence_refs": ["current_turn:0"],
                "source_kinds": ["current_turn"],
                "confidence": 7,
            }
        ],
    }
    result, err, _ = _run(raw)
    assert result is None
    assert err == "semantic_schema_invalid"


def test_on_contract_reply_is_not_marked_off_vocab() -> None:
    from app.synthesis import OFF_VOCAB_METADATA_KEY, OFF_VOCAB_NOTE

    raw = {
        "schema_version": "mind.semantic_synthesis.v1",
        "claims": [
            {
                "claim_id": "c1",
                "label": "shared evening plan",
                "summary": "User mentions watching a show.",
                "claim_kind": "relationship_claim",
                "evidence_refs": ["current_turn:0"],
                "source_kinds": ["current_turn"],
                "anchor": "relationship",
                "recommended_effect": "receive_warmly",
            }
        ],
    }
    result, err, telemetry = _run(raw)
    assert err is None and telemetry.status == "ok"
    assert result is not None
    assert OFF_VOCAB_NOTE not in result.diagnostics.notes
    assert OFF_VOCAB_METADATA_KEY not in result.claims[0].metadata


def test_prompt_names_every_allowed_value_from_the_contract() -> None:
    from app import synthesis
    from orion.mind.synthesis_v1 import ClaimAnchorV1, ClaimKindV1, RecommendedEffectV1

    for literal, valid in (
        (ClaimKindV1, synthesis._VALID_CLAIM_KINDS),
        (ClaimAnchorV1, synthesis._VALID_CLAIM_ANCHORS),
        (RecommendedEffectV1, synthesis._VALID_RECOMMENDED_EFFECTS),
    ):
        values = get_args(literal)
        assert set(values) == set(valid)
        line = next(
            ln for ln in synthesis._SEMANTIC_SYSTEM.splitlines() if ln.endswith(", ".join(values) + ".")
        )
        assert "MUST be exactly one of" in line
