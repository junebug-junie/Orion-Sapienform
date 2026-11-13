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
        assert claim.evidence_refs == original["evidence_refs"]
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


@pytest.mark.parametrize("summary", ["", "   ", None, ["a list"]])
def test_off_vocab_claim_without_content_still_fails(summary: Any) -> None:
    """Kill means kill: a claim with no real summary is not rescued by the enum coercion."""
    raw = {
        "schema_version": "mind.semantic_synthesis.v1",
        "claims": [
            {
                "claim_id": "c1",
                "label": "event",
                "summary": summary,
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


@pytest.mark.parametrize("confidence, ok", [(7, False), (0.7, True)])
def test_out_of_range_confidence_still_fails(confidence: float, ok: bool) -> None:
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
                "confidence": confidence,
            }
        ],
    }
    result, err, _ = _run(raw)
    if ok:
        assert err is None and result is not None and result.claims
    else:
        assert result is None
        assert err == "semantic_schema_invalid"


def _one_claim(**overrides: Any) -> dict[str, Any]:
    claim = {
        "claim_id": "c1",
        "label": "update issue",
        "summary": "An article reports Windows update install failures.",
        "claim_kind": "event_claim",
        "evidence_refs": ["current_turn:0"],
        "source_kinds": ["current_turn"],
        "anchor": "event",
        "recommended_effect": "acknowledge",
    }
    claim.update(overrides)
    return {"schema_version": "mind.semantic_synthesis.v1", "claims": [claim]}


@pytest.mark.parametrize("field", ["claim_kind", "anchor", "recommended_effect"])
@pytest.mark.parametrize("value", [["event_claim"], {"k": "v"}, 3])
def test_non_string_label_values_do_not_crash(field: str, value: Any) -> None:
    from app.synthesis import OFF_VOCAB_METADATA_KEY

    result, err, telemetry = _run(_one_claim(**{field: value}))
    if field == "claim_kind":
        # claim_kind is required content: a non-string kind is not a real claim.
        assert result is None and err == "semantic_schema_invalid"
    else:
        assert err is None and telemetry.status == "ok"
        assert result is not None
        assert result.claims[0].metadata[OFF_VOCAB_METADATA_KEY][field] == value


def test_non_dict_metadata_still_fails() -> None:
    result, err, _ = _run(_one_claim(metadata="keep me"))
    assert result is None
    assert err == "semantic_schema_invalid"


def test_model_written_provenance_key_is_overwritten_by_the_real_record() -> None:
    from app.synthesis import OFF_VOCAB_METADATA_KEY

    result, err, _ = _run(_one_claim(metadata={"off_vocab_fields": "model wrote this", "x": 1}))
    assert err is None and result is not None
    meta = result.claims[0].metadata
    assert meta["off_vocab_fields"] == "model wrote this"
    assert meta["x"] == 1
    assert meta[OFF_VOCAB_METADATA_KEY]["claim_kind"] == "event_claim"


def test_off_vocab_claim_citing_only_situation_is_not_promoted_to_turn_anchor() -> None:
    from app.synthesis import try_coerce_off_vocab_claim_fields

    raw = _one_claim(claim_kind="state_claim", evidence_refs=["situation_compact:1"], source_kinds=["situation_compact"])
    out, did = try_coerce_off_vocab_claim_fields(raw)
    assert did and out is not None
    assert out["claims"][0]["claim_kind"] == "situation_claim"
    out2, _ = try_coerce_off_vocab_claim_fields(_one_claim())
    assert out2 is not None and out2["claims"][0]["claim_kind"] == "current_turn_claim"


def test_legacy_claim_with_invented_labels_is_normalized_then_coerced() -> None:
    from app.synthesis import OFF_VOCAB_NOTE, _validate_semantic_raw

    raw = {
        "schema_version": "mind.semantic_synthesis.v1",
        "claims": [{"claim": "An article reports update failures.", "evidence_refs": ["current_turn:0"],
                    "anchor": "event", "recommended_effect": "follow"}],
    }
    synthesis, normalized = _validate_semantic_raw(raw)
    assert normalized is True and synthesis is not None
    assert "normalized_from_legacy_claim_shape" in synthesis.diagnostics.notes
    assert OFF_VOCAB_NOTE in synthesis.diagnostics.notes
    assert synthesis.claims[0].anchor == "unknown"
    assert synthesis.claims[0].recommended_effect == "no_effect"


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
        listed = [
            {v.strip() for v in ln.split("one of: ", 1)[1].rstrip(".").split(",")}
            for ln in synthesis._SEMANTIC_SYSTEM.splitlines()
            if "MUST be exactly one of: " in ln
        ]
        assert set(values) in listed


def test_singleton_root_with_invented_labels_keeps_provenance() -> None:
    from app.synthesis import OFF_VOCAB_METADATA_KEY, _validate_semantic_raw

    claim = _one_claim()["claims"][0]
    synthesis, normalized = _validate_semantic_raw(claim)
    assert normalized is True and synthesis is not None
    record = synthesis.claims[0].metadata[OFF_VOCAB_METADATA_KEY]
    assert record == {"claim_kind": "event_claim", "anchor": "event", "recommended_effect": "acknowledge"}
