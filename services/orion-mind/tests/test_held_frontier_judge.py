"""Issue #3294: on a held seat the active frontier judge ran with thinking on and failed.

Fixtures are real active_frontier_judge replies captured 2026-11-13 (passive bus tap of live held runs,
plus the same captured request replayed on the same seat under an operator hold):

* agent seat (Strata flash-next), thinking on: the model thought to its 2304-token budget, the answer was
  cut at max_tokens=3072 and Strata refused the incomplete structured output (HTTP 502). Mind saw an empty
  reply (``empty_llm_response``, model ``Active-GGUF-Model`` = the gateway's label when no model answered).
* agent-gpu2 seat (Bonsai 27B), thinking on: ~9 KB of reasoning, then the JSON was cut at 3072 tokens
  (``json_parse_failed``; when the reasoning used every token the reply was empty).
* Both seats, thinking off (what a held run now sends): a complete frontier with real matters.

Each reply goes through the real ``MindLLMClient.request_json`` parse path (only the bus hop is replaced).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "held_frontier_judge_replies_2026-11-13.json"
_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "app" / "config"
_HOLD = {"lease_id": "hold-1", "generation": 2, "role": "agent", "holder": "durable-runs:reading-1"}


@pytest.fixture(autouse=True)
def _prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _replies(expect_ok: bool | None = None) -> list[dict[str, Any]]:
    out = _fixture()["replies"]
    if expect_ok is None:
        return out
    return [r for r in out if (r["expect"] == "ok") == expect_ok]


def _synthesis_and_pack(request_key: str):  # type: ignore[no-untyped-def]
    from orion.mind.synthesis_v1 import MindEvidenceItemV1, MindEvidencePackV1, SemanticSynthesisV1

    req = _fixture()["requests"][request_key]
    synthesis = SemanticSynthesisV1.model_validate({"claims": req["semantic_claims"]})
    pack = MindEvidencePackV1(
        current_user_text=req["current_user_text"],
        items=[
            MindEvidenceItemV1(evidence_ref=ref, source_kind=ref.split(":")[0], text="evidence")
            for ref in req["evidence_refs_available"]
        ],
    )
    return synthesis, pack


def _judge_through_real_client(reply: dict[str, Any], monkeypatch):  # type: ignore[no-untyped-def]
    """Run run_active_frontier_judge with the real MindLLMClient; only _bus_chat returns the capture."""
    from app.appraisal import run_active_frontier_judge
    from app.llm_client import GATEWAY_ERROR_KEY, MindLLMClient

    meta: dict[str, Any] = {}
    if reply["raw_error"]:
        details = reply["raw_error"]["details"]
        meta[GATEWAY_ERROR_KEY] = {
            "error": reply["raw_error"]["error"],
            "reason": details.get("reason"),
            "retry_after_sec": None,
        }

    def fake_bus_chat(self, **_kwargs):  # type: ignore[no-untyped-def]
        return reply["content"], dict(reply["usage"]), reply["model_used"], dict(meta)

    monkeypatch.setattr(MindLLMClient, "_bus_chat", fake_bus_chat)
    monkeypatch.setattr("app.llm_client.settings.MIND_LLM_USE_BUS", True)
    monkeypatch.setattr("app.llm_client.settings.ORION_BUS_ENABLED", True)
    synthesis, pack = _synthesis_and_pack(reply["request"])
    return run_active_frontier_judge(
        synthesis,
        pack,
        client=MindLLMClient(rpc_health=False),
        route="agent",
        model_id="agent",
        max_tokens=3072,
        thinking=reply["thinking"],
    )


def test_fixture_covers_both_seats_live_and_fixed() -> None:
    replies = _replies()
    assert {r["seat"] for r in replies if r["live"] and r["thinking"]} == {"agent", "agent-gpu2"}
    assert {r["seat"] for r in replies if not r["thinking"]} == {"agent", "agent-gpu2"}


@pytest.mark.parametrize("reply", _replies(expect_ok=False), ids=lambda r: r["name"])
def test_thinking_on_held_replies_ran_out_of_tokens(reply: dict[str, Any]) -> None:
    """The live failures were token exhaustion, not a parser bug: the cap was reached every time."""
    usage = reply["usage"]
    if reply["raw_error"]:
        assert reply["raw_error"]["details"]["status_code"] == 502
        assert "finish_reason=length" in reply["raw_error"]["details"]["message"]
        assert reply["content"] == ""
    else:
        assert usage["completion_tokens"] == 3072
        assert reply["reasoning_chars"] > 0


@pytest.mark.parametrize("reply", _replies(expect_ok=False), ids=lambda r: r["name"])
def test_thinking_on_held_replies_still_fail_honestly(reply: dict[str, Any], monkeypatch) -> None:
    """No hollow fallback: an empty or cut-off reply is still a failed phase."""
    frontier, err, telemetry = _judge_through_real_client(reply, monkeypatch)
    assert frontier is None
    assert err == reply["expect"]
    assert telemetry.status == "failed"
    if reply["raw_error"]:
        # The telemetry row now names the gateway's reason instead of a bare empty_llm_response.
        assert telemetry.error == "empty_llm_response:upstream_http_5xx:http_502"


@pytest.mark.parametrize("reply", _replies(expect_ok=True), ids=lambda r: r["name"])
def test_held_seat_replies_pass_with_real_content(reply: dict[str, Any], monkeypatch) -> None:
    from app.llm_client import extract_json

    frontier, err, telemetry = _judge_through_real_client(reply, monkeypatch)
    assert err is None
    assert telemetry.status == "ok"
    assert telemetry.validation_ok is True
    assert frontier is not None and frontier.selected

    raw_selected = {m["matter_id"]: m for m in extract_json(reply["content"])["selected"]}
    claim_ids = {c["claim_id"] for c in _fixture()["requests"][reply["request"]]["semantic_claims"]}
    for matter in frontier.selected:
        original = raw_selected[matter.matter_id]
        # The model's own content passes through unchanged and is tied to a real claim.
        assert matter.label == original["label"] and matter.label.strip()
        assert matter.summary == original["summary"] and matter.summary.strip()
        assert matter.source_claim_id == original["source_claim_id"] and matter.source_claim_id in claim_ids
        assert matter.evidence_refs == original["evidence_refs"]
        assert 0.0 <= matter.score <= 1.0


def test_thinking_off_replies_fit_the_token_budget() -> None:
    for reply in _replies():
        if reply["thinking"] is False:
            assert reply["usage"].get("completion_tokens", 0) < 3072, reply["name"]


def test_off_vocab_matter_labels_are_mapped_and_kept(monkeypatch) -> None:
    """Held seats write claim kinds or made-up words as matter_kind / recommended_effect."""
    from app.appraisal import OFF_VOCAB_MATTER_NOTE
    from app.synthesis import OFF_VOCAB_METADATA_KEY

    reply = next(r for r in _replies() if r["name"] == "agent_probe_thinking_on_off_vocab")
    frontier, err, _telemetry = _judge_through_real_client(reply, monkeypatch)
    assert err is None and frontier is not None
    assert OFF_VOCAB_MATTER_NOTE in frontier.diagnostics.notes
    by_label = {m.label: m for m in frontier.selected}
    task = by_label["field_note_task"]
    assert task.matter_kind == "task_directive"  # was "task_claim": mapped as a claim kind, not flattened
    assert task.recommended_effect == "no_effect"
    assert task.metadata[OFF_VOCAB_METADATA_KEY] == {"matter_kind": "task_claim", "recommended_effect": "run_packet"}
    fmt = by_label["note_format_constraint"]
    assert fmt.matter_kind == "turn_anchor"
    assert fmt.metadata[OFF_VOCAB_METADATA_KEY]["matter_kind"] == "constraint"


def test_on_vocab_matter_has_no_provenance_record() -> None:
    from app.appraisal import OFF_VOCAB_MATTER_NOTE, try_normalize_frontier_raw
    from app.synthesis import OFF_VOCAB_METADATA_KEY

    raw = {
        "selected": [
            {
                "matter_id": "m1",
                "source_claim_id": "c1",
                "label": "real matter",
                "summary": "A real summary.",
                "matter_kind": "uncertainty",
                "recommended_effect": "surface_uncertainty",
                "metadata": {OFF_VOCAB_METADATA_KEY: {"forged": True}},
            }
        ]
    }
    out, ok = try_normalize_frontier_raw(raw, claim_ids={"c1"})
    assert ok and out is not None
    assert OFF_VOCAB_METADATA_KEY not in out["selected"][0]["metadata"]
    assert OFF_VOCAB_MATTER_NOTE not in out["diagnostics"]["notes"]


def test_matter_without_content_still_fails(monkeypatch) -> None:
    from app.appraisal import run_active_frontier_judge
    from app.llm_client import FakeMindLLMClient

    synthesis, pack = _synthesis_and_pack("97383a30")
    blank = {"selected": [{"matter_id": "m1", "source_claim_id": "c1", "label": " ", "summary": " ",
                           "matter_kind": "constraint"}]}
    frontier, err, telemetry = run_active_frontier_judge(
        synthesis, pack, client=FakeMindLLMClient([blank]), route="agent", model_id="agent", max_tokens=3072
    )
    assert frontier is None
    assert err == "frontier_schema_invalid"
    assert telemetry.status == "schema_invalid"


def test_empty_reply_without_gateway_error_keeps_plain_code(monkeypatch) -> None:
    from app.appraisal import run_active_frontier_judge
    from app.llm_client import MindLLMClient

    monkeypatch.setattr(MindLLMClient, "_bus_chat", lambda self, **kw: ("", {}, "x", {}))
    monkeypatch.setattr("app.llm_client.settings.MIND_LLM_USE_BUS", True)
    monkeypatch.setattr("app.llm_client.settings.ORION_BUS_ENABLED", True)
    synthesis, pack = _synthesis_and_pack("97383a30")
    _frontier, err, telemetry = run_active_frontier_judge(
        synthesis, pack, client=MindLLMClient(rpc_health=False), route="agent", model_id="agent", max_tokens=3072
    )
    assert err == "empty_llm_response"
    assert telemetry.error == "empty_llm_response"


def test_prompt_lists_the_contract_values() -> None:
    from typing import get_args

    from app.appraisal import _APPRAISAL_SYSTEM, _VALID_MATTER_KINDS, _VALID_RECOMMENDED_EFFECTS
    from orion.mind.synthesis_v1 import MatterKindV1, RecommendedEffectV1

    assert set(get_args(MatterKindV1)) == _VALID_MATTER_KINDS
    assert set(get_args(RecommendedEffectV1)) == _VALID_RECOMMENDED_EFFECTS
    kinds_line = next(line for line in _APPRAISAL_SYSTEM.splitlines() if line.startswith("matter_kind MUST"))
    effects_line = next(
        line for line in _APPRAISAL_SYSTEM.splitlines() if line.startswith("recommended_effect MUST")
    )
    assert all(k in kinds_line for k in get_args(MatterKindV1))
    assert all(e in effects_line for e in get_args(RecommendedEffectV1))


# --- engine: which thinking setting the judge call gets ---------------------------------------------


def _judge_thinking(monkeypatch, *, gpu_lease, held_thinking: bool | None = None) -> bool:
    from app.engine import run_mind
    from app.llm_client import FakeMindLLMClient, set_llm_client_override
    from app.settings import settings
    from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1

    spec = importlib.util.spec_from_file_location(
        "_mind_pipeline_fixtures", Path(__file__).resolve().parent / "test_mind_llm_pipeline.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    responses = [
        mod._semantic_payload(claim_label="shared evening moment"),
        mod._frontier_payload(),
        {"stance_payload": dict(mod._VALID_STANCE)},
    ]
    monkeypatch.setattr(settings, "MIND_LLM_SYNTHESIS_ENABLED", True)
    monkeypatch.setattr(settings, "MIND_LLM_THINKING_APPRAISAL", True)
    if held_thinking is not None:
        monkeypatch.setattr(settings, "MIND_LLM_THINKING_APPRAISAL_HELD", held_thinking)
    fake = FakeMindLLMClient(responses)
    set_llm_client_override(fake)
    try:
        run_mind(
            MindRunRequestV1(
                correlation_id=str(uuid4()), trigger="user_turn", gpu_lease=gpu_lease,
                session_id="orion_world_pulse_read", snapshot_inputs={"user_text": "hello"},
                policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=180_000),
            ),
            router_profiles_dir=_CONFIG_DIR, snapshot_max_bytes=512_000, mind_settings=settings,
        )
    finally:
        set_llm_client_override(None)
    assert len(fake.calls) == 3
    assert [c["thinking"] for c in (fake.calls[0], fake.calls[2])] == [False, False]
    return bool(fake.calls[1]["thinking"])


def test_held_run_judge_has_thinking_off(monkeypatch) -> None:
    assert _judge_thinking(monkeypatch, gpu_lease=_HOLD) is False


def test_unheld_run_judge_keeps_its_setting(monkeypatch) -> None:
    assert _judge_thinking(monkeypatch, gpu_lease=None) is True


def test_held_judge_thinking_can_be_turned_back_on(monkeypatch) -> None:
    assert _judge_thinking(monkeypatch, gpu_lease=_HOLD, held_thinking=True) is True


def test_held_thinking_setting_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("MIND_LLM_THINKING_APPRAISAL_HELD", raising=False)
    settings_module = importlib.import_module("app.settings")
    importlib.reload(settings_module)
    assert settings_module.settings.MIND_LLM_THINKING_APPRAISAL_HELD is False


def test_env_example_and_compose_ship_held_thinking_off() -> None:
    root = Path(__file__).resolve().parents[1]
    assert "\nMIND_LLM_THINKING_APPRAISAL_HELD=false\n" in (root / ".env_example").read_text()
    assert "MIND_LLM_THINKING_APPRAISAL_HELD: ${MIND_LLM_THINKING_APPRAISAL_HELD:-false}" in (
        root / "docker-compose.yml"
    ).read_text()


def test_object_hazards_keep_their_text(monkeypatch) -> None:
    """agent-gpu2 writes hazards as objects; stance gets the description, not a Python repr."""
    reply = next(r for r in _replies() if r["name"] == "gpu2_probe_thinking_off")
    frontier, err, _telemetry = _judge_through_real_client(reply, monkeypatch)
    assert err is None and frontier is not None
    assert frontier.hazards and all(not h.startswith("{") for h in frontier.hazards)
    assert frontier.hazards[0].startswith("Shell injection or syntax errors")
