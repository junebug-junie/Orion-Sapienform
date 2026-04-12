from __future__ import annotations

from orion.cognition.answer_contract_normalize import (
    bootstrap_answer_contract_on_request,
    enrich_answer_contract_after_routing,
    heuristic_answer_contract,
    output_modes_for_answer_contract_style,
)
from orion.schemas.cortex.contracts import (
    CortexClientContext,
    CortexClientRequest,
    OutputModeDecisionV1,
    RecallDirective,
)
from orion.core.bus.bus_schemas import LLMMessage
from orion.schemas.cognition.answer_contract import AnswerContractDraft


def _req(*, prompt: str, draft: dict | None = None, options: dict | None = None) -> CortexClientRequest:
    meta = {"answer_contract_draft": draft} if draft is not None else {}
    return CortexClientRequest(
        mode="brain",
        verb="chat_general",
        packs=[],
        recall=RecallDirective(),
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content=prompt)],
            raw_user_text=prompt,
            metadata=meta,
        ),
        options=options or {},
    )


def test_heuristic_runtime_debug() -> None:
    h = heuristic_answer_contract("docker compose logs show traceback")
    assert h.request_kind == "runtime_debug"
    assert h.requires_runtime_grounding is True


def test_bootstrap_merge_hub_draft() -> None:
    draft = AnswerContractDraft(request_kind="conceptual", preferred_render_style="comparison")
    req = _req(prompt="hello", draft=draft.model_dump(mode="json"))
    out = bootstrap_answer_contract_on_request(req)
    ac = (out.options or {}).get("answer_contract")
    assert isinstance(ac, dict)
    assert ac["preferred_render_style"] == "comparison"


def test_enrich_classifier_style() -> None:
    req = _req(prompt="how to deploy")
    req = bootstrap_answer_contract_on_request(req)
    opts = dict(req.options or {})
    opts["output_mode_decision"] = OutputModeDecisionV1(
        output_mode="implementation_guide",
        response_profile="technical_delivery",
        direct_answer_bypass_used=False,
    ).model_dump(mode="json")
    req = req.model_copy(update={"options": opts})
    out = enrich_answer_contract_after_routing(req)
    ac = (out.options or {}).get("answer_contract")
    assert ac["preferred_render_style"] == "steps"


def test_output_modes_for_style() -> None:
    assert output_modes_for_answer_contract_style("comparison")[0] == "comparative_analysis"
