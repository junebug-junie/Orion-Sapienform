"""Normalize hub `answer_contract_draft` + heuristics into a bus-safe AnswerContract."""

from __future__ import annotations

import logging
from typing import Any

from orion.cognition.output_mode_classifier import classify_output_mode, preferred_render_style_from_classifier
from orion.schemas.cortex.contracts import CortexClientRequest
from orion.schemas.cognition.answer_contract import AnswerContract, AnswerContractDraft, InvestigationState

logger = logging.getLogger("orion.cognition.answer_contract")

_PERSONAL_HINTS = (
    "motivat",
    "better version",
    "myself",
    "procrastin",
    "burnout",
    "feel stuck",
    " i feel ",
    "i'm stuck",
    "im stuck",
    "anxiety",
    "depress",
    "self-care",
    "self care",
    "mental health",
    "can't focus",
    "cant focus",
    "lazy",
    "habit",
    " self-discipline",
    " self discipline",
    "personal discipline",
)

_REPO_HINTS = (
    "`services/",
    "services/orion",
    "orion/",
    "grep ",
    "import ",
    ".py",
    "pytest",
    "pull request",
    "github",
    "refactor",
    "codebase",
    "file path",
    "stack in repo",
)

_RUNTIME_HINTS = (
    "docker",
    "compose",
    "log tail",
    "traceback",
    "stack trace",
    "exception",
    "timeout",
    "hanging",
    "bus channel",
    "rpc",
)


def _norm_user_text(req: CortexClientRequest) -> str:
    parts = [req.context.raw_user_text or "", req.context.user_message or ""]
    return " ".join(p for p in parts if p).strip()


def heuristic_answer_contract(user_text: str) -> AnswerContract:
    t = " " + " ".join((user_text or "").lower().split()) + " "
    personal = any(h in t for h in _PERSONAL_HINTS)
    repoish = any(h in t for h in _REPO_HINTS) or ("orion" in t and ("file" in t or "import" in t or "code" in t))
    runtimeish = any(h in t for h in _RUNTIME_HINTS)

    if personal and not (repoish or runtimeish):
        return AnswerContract(
            request_kind="personal",
            asks_for_explanation=True,
            requires_repo_grounding=False,
            requires_runtime_grounding=False,
            requires_user_artifact_grounding=False,
            allow_inference=True,
            allow_unverified_specifics=False,
            max_unverified_claims=0,
            preferred_render_style="answer",
        )

    if repoish and runtimeish:
        return AnswerContract(
            request_kind="mixed",
            asks_for_explanation=True,
            requires_repo_grounding=True,
            requires_runtime_grounding=True,
            preferred_render_style="steps",
        )
    if repoish:
        return AnswerContract(
            request_kind="repo_technical",
            asks_for_explanation=True,
            requires_repo_grounding=True,
            preferred_render_style="steps",
        )
    if runtimeish:
        return AnswerContract(
            request_kind="runtime_debug",
            asks_for_explanation=True,
            requires_runtime_grounding=True,
            preferred_render_style="steps",
        )
    return AnswerContract(
        request_kind="conceptual",
        asks_for_explanation=True,
        preferred_render_style="answer",
    )


def merge_draft(base: AnswerContract, draft: AnswerContractDraft | dict[str, Any] | None) -> AnswerContract:
    if draft is None:
        return base
    if isinstance(draft, dict):
        draft = AnswerContractDraft.model_validate(draft)
    merged = base.model_dump()
    for key, value in draft.model_dump(exclude_none=True).items():
        merged[key] = value
    return AnswerContract.model_validate(merged)


def build_answer_contract_draft_for_hub(user_text: str) -> dict[str, Any]:
    """Light hub-side draft (metadata only)."""
    draft = AnswerContractDraft.model_validate(heuristic_answer_contract(user_text).model_dump())
    return draft.model_dump(mode="json")


def bootstrap_answer_contract_on_request(req: CortexClientRequest) -> CortexClientRequest:
    """Phase 1: attach normalized AnswerContract from heuristic + hub draft (pre-router)."""
    ut = _norm_user_text(req)
    meta = req.context.metadata if isinstance(req.context.metadata, dict) else {}
    raw_draft = meta.get("answer_contract_draft")
    base = heuristic_answer_contract(ut)
    contract = merge_draft(base, raw_draft if isinstance(raw_draft, (dict, type(None))) else None)
    opts = dict(req.options or {})
    opts["answer_contract"] = contract.model_dump(mode="json")
    logger.info(
        "answer_contract_built request_kind=%s requires_repo=%s requires_runtime=%s style=%s",
        contract.request_kind,
        contract.requires_repo_grounding,
        contract.requires_runtime_grounding,
        contract.preferred_render_style,
    )
    return req.model_copy(update={"options": opts})


def enrich_answer_contract_after_routing(req: CortexClientRequest) -> CortexClientRequest:
    """Weaken classifier: fold output_mode_decision into contract hints without removing parallel fields."""
    opts = dict(req.options or {})
    raw = opts.get("answer_contract")
    if not isinstance(raw, dict):
        return req
    contract = AnswerContract.model_validate(raw)
    omd = opts.get("output_mode_decision")
    if not isinstance(omd, dict):
        omd = classify_output_mode(_norm_user_text(req) or (req.context.raw_user_text or "")).model_dump(mode="json")
    style_hint = preferred_render_style_from_classifier(omd)
    meta = dict(req.context.metadata) if isinstance(req.context.metadata, dict) else {}
    meta["classifier_preferred_render_style"] = style_hint
    hub_draft = meta.get("answer_contract_draft") if isinstance(meta.get("answer_contract_draft"), dict) else {}
    hub_style = hub_draft.get("preferred_render_style") if isinstance(hub_draft, dict) else None
    if not hub_style:
        contract = contract.model_copy(update={"preferred_render_style": style_hint})
    opts["answer_contract"] = contract.model_dump(mode="json")
    new_ctx = req.context.model_copy(update={"metadata": meta})
    return req.model_copy(update={"options": opts, "context": new_ctx})


def output_modes_for_answer_contract_style(style: str | None) -> tuple[str, str]:
    """Map contract render style to legacy output_mode/response_profile for pack merge."""
    s = (style or "answer").strip().lower()
    if s == "steps":
        return "implementation_guide", "technical_delivery"
    if s == "comparison":
        return "comparative_analysis", "reflective_depth"
    if s == "recommendation":
        return "decision_support", "reflective_depth"
    return "direct_answer", "direct_answer"


def investigation_state_for_contract(contract: AnswerContract) -> dict[str, Any]:
    if contract.requires_repo_grounding and contract.requires_runtime_grounding:
        st: InvestigationState = InvestigationState(status="mixed_required", evidence_acquired=False, findings_count=0)
    elif contract.requires_repo_grounding:
        st = InvestigationState(status="repo_required", evidence_acquired=False, findings_count=0)
    elif contract.requires_runtime_grounding:
        st = InvestigationState(status="runtime_required", evidence_acquired=False, findings_count=0)
    else:
        st = InvestigationState(status="not_needed", evidence_acquired=False, findings_count=0)
    return st.model_dump(mode="json")
