"""Lightweight answer-quality evaluator.

Flags meta-plan or shallow outputs that should be rewritten before returning.
"""

from __future__ import annotations

import re
from typing import Any, Tuple

# Phrases that indicate meta-plan / executive scaffolding rather than actual answer
META_PLAN_PHRASES = [
    r"gather\s+requirements",
    r"create\s+a\s+guide",
    r"review\s+and\s+refine",
    r"test\s+deployment",
    r"implement\s+the\s+solution",
    r"break\s+down\s+into\s+steps",
    r"outline\s+the\s+process",
    r"define\s+the\s+approach",
    r"analyze\s+the\s+requirements",
    r"develop\s+a\s+plan",
    r"clarified\s+goal",  # plan_action scaffold
    r"^\s*steps:\s*$",
]

META_PLAN_PATTERN = re.compile(
    "|".join(f"({p})" for p in META_PLAN_PHRASES),
    re.IGNORECASE,
)

GENERIC_STACK_PATTERN = re.compile(r"\b(flask|ubuntu|gunicorn|nginx|wsgi|systemd)\b", re.IGNORECASE)
ORION_RUNTIME_PATTERN = re.compile(
    r"\b(orion|hub|orch|exec|plannerreact|planner react|agentchain|agent chain|llm gateway|discord)\b",
    re.IGNORECASE,
)

PIP_INSTALL_PATTERN = re.compile(r"\bpip\s+install\b", re.IGNORECASE)
APT_INSTALL_PATTERN = re.compile(r"\bapt(?:-get)?\s+install\b", re.IGNORECASE)


def verified_tokens_from_findings_bundle(fb: dict[str, Any] | None) -> set[str]:
    """Lowercased tokens from findings the pipeline treated as verified evidence."""
    toks: set[str] = set()
    if not isinstance(fb, dict):
        return toks
    for item in fb.get("findings") or []:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip().lower()
        if claim:
            toks.add(claim)
        ref = str(item.get("source_ref") or "").strip().lower()
        if ref:
            toks.add(ref)
        if item.get("verified") is True and claim:
            toks.add("verified:" + claim[:200])
    return toks


def detect_unverified_install_commands(
    text: str,
    verified_tokens: set[str],
) -> Tuple[bool, str]:
    """
    Deterministic guardrail: block pip/apt install specifics unless echoed in findings.
    """
    if not text or not isinstance(text, str):
        return False, ""
    if PIP_INSTALL_PATTERN.search(text):
        for v in verified_tokens:
            if "pip install" in v:
                return False, ""
        return True, "unsupported_specific_pip_install"
    if APT_INSTALL_PATTERN.search(text):
        for v in verified_tokens:
            if "apt install" in v or "apt-get install" in v:
                return False, ""
        return True, "unsupported_specific_apt_install"
    return False, ""


def looks_like_meta_plan(text: str) -> bool:
    """Return True if output appears to be meta-planning rather than concrete answer."""
    if not text or not isinstance(text, str):
        return False
    return bool(META_PLAN_PATTERN.search(text))


def detect_generic_delivery_drift(
    text: str,
    *,
    request_text: str = "",
    grounding_mode: str | None = None,
) -> Tuple[bool, str]:
    if not text or grounding_mode != "orion_repo_architecture":
        return False, ""
    request_norm = (request_text or "").lower()
    if "orion" not in request_norm:
        return False, ""
    if "discord" not in request_norm:
        return False, ""
    if not GENERIC_STACK_PATTERN.search(text):
        return False, ""
    if ORION_RUNTIME_PATTERN.search(text):
        return False, ""
    return True, "generic_architecture_drift"


def should_rewrite_for_instructional(
    text: str,
    output_mode: str | None,
    *,
    request_text: str = "",
    grounding_mode: str | None = None,
    findings_bundle: dict[str, Any] | None = None,
    answer_contract: dict[str, Any] | None = None,
) -> Tuple[bool, str]:
    """
    Return (should_rewrite, reason).
    For tutorial/implementation output modes, flag meta-plan for rewrite.
    """
    if not text:
        return False, ""
    instructional_modes = {
        "implementation_guide",
        "tutorial",
        "code_delivery",
        "direct_answer",
        "comparative_analysis",
        "decision_support",
        "reflective_depth",
        "debug_diagnosis",
        "project_planning",
    }
    if output_mode not in instructional_modes:
        return False, ""
    if looks_like_meta_plan(text):
        return True, "meta_plan_detected"
    generic_drift, generic_reason = detect_generic_delivery_drift(
        text,
        request_text=request_text,
        grounding_mode=grounding_mode,
    )
    if generic_drift:
        return True, generic_reason
    ac_gate = answer_contract if isinstance(answer_contract, dict) else {}
    if ac_gate.get("requires_repo_grounding") or ac_gate.get("requires_runtime_grounding"):
        unsafe, unsafe_reason = should_block_unsupported_specifics(
            text,
            findings_bundle=findings_bundle if isinstance(findings_bundle, dict) else {},
            answer_contract=ac_gate,
        )
        if unsafe:
            return True, unsafe_reason
    # Short generic outputs that lack specifics
    if len(text.strip()) < 120 and not any(c in text for c in ["`", "1.", "2.", "Step", "```"]):
        return False, ""
    return False, ""


def should_block_unsupported_specifics(
    text: str,
    *,
    findings_bundle: dict[str, Any] | None,
    answer_contract: dict[str, Any] | None,
) -> Tuple[bool, str]:
    """Fail-closed when contract disallows unverified specifics and install commands appear without evidence."""
    if not text:
        return False, ""
    contract = answer_contract if isinstance(answer_contract, dict) else {}
    if contract.get("allow_unverified_specifics") is True:
        return False, ""
    fb = findings_bundle if isinstance(findings_bundle, dict) else {}
    toks = verified_tokens_from_findings_bundle(fb)
    bad, reason = detect_unverified_install_commands(text, toks)
    return bad, reason
