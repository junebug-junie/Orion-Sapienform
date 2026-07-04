from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import yaml

from orion.schemas.pre_turn_appraisal import (
    PreTurnAppraisalRequestV1,
    TurnAppraisalParadigmSliceV1,
    TurnWindowMessageV1,
)
from orion.substrate.appraisal.contract import assemble_repair_contract_delta
from orion.substrate.appraisal.models import EvidenceKind, RepairEvidenceV1
from orion.substrate.appraisal.probe.logprob_runner import (
    kind_probe_to_evidence,
    parse_yes_no_lines,
    score_kind_from_answer_token,
)

_EVIDENCE_KINDS: tuple[EvidenceKind, ...] = (
    "specificity_demand",
    "trust_rupture",
    "coherence_gap",
    "repetition_failure",
    "operational_block",
    "explicit_repair_command",
    "assistant_accountability_demand",
)

_LLMCaller = Callable[[str], dict[str, Any] | Awaitable[dict[str, Any]]]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "orion").is_dir() and (parent / "config").is_dir():
            return parent
    return Path.cwd()


def _resolve_weights_path(weights_path: str) -> Path:
    path = Path(weights_path)
    if path.is_file():
        return path
    return _repo_root() / weights_path


def load_repair_weights(weights_path: str) -> dict[str, float]:
    resolved = _resolve_weights_path(weights_path)
    if not resolved.is_file():
        return {}
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def build_repair_probe_prompt(window: list[TurnWindowMessageV1]) -> str:
    thread_lines = [f"{m.role.upper()}: {m.content}" for m in window]
    thread = "\n".join(thread_lines)
    kind_lines = "\n".join(f"{kind}: YES or NO" for kind in _EVIDENCE_KINDS)
    return (
        "You are a repair-pressure classifier. Read the paired conversation thread.\n"
        "Do NOT answer the user. Reply with exactly seven lines, one per kind, YES or NO only.\n\n"
        f"THREAD:\n{thread}\n\n"
        "Answer each line using this exact format (kind: YES or NO):\n"
        f"{kind_lines}\n"
    )


def reduce_repair_level(
    kind_scores: dict[str, float],
    *,
    confidences: dict[str, float],
    weights: dict[str, float],
) -> tuple[float, float]:
    level = 0.0
    for kind, score in kind_scores.items():
        weight = float(weights.get(kind, 0.0))
        level += weight * float(score)
    level = max(0.0, min(1.0, level))

    active_confidences = [
        float(confidences[kind])
        for kind, score in kind_scores.items()
        if float(score) > 0.5 and kind in confidences
    ]
    confidence = min(active_confidences) if active_confidences else 0.0
    confidence = max(0.0, min(1.0, confidence))
    return level, confidence


def _normalize_logprob_content(payload: dict[str, Any]) -> list[dict[str, Any]]:
    unc = payload.get("llm_uncertainty")
    if isinstance(unc, dict):
        content = unc.get("content") or unc.get("per_token")
        if isinstance(content, list) and content:
            return [entry for entry in content if isinstance(entry, dict)]

    raw = payload.get("raw")
    if isinstance(raw, dict):
        probs = raw.get("probs")
        if not isinstance(probs, list) and isinstance(raw.get("completion_probabilities"), list):
            first = raw["completion_probabilities"][0]
            if isinstance(first, dict):
                probs = first.get("probs")
        if isinstance(probs, list):
            content: list[dict[str, Any]] = []
            for entry in probs:
                if not isinstance(entry, dict):
                    continue
                token = entry.get("token")
                if token is None:
                    continue
                tops = entry.get("top_logprobs") or entry.get("top_probs") or []
                normalized_tops: list[dict[str, Any]] = []
                if isinstance(tops, list):
                    for t in tops:
                        if isinstance(t, dict):
                            normalized_tops.append(
                                {"token": t.get("token"), "logprob": t.get("logprob")}
                            )
                content.append(
                    {
                        "token": token,
                        "logprob": entry.get("logprob"),
                        "top_logprobs": normalized_tops,
                    }
                )
            return content
    return []


def _answer_token_entries(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in content:
        token = str(entry.get("token") or "").strip().upper()
        if token not in {"YES", "NO"}:
            continue
        tops = entry.get("top_logprobs")
        if not isinstance(tops, list) or len(tops) < 2:
            continue
        out.append(entry)
    return out


def _evidence_from_probe_response(payload: dict[str, Any]) -> list[RepairEvidenceV1]:
    unc = payload.get("llm_uncertainty")
    if isinstance(unc, dict) and unc.get("available") is False:
        return []

    text = str(payload.get("text") or "")
    parsed = parse_yes_no_lines(text)
    if not parsed:
        return []

    content = _normalize_logprob_content(payload)
    if not content:
        return []

    answer_entries = _answer_token_entries(content)
    if not answer_entries:
        return []

    evidence: list[RepairEvidenceV1] = []
    entry_idx = 0
    for kind in _EVIDENCE_KINDS:
        if kind not in parsed:
            continue
        if entry_idx >= len(answer_entries):
            break
        entry = answer_entries[entry_idx]
        entry_idx += 1
        scored = score_kind_from_answer_token(kind, entry)
        if scored is None:
            continue
        evidence.append(kind_probe_to_evidence(scored))
    return evidence


class RepairPressureV2Paradigm:
    name = "repair_pressure"

    def __init__(
        self,
        llm_caller: _LLMCaller,
        *,
        weights_path: str = "config/substrate/repair_pressure_weights.v2.yaml",
    ) -> None:
        self._llm_caller = llm_caller
        self._weights_path = weights_path

    async def _call_llm(self, prompt: str) -> dict[str, Any]:
        result = self._llm_caller(prompt)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, dict) else {"text": "", "llm_uncertainty": {"available": False}}

    async def run(self, req: PreTurnAppraisalRequestV1) -> TurnAppraisalParadigmSliceV1:
        prompt = build_repair_probe_prompt(req.turn_window)
        payload = await self._call_llm(prompt)
        evidence = _evidence_from_probe_response(payload)

        if not evidence:
            return TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.0,
                confidence=0.0,
                dimensions={},
                evidence=[],
                contract_delta=dict(req.contract_before or {"mode": "default"}),
                notes=["no_repair_evidence"],
            )

        kind_scores = {ev.evidence_kind: float(ev.score) for ev in evidence}
        confidences = {ev.evidence_kind: float(ev.confidence) for ev in evidence}
        weights = load_repair_weights(self._weights_path)
        if not weights:
            return TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.0,
                confidence=0.0,
                dimensions=dict(kind_scores),
                evidence=evidence,
                contract_delta=dict(req.contract_before or {"mode": "default"}),
                notes=["weights_file_missing"],
            )
        level, confidence = reduce_repair_level(
            kind_scores,
            confidences=confidences,
            weights=weights,
        )
        contract_delta = assemble_repair_contract_delta(
            contract_before=dict(req.contract_before or {"mode": "default"}),
            level=level,
            confidence=confidence,
            kind_scores=kind_scores,
        )
        return TurnAppraisalParadigmSliceV1(
            appraisal_kind="repair_pressure",
            level=level,
            confidence=confidence,
            dimensions=dict(kind_scores),
            evidence=evidence,
            contract_delta=contract_delta,
            notes=[],
        )
