"""Juniper's rule (2026-11-12): a world-pulse reading turn never sends LLM work to the fast or the
metacog lane. Reading turns are durable runs and always carry a GPU pool hold; the one rule is
"a held turn runs all its LLM work on its hold" (orion.llm.resource_lease.held_call_route).

Also pins the probe-parsing fix the live gate needed: until 11-12 every repair-pressure probe on
every lane fell to the fixed text-only fallback scores (100% of repair_pressure_appraisal_log rows
since 10-23 had confidence 0.65), because the probe never read the per-token logprobs the
gateway actually returns.
"""
from __future__ import annotations

import math

import pytest

from orion.llm.resource_lease import GPU_LEASE_ROUTE, held_call_route
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnWindowMessageV1
from orion.substrate.appraisal.paradigms.repair_pressure_v2 import (
    _EVIDENCE_KINDS,
    _TEXT_FALLBACK_DETECTOR,
    _evidence_from_probe_response,
)
from orion.substrate.appraisal.probe.logprob_runner import DETECTOR_NAME, score_kind_from_answer_token

_HOLD = GpuLeaseRefV1(lease_id="hold-1", generation=2, role="agent-gpu2", holder="durable-runs:reading-1")


# --- the routing rule ---------------------------------------------------------------------------

def test_unheld_call_keeps_its_own_route() -> None:
    assert held_call_route(None, "quick_turn") == ("quick_turn", None)
    assert held_call_route(None, "metacog") == ("metacog", None)


@pytest.mark.parametrize("default_route", ["quick_turn", "quick", "metacog", "metacog_turn", "metacog_background"])
def test_held_call_never_names_a_small_lane(default_route) -> None:
    route, lease = held_call_route(_HOLD, default_route)
    assert route == GPU_LEASE_ROUTE == "agent"
    assert lease == _HOLD.model_dump(mode="json")


def test_held_call_accepts_the_wire_dict() -> None:
    assert held_call_route(_HOLD.model_dump(mode="json"), "quick_turn")[1] == _HOLD.model_dump(mode="json")


# --- the request contract (consumer-first deploy) -----------------------------------------------

def _req(**kw) -> PreTurnAppraisalRequestV1:
    return PreTurnAppraisalRequestV1(
        correlation_id="c", session_id="s", turn_window=[TurnWindowMessageV1(role="user", content="x")], **kw
    )


def test_unheld_request_is_byte_identical_to_before_the_field() -> None:
    """Juniper's turns hitting a not-yet-rebuilt cortex-exec (extra="forbid") must still validate."""
    dumped = _req().model_dump(mode="json")
    assert "gpu_lease" not in dumped
    assert set(dumped) == {"correlation_id", "session_id", "turn_window", "paradigms_requested",
                           "contract_before", "options"}


def test_held_request_round_trips_its_hold() -> None:
    dumped = _req(gpu_lease=_HOLD).model_dump(mode="json")
    assert dumped["gpu_lease"] == _HOLD.model_dump(mode="json")
    assert PreTurnAppraisalRequestV1.model_validate(dumped).gpu_lease == _HOLD


# --- the probe parser ---------------------------------------------------------------------------

def _answer_token(answer: str, yes_lp: float, no_lp: float) -> dict:
    tops = sorted([{"token": " YES", "logprob": yes_lp}, {"token": " NO", "logprob": no_lp},
                   {"token": " no", "logprob": -8.0}, {"token": "\n", "logprob": -9.0}],
                  key=lambda t: -t["logprob"])
    return {"token": f" {answer}", "logprob": max(yes_lp, no_lp), "top_logprobs": tops}


def _seven_lines(answers: dict[str, tuple[float, float]]) -> tuple[str, list[dict]]:
    text_lines, tokens = [], []
    for kind in _EVIDENCE_KINDS:
        yes_lp, no_lp = answers[kind]
        answer = "YES" if yes_lp > no_lp else "NO"
        text_lines.append(f"{kind}: {answer}")
        tokens += [{"token": kind, "logprob": 0.0, "top_logprobs": [{"token": kind, "logprob": 0.0},
                                                                     {"token": "x", "logprob": -9.0}]},
                   {"token": ":", "logprob": 0.0, "top_logprobs": [{"token": ":", "logprob": 0.0},
                                                                   {"token": " NO", "logprob": -7.0}]},
                   _answer_token(answer, yes_lp, no_lp),
                   {"token": "\n", "logprob": 0.0, "top_logprobs": [{"token": "\n", "logprob": 0.0},
                                                                    {"token": " ", "logprob": -9.0}]}]
    return "\n".join(text_lines), tokens


_ANSWERS = {kind: (-3.0, -0.05) for kind in _EVIDENCE_KINDS}
_ANSWERS["specificity_demand"] = (-0.88, -0.54)   # the live Strata numbers, 2026-11-12
_ANSWERS["operational_block"] = (-0.10, -2.40)
_SUMMARY_ONLY_UNCERTAINTY = {"available": True, "mean_logprob": -0.1, "source": "openai_compat"}


def _assert_real_scores(evidence) -> None:
    assert evidence and all(e.detector == DETECTOR_NAME for e in evidence)
    assert not any(e.detector == _TEXT_FALLBACK_DETECTOR for e in evidence)
    scores = {e.evidence_kind: e.score for e in evidence}
    assert len(set(round(s, 6) for s in scores.values())) >= 3   # varies, not one fixed constant
    assert scores["operational_block"] > 0.5 > scores["specificity_demand"] > 0.3


def test_openai_compat_reply_scores_from_choices_logprobs() -> None:
    """Strata (agent seat) reply: llm_uncertainty is only a summary; the per-token list lives in
    raw.choices[0].logprobs.content. Before 11-12 this fell to the text fallback every time."""
    text, tokens = _seven_lines(_ANSWERS)
    payload = {"text": text, "llm_uncertainty": dict(_SUMMARY_ONLY_UNCERTAINTY),
               "raw": {"choices": [{"index": 0, "message": {"content": text}, "logprobs": {"content": tokens}}]}}
    _assert_real_scores(_evidence_from_probe_response(payload))


def test_llamacpp_flat_completion_probabilities_score() -> None:
    """llama.cpp /completion (fast lane, gpu2 seat): a flat per-token list, verified live 11-12."""
    text, tokens = _seven_lines(_ANSWERS)
    payload = {"text": text, "llm_uncertainty": dict(_SUMMARY_ONLY_UNCERTAINTY),
               "raw": {"content": text, "completion_probabilities": tokens}}
    _assert_real_scores(_evidence_from_probe_response(payload))


def test_reply_without_token_logprobs_still_falls_back_to_text() -> None:
    text, _ = _seven_lines(_ANSWERS)
    evidence = _evidence_from_probe_response({"text": text, "llm_uncertainty": {"available": False}, "raw": {}})
    assert evidence and all(e.detector == _TEXT_FALLBACK_DETECTOR for e in evidence)


def test_lowercase_spelling_no_longer_overwrites_the_real_answer() -> None:
    """Live 11-12: ' NO' -0.54, ' YES' -0.88, ' no' -8.08. The last spelling used to win, so this
    near-even NO scored as a confident YES. Spellings of one answer add up."""
    entry = {"token": " NO", "logprob": -0.54,
             "top_logprobs": [{"token": " NO", "logprob": -0.54}, {"token": " YES", "logprob": -0.88},
                              {"token": " no", "logprob": -8.08}]}
    scored = score_kind_from_answer_token("specificity_demand", entry)
    assert scored is not None
    log_no = math.log(math.exp(-0.54) + math.exp(-8.08))
    assert scored.score == pytest.approx(1.0 / (1.0 + math.exp(-(-0.88 - log_no))))
    assert scored.score < 0.5
