from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnWindowMessageV1
from orion.substrate.appraisal.paradigms.repair_pressure_v2 import RepairPressureV2Paradigm

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "repair_pressure"


def _load_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(_FIXTURES_DIR.glob("*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def _mock_llm_response(fixture: dict[str, Any]) -> dict[str, Any]:
    lines = fixture.get("mock_probe_lines") or {}
    if not lines:
        return {"text": "", "llm_uncertainty": {"available": False}}

    text_lines = []
    content = []
    for kind, probe in lines.items():
        yes_lp = float(probe["yes_lp"])
        no_lp = float(probe["no_lp"])
        margin = float(probe.get("margin", yes_lp - no_lp))
        answer = "YES" if yes_lp > no_lp else "NO"
        text_lines.append(f"{kind}: {answer}")
        content.append(
            {
                "token": answer,
                "logprob": yes_lp if answer == "YES" else no_lp,
                "top_logprobs": [
                    {"token": "YES", "logprob": yes_lp},
                    {"token": "NO", "logprob": no_lp},
                ],
            }
        )
    return {
        "text": "\n".join(text_lines),
        "llm_uncertainty": {"available": True, "content": content},
        "raw": {"probs": content},
    }


def _assert_fixture(fixture: dict[str, Any], slice_) -> None:
    expect = fixture.get("expect") or {}
    if "level_min" in expect:
        assert slice_.level >= float(expect["level_min"]), fixture["id"]
    if "level_max" in expect:
        assert slice_.level <= float(expect["level_max"]), fixture["id"]
    if "confidence_min" in expect:
        assert slice_.confidence >= float(expect["confidence_min"]), fixture["id"]
    if "mode" in expect:
        assert slice_.contract_delta.get("mode") == expect["mode"], fixture["id"]
    if expect.get("mode_unchanged"):
        assert slice_.contract_delta.get("mode") == "default", fixture["id"]
    if "active_kinds_min" in expect:
        active = sum(1 for v in (slice_.dimensions or {}).values() if float(v) >= 0.65)
        assert active >= int(expect["active_kinds_min"]), fixture["id"]
    if "trust_rupture_max" in expect:
        assert float((slice_.dimensions or {}).get("trust_rupture", 0.0)) <= float(expect["trust_rupture_max"])
    if "coherence_gap_max" in expect:
        assert float((slice_.dimensions or {}).get("coherence_gap", 0.0)) <= float(expect["coherence_gap_max"])


@pytest.mark.parametrize("fixture", _load_fixtures(), ids=lambda f: f["id"])
@pytest.mark.asyncio
async def test_repair_pressure_v2_fixture(fixture: dict[str, Any]) -> None:
    window = [TurnWindowMessageV1.model_validate(m) for m in fixture["turn_window"]]
    paradigm = RepairPressureV2Paradigm(
        llm_caller=lambda _prompt: _mock_llm_response(fixture),
        weights_path="config/substrate/repair_pressure_weights.v2.yaml",
    )
    req = PreTurnAppraisalRequestV1(
        correlation_id="00000000-0000-4000-8000-000000000003",
        session_id="eval",
        turn_window=window,
    )
    slice_ = await paradigm.run(req)
    _assert_fixture(fixture, slice_)


def main() -> int:
    failures = 0
    for fixture in _load_fixtures():
        window = [TurnWindowMessageV1.model_validate(m) for m in fixture["turn_window"]]
        paradigm = RepairPressureV2Paradigm(
            llm_caller=lambda _prompt, fx=fixture: _mock_llm_response(fx),
            weights_path="config/substrate/repair_pressure_weights.v2.yaml",
        )
        req = PreTurnAppraisalRequestV1(
            correlation_id="00000000-0000-4000-8000-000000000003",
            session_id="eval",
            turn_window=window,
        )
        slice_ = asyncio.run(paradigm.run(req))
        try:
            _assert_fixture(fixture, slice_)
            print(f"PASS {fixture['id']} level={slice_.level:.3f} mode={slice_.contract_delta.get('mode')}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fixture['id']}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
