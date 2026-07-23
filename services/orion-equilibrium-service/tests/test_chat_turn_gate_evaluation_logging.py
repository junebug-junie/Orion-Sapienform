from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import EquilibriumService


def _thought_event(*, disposition="proceed", boundary_register=False) -> dict:
    return {
        "correlation_id": "corr-1",
        "disposition": disposition,
        "disposition_reasons": [],
        "boundary_register": boundary_register,
        "grounding_capsule": None,
        "autonomy_slice": None,
    }


def _run_artifact(*, compliance_verdict="completed", exit_code=0) -> dict:
    return {
        "correlation_id": "corr-1",
        "compliance_verdict": compliance_verdict,
        "grounding_status": "grounded",
        "exit_code": exit_code,
        "finalize_degraded_reason": None,
        "reflection": {"alignment_verdict": "aligned", "strain_unresolved": False},
        "substrate_appraisal": {"surprise_level": 0.1},
    }


@pytest.mark.asyncio
async def test_no_fire_terminal_evaluation_is_logged(caplog):
    """Regression for the live-verification gap found post-deploy: a terminal
    evaluation that fires nothing must still produce one clear, greppable log
    line -- not silence indistinguishable from the gate never having run."""
    svc = EquilibriumService()
    svc._chat_turn_correlator = AsyncMock()
    svc._chat_turn_correlator.accumulate.return_value = (
        _thought_event(),
        _run_artifact(),
        False,
        None,
    )
    svc._publish_metacog_trigger = AsyncMock()

    with caplog.at_level("INFO", logger="orion-equilibrium"):
        await svc._handle_chat_turn_evidence(
            distress=0.0,
            zen=1.0,
            correlation_id="corr-1",
            run_artifact=_run_artifact(),
        )

    svc._publish_metacog_trigger.assert_not_called()
    messages = [r.getMessage() for r in caplog.records]
    evaluated = [m for m in messages if "chat_turn_gate_evaluated" in m]
    assert len(evaluated) == 1
    assert "fired=False" in evaluated[0]
    assert "corr_id=corr-1" in evaluated[0]
    assert "disposition=proceed" in evaluated[0]
    assert "compliance_verdict=completed" in evaluated[0]


@pytest.mark.asyncio
async def test_fire_terminal_evaluation_is_logged_before_publish(caplog):
    svc = EquilibriumService()
    svc._chat_turn_correlator = AsyncMock()
    svc._chat_turn_correlator.accumulate.return_value = (
        _thought_event(disposition="defer"),
        None,
        False,
        None,
    )
    svc._publish_metacog_trigger = AsyncMock()

    with caplog.at_level("INFO", logger="orion-equilibrium"):
        await svc._handle_chat_turn_evidence(
            distress=0.0,
            zen=1.0,
            correlation_id="corr-1",
            thought_event=_thought_event(disposition="defer"),
        )

    svc._publish_metacog_trigger.assert_called_once()
    messages = [r.getMessage() for r in caplog.records]
    evaluated = [m for m in messages if "chat_turn_gate_evaluated" in m]
    assert len(evaluated) == 1
    assert "fired=True" in evaluated[0]
    assert "disposition=defer" in evaluated[0]


@pytest.mark.asyncio
async def test_non_terminal_evidence_does_not_log_or_publish(caplog):
    svc = EquilibriumService()
    svc._chat_turn_correlator = AsyncMock()
    svc._chat_turn_correlator.accumulate.return_value = (
        _thought_event(),
        None,
        False,
        None,
    )
    svc._publish_metacog_trigger = AsyncMock()

    with caplog.at_level("INFO", logger="orion-equilibrium"):
        await svc._handle_chat_turn_evidence(
            distress=0.0,
            zen=1.0,
            correlation_id="corr-1",
            thought_event=_thought_event(),
        )

    svc._publish_metacog_trigger.assert_not_called()
    messages = [r.getMessage() for r in caplog.records]
    assert not [m for m in messages if "chat_turn_gate_evaluated" in m]
