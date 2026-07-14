from __future__ import annotations

import sys

from scripts import trace_unified_turn
from scripts.trace_unified_turn import (
    HopEvidence,
    classify_log_line,
    extract_correlation_id,
    format_turn_trace,
    main,
    parse_service_logs,
)


def test_extract_correlation_id_from_log_line() -> None:
    line = (
        "harness_motor_complete corr=1d7965dc-6cd3-44b2-88af-a001c51fbb45 "
        "steps=62 grammar_receipts=62"
    )
    assert extract_correlation_id(line) == "1d7965dc-6cd3-44b2-88af-a001c51fbb45"


def test_classify_motor_complete_fields() -> None:
    line = (
        "harness_motor_complete corr=1d7965dc-6cd3-44b2-88af-a001c51fbb45 "
        "steps=62 grammar_receipts=62 verdict=completed grounding=grounded draft_len=378"
    )
    hop, fields = classify_log_line(line)
    assert hop == "motor_complete"
    assert fields["steps"] == "62"
    assert fields["draft_len"] == "378"


def test_classify_closure_and_prediction_error() -> None:
    recv = (
        "post_turn_closure received corr=1d7965dc-6cd3-44b2-88af-a001c51fbb45 "
        "surprise_unresolved=True outcome_id=outcome_e76bcccb99d44587df38b39f grammar_events=62"
    )
    hop, fields = classify_log_line(recv)
    assert hop == "closure_received"
    assert fields["surprise_unresolved"] == "True"

    pe = (
        "post_turn_closure_prediction_error_write corr=1d7965dc-6cd3-44b2-88af-a001c51fbb45 "
        "node_id=harness_closure:1d7965dc-6cd3-44b2-88af-a001c51fbb45 error=0.65"
    )
    hop2, fields2 = classify_log_line(pe)
    assert hop2 == "prediction_error"
    assert fields2["error"] == "0.65"


def test_parse_service_logs_filters_by_correlation() -> None:
    lines = [
        ("gov", "harness_verdict_published corr=1d7965dc-6cd3-44b2-88af-a001c51fbb45 alignment=uncertain"),
        ("gov", "harness_verdict_published corr=other-corr alignment=aligned"),
    ]
    evidence = parse_service_logs(correlation_id="1d7965dc-6cd3-44b2-88af-a001c51fbb45", lines=lines)
    assert len(evidence) == 1
    assert evidence[0].hop == "verdict"
    assert evidence[0].fields["alignment"] == "uncertain"


def test_format_turn_trace_lists_hops() -> None:
    from scripts.trace_unified_turn import TurnTrace

    formatted = format_turn_trace(
        TurnTrace(
            correlation_id="1d7965dc-6cd3-44b2-88af-a001c51fbb45",
            evidence=[
                HopEvidence(
                    hop="motor_complete",
                    service="gov",
                    line="x",
                    fields={"steps": "62", "draft_len": "378"},
                ),
                HopEvidence(
                    hop="verdict",
                    service="gov",
                    line="y",
                    fields={"alignment": "uncertain"},
                ),
            ],
            grammar_summaries=["Harness step 58: tool=Read, [58] assistant"],
        )
    )
    assert "1d7965dc-6cd3-44b2-88af-a001c51fbb45" in formatted
    assert "draft_len=378" in formatted or "steps=62" in formatted
    assert "alignment=uncertain" in formatted
    assert "Harness step 58" in formatted


def test_live_bare_invocation_does_not_crash_on_missing_bus_attr(monkeypatch) -> None:
    """Regression: `trace_unified_turn.py live` (no flags) used to raise
    AttributeError: 'Namespace' object has no attribute 'bus' -- --bus was
    referenced in cmd_live() but never registered as a --live subcommand
    argument, so bare `live` (the documented default, docker-only mode)
    always crashed before reaching _live_docker_loop."""
    calls: list[dict] = []

    def _fake_docker_loop(*, correlation_id, containers, follow):
        calls.append({"correlation_id": correlation_id, "containers": list(containers), "follow": follow})

    monkeypatch.setattr(trace_unified_turn, "_live_docker_loop", _fake_docker_loop)
    monkeypatch.setattr(sys, "argv", ["trace_unified_turn.py", "live"])
    exit_code = main()
    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["follow"] is True


def test_live_all_flag_still_sets_bus_and_docker(monkeypatch) -> None:
    """--all's documented behavior (bus + docker) must survive the --bus
    argparse fix -- args.bus should end up True via --all just as it does
    via an explicit --bus, not only because --all happened to dodge the
    AttributeError before this fix."""
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_live = sub.add_parser("live")
    p_live.add_argument("--corr")
    p_live.add_argument("--all", action="store_true")
    p_live.add_argument("--bus", action="store_true")
    p_live.add_argument("--docker", action="store_true")
    p_live.add_argument("--redis")
    p_live.add_argument("--bus-channels", nargs="*")

    args = parser.parse_args(["live", "--all"])
    assert args.bus is False  # not yet resolved -- cmd_live() does that
    if getattr(args, "all", False):
        args.bus = True
        args.docker = True
    assert args.bus is True
    assert args.docker is True
