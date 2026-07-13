from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_single_consumer_channels as gate

# The four cortex-exec request lane channels (the seam behind the PR #994
# duplicate-execution bug this gate exists to prevent).
CORTEX_EXEC_LANES = (
    "orion:cortex:exec:request",
    "orion:cortex:exec:request:chat",
    "orion:cortex:exec:request:spark",
    "orion:cortex:exec:request:background",
)


# --- evaluate_counts: the pure decision core, no Redis, no I/O ---------------


def test_evaluate_counts_all_ones_is_clean() -> None:
    counts = {"orion:verb:request": 1, "orion:cortex:exec:request": 1}

    violations, warnings = gate.evaluate_counts(counts)

    assert violations == []
    assert warnings == []


def test_evaluate_counts_two_subscribers_is_a_violation_naming_the_channel() -> None:
    counts = {"orion:verb:request": 2, "orion:cortex:exec:request": 1}

    violations, warnings = gate.evaluate_counts(counts)

    assert len(violations) == 1
    assert "orion:verb:request" in violations[0]
    assert "2 subscribers" in violations[0]
    assert warnings == []


def test_evaluate_counts_zero_is_a_warning_by_default() -> None:
    counts = {"orion:dream:trigger": 0}

    violations, warnings = gate.evaluate_counts(counts)

    assert violations == []
    assert len(warnings) == 1
    assert "orion:dream:trigger" in warnings[0]


def test_evaluate_counts_zero_is_a_violation_under_strict_zero() -> None:
    counts = {"orion:dream:trigger": 0}

    violations, warnings = gate.evaluate_counts(counts, strict_zero=True)

    assert len(violations) == 1
    assert "orion:dream:trigger" in violations[0]
    assert warnings == []


# --- catalog wiring: the real channels.yaml carries the annotations ----------


def test_verb_request_is_marked_single_consumer() -> None:
    channels = gate.load_single_consumer_channels()

    assert "orion:verb:request" in channels


def test_all_four_cortex_exec_request_lanes_are_marked() -> None:
    channels = set(gate.load_single_consumer_channels())

    missing = [ch for ch in CORTEX_EXEC_LANES if ch not in channels]
    assert missing == []


def test_no_single_consumer_channel_contains_a_glob_char() -> None:
    channels = gate.load_single_consumer_channels()

    globby = [ch for ch in channels if any(c in ch for c in "*?[")]
    assert globby == []


def test_collapse_mirror_service_entry_exists_and_is_marked() -> None:
    # Regression for the catalog gap: orion-collapse-mirror registers a Rabbit
    # RPC on this channel (services/orion-collapse-mirror/app/bus_runtime.py)
    # but the channel was missing from orion/bus/channels.yaml entirely.
    channels = gate.load_single_consumer_channels()

    assert "orion:exec:request:CollapseMirrorService" in channels


def test_fixture_catalog_glob_names_are_skipped(tmp_path) -> None:
    fixture = tmp_path / "channels.yaml"
    fixture.write_text(
        """
channels:
  - name: "orion:test:reply*"
    kind: "result"
    schema_id: "GenericPayloadV1"
    producer_services: ["orion-test"]
    consumer_services: ["orion-test"]
    single_consumer: true
    stability: "experimental"
    since: "2026-07-13"

  - name: "orion:test:request"
    kind: "request"
    schema_id: "GenericPayloadV1"
    producer_services: ["orion-test"]
    consumer_services: ["orion-test"]
    single_consumer: true
    stability: "experimental"
    since: "2026-07-13"

  - name: "orion:test:unmarked"
    kind: "request"
    schema_id: "GenericPayloadV1"
    producer_services: ["orion-test"]
    consumer_services: ["orion-test"]
    stability: "experimental"
    since: "2026-07-13"
"""
    )

    channels = gate.load_single_consumer_channels(fixture)

    assert channels == ["orion:test:request"]


# --- redis-cli NUMSUB output parsing ------------------------------------------


def test_parse_numsub_output_plain_raw_form() -> None:
    stdout = (
        "orion:verb:request\n"
        "1\n"
        "orion:cortex:exec:request\n"
        "2\n"
        "orion:dream:trigger\n"
        "0\n"
    )

    counts = gate.parse_numsub_output(stdout)

    assert counts == {
        "orion:verb:request": 1,
        "orion:cortex:exec:request": 2,
        "orion:dream:trigger": 0,
    }


def test_parse_numsub_output_numbered_interactive_form() -> None:
    stdout = (
        '1) "orion:verb:request"\n'
        "2) (integer) 1\n"
        '3) "orion:cortex:exec:request"\n'
        "4) (integer) 3\n"
    )

    counts = gate.parse_numsub_output(stdout)

    assert counts == {
        "orion:verb:request": 1,
        "orion:cortex:exec:request": 3,
    }


def test_parse_numsub_output_empty_and_garbage_are_safe() -> None:
    assert gate.parse_numsub_output("") == {}
    assert gate.parse_numsub_output("\n\n") == {}
    # A dangling channel line with no count line is dropped, not crashed on.
    assert gate.parse_numsub_output("orion:verb:request\n") == {}


# --- main() infra-error paths: exit 2, never a false violation ----------------


def test_main_without_bus_url_exits_two(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ORION_BUS_URL", raising=False)

    rc = gate.main(["--bus-url", ""])

    assert rc == 2
    assert "ORION_BUS_URL" in capsys.readouterr().err


def test_main_infra_failure_exits_two_not_one(monkeypatch, capsys) -> None:
    def boom(bus_url, channels):
        raise RuntimeError("Could not connect to Redis at nope:6379")

    monkeypatch.setattr(gate, "fetch_live_counts", boom)

    rc = gate.main(["--bus-url", "redis://nope:6379/0"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "infra error" in err
    assert "Could not connect" in err


def test_main_duplicate_subscriber_exits_one(monkeypatch, capsys) -> None:
    def fake_counts(bus_url, channels):
        return {ch: 1 for ch in channels} | {"orion:verb:request": 2}

    monkeypatch.setattr(gate, "fetch_live_counts", fake_counts)

    rc = gate.main(["--bus-url", "redis://fake:6379/0"])

    assert rc == 1
    out, err = capsys.readouterr()
    assert "FAIL 2 orion:verb:request" in out
    assert "gate FAILED" in err


def test_main_all_single_subscribers_exits_zero(monkeypatch, capsys) -> None:
    def fake_counts(bus_url, channels):
        return {ch: 1 for ch in channels}

    monkeypatch.setattr(gate, "fetch_live_counts", fake_counts)

    rc = gate.main(["--bus-url", "redis://fake:6379/0"])

    assert rc == 0
    assert "gate OK" in capsys.readouterr().out
