"""Reading back the `:IncidentReport` an urgent run wrote into Orion's own graph.

Orion writes the node by hand, in Cypher. The reader turns it into a verdict or
says plainly that there is no structured verdict -- it never guesses one from a
half-written node, and it never raises into the durable run that calls it.
"""

from __future__ import annotations

import pytest

from orion.curiosity.incident_report import (
    NO_STRUCTURED_VERDICT,
    IncidentReport,
    incident_report_for_run_cypher,
    read_incident_report,
)
from orion.curiosity.worldview import WorldviewReader, WorldviewUnavailable

RUN_ID = "a1b2c3d4e5f6"


class _FakeReader(WorldviewReader):
    """Answers by matching on the query text, the way FalkorDB would by shape."""

    def __init__(self, *, answers=None, raises=None) -> None:
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.answers = answers or {}
        self.raises = raises
        self.queries: list[str] = []

    def query(self, cypher: str):
        self.queries.append(cypher)
        if self.raises is not None:
            raise self.raises
        for needle, rows in self.answers.items():
            if needle in cypher:
                return rows
        return []


def _rows(node_id=1, written_at=1759050000000, evidence=("gpu2 read 91C at 03:02", "fan_pct=0"), **fields):
    """One row per evidence item, the shape the UNWIND query returns."""
    base = {
        "node_id": node_id,
        "run_id": RUN_ID,
        "incident_id": "0123456789abcdef",
        "is_real": "real",
        "likely_cause": "gpu2 fan stopped",
        "severity": "high",
        "operator_action": "Reseat the gpu2 fan header",
        "confidence": "0.8",
        "written_at": written_at,
    }
    base.update(fields)
    return [{**base, "evidence": item} for item in evidence]


def _reader(rows):
    return _FakeReader(answers={"IncidentReport": rows})


def test_valid_row_reads_as_report():
    report, flag = read_incident_report(_reader(_rows()), RUN_ID)
    assert flag is None
    assert report == IncidentReport(
        incident_id="0123456789abcdef",
        is_real="real",
        likely_cause="gpu2 fan stopped",
        evidence=("gpu2 read 91C at 03:02", "fan_pct=0"),
        severity="high",
        operator_action="Reseat the gpu2 fan header",
        confidence=0.8,
    )


def test_confidence_is_clamped():
    report, flag = read_incident_report(_reader(_rows(confidence="1.7")), RUN_ID)
    assert flag is None and report.confidence == 1.0
    report, flag = read_incident_report(_reader(_rows(confidence=-0.3)), RUN_ID)
    assert flag is None and report.confidence == 0.0


@pytest.mark.parametrize("confidence", [None, "", "high", "nan"])
def test_unreadable_confidence_is_no_verdict(confidence):
    assert read_incident_report(_reader(_rows(confidence=confidence)), RUN_ID) == (
        None,
        NO_STRUCTURED_VERDICT,
    )


def test_case_and_whitespace_tolerated_on_enums():
    report, flag = read_incident_report(
        _reader(_rows(is_real=" Sensor_Fault ", severity="CRITICAL")), RUN_ID
    )
    assert flag is None
    assert report.is_real == "sensor_fault"
    assert report.severity == "critical"


def test_unknown_is_real_is_no_verdict():
    assert read_incident_report(_reader(_rows(is_real="maybe")), RUN_ID) == (None, NO_STRUCTURED_VERDICT)


def test_unknown_severity_is_no_verdict():
    assert read_incident_report(_reader(_rows(severity="medium")), RUN_ID) == (None, NO_STRUCTURED_VERDICT)


@pytest.mark.parametrize("field", ["likely_cause", "operator_action"])
def test_missing_text_field_is_no_verdict(field):
    assert read_incident_report(_reader(_rows(**{field: "  "})), RUN_ID) == (None, NO_STRUCTURED_VERDICT)


def test_empty_evidence_is_no_verdict():
    # The query appends a null to every node's evidence, so an empty or
    # missing list arrives as one row with evidence None.
    assert read_incident_report(_reader(_rows(evidence=(None,))), RUN_ID) == (None, NO_STRUCTURED_VERDICT)
    assert read_incident_report(_reader(_rows(evidence=())), RUN_ID) == (None, NO_STRUCTURED_VERDICT)
    # Blank strings are not evidence either.
    assert read_incident_report(_reader(_rows(evidence=("", "  "))), RUN_ID) == (None, NO_STRUCTURED_VERDICT)


def test_null_sentinel_row_is_ignored():
    report, flag = read_incident_report(_reader(_rows(evidence=("a", None))), RUN_ID)
    assert flag is None and report.evidence == ("a",)


def test_newer_evidence_less_report_is_not_hidden_by_older_valid_one():
    good_old = _rows(node_id=1, written_at=1000)
    empty_new = _rows(node_id=2, written_at=2000, evidence=(None,))
    assert read_incident_report(_reader(good_old + empty_new), RUN_ID) == (None, NO_STRUCTURED_VERDICT)


def test_list_valued_evidence_column_is_flattened():
    rows = _rows(evidence=(["a, with comma", "b"],))
    report, flag = read_incident_report(_reader(rows), RUN_ID)
    assert flag is None
    assert report.evidence == ("a, with comma", "b")


def test_duplicate_evidence_collapses():
    report, _ = read_incident_report(_reader(_rows(evidence=("x", "x", "y"))), RUN_ID)
    assert report.evidence == ("x", "y")


def test_no_rows_is_no_verdict():
    assert read_incident_report(_reader([]), RUN_ID) == (None, NO_STRUCTURED_VERDICT)


@pytest.mark.parametrize("exc", [WorldviewUnavailable("down"), RuntimeError("boom"), TypeError("odd")])
def test_reader_error_is_no_verdict(exc):
    assert read_incident_report(_FakeReader(raises=exc), RUN_ID) == (None, NO_STRUCTURED_VERDICT)


def test_newest_written_at_wins():
    old = _rows(node_id=1, written_at=1000, is_real="sensor_fault", evidence=("old",))
    new = _rows(node_id=2, written_at=2000, is_real="real", evidence=("new-1", "new-2"))
    for rows in (old + new, new + old):
        report, flag = read_incident_report(_reader(rows), RUN_ID)
        assert flag is None
        assert report.is_real == "real"
        assert report.evidence == ("new-1", "new-2")


def test_newest_wins_even_when_it_is_malformed():
    good_old = _rows(node_id=1, written_at=1000)
    bad_new = _rows(node_id=2, written_at=2000, is_real="maybe")
    assert read_incident_report(_reader(good_old + bad_new), RUN_ID) == (None, NO_STRUCTURED_VERDICT)


def test_iso_written_at_orders_with_epoch_ms():
    # Orion is asked for timestamp() but has written ISO before.
    old = _rows(node_id=1, written_at="2026-09-28T09:00:00Z", severity="low")
    new = _rows(node_id=2, written_at=1790589600000, severity="critical")  # 10:00Z
    for rows in (old + new, new + old):
        report, _ = read_incident_report(_reader(rows), RUN_ID)
        assert report.severity == "critical"


def test_bad_run_id_refused_by_cypher_builder():
    for bad in ("", "not-hex", "ABCDEF123456", "abc' OR 1=1 //", None):
        with pytest.raises(ValueError):
            incident_report_for_run_cypher(bad)


def test_bad_run_id_reads_as_no_verdict_without_querying():
    reader = _reader(_rows())
    assert read_incident_report(reader, "not-hex") == (None, NO_STRUCTURED_VERDICT)
    assert reader.queries == []


def test_cypher_scopes_to_run_and_unwinds_evidence():
    cypher = incident_report_for_run_cypher(RUN_ID)
    assert "MATCH (r:IncidentReport)" in cypher
    assert f"r.run_id = '{RUN_ID}'" in cypher
    assert "UNWIND coalesce(r.evidence, []) + [null] AS e" in cypher
    assert "id(r) AS node_id" in cypher
