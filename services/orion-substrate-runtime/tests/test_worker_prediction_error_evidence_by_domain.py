"""Unit tests for `_brain_frame_prediction_error_evidence_by_domain()`
(2026-08-11) -- the durable-evidence companion to the existing
`_brain_frame_prediction_error_by_domain()`. Same node snapshot, same
zero-extra-I/O shape; reads `metadata['prediction_error_evidence_event_ids']`
instead of `metadata['prediction_error']`.

Closes the retention gap PR #1547/#1551 shipped with: `caused_by_event_ids`
on the prediction-error receipt is real but prunes in 30 minutes
(`ORION_RECEIPT_RETENTION_SUCCESS_MINUTES`); this reads the same evidence
back off the durable FalkorDB node instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

import pytest

from app.worker import BiometricsSubstrateWorker


@pytest.fixture(autouse=True)
def _probe_domain(monkeypatch):
    """No live domain is left since node:substrate.execution was retired 2026-11-24 (#3576):
    the reader is exercised on a test-only "probe" domain."""
    # Patch the globals the methods actually read (another test can re-import `app`).
    g = BiometricsSubstrateWorker._brain_frame_prediction_error_and_evidence_by_domain.__globals__
    monkeypatch.setitem(g, "_PREDICTION_ERROR_DOMAIN_NODE_IDS",
                        {**g["_PREDICTION_ERROR_DOMAIN_NODE_IDS"], "node:substrate.probe": "probe"})


def _node(node_id: str, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(node_id=node_id, metadata=metadata)


def _make_worker() -> BiometricsSubstrateWorker:
    return BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)


def test_reads_evidence_off_known_domain_nodes() -> None:
    worker = _make_worker()
    nodes = [
        _node(
            "node:substrate.biometrics",
            {"prediction_error": 0.4, "prediction_error_evidence_event_ids": ["gev_1", "gev_2"]},
        ),
        _node(
            "node:substrate.probe",
            {"prediction_error": 0.0, "prediction_error_evidence_event_ids": ["gev_3"]},
        ),
        _node(
            "node:substrate.execution",
            {"prediction_error": 0.7, "prediction_error_evidence_event_ids": ["gev_4"]},
        ),
    ]
    out = worker._brain_frame_prediction_error_evidence_by_domain(nodes)
    # node:substrate.biometrics (2026-11-18, #3575) and node:substrate.execution (2026-11-24,
    # #3576) were retired: a lingering node is not a domain.
    assert out == {"probe": ["gev_3"]}


def test_the_real_map_is_empty_so_a_lingering_execution_node_reads_nothing(monkeypatch) -> None:
    monkeypatch.undo()
    g = BiometricsSubstrateWorker._brain_frame_prediction_error_and_evidence_by_domain.__globals__
    assert g["_PREDICTION_ERROR_DOMAIN_NODE_IDS"] == {}
    nodes = [_node("node:substrate.execution",
                   {"prediction_error": 0.7, "prediction_error_evidence_event_ids": ["gev_4"]})]
    assert _make_worker()._brain_frame_prediction_error_and_evidence_by_domain(nodes) == ({}, {})


def test_skips_unknown_node_ids() -> None:
    worker = _make_worker()
    nodes = [
        _node("node:something:unrelated", {"prediction_error_evidence_event_ids": ["gev_1"]}),
    ]
    assert worker._brain_frame_prediction_error_evidence_by_domain(nodes) == {}


def test_skips_domains_with_no_evidence_key() -> None:
    """A node whose prediction_error was written but never got
    `evidence_event_ids` passed (e.g. bus_synaptic/codebase call sites,
    which never populate this) must not appear -- not a fabricated `[]`."""
    worker = _make_worker()
    nodes = [_node("node:substrate.bus_synaptic", {"prediction_error": 0.02})]
    assert worker._brain_frame_prediction_error_evidence_by_domain(nodes) == {}


def test_skips_empty_evidence_list() -> None:
    worker = _make_worker()
    nodes = [
        _node(
            "node:substrate.probe",
            {"prediction_error": 0.0, "prediction_error_evidence_event_ids": []},
        ),
    ]
    assert worker._brain_frame_prediction_error_evidence_by_domain(nodes) == {}


def test_tolerates_malformed_evidence_value() -> None:
    """Fail-open: a non-list value (e.g. corrupted/legacy metadata) must be
    skipped, never raise, mirroring the sibling scalar method's own
    (TypeError, ValueError) tolerance."""
    worker = _make_worker()
    nodes = [
        _node(
            "node:substrate.probe",
            {"prediction_error": 0.1, "prediction_error_evidence_event_ids": "not-a-list"},
        ),
    ]
    assert worker._brain_frame_prediction_error_evidence_by_domain(nodes) == {}


def test_empty_node_list_yields_empty_dict() -> None:
    worker = _make_worker()
    assert worker._brain_frame_prediction_error_evidence_by_domain([]) == {}


class TestCombinedSinglePassHelper:
    """`_brain_frame_prediction_error_and_evidence_by_domain()` (review
    finding, 2026-08-11): the two live per-tick callers each called both
    individual methods back-to-back over the identical node snapshot,
    walking it twice for no reason. This combined method returns both dicts
    from a single pass, and both individual methods now delegate to it."""

    def test_returns_both_dicts_matching_the_individual_methods(self) -> None:
        worker = _make_worker()
        nodes = [
            _node(
                "node:substrate.probe",
                {
                    "prediction_error": 0.4,
                    "prediction_error_evidence_event_ids": ["gev_1", "gev_2"],
                },
            ),
            # Retired 2026-11 (PE step 5; chat and biometrics step 6): never read as a domain.
            _node("node:substrate.route", {"prediction_error": 0.9}),
            _node("node:substrate.chat", {"prediction_error": 0.7}),
            _node("node:substrate.execution", {"prediction_error": 0.6,
                                               "prediction_error_evidence_event_ids": ["gev_8"]}),
            _node("node:substrate.biometrics", {"prediction_error": 0.2,
                                                "prediction_error_evidence_event_ids": ["gev_9"]}),
        ]

        pe_by_domain, evidence_by_domain = (
            worker._brain_frame_prediction_error_and_evidence_by_domain(nodes)
        )

        assert pe_by_domain == {"probe": 0.4}
        assert evidence_by_domain == {"probe": ["gev_1", "gev_2"]}

    def test_individual_methods_delegate_to_the_same_single_pass(self) -> None:
        """The two individually-named methods must still return exactly what
        they returned before this refactor -- delegating to the combined
        helper must be behavior-preserving, not just internally cleaner."""
        worker = _make_worker()
        nodes = [
            _node(
                "node:substrate.probe",
                {
                    "prediction_error": 0.1,
                    "prediction_error_evidence_event_ids": ["gev_3"],
                },
            ),
        ]

        combined_pe, combined_evidence = (
            worker._brain_frame_prediction_error_and_evidence_by_domain(nodes)
        )
        assert worker._brain_frame_prediction_error_by_domain(nodes) == combined_pe
        assert (
            worker._brain_frame_prediction_error_evidence_by_domain(nodes)
            == combined_evidence
        )
