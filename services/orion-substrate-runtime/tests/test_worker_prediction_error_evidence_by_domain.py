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

from app.worker import BiometricsSubstrateWorker


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
            "node:substrate.execution",
            {"prediction_error": 0.0, "prediction_error_evidence_event_ids": ["gev_3"]},
        ),
    ]
    out = worker._brain_frame_prediction_error_evidence_by_domain(nodes)
    assert out == {"biometrics": ["gev_1", "gev_2"], "execution": ["gev_3"]}


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
            "node:substrate.chat",
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
            "node:substrate.route",
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
                "node:substrate.biometrics",
                {
                    "prediction_error": 0.4,
                    "prediction_error_evidence_event_ids": ["gev_1", "gev_2"],
                },
            ),
            _node("node:substrate.route", {"prediction_error": 0.0}),
        ]

        pe_by_domain, evidence_by_domain = (
            worker._brain_frame_prediction_error_and_evidence_by_domain(nodes)
        )

        assert pe_by_domain == {"biometrics": 0.4, "route": 0.0}
        assert evidence_by_domain == {"biometrics": ["gev_1", "gev_2"]}

    def test_individual_methods_delegate_to_the_same_single_pass(self) -> None:
        """The two individually-named methods must still return exactly what
        they returned before this refactor -- delegating to the combined
        helper must be behavior-preserving, not just internally cleaner."""
        worker = _make_worker()
        nodes = [
            _node(
                "node:substrate.execution",
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
