"""Unit tests for the codebase-mass consumer patch's message handler
(docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md, "Producer
+ consumer patch design"): `_handle_codebase_delta_message`.

Uses the real OrionCodec to encode/decode a real CodebaseDeltaV1-carrying
BaseEnvelope (genuine wire round-trip), with the store and
_write_prediction_error_node mocked -- those are already covered by
test_store_codebase_mass_baseline.py and test_worker_prediction_error_node.py
respectively.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.worker import BiometricsSubstrateWorker

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.codebase_delta import (
    CodebaseDeltaV1,
    GitDeltaPayloadV1,
    GraphDeltaPayloadV1,
    PrLifecycleDeltaPayloadV1,
)
from orion.substrate.prediction_error import CodebaseMassBaseline, _DomainEwmaBaseline

_NOW = datetime(2026, 7, 31, tzinfo=timezone.utc)
_SOURCE = ServiceRef(name="cocreation-signals", version="0.1.0", node="athena")


def _encode(event: CodebaseDeltaV1) -> dict:
    codec = OrionCodec()
    envelope = BaseEnvelope(
        kind="substrate.codebase_delta.v1", source=_SOURCE, payload=event.model_dump(mode="json")
    )
    return {"type": "message", "data": codec.encode(envelope)}


def _make_worker(*, baseline: CodebaseMassBaseline | None = None) -> tuple[BiometricsSubstrateWorker, MagicMock, MagicMock]:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._bus = MagicMock()
    worker._bus.codec = OrionCodec()
    fake_store = MagicMock()
    fake_store.get_latest_codebase_mass_baseline.return_value = baseline or CodebaseMassBaseline()
    worker._store = fake_store
    worker._settings = MagicMock()
    worker._settings.codebase_mass_baseline_retention_days = 30.0
    fake_write = MagicMock()
    worker._write_prediction_error_node = fake_write
    return worker, fake_store, fake_write


def test_git_domain_scores_and_persists() -> None:
    worker, fake_store, fake_write = _make_worker(
        baseline=CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=500.0, variance=10_000.0, n=5))
    )
    event = CodebaseDeltaV1(
        domain="git",
        observed_at=_NOW,
        git=GitDeltaPayloadV1(
            prev_sha="a" * 40, head_sha="b" * 40, commit_count=1,
            files_added=0, files_deleted=0, files_modified=1,
            lines_added=520, lines_removed=0,
        ),
    )
    worker._handle_codebase_delta_message(_encode(event))

    fake_write.assert_called_once()
    _, kwargs = fake_write.call_args
    assert kwargs["node_id"] == "node:substrate.codebase"
    assert kwargs["reducer_key"] == "codebase"
    assert kwargs["error"] == pytest.approx(0.2 / 3.0)  # zscore (520-500)/sqrt(10000)=0.2

    fake_store.save_codebase_mass_baseline.assert_called_once()
    saved_baseline, save_kwargs = fake_store.save_codebase_mass_baseline.call_args
    assert saved_baseline[0].git.n == 6
    assert save_kwargs["retention_days"] == 30.0

    # Regression guard: this call was originally missing entirely (found live
    # 2026-07-31) -- without it, field-digester's state_deltas.py ingestion
    # (which reads ReductionReceiptV1's target_kind="prediction_signal", not
    # the FalkorDB node directly) would never see this domain's
    # prediction_error at all, despite the node existing.
    fake_store.save_receipt.assert_called_once()
    receipt = fake_store.save_receipt.call_args[0][0]
    assert receipt.state_deltas[0].target_kind == "prediction_signal"
    assert receipt.state_deltas[0].target_id == "node:substrate.codebase"
    assert receipt.state_deltas[0].after["pressure_hints"]["prediction_error"] == pytest.approx(
        0.2 / 3.0, abs=1e-4
    )


def test_calm_tick_writes_node_but_skips_receipt() -> None:
    """A below-baseline (score == 0.0) tick must still write the FalkorDB
    node (a genuine calm reading, not skipped -- matches every other
    domain's 'write every tick' fix) but must NOT emit a save_receipt audit
    entry -- that's an audit trail of notable events, gated on error > 0.0,
    same convention every sibling domain's tick already uses (e.g. the
    biometrics tick, same file)."""
    worker, fake_store, fake_write = _make_worker(
        baseline=CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=5000.0, variance=1_000_000.0, n=10))
    )
    event = CodebaseDeltaV1(
        domain="git",
        observed_at=_NOW,
        git=GitDeltaPayloadV1(
            prev_sha="a" * 40, head_sha="b" * 40, commit_count=1,
            files_added=0, files_deleted=0, files_modified=1,
            lines_added=10, lines_removed=0,  # well below baseline -> score clamps to 0.0
        ),
    )
    worker._handle_codebase_delta_message(_encode(event))

    fake_write.assert_called_once()
    assert fake_write.call_args[1]["error"] == 0.0
    fake_store.save_receipt.assert_not_called()
    fake_store.save_codebase_mass_baseline.assert_called_once()


def test_pr_lifecycle_domain_scores() -> None:
    worker, fake_store, fake_write = _make_worker()
    event = CodebaseDeltaV1(
        domain="pr_lifecycle",
        observed_at=_NOW,
        pr_lifecycle=PrLifecycleDeltaPayloadV1(
            since=_NOW, until=_NOW, submitted_count=3, merged_count=2, closed_without_merge_count=0,
        ),
    )
    worker._handle_codebase_delta_message(_encode(event))

    fake_write.assert_called_once()
    fake_store.save_codebase_mass_baseline.assert_called_once()
    saved_baseline = fake_store.save_codebase_mass_baseline.call_args[0][0]
    assert saved_baseline.pr.n == 1
    assert saved_baseline.git.n == 0  # untouched -- this tick only carried pr_lifecycle


def test_graph_domain_scores_with_none_jaccard() -> None:
    """god_node_jaccard_similarity=None (unparseable upstream) must survive
    through the whole real decode -> reconstruct -> score path without
    crashing or getting coerced."""
    worker, fake_store, fake_write = _make_worker()
    event = CodebaseDeltaV1(
        domain="graph",
        observed_at=_NOW,
        graph=GraphDeltaPayloadV1(
            node_count_delta=-40, edge_count_delta=-110, community_count_delta=-3,
            god_node_jaccard_similarity=None,
        ),
    )
    worker._handle_codebase_delta_message(_encode(event))

    fake_write.assert_called_once()
    fake_store.save_codebase_mass_baseline.assert_called_once()


def test_malformed_wire_payload_rejected_by_schema_validation() -> None:
    """A wire payload with domain='git' but no real git field (violates
    CodebaseDeltaV1's own model_validator) is rejected at the
    `CodebaseDeltaV1.model_validate()` call itself (ValidationError, a
    ValueError subclass) -- proves the handler's decode/validate step, not
    the separate domain-dispatch `else` branch in the handler (that branch
    is unreachable given any *validated* CodebaseDeltaV1 instance today --
    kept only as forward-compatible defense against a future domain Literal
    value this consumer hasn't been updated to handle, not something a
    current unit test can construct without bypassing pydantic entirely)."""
    worker, fake_store, fake_write = _make_worker()
    raw_payload = {"domain": "git", "observed_at": _NOW.isoformat(), "git": None}
    codec = OrionCodec()
    envelope = BaseEnvelope(kind="substrate.codebase_delta.v1", source=_SOURCE, payload=raw_payload)
    raw_msg = {"type": "message", "data": codec.encode(envelope)}

    worker._handle_codebase_delta_message(raw_msg)

    fake_write.assert_not_called()
    fake_store.get_latest_codebase_mass_baseline.assert_not_called()
    fake_store.save_codebase_mass_baseline.assert_not_called()


def test_decode_failure_does_not_raise() -> None:
    worker, fake_store, fake_write = _make_worker()
    worker._handle_codebase_delta_message({"type": "message", "data": b"not-valid-envelope-bytes"})
    fake_write.assert_not_called()
    fake_store.save_codebase_mass_baseline.assert_not_called()
