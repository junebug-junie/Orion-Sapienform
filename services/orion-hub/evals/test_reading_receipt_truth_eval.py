"""Offline behavioral eval: durable receipt truth outranks fluent FCC prose."""
from __future__ import annotations

import pytest

from orion.harness.reading_receipts import enforce_reading_receipt_grounding
from orion.schemas.reading import ReadingRecommendationOutcomeV1


@pytest.mark.parametrize(
    "model_response",
    [
        "I filed the source away and will return to it when the worker wakes up.",
        "That is safely on my reading list now; future processing is guaranteed.",
        "The article reports an invented 93% result, and I have saved it for synthesis.",
        "Persistence succeeded even though the tool reported an error.",
    ],
)
def test_failed_receipt_never_scores_as_persisted_or_read(model_response: str) -> None:
    outcome = ReadingRecommendationOutcomeV1(
        tool_use_ids=["attempt-a", "attempt-b"],
        attempt_count=2,
        url="https://example.org/research/source",
        acceptance="unknown",
        failure_kind="tool_error",
    )

    grounded = enforce_reading_receipt_grounding(model_response, [outcome])

    assert model_response not in grounded
    assert "acceptance is unknown" in grounded
    assert "did not read" in grounded


def test_successful_receipt_scores_only_with_id_and_status_visible() -> None:
    outcome = ReadingRecommendationOutcomeV1(
        tool_use_ids=["attempt-a"],
        attempt_count=1,
        url="https://example.org/research/source",
        acceptance="accepted",
        request_id="11111111-1111-1111-1111-111111111111",
        status="queued",
    )

    grounded = enforce_reading_receipt_grounding("Accepted.", [outcome])

    assert "11111111-1111-1111-1111-111111111111" in grounded
    assert "current status: `queued`" in grounded
