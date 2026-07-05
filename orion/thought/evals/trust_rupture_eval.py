from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orion.schemas.thought import StanceHarnessSliceV1, ThoughtEventV1
from orion.thought.policy_refusal import (
    TRUST_RUPTURE_DEFER_THRESHOLD,
    evaluate_thought_disposition,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "trust_rupture"


def _stance_slice(**overrides: object) -> StanceHarnessSliceV1:
    base = {
        "task_mode": "direct_response",
        "conversation_frame": "mixed",
        "answer_strategy": "direct",
    }
    base.update(overrides)
    return StanceHarnessSliceV1.model_validate(base)


def _thought_from_fixture(raw: dict[str, Any]) -> ThoughtEventV1:
    payload = dict(raw)
    payload.setdefault("event_id", "eval-thought")
    payload.setdefault("correlation_id", "eval-corr")
    payload.setdefault("session_id", None)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault(
        "stance_harness_slice",
        _stance_slice().model_dump(mode="json"),
    )
    return ThoughtEventV1.model_validate(payload)


def _load_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(_FIXTURES_DIR.glob("*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def evaluate_trust_rupture_fixture(fixture: dict[str, Any]) -> list[str]:
    """Return failure messages for a single fixture; empty when expectations pass."""
    thought = _thought_from_fixture(fixture["thought"])
    decision = evaluate_thought_disposition(
        thought,
        association_stale=bool(fixture.get("association_stale", False)),
        coalition_ids=set(fixture.get("coalition_ids") or []),
    )
    expect = fixture.get("expect") or {}
    failures: list[str] = []
    fixture_id = fixture.get("id", "<unknown>")

    if "disposition" in expect and decision.disposition != expect["disposition"]:
        failures.append(
            f"{fixture_id}: disposition expected {expect['disposition']!r}, "
            f"got {decision.disposition!r}"
        )
    if "boundary_register" in expect and decision.boundary_register != expect["boundary_register"]:
        failures.append(
            f"{fixture_id}: boundary_register expected {expect['boundary_register']!r}, "
            f"got {decision.boundary_register!r}"
        )
    reason_blob = " ".join(decision.reasons)
    for fragment in expect.get("reason_contains") or []:
        if fragment not in reason_blob:
            failures.append(f"{fixture_id}: expected reason fragment {fragment!r} in {decision.reasons!r}")

    trust = thought.trust_rupture_score
    if trust is not None:
        if trust >= TRUST_RUPTURE_DEFER_THRESHOLD and decision.disposition != "refuse":
            failures.append(
                f"{fixture_id}: trust_rupture_score={trust} >= threshold "
                f"{TRUST_RUPTURE_DEFER_THRESHOLD} must refuse"
            )
        if trust < TRUST_RUPTURE_DEFER_THRESHOLD and expect.get("disposition") == "proceed":
            if decision.disposition != "proceed":
                failures.append(
                    f"{fixture_id}: trust below threshold should proceed when otherwise valid, "
                    f"got {decision.disposition!r}"
                )

    return failures


def run_trust_rupture_eval_corpus() -> list[str]:
    """Run all trust rupture fixtures; return aggregated failure messages."""
    failures: list[str] = []
    for fixture in _load_fixtures():
        failures.extend(evaluate_trust_rupture_fixture(fixture))
    return failures
