"""Gate: TRUST_RUPTURE_DEFER_THRESHOLD is eval-calibrated at 0.65.

Changing the constant in ``orion/thought/policy_refusal.py`` requires re-running
``orion/thought/evals/trust_rupture_eval.py`` and updating this frozen value.
"""

from __future__ import annotations

import pytest

from orion.thought.evals.trust_rupture_eval import (
    _load_fixtures,
    evaluate_trust_rupture_fixture,
    run_trust_rupture_eval_corpus,
)
from orion.thought.policy_refusal import TRUST_RUPTURE_DEFER_THRESHOLD

FROZEN_TRUST_RUPTURE_DEFER_THRESHOLD = 0.65


def test_trust_rupture_threshold_frozen_at_eval_value() -> None:
    assert TRUST_RUPTURE_DEFER_THRESHOLD == FROZEN_TRUST_RUPTURE_DEFER_THRESHOLD


def test_trust_rupture_eval_corpus_passes_at_frozen_threshold() -> None:
    failures = run_trust_rupture_eval_corpus()
    assert failures == [], "\n".join(failures)


@pytest.mark.parametrize("fixture", _load_fixtures(), ids=lambda f: f["id"])
def test_trust_rupture_fixture(fixture: dict) -> None:
    failures = evaluate_trust_rupture_fixture(fixture)
    assert failures == [], "\n".join(failures)
