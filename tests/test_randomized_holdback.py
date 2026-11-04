"""The only arm that licenses the word "causal".

The `no_action` control arm is quasi-experimental: ticks where nothing was
proposed are systematically calmer ticks, and baseline binning absorbs most of
that selection and provably not all of it. A randomized holdback withholds
ticks that WOULD have acted, so assignment is independent of the signal.

The design decision these tests exist to protect is PER-TICK vs PER-CANDIDATE.
The field delta is frame-wide. Withholding one candidate while its siblings run
gives a control observation contaminated by those siblings -- which is exactly
what made the capacity-blocked arm unusable. Repeating that mistake here would
produce a worse-than-useless arm wearing the word "randomized".
"""

from __future__ import annotations


import pytest

from orion.autonomy.contrast import (
    CONTROL_ARM_PRECEDENCE,
    ControlCell,
    contrast,
)
from orion.autonomy.prediction import EffectPosterior



class TestTheHoldbackArmWinsThePrecedence:
    def test_it_outranks_no_action(self):
        assert CONTROL_ARM_PRECEDENCE[0] == "randomized_holdback"

    def test_contrast_prefers_it_even_with_far_less_data(self):
        """A small causal arm beats a large confounded one. That is the whole
        reason to spend capability on it."""
        treated = {
            ("maintain", "host:docker_images", "resource_pressure", 8):
                EffectPosterior(mean=-0.15, variance=0.001, n=3000)
        }
        control = {
            ("resource_pressure", "no_action", 8):
                ControlCell(EffectPosterior(-0.14, 0.0005, 20000), moved_n=18000, move_rate=0.9),
            ("resource_pressure", "randomized_holdback", 8):
                ControlCell(EffectPosterior(-0.02, 0.01, 40), moved_n=35, move_rate=0.9),
        }
        est = contrast(treated, control, "maintain", "host:docker_images", "resource_pressure")
        assert est.control_arm == "randomized_holdback"
        assert est.evidence_class == "experimental"
        # -0.15 - (-0.02) = -0.13, not the -0.01 the confounded arm would give.
        assert est.value == pytest.approx(-0.13, abs=1e-9)
        assert est.control_n == 40

    def test_without_a_holdback_arm_it_falls_back_and_says_so(self):
        treated = {
            ("maintain", "t", "resource_pressure", 8): EffectPosterior(-0.15, 0.001, 3000)
        }
        control = {
            ("resource_pressure", "no_action", 8):
                ControlCell(EffectPosterior(-0.14, 0.0005, 20000), moved_n=18000, move_rate=0.9)
        }
        est = contrast(treated, control, "maintain", "t", "resource_pressure")
        assert est.control_arm == "no_action"
        assert est.evidence_class == "quasi_experimental"
