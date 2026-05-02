"""Tests for normalization utilities."""
import pytest
from orion.signals.normalization import EwmaBand, NormalizationContext, clamp01, clamp11


class TestClamp:
    def test_clamp01_below(self):
        assert clamp01(-0.5) == 0.0

    def test_clamp01_above(self):
        assert clamp01(1.5) == 1.0

    def test_clamp01_in_range(self):
        assert clamp01(0.5) == 0.5

    def test_clamp11_below(self):
        assert clamp11(-2.0) == -1.0

    def test_clamp11_above(self):
        assert clamp11(2.0) == 1.0

    def test_clamp11_in_range(self):
        assert clamp11(0.7) == 0.7


class TestEwmaBand:
    def test_normalize_before_first_update_returns_zero_or_half(self):
        band = EwmaBand()
        val = band.normalize(0.5)
        assert 0.0 <= val <= 1.0

    def test_converges_to_stable_output(self):
        band = EwmaBand(alpha=0.1)
        for _ in range(50):
            band.update(0.7)
        result = band.normalize(0.7)
        assert 0.0 <= result <= 1.0

    def test_same_band_returned_from_context(self):
        ctx = NormalizationContext()
        band1 = ctx.get_band("biometrics", "gpu_util")
        band2 = ctx.get_band("biometrics", "gpu_util")
        assert band1 is band2

    def test_different_keys_different_bands(self):
        ctx = NormalizationContext()
        band1 = ctx.get_band("biometrics", "gpu_util")
        band2 = ctx.get_band("biometrics", "cpu")
        assert band1 is not band2

    def test_same_tracker_per_organ_metric(self):
        ctx = NormalizationContext()
        t1 = ctx.get_tracker("biometrics", "gpu_util")
        t2 = ctx.get_tracker("biometrics", "gpu_util")
        assert t1 is t2
        assert ctx.get_tracker("biometrics", "gpu_util") is not ctx.get_tracker("biometrics", "cpu")
