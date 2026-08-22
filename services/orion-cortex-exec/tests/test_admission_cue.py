"""ROADMAP A5 -- the read side, and the four states a zero can mean.

The thing under test is not arithmetic. It is that "asked and was never made to wait", "made no
requests at all", and "could not read the gateway" stay three different answers all the way into
the rendered cue, instead of collapsing into one `0` that would let Orion conclude nothing is
constraining it from an unreachable gateway.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app import admission_cue
from app.admission_cue import (
    admission_cue_for_settings,
    render_admission_cue,
    reset_cache,
)
from app.executor import _metacog_biometrics_cue


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_cache()
    yield
    reset_cache()


def _settings(**over):
    base = dict(
        cortex_exec_admission_cue_enabled=True,
        cortex_exec_llm_gateway_url="http://llm-gateway:8210",
        cortex_exec_admission_cue_window_s=21600.0,
        cortex_exec_admission_cue_ttl_sec=60.0,
        cortex_exec_admission_cue_timeout_sec=2.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _snapshot(**over):
    base = dict(
        window_s=21600.0, via="bus", checked=294, deferrals=0, timeouts=0, unchecked=0,
        queued=0, deferred_s_total=0.0, longest_wait_s=0.0, last_deferral_ts=None,
        truncated=False, routes=["quick_background"],
    )
    base.update(over)
    return base


class TestRender:
    def test_quiet_window_reports_the_denominator(self):
        """The live 2026-08-19 case: 294 asks, zero waits. `of` is what makes that readable."""
        assert render_admission_cue(_snapshot()) == {"n": 0, "of": 294, "h": 6.0}

    def test_no_requests_at_all_is_not_the_same_as_never_waiting(self):
        idle = render_admission_cue(_snapshot(checked=0))
        quiet = render_admission_cue(_snapshot(checked=294))
        assert idle == {"n": 0, "of": 0, "h": 6.0}
        assert idle != quiet

    def test_a_real_deferral_carries_its_duration(self):
        out = render_admission_cue(_snapshot(checked=291, deferrals=3, longest_wait_s=4.23))
        assert out == {"n": 3, "of": 291, "h": 6.0, "max_s": 4.2}

    def test_max_s_is_omitted_when_nothing_waited(self):
        """`longest_wait_s` is 0 at rest, but shipping a `max_s` key at all invites reading the
        /slots round trip as a wait. The ledger refuses to call it one; so does the renderer."""
        assert "max_s" not in render_admission_cue(_snapshot(longest_wait_s=0.021))

    @pytest.mark.parametrize(
        "bad", [None, "nope", 42, [], {}, {"deferrals": 1},
                {"checked": 1, "deferrals": 0, "window_s": 60.0}]  # no `unchecked`
    )
    def test_unreadable_payload_is_none_not_zero(self, bad):
        assert render_admission_cue(bad) is None

    def test_non_numeric_fields_degrade_to_none(self):
        assert render_admission_cue(_snapshot(checked="many")) is None


class TestFetchAndCache:
    def test_disabled_returns_none_and_never_fetches(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("must not fetch when disabled")

        monkeypatch.setattr(admission_cue, "fetch_admission_snapshot", _boom)
        assert admission_cue_for_settings(_settings(cortex_exec_admission_cue_enabled=False)) is None

    def test_unreachable_gateway_returns_none(self, monkeypatch):
        def _boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(admission_cue, "fetch_admission_snapshot", _boom)
        assert admission_cue_for_settings(_settings()) is None

    def test_result_is_cached_within_ttl(self, monkeypatch):
        calls = []

        def _fake(base_url, *, window_s, timeout_sec):
            calls.append(base_url)
            return _snapshot()

        monkeypatch.setattr(admission_cue, "fetch_admission_snapshot", _fake)
        first = admission_cue_for_settings(_settings())
        second = admission_cue_for_settings(_settings())
        assert first == second == {"n": 0, "of": 294, "h": 6.0}
        assert len(calls) == 1

    def test_a_failure_is_cached_too(self, monkeypatch):
        """Otherwise an unreachable gateway means a blocking urlopen on every metacog pass."""
        calls = []

        def _boom(*a, **k):
            calls.append(1)
            raise OSError("refused")

        monkeypatch.setattr(admission_cue, "fetch_admission_snapshot", _boom)
        assert admission_cue_for_settings(_settings()) is None
        assert admission_cue_for_settings(_settings()) is None
        assert len(calls) == 1

    def test_the_configured_window_reaches_the_request(self, monkeypatch):
        seen = {}

        def _fake(base_url, *, window_s, timeout_sec):
            seen.update(base_url=base_url, window_s=window_s, timeout_sec=timeout_sec)
            return _snapshot(window_s=window_s)

        monkeypatch.setattr(admission_cue, "fetch_admission_snapshot", _fake)
        admission_cue_for_settings(_settings(cortex_exec_admission_cue_window_s=3600.0))
        assert seen == {
            "base_url": "http://llm-gateway:8210", "window_s": 3600.0, "timeout_sec": 2.0,
        }


class TestCueRendering:
    """The cue builder is a pure function of ctx, so these drive it directly."""

    @staticmethod
    def _ctx(admission):
        return {
            "biometrics": {"status": "ok", "cluster": {"constraint": "NONE"}},
            "admission": admission,
        }

    def test_waited_key_appears_in_the_cue(self):
        cue = json.loads(_metacog_biometrics_cue(self._ctx({"n": 2, "of": 100, "h": 6.0, "max_s": 4.2})))
        assert cue["waited"] == {"n": 2, "of": 100, "h": 6.0, "max_s": 4.2}

    def test_quiet_window_still_renders_the_key(self):
        """`waited:{"n":0,...}` is a claim Orion can act on. Silence is not."""
        cue = json.loads(_metacog_biometrics_cue(self._ctx({"n": 0, "of": 294, "h": 6.0})))
        assert cue["waited"]["n"] == 0 and cue["waited"]["of"] == 294

    @pytest.mark.parametrize("absent", [None, {}, "unknown", 0])
    def test_unknown_omits_the_key_entirely(self, absent):
        """The failure this whole module is written against: unknown must not render as calm."""
        cue = json.loads(_metacog_biometrics_cue(self._ctx(absent)))
        assert "waited" not in cue

    def test_cue_stays_inside_its_char_budget(self):
        """The cue truncates to `{"status":...}` if it overruns, which would drop this and every
        other signal. A realistic full payload must not push it over."""
        from app.executor import _METACOG_BIOMETRICS_CUE_DRAFT_MAX_CHARS

        ctx = {
            "biometrics": {
                "status": "ok",
                "cluster": {
                    "constraint": "NONE",
                    "composite": {"strain": 0.11, "homeostasis": 0.89, "vitality": 0.77},
                    "peak_pressure": 1.0,
                    "peak_pressure_channel": "power",
                    "peak_pressure_node": "athena",
                    "measurements": {"chassis_watts": 786.0},
                    "measurements_missing": {"chassis_watts": ["circe"]},
                },
                "freshness_s": 12,
            },
            "admission": {"n": 12, "of": 2941, "h": 6.0, "max_s": 41.7},
        }
        cue = _metacog_biometrics_cue(ctx)
        assert len(cue) <= _METACOG_BIOMETRICS_CUE_DRAFT_MAX_CHARS
        assert "waited" in json.loads(cue)


class TestUncheckedIsNotCalm:
    """The gate FAILS OPEN. When llama.cpp's /slots is unreachable the request forwards
    unchecked, which counts toward `checked` and can never be a deferral. Rendering that as
    "asked N times, never constrained" would be a confident first-person claim assembled
    entirely from measurements that did not happen."""

    def test_a_fully_unmeasurable_window_is_unknown_not_zero(self):
        assert render_admission_cue(_snapshot(checked=294, unchecked=294)) is None

    def test_a_partially_unmeasurable_window_reports_the_gap(self):
        out = render_admission_cue(_snapshot(checked=294, unchecked=40))
        assert out == {"n": 0, "of": 294, "h": 6.0, "unk": 40}

    def test_a_fully_measured_window_carries_no_unk_key(self):
        assert "unk" not in render_admission_cue(_snapshot(checked=294, unchecked=0))

    def test_an_empty_window_is_still_reportable(self):
        """checked==0 and unchecked==0 is "made no requests", not "nothing measurable"."""
        assert render_admission_cue(_snapshot(checked=0, unchecked=0)) == {
            "n": 0, "of": 0, "h": 6.0,
        }


class TestMaxSNeverContradictsItself:
    def test_a_sub_decisecond_wait_is_not_rounded_to_zero(self):
        """Reachable with LLM_GATEWAY_BACKGROUND_MAX_WAIT_SEC=0: the first busy poll times out
        immediately, which IS a deferral, with a wait of ~0.02s. `round(0.02, 1) == 0.0` would
        render "I was made to wait, for zero seconds"."""
        out = render_admission_cue(_snapshot(checked=1, deferrals=1, longest_wait_s=0.021))
        assert out["max_s"] == 0.021
        assert out["max_s"] > 0.0

    def test_a_normal_wait_stays_at_one_decimal(self):
        out = render_admission_cue(_snapshot(checked=9, deferrals=3, longest_wait_s=4.2761))
        assert out["max_s"] == 4.3

    def test_max_s_is_absent_when_the_duration_is_genuinely_zero(self):
        assert "max_s" not in render_admission_cue(
            _snapshot(checked=1, deferrals=1, longest_wait_s=0.0)
        )


class TestFirstPersonScope:
    def test_the_fetch_asks_only_for_orions_own_call_path(self, monkeypatch):
        """`quick_background` also carries AI Town's NPC dialogue. A cue that says "I waited"
        must not be counting somebody else's wait."""
        import app.admission_cue as mod

        seen = {}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"window_s":60.0,"checked":0,"deferrals":0,"unchecked":0}'

        def _fake_urlopen(url, timeout=None):
            seen["url"] = url
            return _Resp()

        monkeypatch.setattr(mod, "urlopen", _fake_urlopen)
        mod.fetch_admission_snapshot("http://gw:8210", window_s=21600.0, timeout_sec=2.0)
        assert "via=bus" in seen["url"]
        assert "window_s=21600" in seen["url"]


class TestCueSurvivesTruncation:
    def test_waited_is_kept_when_the_payload_overruns(self):
        """An ABSENT `waited` means "gateway unreadable". Dropping a successfully-read,
        non-zero deferral count in the overrun fallback would manufacture that, with no trace."""
        ctx = {
            "biometrics": {
                "status": "ok",
                "cluster": {
                    "constraint": "THERMAL_HEADROOM_EXHAUSTED_ON_A_VERY_LONG_CHANNEL_NAME",
                    "composite": {"strain": 0.11, "homeostasis": 0.89, "vitality": 0.77},
                    "peak_pressure": 1.0,
                    "peak_pressure_channel": "gpu_mem_bandwidth_saturation_channel",
                    "peak_pressure_node": "athena-orchestration-warhorse",
                    "measurements": {"chassis_watts": 786.0},
                    # A long absent-node list is the realistic way this cue overruns: it
                    # names every machine missing from the fleet wattage total.
                    "measurements_missing": {
                        "chassis_watts": [f"node-{i:02d}-with-a-long-hostname" for i in range(8)]
                    },
                },
                "freshness_s": 12,
            },
            "admission": {"n": 7, "of": 2941, "h": 6.0, "max_s": 41.7},
        }
        cue = _metacog_biometrics_cue(ctx)
        from app.executor import _METACOG_BIOMETRICS_CUE_DRAFT_MAX_CHARS

        parsed = json.loads(cue)
        assert len(cue) <= _METACOG_BIOMETRICS_CUE_DRAFT_MAX_CHARS
        assert "peak_at" not in parsed, "this payload must actually overrun, or the test is moot"
        assert parsed["waited"] == {"n": 7, "of": 2941, "h": 6.0, "max_s": 41.7}
