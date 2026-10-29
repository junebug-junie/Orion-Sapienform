"""scripts/gpu_power_bench.py: the energy math, and that the card's limit always goes back."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gpu_power_bench.py"
_spec = importlib.util.spec_from_file_location("gpu_power_bench", _PATH)
bench = importlib.util.module_from_spec(_spec)
sys.modules["gpu_power_bench"] = bench  # dataclasses look the module up while the class is built
_spec.loader.exec_module(bench)


def test_constant_power_integrates_to_watts_times_seconds() -> None:
    samples = [(t / 10, 200.0) for t in range(0, 31)]  # 0..3 s at 200 W
    assert bench.energy_joules(samples, 0.5, 2.5) == pytest.approx(400.0)


def test_window_edges_are_interpolated() -> None:
    samples = [(0.0, 100.0), (1.0, 200.0), (2.0, 200.0)]
    # 0.5..1.5: 0.5 s ramping 150->200 (87.5 J) + 0.5 s at 200 (100 J)
    assert bench.energy_joules(samples, 0.5, 1.5) == pytest.approx(187.5)


@pytest.mark.parametrize("t0,t1", [(-1.0, 1.0), (0.5, 9.0), (1.0, 1.0), (2.0, 1.0)])
def test_uncovered_or_empty_window_is_none_not_a_guess(t0: float, t1: float) -> None:
    samples = [(0.0, 100.0), (1.0, 100.0), (2.0, 100.0)]
    assert bench.energy_joules(samples, t0, t1) is None


def test_sample_splits_reading_and_writing_energy() -> None:
    s = bench.Sampler(2)
    s.power = [(t / 10, 100.0 if t < 10 else 200.0) for t in range(0, 41)]  # 100 W then 200 W
    s.temp = [(t / 10, 60) for t in range(0, 41)]
    s.clock = [(t / 10, 1300) for t in range(0, 41)]
    timings = {"prompt_ms": 900.0, "predicted_ms": 2000.0, "prompt_n": 1800, "predicted_n": 64}
    out = bench.to_sample(150, 0, s, 0.0, timings)
    assert out.prompt_tok_s == 2000.0
    assert out.gen_tok_s == 32.0
    assert out.gen_avg_w == pytest.approx(200.0, abs=3)  # ramp sits at the start of the window
    assert out.gen_j_per_tok == pytest.approx(out.gen_avg_w * 2.0 / 64, abs=0.01)
    assert out.max_temp_c == 60


def test_summary_compares_each_cap_to_the_top_cap() -> None:
    def mk(cap: int, tok_s: float, jpt: float) -> object:
        return bench.Sample(cap, 0, 1000, 1.0, 256, 256 / tok_s, 1000.0, tok_s, 0.1, jpt, 200.0, 70, 1400)

    rows = bench.summarize([mk(250, 40.0, 5.0), mk(250, 40.0, 5.0), mk(150, 36.0, 4.0)])
    assert [r["cap_w"] for r in rows] == [250, 150]
    assert rows[1]["gen_tok_s_vs_top"] == 0.9
    assert rows[1]["gen_j_per_tok_vs_top"] == 0.8


class _Boom(RuntimeError):
    pass


def test_limit_is_restored_when_a_request_fails(monkeypatch, tmp_path) -> None:
    calls: list[float] = []
    monkeypatch.setattr(bench, "read_limits", lambda gpu: (250.0, 100.0, 250.0))
    monkeypatch.setattr(bench, "set_limit", lambda gpu, w: calls.append(w))
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    monkeypatch.setattr(bench.Sampler, "run", lambda self: None)

    def fail(*_a, **_k):
        raise _Boom("server went away")

    monkeypatch.setattr(bench, "one_request", fail)
    with pytest.raises(_Boom):
        bench.main(["--gpu", "2", "--port", "8016", "--caps", "200,150", "--out", str(tmp_path)])
    assert calls == [200, 250.0]  # set the first cap, then put the original back


def test_caps_outside_the_card_range_are_refused_before_touching_it(monkeypatch, tmp_path) -> None:
    calls: list[float] = []
    monkeypatch.setattr(bench, "read_limits", lambda gpu: (300.0, 150.0, 300.0))
    monkeypatch.setattr(bench, "set_limit", lambda gpu, w: calls.append(w))
    assert bench.main(["--gpu", "3", "--port", "8012", "--caps", "250,100", "--out", str(tmp_path)]) == 2
    assert calls == []


def test_prompt_is_identical_every_run() -> None:
    assert bench.fixed_prompt() == bench.fixed_prompt()
    assert len(bench.fixed_prompt()) > 4000
