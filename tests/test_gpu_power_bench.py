"""scripts/gpu_power_bench.py: the energy math, dirty-sample rules, and that the card's limit always goes back."""
from __future__ import annotations

import importlib.util
import signal
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


def _sampler(power: list[tuple[float, float]]) -> object:
    s = bench.Sampler(2)
    s.power = power
    s.temp = [(t, 60) for t, _ in power]
    s.clock = [(t, 1300) for t, _ in power]
    return s


def test_sample_windows_are_anchored_to_the_reply() -> None:
    # Reading 1.0..2.0 s at 100 W, writing 2.0..4.0 s at 200 W; reply arrives at 4.0.
    power = [(t / 100, 100.0 if t < 200 else 200.0) for t in range(0, 501)]
    timings = {"prompt_ms": 1000.0, "predicted_ms": 2000.0, "prompt_n": 2000, "predicted_n": 64}
    out = bench.to_sample(150, 0, _sampler(power), 4.0, timings)
    assert out.prompt_tok_s == 2000.0
    assert out.gen_tok_s == 32.0
    assert out.prompt_j_per_tok == pytest.approx(100.0 / 2000, abs=1e-3)
    assert out.gen_avg_w == pytest.approx(200.0, abs=0.6)
    assert out.gen_j_per_tok == pytest.approx(400.0 / 64, abs=0.02)


def test_waited_time_is_round_trip_minus_server_clocks() -> None:
    assert bench.waited_s(10.0, 13.4, {"prompt_ms": 1000.0, "predicted_ms": 2000.0}) == pytest.approx(0.4)


def test_summary_compares_to_the_first_cap_and_never_switches_baseline() -> None:
    def mk(cap: int, tok_s: float, jpt: float) -> object:
        return bench.Sample(cap, 0, 1000, 1.0, 256, 256 / tok_s, 1000.0, tok_s, 0.1, jpt, 200.0, 70, 1400)

    rows = bench.summarize([mk(250, 40.0, 5.0), mk(250, 40.0, 5.0), mk(150, 36.0, 4.0)], [250, 150])
    assert rows[1]["gen_tok_s_vs_top"] == 0.9
    assert rows[1]["gen_j_per_tok_vs_top"] == 0.8
    # 250 W got no clean samples: ratios are None, not silently relative to 200 W.
    rows = bench.summarize([mk(200, 38.0, 4.5), mk(150, 36.0, 4.0)], [250, 200, 150])
    assert rows[0]["n"] == 0
    assert all(r["gen_tok_s_vs_top"] is None for r in rows)


def test_slots_without_is_processing_abort_instead_of_reading_clean(monkeypatch) -> None:
    monkeypatch.setattr(bench, "http_json", lambda *a, **k: [{"id": 0, "state": 0}])
    with pytest.raises(bench.BenchAbort):
        bench.busy_slots("http://x")


def test_busy_slots_counts_our_own_slot_unless_ignored(monkeypatch) -> None:
    monkeypatch.setattr(bench, "http_json", lambda *a, **k: [{"id": 0, "is_processing": True},
                                                              {"id": 1, "is_processing": False}])
    assert bench.busy_slots("http://x") == [0]
    assert bench.busy_slots("http://x", ignore=0) == []


def test_dead_sampler_aborts_the_quiet_wait() -> None:
    s = bench.Sampler(2)  # never started: not alive, no readings
    with pytest.raises(bench.BenchAbort):
        bench.wait_until_quiet("http://x", s, 80.0, timeout_s=5)


def test_thread_join_works() -> None:
    # Regression: an Event named `_stop` shadows Thread._stop and breaks join().
    s = bench.Sampler(2)
    s.run = lambda: None
    s.start()
    s.join(timeout=2)
    w = bench.SlotWatch("http://x", (0,))
    w.run = lambda: None
    w.start()
    w.join(timeout=2)


def test_restore_retries_after_an_interrupt_and_ignores_signals_meanwhile(monkeypatch) -> None:
    calls: list[float] = []
    seen_handlers: list[object] = []

    def flaky(gpu, w):
        seen_handlers.append(signal.getsignal(signal.SIGTERM))
        calls.append(w)
        if len(calls) == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(bench, "set_limit", flaky)
    assert bench.restore_limit(2, 250.0) is True
    assert calls == [250.0, 250.0]
    assert all(h == signal.SIG_IGN for h in seen_handlers)
    assert signal.getsignal(signal.SIGTERM) != signal.SIG_IGN  # put back afterwards


def _patch_main(monkeypatch, *, limits=(250.0, 250.0, 100.0, 250.0)) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(bench, "read_limits", lambda gpu: limits)
    monkeypatch.setattr(bench, "set_limit", lambda gpu, w: calls.append(w))
    monkeypatch.setattr(bench, "busy_slots", lambda base, ignore=None: [])
    monkeypatch.setattr(bench, "http_json", lambda *a, **k: {"model_path": "m"})
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    monkeypatch.setattr(bench.Sampler, "run", lambda self: None)
    return calls


def _argv(tmp_path: Path, caps: str = "200,150") -> list[str]:
    return ["--gpu", "2", "--port", "8016", "--caps", caps, "--out", str(tmp_path / "out"),
            "--lock-dir", str(tmp_path)]


@pytest.mark.parametrize("exc", [RuntimeError("unexpected bug"), SystemExit(143), KeyboardInterrupt()])
def test_limit_is_restored_on_error_sigterm_and_ctrl_c(monkeypatch, tmp_path, exc) -> None:
    calls = _patch_main(monkeypatch)

    def fail(*_a, **_k):
        raise exc

    monkeypatch.setattr(bench, "one_request", fail)
    with pytest.raises(type(exc)):
        bench.main(_argv(tmp_path))
    assert calls == [200, 250.0]  # set the first cap, then put the original back


def test_all_dirty_attempts_end_the_cap_and_still_restore(monkeypatch, tmp_path) -> None:
    calls = _patch_main(monkeypatch)
    monkeypatch.setattr(bench, "wait_until_quiet", lambda *a, **k: True)
    monkeypatch.setattr(bench, "token_counter", lambda base: None)
    timings = {"prompt_ms": 1000.0, "predicted_ms": 1000.0, "prompt_n": 10, "predicted_n": 10}
    # Every request reports a 5 s round trip for 2 s of server work: it waited, so it is dropped.
    monkeypatch.setattr(bench, "one_request", lambda *a, **k: (0.0, 5.0, timings))
    monkeypatch.setattr(bench.SlotWatch, "run", lambda self: None)
    assert bench.main(_argv(tmp_path, caps="200") + ["--repeats", "2", "--max-retries", "3"]) == 0
    assert calls == [200, 250.0]
    import json
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["by_cap"][0]["n"] == 0
    assert len(summary["meta"]["dropped"]) == 5  # 2 repeats + 3 retries, all thrown away


def test_card_not_at_default_is_refused_before_touching_it(monkeypatch, tmp_path) -> None:
    calls = _patch_main(monkeypatch, limits=(150.0, 250.0, 100.0, 250.0))
    assert bench.main(_argv(tmp_path)) == 2
    assert calls == []


def test_caps_outside_the_card_range_are_refused_before_touching_it(monkeypatch, tmp_path) -> None:
    calls = _patch_main(monkeypatch, limits=(300.0, 300.0, 150.0, 300.0))
    assert bench.main(_argv(tmp_path, caps="250,100")) == 2
    assert calls == []


def test_second_bench_on_the_same_card_is_refused(monkeypatch, tmp_path) -> None:
    import fcntl

    calls = _patch_main(monkeypatch)
    held = open(tmp_path / "gpu-power-bench-2.lock", "w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert bench.main(_argv(tmp_path)) == 2
    finally:
        held.close()
    assert calls == []


def test_prompt_is_identical_every_run() -> None:
    assert bench.fixed_prompt() == bench.fixed_prompt()
    assert len(bench.fixed_prompt()) > 4000


def test_server_down_at_start_changes_nothing(monkeypatch, tmp_path) -> None:
    calls = _patch_main(monkeypatch)
    monkeypatch.setattr(bench, "wait_for_server", lambda base, wait_s: False)
    assert bench.main(_argv(tmp_path)) == 3
    assert calls == []


def test_wait_for_server_retries_until_it_answers(monkeypatch) -> None:
    answers = iter([bench.urllib.error.URLError("refused"), bench.urllib.error.URLError("refused"), {}])

    def health(*_a, **_k):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(bench, "http_json", health)
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    assert bench.wait_for_server("http://x", 60) is True


@pytest.mark.parametrize("err", [bench.urllib.error.URLError(ConnectionRefusedError(111, "refused")),
                                 TimeoutError("timed out"), bench.http.client.IncompleteRead(b""),
                                 KeyError("timings")])
def test_server_leaving_mid_run_keeps_samples_and_restores(monkeypatch, tmp_path, err) -> None:
    calls = _patch_main(monkeypatch)

    def gone(*_a, **_k):
        raise err

    monkeypatch.setattr(bench, "one_request", gone)
    assert bench.main(_argv(tmp_path)) == 1
    assert calls == [200, 250.0]
    import json
    meta = json.loads((tmp_path / "out" / "summary.json").read_text())["meta"]
    assert meta["aborted"].startswith("model server went away")


def test_slots_failing_before_start_is_a_clean_refusal(monkeypatch, tmp_path) -> None:
    calls = _patch_main(monkeypatch)

    def no_slots(base, ignore=None):
        raise bench.BenchAbort("no is_processing")

    monkeypatch.setattr(bench, "busy_slots", no_slots)
    assert bench.main(_argv(tmp_path)) == 2
    monkeypatch.setattr(bench, "busy_slots", lambda base, ignore=None: (_ for _ in ()).throw(TimeoutError()))
    assert bench.main(_argv(tmp_path)) == 3
    assert calls == []


def test_ctrl_c_while_waiting_for_the_server_is_clean(monkeypatch, tmp_path) -> None:
    calls = _patch_main(monkeypatch)

    def interrupted(base, wait_s):
        raise KeyboardInterrupt

    monkeypatch.setattr(bench, "wait_for_server", interrupted)
    assert bench.main(_argv(tmp_path)) == 130
    assert calls == []


def test_prompt_size_is_a_flag() -> None:
    assert len(bench.fixed_prompt(480)) > 7 * len(bench.fixed_prompt(60))
    assert bench.fixed_prompt(480) == bench.fixed_prompt(480)


def test_busy_slots_ignores_every_slot_of_ours(monkeypatch) -> None:
    monkeypatch.setattr(bench, "http_json", lambda *a, **k: [{"id": 0, "is_processing": True},
                                                              {"id": 1, "is_processing": True}])
    assert bench.busy_slots("http://x", ignore=(0, 1)) == []
    assert bench.busy_slots("http://x", ignore=(0,)) == [1]


def test_one_round_sends_every_slot_at_once(monkeypatch) -> None:
    import threading

    barrier = threading.Barrier(2, timeout=2)  # both requests must be in flight together

    def req(base, prompt, n, slot):
        barrier.wait()
        return (0.0, 1.0, {"slot": slot})

    monkeypatch.setattr(bench, "one_request", req)
    assert [r[2]["slot"] for r in bench.one_round("http://x", "p", 8, (0, 1))] == [0, 1]


def test_one_round_reraises_a_request_failure(monkeypatch) -> None:
    def req(base, prompt, n, slot):
        if slot == 1:
            raise TimeoutError("stalled")
        return (0.0, 1.0, {})

    monkeypatch.setattr(bench, "one_request", req)
    with pytest.raises(TimeoutError):
        bench.one_round("http://x", "p", 8, (0, 1))


def test_round_energy_spans_first_read_to_last_reply() -> None:
    # Constant 200 W, 0..20 s. Request A: reads 1 s, writes 9 s, reply at 12. B: reads 2 s, writes 8 s, reply at 13.
    power = [(t / 10, 200.0) for t in range(0, 201)]
    a = (0.0, 12.0, {"prompt_ms": 1000.0, "predicted_ms": 9000.0, "prompt_n": 1000, "predicted_n": 270})
    b = (0.0, 13.0, {"prompt_ms": 2000.0, "predicted_ms": 8000.0, "prompt_n": 1000, "predicted_n": 240})
    s = bench.to_round_sample(150, 0, _sampler(power), [a, b])
    # Round runs 2.0..13.0 s = 11 s at 200 W = 2200 J over 510 written tokens.
    assert s.slots == 2
    assert s.all_in_j_per_gen_tok == pytest.approx(2200 / 510, abs=1e-3)
    assert s.total_gen_tok_s == pytest.approx(510 / 11, abs=0.01)
    assert s.gen_tok_s == pytest.approx(30.0)  # median of 30 and 30: what one turn sees
    assert s.gen_j_per_tok is None and s.prompt_j_per_tok is None  # phases overlap: no per-phase split


def test_single_slot_round_keeps_per_phase_numbers() -> None:
    power = [(t / 100, 100.0 if t < 200 else 200.0) for t in range(0, 501)]
    timings = {"prompt_ms": 1000.0, "predicted_ms": 2000.0, "prompt_n": 2000, "predicted_n": 64}
    s = bench.to_round_sample(150, 0, _sampler(power), [(0.0, 4.0, timings)])
    assert s.gen_j_per_tok == pytest.approx(400.0 / 64, abs=0.02)
    assert s.all_in_j_per_gen_tok == pytest.approx(500.0 / 64, abs=0.03)
    assert s.total_gen_tok_s == pytest.approx(64 / 3, abs=0.01)


def test_two_slot_run_ends_clean_and_restores(monkeypatch, tmp_path) -> None:
    calls = _patch_main(monkeypatch)
    monkeypatch.setattr(bench, "http_json", lambda *a, **k: [{"id": 0}, {"id": 1}])
    monkeypatch.setattr(bench, "wait_until_quiet", lambda *a, **k: True)
    monkeypatch.setattr(bench, "token_counter", lambda base: None)
    monkeypatch.setattr(bench.SlotWatch, "run", lambda self: None)
    monkeypatch.setattr(bench, "to_round_sample", lambda cap, rep, sampler, results: bench.Sample(
        cap, rep, 10, 1.0, 10, 1.0, 10.0, 10.0, None, None, 100.0, 60, 1300, slots=len(results)))
    sent: list[int] = []

    def req(base, prompt, n, slot):
        sent.append(slot)
        return (0.0, 2.0, {"prompt_ms": 1000.0, "predicted_ms": 1000.0, "prompt_n": 10, "predicted_n": 10})

    monkeypatch.setattr(bench, "one_request", req)
    assert bench.main(_argv(tmp_path, caps="200") + ["--repeats", "1", "--slots", "0,1"]) == 0
    assert calls == [200, 250.0]
    assert sorted(sent[1:]) == [0, 1]  # after the one-slot warm-up, one request per slot
    import json
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["meta"]["slots"] == [0, 1]


@pytest.mark.parametrize("slots", ["0,0", "0,5"])
def test_bad_slots_are_refused_before_touching_the_card(monkeypatch, tmp_path, slots) -> None:
    calls = _patch_main(monkeypatch)
    monkeypatch.setattr(bench, "http_json", lambda *a, **k: [{"id": 0}, {"id": 1}])
    assert bench.main(_argv(tmp_path) + ["--slots", slots]) == 2
    assert calls == []


def test_queueing_between_our_own_slots_is_not_a_dirty_round(monkeypatch, tmp_path) -> None:
    # Live gpu2: the server reads the two prompts one after another, so slot 1 waits ~25 s.
    calls = _patch_main(monkeypatch)
    monkeypatch.setattr(bench, "http_json", lambda *a, **k: [{"id": 0}, {"id": 1}])
    monkeypatch.setattr(bench, "wait_until_quiet", lambda *a, **k: True)
    monkeypatch.setattr(bench, "token_counter", lambda base: None)
    monkeypatch.setattr(bench.SlotWatch, "run", lambda self: None)
    monkeypatch.setattr(bench, "to_round_sample", lambda cap, rep, sampler, results: bench.Sample(
        cap, rep, 10, 1.0, 10, 1.0, 10.0, 10.0, None, None, 100.0, 60, 1300, slots=len(results)))
    tm = {"prompt_ms": 1000.0, "predicted_ms": 1000.0, "prompt_n": 10, "predicted_n": 10}
    monkeypatch.setattr(bench, "one_request",
                        lambda base, prompt, n, slot: (0.0, 2.0 if slot == 0 else 27.0, tm))
    assert bench.main(_argv(tmp_path, caps="200") + ["--repeats", "1", "--max-retries", "0",
                                                      "--slots", "0,1"]) == 0
    import json
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["by_cap"][0]["n"] == 1 and summary["meta"]["dropped"] == []
    assert calls == [200, 250.0]


def test_foreign_work_on_a_slot_we_already_freed_is_dirty(monkeypatch) -> None:
    # Both slots are ours. Slot 0's reply came back 5 s ago and it is busy again: someone else's request.
    monkeypatch.setattr(bench, "http_json", lambda *a, **k: [{"id": 0, "is_processing": True},
                                                              {"id": 1, "is_processing": True}])
    w = bench.SlotWatch("http://x", (0, 1))
    w._poll(0)
    assert not w.dirty  # both still running our own requests
    w.finished[0] = bench.time.monotonic() - 5
    w._poll(0)
    assert w.dirty


def test_a_slot_busy_just_after_its_reply_is_not_dirty(monkeypatch) -> None:
    monkeypatch.setattr(bench, "http_json", lambda *a, **k: [{"id": 0, "is_processing": True}])
    w = bench.SlotWatch("http://x", (0, 1))
    w.finished[0] = bench.time.monotonic()  # the server has not marked it idle yet
    w._poll(0)
    assert not w.dirty


def test_one_round_records_when_each_reply_arrived(monkeypatch) -> None:
    monkeypatch.setattr(bench, "one_request", lambda base, prompt, n, slot: (0.0, 10.0 + slot, {}))
    finished: dict = {}
    bench.one_round("http://x", "p", 8, (0, 1), finished)
    assert finished == {0: 10.0, 1: 11.0}


def test_one_round_fails_fast_without_waiting_for_the_other_slot(monkeypatch) -> None:
    import threading

    release = threading.Event()

    def req(base, prompt, n, slot):
        if slot == 0:
            raise TimeoutError("stalled")
        release.wait(10)  # the other request would run for minutes
        return (0.0, 1.0, {})

    monkeypatch.setattr(bench, "one_request", req)
    t0 = bench.time.monotonic()
    with pytest.raises(TimeoutError):
        bench.one_round("http://x", "p", 8, (0, 1))
    assert bench.time.monotonic() - t0 < 3
    release.set()


def test_zero_server_time_does_not_crash_the_round() -> None:
    power = [(t / 10, 200.0) for t in range(0, 201)]
    a = (0.0, 12.0, {"prompt_ms": 0.0, "predicted_ms": 9000.0, "prompt_n": 0, "predicted_n": 270})
    b = (0.0, 13.0, {"prompt_ms": 2000.0, "predicted_ms": 8000.0, "prompt_n": 1000, "predicted_n": 240})
    assert bench.to_round_sample(150, 0, _sampler(power), [a, b]).slots == 2


@pytest.mark.parametrize("extra", [["--slots", ""], ["--slots", "0,"], ["--prompt-sentences", "0"]])
def test_bad_flags_are_refused_cleanly(monkeypatch, tmp_path, extra) -> None:
    calls = _patch_main(monkeypatch)
    assert bench.main(_argv(tmp_path) + extra) == 2
    assert calls == []


def test_expected_limit_is_the_configured_target_on_its_host(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "limits.json"
    cfg.write_text('{"host": "circe", "cards": {"2": {"watts": 150}}}')
    monkeypatch.setattr(bench.socket, "gethostname", lambda: "circe")
    assert bench.expected_limit(2, 250.0, cfg) == 150.0
    assert bench.expected_limit(1, 250.0, cfg) == 250.0           # not listed: default
    monkeypatch.setattr(bench.socket, "gethostname", lambda: "athena")
    assert bench.expected_limit(2, 250.0, cfg) == 250.0           # another host's config
    assert bench.expected_limit(2, 250.0, tmp_path / "none") == 250.0


def test_capped_card_at_its_configured_limit_runs_and_restores_it(monkeypatch, tmp_path) -> None:
    calls = _patch_main(monkeypatch, limits=(150.0, 250.0, 100.0, 250.0))
    cfg = tmp_path / "limits.json"
    cfg.write_text('{"host": "circe", "cards": {"2": {"watts": 150}}}')
    monkeypatch.setattr(bench.socket, "gethostname", lambda: "circe")

    def fail(*_a, **_k):
        raise RuntimeError("stop after the first cap")

    monkeypatch.setattr(bench, "one_request", fail)
    with pytest.raises(RuntimeError):
        bench.main(_argv(tmp_path, caps="250,150") + ["--limits-config", str(cfg)])
    assert calls == [250, 150.0]  # ran, then put the card back at 150 W, not the default


def test_lock_is_taken_before_the_limit_is_read(monkeypatch, tmp_path) -> None:
    import fcntl

    seen = []
    _patch_main(monkeypatch)

    def limits(gpu):
        probe = open(tmp_path / "gpu-power-bench-2.lock", "w")
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            seen.append("free")
        except BlockingIOError:
            seen.append("held")
        finally:
            probe.close()
        return (150.0, 250.0, 100.0, 250.0)

    monkeypatch.setattr(bench, "read_limits", limits)
    assert bench.main(_argv(tmp_path) + ["--limits-config", str(tmp_path / "none")]) == 2
    assert seen == ["held"]  # orion-gpu-power-limits cannot change it between our read and our run
