"""evals/run_qwen35_fast_eval.py is leased: single checks under one operator hold through the gateway,
parallel isolation as ten separate gateway leases, no direct-URL mode (leased-evals lane-door PR 1)."""
from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import threading
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1] / "evals" / "run_qwen35_fast_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_qwen35_fast_eval", EVAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_no_direct_url_mode_and_no_worker_address():
    mod = _load()
    with pytest.raises(SystemExit):
        mod.main(["--base-url", "http://127.0.0.1:18013"])
    src = inspect.getsource(mod)
    assert not re.search(r"https?://[\w.\-]+:\d+", src), "the eval names no worker or tunnel address"
    assert "operator_lane_sync" in src and "gateway_call" in src


def test_parallel_load_reports_overlap_and_checks_isolation():
    mod = _load()
    busy = {"n": 0}
    lock = threading.Lock()

    def call(body):
        import time

        code = body["messages"][0]["content"].rsplit("code is ", 1)[1].split(".")[0]
        with lock:
            busy["n"] += 1
        time.sleep(0.2)
        with lock:
            busy["n"] -= 1
        return {"choices": [{"message": {"content": code}}]}

    peak = mod.parallel_load(call, lambda: [{"is_processing": i < busy["n"]} for i in range(10)], baseline_sec=0.1)
    assert peak >= 2


def test_parallel_load_fails_on_a_foreign_code():
    mod = _load()
    with pytest.raises(AssertionError):
        mod.parallel_load(lambda body: {"choices": [{"message": {"content": "AX17"}}]}, lambda: [], baseline_sec=0)


def test_queued_calls_read_as_no_overlap():
    mod = _load()
    lock = threading.Lock()

    def one_at_a_time(body):
        with lock:
            code = body["messages"][0]["content"].rsplit("code is ", 1)[1].split(".")[0]
            return {"choices": [{"message": {"content": code}}]}

    assert mod.parallel_load(one_at_a_time, lambda: [{"is_processing": False}] * 10, baseline_sec=0) < 2


def test_orions_own_busy_slots_do_not_count_as_the_evals_overlap():
    """Review finding: the fast worker is shared; Orion's own traffic showing in /slots made a queued
    run look parallel. The baseline sampled before the calls is subtracted."""
    mod = _load()
    lock = threading.Lock()

    def one_at_a_time(body):
        with lock:
            code = body["messages"][0]["content"].rsplit("code is ", 1)[1].split(".")[0]
            return {"choices": [{"message": {"content": code}}]}

    orion_busy = [{"is_processing": i < 4} for i in range(10)]   # four of Orion's calls, all along
    assert mod.parallel_load(one_at_a_time, lambda: orion_busy, baseline_sec=0.1) < 2
