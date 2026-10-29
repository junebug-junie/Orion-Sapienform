from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("run_call_usage_eval", Path(__file__).with_name("run_call_usage_eval.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _row(recorder="gateway", path="bus_rpc", corr="c1", outcome="served", prompt="14", completion="8",
         source="usage", duration="230", tps="70.0") -> str:
    return "\t".join([recorder, path, corr, outcome, prompt, completion, source, duration, tps]) + "\n"


def test_healthy_rows_pass():
    r = ev.evaluate([_row(), _row(path="http_openai", corr=""), _row(outcome="upstream_http_5xx", completion="")])
    assert r["ok"] and r["groups"]["gateway/bus_rpc"]["coverage"] == 1.0


def test_served_rows_without_tokens_fail_coverage():
    assert not ev.evaluate([_row(completion="")] + [_row()] * 3)["ok"]


def test_more_tokens_than_the_clock_allows_is_implausible():
    # 8000 tokens at 70 tok/s needs >= 114 s, not 230 ms
    assert not ev.evaluate([_row(completion="8000")])["ok"]


def test_no_rows_is_not_a_pass():
    assert not ev.evaluate([])["ok"]
