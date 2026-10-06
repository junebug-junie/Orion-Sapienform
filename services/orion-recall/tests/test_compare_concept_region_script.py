"""Review finding 4: the live comparison script must refuse to count equality
against an incomplete cache hydrate, and report matched vs empty separately
with true maxima (not the max of per-query medians)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_concept_region_direct_vs_cache.py"


def _load():
    spec = importlib.util.spec_from_file_location("compare_concept_region_script", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_incomplete_hydrate_is_not_comparable() -> None:
    mod = _load()
    ok = SimpleNamespace(last_hydrate_ok=True, last_scan_receipt=SimpleNamespace(complete=True))
    assert mod.hydrate_is_complete(ok)
    for bad in (
        SimpleNamespace(last_hydrate_ok=False, last_scan_receipt=SimpleNamespace(complete=False)),
        SimpleNamespace(last_hydrate_ok=None, last_scan_receipt=None),
        SimpleNamespace(last_hydrate_ok=True, last_scan_receipt=SimpleNamespace(complete=False)),
    ):
        assert not mod.hydrate_is_complete(bad)


def test_main_exits_2_before_comparing_on_incomplete_hydrate(monkeypatch, tmp_path, capsys) -> None:
    mod = _load()
    queries = tmp_path / "q.txt"
    queries.write_text("hello\n")
    compared: list = []

    class _Cache:
        last_hydrate_ok = False
        last_scan_receipt = SimpleNamespace(complete=False, reason="local mutation during scan")

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(mod, "FalkorSubstrateStore", _Cache)
    monkeypatch.setattr(mod, "RedisGraphQueryClient", lambda **k: object())
    monkeypatch.setattr(mod, "fetch_concept_region_fragment", lambda *a, **k: compared.append(1) or [])
    assert mod.main(["--uri", "redis://127.0.0.1:16401", "--queries-file", str(queries)]) == 2
    assert compared == []
    assert "hydrate incomplete" in capsys.readouterr().err


def test_summary_uses_every_sample_for_the_max() -> None:
    mod = _load()
    rows = [
        {"cache": 1, "direct": 1, "identical_ordered": True, "samples_ms": [10.0, 11.0, 900.0]},
        {"cache": 1, "direct": 1, "identical_ordered": True, "samples_ms": [20.0, 21.0, 22.0]},
    ]
    summary = mod.summarize(rows)
    assert summary["ms_max"] == 900.0  # max of medians would say 21.0
    assert summary["samples"] == 6 and summary["identical"] == 2


def test_reinforce_mode_refuses_production() -> None:
    mod = _load()
    assert not mod.reinforce_allowed("redis://127.0.0.1:6380", True)
    assert not mod.reinforce_allowed("redis://127.0.0.1:16401", False)
    assert mod.reinforce_allowed("redis://127.0.0.1:16401", True)


def test_production_guard_covers_hostname_port_and_configured_uri(monkeypatch) -> None:
    mod = _load()
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    assert mod.is_production_uri("redis://orion-athena-falkordb:6379")
    assert mod.is_production_uri("redis://ORION-ATHENA-FALKORDB:6379")
    assert mod.is_production_uri("redis://127.0.0.1:6380")
    assert not mod.is_production_uri("redis://127.0.0.1:16401")
    # Same resolved address and port as the configured production URI.
    monkeypatch.setenv("FALKORDB_URI", "redis://127.0.0.1:17777")
    assert mod.is_production_uri("redis://localhost:17777")
    assert not mod.reinforce_allowed("redis://localhost:17777", True)
    assert mod.reinforce_allowed("redis://127.0.0.1:16401", True)
