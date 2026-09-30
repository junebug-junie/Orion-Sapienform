"""Deterministic tests for grade_transport_baseline.py on fixture rows (no DB)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "grade_transport_baseline.py"
_spec = importlib.util.spec_from_file_location("grade_transport_baseline", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["grade_transport_baseline"] = mod
_spec.loader.exec_module(mod)

DAY = datetime(2026, 9, 30, tzinfo=timezone.utc)


def _row(hour: int, *, key="orion:exec:request:LLMGatewayService", z=0.0, ratio=1.0, floor=1000.0,
         floor_start=None, evaluated=40, day=0, opened=None, would=None, excluded=False, fp="f1"):
    return {
        "service": "cortex-exec", "instance": "chat", "key": key,
        "hour_start": (DAY + timedelta(days=day, hours=hour)).isoformat(),
        "flush_reason": "hour_end", "windows_seen": 120, "windows_evaluated": evaluated,
        "success_count": 200, "timeout_count": 0, "z_p50": z, "z_p90": z + 1,
        "saturation_ratio_p50": ratio, "baseline_ms": floor, "floor_ms_start": floor_start or floor,
        "floor_ms": floor, "calls_per_min_mean": 3.0, "conditions_opened": opened or {},
        "open_at_hour_end": [], "would_emit_by_condition": would or {}, "excluded": excluded,
        "warm": True, "emit_effective": False, "config_fingerprint": fp,
    }


def _grade(rows):
    r = mod.grade(rows)
    return r, {g.ident.split("|")[2]: g for g in r["keys"]}


def test_weighted_median():
    assert mod.weighted_median([]) is None
    assert mod.weighted_median([(1.0, 1), (5.0, 3)]) == 5.0
    assert mod.weighted_median([(1.0, 1), (3.0, 1)]) == 2.0
    assert mod.weighted_median([(9.0, 0), (1.0, 2)]) == 1.0


def test_calm_hop_passes_and_only_quiet_hours_are_judged():
    # daytime (15:00 UTC) is wild; quiet hours 07-11 UTC are calm -> PASS
    rows = [_row(h, z=0.1, ratio=1.05) for h in range(7, 12)] + [_row(15, z=4.0, ratio=2.5)]
    report, g = _grade(rows)
    assert report["overall"] == "PASS"
    assert g["orion:exec:request:LLMGatewayService"].verdict == "PASS"
    assert "rests where it should" in mod.render(report)


def test_quiet_hour_z_or_ratio_out_of_band_fails():
    _, g = _grade([_row(h, z=0.8, ratio=1.0) for h in range(7, 12)])
    assert g["orion:exec:request:LLMGatewayService"].verdict == "FAIL"
    _, g = _grade([_row(h, z=0.0, ratio=1.4) for h in range(7, 12)])
    assert "slow-vs-best ratio was 1.40" in g["orion:exec:request:LLMGatewayService"].sentence


def test_quiet_hour_boundaries_are_07_to_11_utc():
    # 06:00 and 12:00 UTC are outside 01:00-06:00 MDT
    _, g = _grade([_row(6, z=3.0), _row(12, z=3.0), _row(8, z=0.1)])
    assert g["orion:exec:request:LLMGatewayService"].verdict == "PASS"


def test_no_quiet_hour_traffic_is_not_gradable_not_pass():
    report, g = _grade([_row(15), _row(9, evaluated=0)])
    assert g["orion:exec:request:LLMGatewayService"].verdict == "NOT_GRADABLE"
    assert report["overall"] == "NO_DATA"


def test_floor_creeping_up_without_regime_shift_is_flagged():
    rows = [_row(h, floor=1000.0 + 60 * h) for h in range(0, 12)]  # 1000 -> 1660
    _, g = _grade(rows)
    k = g["orion:exec:request:LLMGatewayService"]
    assert k.verdict == "FAIL" and k.floor_flags and "no regime_shift" in k.floor_flags[0]


def test_floor_rise_after_a_stated_regime_shift_is_fine():
    rows = [_row(h, floor=1000.0) for h in range(0, 5)]
    rows.append(_row(5, floor=2000.0, floor_start=1000.0, opened={"regime_shift": 1}))
    rows += [_row(h, floor=2000.0) for h in range(6, 12)]
    _, g = _grade(rows)
    assert g["orion:exec:request:LLMGatewayService"].floor_flags == []


def test_floor_falling_is_not_flagged_and_config_change_restarts_segment():
    rows = [_row(h, floor=2000.0 - 100 * h) for h in range(0, 8)]
    rows += [_row(h, floor=2500.0, fp="f2") for h in range(8, 12)]
    report, g = _grade(rows)
    assert g["orion:exec:request:LLMGatewayService"].floor_flags == []
    assert "config fingerprints" in mod.render(report)


def test_would_emit_table_sums_per_day():
    rows = [
        _row(1, would={"timeout:open": 2, "timeout:close": 1}),
        _row(2, would={"timeout:open": 1}, key="orion:state:request"),
        _row(3, day=1, would={"spike:open": 1}),
    ]
    table = mod.would_emit_by_day(rows)
    assert table == {"2026-09-30": {"timeout:open": 3, "timeout:close": 1}, "2026-10-01": {"spike:open": 1}}
    assert "2026-09-30: 4 rows" in mod.render(mod.grade(rows))


def test_json_string_columns_from_postgres_are_read():
    r = _row(9, would=None)
    r["would_emit_by_condition"] = json.dumps({"timeout:open": 1})
    assert mod.would_emit_by_day([r]) == {"2026-09-30": {"timeout:open": 1}}


def test_cli_on_jsonl_fixture(tmp_path, capsys):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(_row(h, excluded=True)) for h in range(7, 12)))
    assert mod.main(["--input", str(p)]) == 0
    out = capsys.readouterr().out
    assert "Overall: PASS" in out and "excluded: measured, never triggers" in out
    assert "provisional" in out  # < 7 days


def test_empty_input_says_so(tmp_path, capsys):
    p = tmp_path / "rows.jsonl"
    p.write_text("")
    assert mod.main(["--input", str(p)]) == 3
    assert "UNVERIFIED" in capsys.readouterr().out
