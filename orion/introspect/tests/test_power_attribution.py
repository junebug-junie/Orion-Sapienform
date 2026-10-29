"""power_attribution: SQL rows -> analysis -> introspect items, and the tool's argument contract."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.introspect import power_attribution as pa
from orion.introspect.tests.power_attribution_fake import FakePowerConn
from orion.physiology.tests import heat_fixture as fx
from orion.schemas.heat_archaeology import (
    EvidenceRefV1, FitTermV1, NodeHeatArchaeologyV1, PowerStepV1, RegimeShiftV1, StepAttributionV1, WorkloadFitV1,
)
from orion.schemas.introspect import IntrospectResultV1, PowerAttributionArguments

MCP_TOOL_RESULT_MAX_CHARS = 12000


def _read(node=None):
    conn = FakePowerConn(fx.db_rows())
    out = asyncio.run(pa.read_power_attribution(conn, fx.START, fx.END, node))
    return conn, out


def test_rows_from_the_three_queries_reproduce_the_step():
    conn, out = _read()
    assert [q for q, _ in conn.queries] == ["energy_hourly", "biometrics_hourly", "pool_busy_hours"]
    circe = next(r for r in out if r.node == "circe")
    assert len(circe.steps) == 1 and abs((circe.steps[0].at - fx.STEP_AT).total_seconds()) <= 3 * 3600
    # http:anthropic arrives as harness; the durable-run hold as a reservation, never fitted.
    names = {t.name for t in circe.fit.terms}
    assert "harness" in names and "run:curiosity.investigate" not in names
    assert "run:curiosity.investigate" in circe.reservations


def test_node_filter_skips_the_pool_query_off_the_pool_node():
    conn, out = _read("athena")
    assert [q for q, _ in conn.queries] == ["energy_hourly", "biometrics_hourly"]
    assert [r.node for r in out] == ["athena"]


def test_energy_rows_convert_kwh_over_covered_seconds_to_watts():
    t = datetime(2026, 10, 14, 3, tzinfo=timezone.utc)
    [r] = pa.merge_rows([{"hour_start": t, "source": "circe", "kwh": 0.55, "covered_sec": 1800.0}], [])
    assert r.wall_w == pytest.approx(1100.0)
    assert r.gpu_w is None  # no biometrics row: unknown, not 0


def test_items_are_unsettled_text_plus_numbers_plus_the_sql():
    _, out = _read()
    items = pa.build_items(out, fx.START, fx.END)
    assert [i.kind for i in items] == ["power_attribution", "power_attribution", "power_attribution_queries"]
    circe = items[0]
    assert circe.epistemic_status == "unsettled" and "UNVERIFIED" in circe.text
    assert circe.extra["verified"] is False
    assert circe.extra["steps"][0]["delta_w"] > 100
    queries = items[-1]
    assert set(queries.extra["queries"]) == {"energy_hourly", "biometrics_hourly", "pool_busy_hours"}
    assert "UNVERIFIED" in queries.text
    IntrospectResultV1(ok=True, operation="power_attribution", as_of=fx.END, total_available=len(items), items=items)


def _worst_node(name: str, n_terms: int = 11, summary_chars: int = 1600) -> NodeHeatArchaeologyV1:
    """The most a node can carry: STEP_MAX steps, every lease group plus other/regime/intercept."""
    t = fx.START
    terms = [FitTermV1(name=f"workload-number-{i:02d}", kind="lease", watts_per_unit=-1234.5, se=999.9,
                       mean_level=0.123) for i in range(n_terms)]
    return NodeHeatArchaeologyV1(
        node=name, window_start=t, window_end=t + timedelta(days=30), hours_with_wall=720,
        mean_wall_w=1234.0, daily_wall_w=[(f"2026-10-{d:02d}", 1234.0) for d in range(1, 32)],
        steps=[PowerStepV1(at=t, before_w=1234.0, after_w=1234.0, delta_w=-1234.0, before_hours=100,
                           after_hours=100, t_stat=-12.3, gpu_delta_w=-123.4, cpu_delta_w=-123.4,
                           rest_delta_w=-123.4, load_15m_before=12.34, load_15m_after=12.34)] * 4,
        fit=WorkloadFitV1(target="wall_w", n_hours=720, r2=0.123, residual_sd_w=123.4, terms=terms),
        gpu_fit=WorkloadFitV1(target="gpu_w", n_hours=720, r2=0.123, residual_sd_w=123.4, terms=[]),
        attributions=[StepAttributionV1(step_at=t, delta_w=-1234.0, unexplained_w=-1234.0,
                                        contributions_w={x.name: -1234.0 for x in terms})] * 4,
        regime_shifts=[RegimeShiftV1(series="load_15m", at=t, before=12.34, after=12.34, delta=-12.34)] * 8,
        reservations={f"run:workflow.number.{i}": 0.123 for i in range(5)},
        load_regime_threshold=12.34,
        evidence=[EvidenceRefV1(claim="wall power step at 2026-10-14 03:00 UTC", query="energy_hourly",
                                values={"from": "2026-10-01 00:00 UTC", "to": "2026-10-31 00:00 UTC", "hours": 720})] * 17,
        summary="x" * summary_chars,
    )


def _body(items):
    return json.dumps(IntrospectResultV1(
        ok=True, operation="power_attribution", as_of=fx.END, total_available=len(items), items=items,
    ).model_dump(mode="json"))


def test_a_thirty_day_four_machine_worst_case_fits_the_mcp_budget_without_cutting_summaries():
    # Only circe (the pool node) has lease terms; the others have at most regime + intercept.
    results = [_worst_node("circe")] + [_worst_node(n, n_terms=2) for n in ("athena", "hecate", "cooling")]
    items = pa.build_items(results, fx.START, fx.START + timedelta(days=30))
    assert len(_body(items)) < MCP_TOOL_RESULT_MAX_CHARS
    assert all(i.text == "x" * 1600 and not i.truncated for i in items if i.kind == "power_attribution")
    assert "daily_wall_w" in items[0].extra["omitted"]


def test_even_an_absurd_result_fits_by_clipping_summaries_and_says_so():
    results = [_worst_node(n, summary_chars=4000) for n in ("circe", "athena", "hecate", "cooling")]
    items = pa.build_items(results, fx.START, fx.START + timedelta(days=30))
    assert len(_body(items)) < MCP_TOOL_RESULT_MAX_CHARS
    node_items = [i for i in items if i.kind == "power_attribution"]
    clipped = [i for i in node_items if i.truncated]
    assert clipped and all(i.text.endswith(pa.CLIP_MARKER) for i in clipped)
    assert all(i.text == "x" * 4000 for i in node_items if not i.truncated)
    assert "steps" in node_items[0].extra["omitted"]


def test_small_results_shed_nothing():
    _, out = _read("athena")
    [item, _] = pa.build_items(out, fx.START, fx.END)
    assert "omitted" not in item.extra and item.extra["daily_wall_w"]


@pytest.mark.parametrize("args", [
    {"days": 1}, {"days": 31}, {"node": "zeus"}, {"extra": 1},
    {"until": "2026-10-14T00:00:00"},  # no timezone
])
def test_arguments_refuse_out_of_contract_values(args):
    with pytest.raises(ValueError):
        PowerAttributionArguments.model_validate(args)


def test_arguments_refuse_future_and_pruned_windows():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="future"):
        PowerAttributionArguments(until=now + timedelta(hours=1))
    with pytest.raises(ValueError, match="30 days"):
        PowerAttributionArguments(until=now - timedelta(days=20), days=14)
    ok = PowerAttributionArguments(until=now - timedelta(days=2), days=14)
    start, end = ok.window(now)
    assert end - start == timedelta(days=14)
    assert PowerAttributionArguments().window(now)[1] == now


def test_uneven_summaries_still_fit_the_budget():
    """Review 2026-10-23: one 9000-char summary beside three short ones overflowed (13007 chars)."""
    results = [_worst_node("circe", summary_chars=9000)] + [
        _worst_node(n, n_terms=2, summary_chars=50) for n in ("athena", "hecate", "cooling")]
    items = pa.build_items(results, fx.START, fx.START + timedelta(days=30))
    assert len(_body(items)) <= pa.RESULT_BUDGET_CHARS + 400
    assert len(_body(items)) < MCP_TOOL_RESULT_MAX_CHARS
    assert items[0].truncated and items[0].text.endswith(pa.CLIP_MARKER)
    assert all(i.text == "x" * 50 for i in items[1:4])


def test_reader_pins_utc_hours_inside_its_transaction():
    conn, _ = _read()
    assert conn.executed == ["set local timezone = 'UTC'"]
    assert "insert" not in pa.POOL_BUSY_SQL.lower() and "update" not in pa.POOL_BUSY_SQL.lower()
    assert "opens as" in pa.POOL_BUSY_SQL  # leases still held at the window's end count
