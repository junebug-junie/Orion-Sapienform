"""scripts/backfill_signal_history_from_pe.py (PR #3781 Patch 1): dry run writes nothing, apply
copies only missing rows with the right kind, re-run is a no-op, the 100k refusal holds, and the
CLAUDE.md section 14 files are written."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("backfill_signal", REPO / "scripts/backfill_signal_history_from_pe.py")
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]


class FakeConn:
    """Interprets the script's four statements over two in-memory tables."""

    def __init__(self, pe):
        self.pe = {(n, t): v for n, t, v in pe}
        self.sig: dict = {}
        self.inserts = 0

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "LEFT JOIN" in sql:
            rows = sorted((n, t, v) for (n, t), v in self.pe.items()
                          if n in params["node_ids"] and t >= params["since"] and (n, t) not in self.sig)
            return _Result(rows[: params["limit"]])
        if "AS wrong_kind" in sql:
            n, since = params["node_id"], params["since"]
            pe = [(k, v) for k, v in self.pe.items() if k[0] == n and k[1] >= since]
            sig = [(k, v) for k, v in self.sig.items() if k[0] == n and k[1] >= since]
            mism = sum(1 for k, v in pe if k in self.sig and self.sig[k][0] != v)
            wrong = sum(1 for _, (val, kind) in sig if kind != params["kind"])
            return _Result([(len(pe), len(sig), mism, wrong)])
        if "INSERT INTO substrate_node_signal_history" in sql:
            assert "ON CONFLICT (node_id, observed_at) DO NOTHING" in sql
            for p in params:
                self.inserts += 1
                self.sig.setdefault((p["node_id"], p["observed_at"]), (p["value"], p["kind"]))
            return _Result([])
        raise AssertionError(sql)


def _pe_rows():
    rows = []
    for i in range(30):
        t = NOW - timedelta(hours=6 * i)  # 7.25 days back at i=29
        rows.append(("node:substrate.strain", t, 1.0 if i % 7 == 0 else 0.0))
        rows.append(("node:substrate.perception", t, 0.1 * (i % 4)))
        rows.append(("node:substrate.execution", t, 0.5))  # not a signal node
    return rows


def test_dry_run_writes_nothing_and_snapshots_targets(tmp_path):
    conn = FakeConn(_pe_rows())
    out = io.StringIO()
    s = bf.run(conn, days=7, apply=False, job_dir=tmp_path, now=NOW, out=out)
    assert conn.sig == {} and conn.inserts == 0
    assert s["targets"] == 2 * 29  # i=0..28 inside 7 days, two signal nodes
    snap = next(tmp_path.glob("snapshot-dry_run-*.jsonl")).read_text().splitlines()
    assert len(snap) == 58 and {json.loads(l)["kind"] for l in snap} == {"alarm", "scene_change"}
    for f in ("progress.log", "report.md", "before_after.csv"):
        assert (tmp_path / f).exists()
    assert "DRY RUN" in (tmp_path / "report.md").read_text()


def test_apply_copies_missing_rows_then_rerun_is_noop(tmp_path):
    conn = FakeConn(_pe_rows())
    # One row the live dual-write already wrote: must not be re-inserted.
    first = ("node:substrate.strain", NOW)
    conn.sig[first] = (1.0, "alarm")
    s = bf.run(conn, days=7, apply=True, job_dir=tmp_path, now=NOW, out=io.StringIO())
    assert s["targets"] == 57 and s["errors"] == 0 and s["matched"] is True
    assert "OK:" in s["verdict"]
    assert all(k[0] != "node:substrate.execution" for k in conn.sig)
    assert conn.sig[("node:substrate.perception", NOW - timedelta(hours=6))] == (0.1, "scene_change")
    again = bf.run(conn, days=7, apply=True, job_dir=tmp_path / "2", now=NOW, out=io.StringIO())
    assert again["targets"] == 0 and again["matched"] is True


def test_refuses_above_the_row_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "MAX_ROWS", 10)
    with pytest.raises(SystemExit, match="ask Juniper"):
        bf.run(FakeConn(_pe_rows()), days=7, apply=True, job_dir=tmp_path, now=NOW, out=io.StringIO())


def test_nan_or_future_rows_are_anomalies_not_copied(tmp_path):
    rows = [("node:substrate.cabinet", NOW - timedelta(hours=1), float("nan")),
            ("node:substrate.cabinet", NOW + timedelta(hours=1), 0.0),
            ("node:substrate.cabinet", NOW - timedelta(hours=2), 0.0)]
    conn = FakeConn(rows)
    s = bf.run(conn, days=7, apply=True, job_dir=tmp_path, now=NOW, out=io.StringIO())
    assert s["anomalies"] == 2 and list(conn.sig) == [("node:substrate.cabinet", NOW - timedelta(hours=2))]
