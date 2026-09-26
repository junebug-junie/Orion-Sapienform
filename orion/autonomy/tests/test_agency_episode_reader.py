import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from orion.autonomy.agency_episode_reader import collect


class Cursor:
    def __init__(self, conn):
        self.conn = conn
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=None):
        self.conn.queries.append((query, params))
        assert query.startswith(("SELECT", "SHOW"))
        if "FROM substrate_dispatch_results" in query:
            if "WHERE dispatch_id" in query:
                self.rows = [{"result_id": "old-r", "dispatch_id": "old", "frame_id": "old-f"}, {"result_id": "new-r", "dispatch_id": "new", "frame_id": "new-f"}]
            else:
                self.rows = [{"result_id": "new-r", "dispatch_id": "new", "frame_id": "new-f"}]
        elif query.startswith("SELECT dispatch_id FROM substrate_action_outcomes"):
            self.rows = [{"dispatch_id": "old"}]
        elif "FROM curiosity_peer_brief" in query and self.conn.fail:
            raise RuntimeError("SECRET exception content")
        else:
            self.rows = []

    def fetchone(self):
        return (self.conn.readonly,)

    def fetchall(self):
        return self.rows


class Connection:
    autocommit = True
    readonly = "on"
    fail = False

    def __init__(self):
        self.queries = []

    def cursor(self, **kwargs):
        return Cursor(self)


class Graph:
    def query(self, query):
        assert query.startswith("MATCH")
        return []


def test_sampling_fetches_matching_old_results_not_only_latest():
    conn = Connection()
    data = collect(conn, Graph(), limit=1)
    assert {r["dispatch_id"] for r in data["results"]["rows"]} == {"old", "new"}
    query, params = next((q, p) for q, p in conn.queries if "FROM substrate_dispatch_results WHERE" in q)
    assert set(params[0]) == {"old", "new"}
    assert "%s" in query


def test_failed_source_is_distinct_from_empty_and_error_is_redacted():
    conn = Connection()
    conn.fail = True
    data = collect(conn, Graph(), limit=1)
    assert data["sql_briefs"]["status"] == "unavailable"
    assert data["graph_briefs"]["status"] == "ok"
    assert "SECRET" not in json.dumps(data)


def test_writable_connection_is_refused_before_data_reads():
    conn = Connection()
    conn.readonly = "off"
    with pytest.raises(ValueError, match="read-only"):
        collect(conn, Graph())
    assert len(conn.queries) == 1


@pytest.mark.parametrize("limit", [0, 51, -1])
def test_limit_is_bounded(limit):
    conn = Connection()
    with pytest.raises(ValueError):
        collect(conn, Graph(), limit=limit)
    assert not conn.queries


def test_relationship_truncation_cannot_be_mistaken_for_missing():
    class ManyBriefs(Graph):
        def query(self, query):
            if "MATCH (b:PeerBrief)" in query:
                return [{"brief_id": str(i)} for i in range(11)]
            return [{"help_id": "h1"}]

    data = collect(Connection(), ManyBriefs(), limit=1)
    assert data["graph_briefs"] == {"status": "truncated", "rows": []}


def test_cli_offline_snapshot_is_owner_only_and_never_overwrites(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"captured_at": "2026-09-26T00:00:00Z"}))
    target = tmp_path / "snapshot.json"
    cmd = [sys.executable, str(root / "scripts/analysis/report_agency_episodes.py"), "--input", str(source), "--snapshot", str(target)]
    env = {**os.environ, "ORION_PG_DSN": "invalid://must-not-connect"}
    first = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert first.returncode == 2  # explicit missing sources
    assert json.loads(first.stdout)["sources"]["asks"]["status"] == "unavailable"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    saved = target.read_bytes()
    second = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert second.returncode == 2
    assert "FileExistsError" in second.stderr
    assert target.read_bytes() == saved
