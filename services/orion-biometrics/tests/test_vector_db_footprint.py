"""Vector DB footprint gauge (app/vector_db_footprint.py): size, counts, alert, wiring.

Counts are fetched from a real local HTTP server speaking the Chroma 0.4.x v1 routes the
poller uses, so URL building and JSON parsing are exercised, not mocked away.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.vector_db_footprint import VectorDbFootprintPoller, dir_size_bytes, sample_footprint

_COLLECTIONS = {"orion_main_store": ("id-main", 9119), "orion_brand_new": ("id-new", 3)}


@pytest.fixture()
def fake_chroma():
    calls: list[str] = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            calls.append(self.path)
            if self.path == "/api/v1/collections":
                body = [{"name": n, "id": i} for n, (i, _) in _COLLECTIONS.items()]
            else:
                by_id = {i: c for _, (i, c) in _COLLECTIONS.items()}
                cid = self.path.split("/")[4]
                body = by_id[cid]
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", calls
    srv.shutdown()


def _chroma_dir(tmp_path: Path) -> Path:
    d = tmp_path / "chroma"
    (d / "seg").mkdir(parents=True)
    (d / "chroma.sqlite3").write_bytes(b"x" * 1000)
    (d / "seg" / "data_level0.bin").write_bytes(b"y" * 500)
    return d


def test_dir_size_sums_nested_files(tmp_path):
    assert dir_size_bytes(str(_chroma_dir(tmp_path))) == 1500


def test_sample_reports_size_counts_and_unpoliced(tmp_path, fake_chroma):
    url, _ = fake_chroma
    snap = sample_footprint(str(_chroma_dir(tmp_path)), url, alert_bytes=10_000, timeout=2.0)
    d = snap.as_details()
    assert d["dir_bytes"] == 1500 and d["sqlite_bytes"] == 1000
    assert d["over_alert"] is False and d["alert_bytes"] == 10_000
    assert d["collection_counts"] == {"orion_main_store": 9119, "orion_brand_new": 3}
    assert d["unpoliced_collections"] == ["orion_brand_new"]
    assert "error" not in d


def test_missing_dir_makes_no_http_calls(tmp_path, fake_chroma):
    url, calls = fake_chroma
    snap = sample_footprint(str(tmp_path / "nope"), url, alert_bytes=1, timeout=2.0)
    assert snap.error == "not_present" and snap.dir_bytes is None
    assert calls == []


def test_api_down_still_reports_size(tmp_path):
    snap = sample_footprint(str(_chroma_dir(tmp_path)), "http://127.0.0.1:9", alert_bytes=1000, timeout=0.5)
    assert snap.dir_bytes == 1500 and snap.over_alert is True
    assert snap.error.startswith("collection_counts_failed")
    assert snap.collection_counts == {}


def test_alert_logs_once_per_crossing(tmp_path, caplog):
    d = _chroma_dir(tmp_path)
    poller = VectorDbFootprintPoller(enabled=True, data_path=str(d), base_url="", alert_bytes=1200)
    with caplog.at_level(logging.INFO, logger="orion-biometrics.vector_db"):
        poller.poll_once()
        poller.poll_once()
        (d / "chroma.sqlite3").write_bytes(b"x")  # shrink back under
        poller.poll_once()
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("vector_db_over_alert" in m for m in msgs) == 1
    assert sum("vector_db_back_under_alert" in m for m in msgs) == 1
    assert poller.details()["vector_db"]["over_alert"] is False


def test_disabled_poller_adds_nothing_to_heartbeat(tmp_path):
    poller = VectorDbFootprintPoller(enabled=False, data_path=str(tmp_path), base_url="", alert_bytes=1)
    assert poller.details() == {}


def test_heartbeat_details_carry_vector_db(monkeypatch, tmp_path):
    import app.main as main

    poller = VectorDbFootprintPoller(enabled=True, data_path=str(_chroma_dir(tmp_path)), base_url="", alert_bytes=1)
    poller.poll_once()
    monkeypatch.setattr(main, "_vector_db_poller", poller)
    details = main._heartbeat_details()
    assert details["vector_db"]["dir_bytes"] == 1500


def test_settings_ship_enabled_with_athena_defaults():
    from app.settings import settings

    assert settings.VECTOR_DB_FOOTPRINT_ENABLED is True
    assert settings.VECTOR_DB_DATA_PATH == "/host_mnt/graphdb/chroma"
    assert settings.VECTOR_DB_ALERT_BYTES > 0
