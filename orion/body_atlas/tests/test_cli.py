from __future__ import annotations

import json
from datetime import datetime, timezone

import yaml

from orion.body_atlas.__main__ import MAX_CHARS, MAX_LINES, main, summarize
from orion.body_atlas.build import build_atlas


def _catalog_root(tmp_path, channels):
    (tmp_path / "orion" / "bus").mkdir(parents=True)
    (tmp_path / "orion" / "bus" / "channels.yaml").write_text(yaml.safe_dump({"channels": channels}))
    return tmp_path


def _pairs(n):
    # n separate two-service organs
    return [{"name": f"orion:ns{i}:x", "producer_services": [f"p{i}"], "consumer_services": [f"c{i}"]}
            for i in range(n)]


def test_summary_stays_inside_the_context_caps(tmp_path):
    atlas = build_atlas(_pairs(120), None, now=datetime(2026, 10, 23, tzinfo=timezone.utc))
    text = summarize(atlas, out="/x/atlas.json")
    assert len(text.splitlines()) <= MAX_LINES and len(text) <= MAX_CHARS
    assert "showing" in text and "of 120 multi-part organs" in text
    assert text.endswith("full atlas: /x/atlas.json")


def test_char_cap_keeps_the_full_atlas_path(tmp_path):
    long = [{"name": f"orion:ns{i}:x", "producer_services": [f"p{i}-" + "x" * 300],
             "consumer_services": [f"c{i}-" + "y" * 300]} for i in range(30)]
    text = summarize(build_atlas(long, None, now=datetime(2026, 10, 23, tzinfo=timezone.utc)),
                     out="/x/atlas.json")
    assert len(text) <= MAX_CHARS and "(cut at 4000 chars)" in text
    assert text.endswith("full atlas: /x/atlas.json")


def test_catalog_only_run_writes_the_full_atlas(tmp_path, capsys):
    root = _catalog_root(tmp_path, _pairs(3))
    out = tmp_path / "out" / "atlas.json"
    assert main(["--root", str(root), "--no-live", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["schema_version"] == "body_atlas.v1" and data["sources"]["live"] is False
    assert "catalog only" in capsys.readouterr().out


def test_live_run_without_a_uri_refuses_instead_of_falling_back(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    root = _catalog_root(tmp_path, _pairs(1))
    assert main(["--root", str(root)]) == 2
    assert "--no-live" in capsys.readouterr().err


def test_missing_catalog_is_an_error(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "--no-live"]) == 2
    assert "no channel catalog" in capsys.readouterr().err
