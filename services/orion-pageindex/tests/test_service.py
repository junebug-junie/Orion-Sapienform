from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.models import JournalEntry
from app.pageindex_cli import PageIndexCliError
from app.service import JournalPageIndexService


def _fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "PageIndex"
    repo.mkdir(parents=True)
    script = repo / "run_pageindex.py"
    script.write_text(
        """
import argparse, json
p = argparse.ArgumentParser()
p.add_argument('--md_path')
p.add_argument('--output_dir')
p.add_argument('--query', default=None)
a = p.parse_args()
if a.query:
    print(json.dumps({"results":[{"node_id":"n1","heading":"2026-04-13 — Balancing Acts","excerpt":"identity continuity reflection","entry_id":"entry-1","created_at":"2026-04-13T00:00:00+00:00","source_kind":"journal","provenance":{"engine":"vectify-pageindex"}}]}))
else:
    print(json.dumps({"status":"built","artifact":a.output_dir}))
""".strip()
    )
    return repo


def test_health_proves_pageindex_presence(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    from app import service as service_module
    service_module.settings.PAGEINDEX_REPO_PATH = str(repo)
    service_module.settings.PAGEINDEX_DATA_DIR = str(tmp_path / "data")

    svc = service_module.JournalPageIndexService()
    health = svc.health()
    assert health["pageindex"]["repo_exists"] is True
    assert health["pageindex"]["run_script_exists"] is True


def test_markdown_export_generation(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    from app import service as service_module
    service_module.settings.PAGEINDEX_REPO_PATH = str(repo)
    service_module.settings.PAGEINDEX_DATA_DIR = str(tmp_path / "data")

    svc = service_module.JournalPageIndexService()
    svc._repo.fetch_entries = lambda: [
        JournalEntry(
            entry_id="entry-1",
            created_at=datetime(2026, 4, 13, tzinfo=timezone.utc),
            mode="reflective",
            source_kind="journal",
            source_ref="src-1",
            title="Balancing Acts",
            body="Body text",
        )
    ]
    out = svc.rebuild_journals()
    md = Path(out.markdown_export_path).read_text(encoding="utf-8")
    assert "# Orion Journal Corpus" in md
    assert "## 2026-04-13 — Balancing Acts" in md
    assert "- entry_id: entry-1" in md


def test_rebuild_status_and_query(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    from app import service as service_module
    service_module.settings.PAGEINDEX_REPO_PATH = str(repo)
    service_module.settings.PAGEINDEX_DATA_DIR = str(tmp_path / "data")

    svc = service_module.JournalPageIndexService()
    svc._repo.fetch_entries = lambda: [
        JournalEntry(
            entry_id="entry-1",
            created_at=datetime(2026, 4, 13, tzinfo=timezone.utc),
            mode="reflective",
            source_kind="journal",
            source_ref="src-1",
            title="Balancing Acts",
            body="Body text",
        )
    ]
    build = svc.rebuild_journals()
    assert build.build_success is True

    status = svc.status()
    assert status.build_success is True
    assert status.journal_corpus_row_count == 1

    query = svc.query_journals("identity continuity")
    assert query.query_result_count == 1
    assert query.results[0].entry_id == "entry-1"


def test_unavailable_pageindex_fails(monkeypatch, tmp_path: Path) -> None:
    from app import service as service_module
    service_module.settings.PAGEINDEX_REPO_PATH = str(tmp_path / "missing")
    service_module.settings.PAGEINDEX_DATA_DIR = str(tmp_path / "data")

    svc = service_module.JournalPageIndexService()
    svc._repo.fetch_entries = lambda: [
        JournalEntry(
            entry_id="entry-1",
            created_at=datetime(2026, 4, 13, tzinfo=timezone.utc),
            body="Body text",
        )
    ]
    build = svc.rebuild_journals()
    assert build.build_success is False
    assert "run script missing" in (build.build_error or "").lower()


def test_actual_pageindex_cli_invocation_path(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    from app import service as service_module
    service_module.settings.PAGEINDEX_REPO_PATH = str(repo)
    service_module.settings.PAGEINDEX_DATA_DIR = str(tmp_path / "data")

    svc = service_module.JournalPageIndexService()
    svc._repo.fetch_entries = lambda: []
    svc.rebuild_journals()
    status_data = json.loads((Path(tmp_path / "data") / "journals_status.json").read_text(encoding="utf-8"))
    assert status_data["pageindex_impl"] == "actual"
    assert status_data["pageindex_installation_mode"] == "cli"
