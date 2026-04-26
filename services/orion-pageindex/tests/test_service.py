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
            trigger_kind="manual",
            trigger_summary="manual summary",
            conversation_frame="reflective",
            task_mode="reflective_dialogue",
            identity_salience="high",
            stance_summary="brief reflective frame",
            active_identity_facets=["identity continuity"],
            active_growth_axes=["stability"],
            active_relationship_facets=["trust"],
            social_posture=["warm"],
            reflective_themes=["integration"],
            active_tensions=["speed_vs_depth"],
            dream_motifs=["bridge"],
            response_hazards=["overgeneralization"],
            title="Balancing Acts",
            body="Body text",
        )
    ]
    out = svc.rebuild_journals()
    md = Path(out.markdown_export_path).read_text(encoding="utf-8")
    assert "# Orion Journal Corpus" in md
    assert "## 2026-04-13 — Balancing Acts" in md
    assert "- entry_id: entry-1" in md
    assert "- trigger_kind: manual" in md
    assert "- trigger_summary: manual summary" in md
    assert "- conversation_frame: reflective" in md
    assert "- task_mode: reflective_dialogue" in md
    assert "- identity_salience: high" in md
    assert "- stance_summary: brief reflective frame" in md
    assert "- active_identity_facets: identity continuity" in md
    assert "- active_growth_axes: stability" in md
    assert "- active_relationship_facets: trust" in md
    assert "- social_posture: warm" in md
    assert "- reflective_themes: integration" in md
    assert "- active_tensions: speed_vs_depth" in md
    assert "- dream_motifs: bridge" in md
    assert "- response_hazards: overgeneralization" in md


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


def test_markdown_export_is_null_safe_for_new_metadata(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    from app import service as service_module

    service_module.settings.PAGEINDEX_REPO_PATH = str(repo)
    service_module.settings.PAGEINDEX_DATA_DIR = str(tmp_path / "data")

    svc = service_module.JournalPageIndexService()
    svc._repo.fetch_entries = lambda: [
        JournalEntry(
            entry_id="entry-2",
            created_at=datetime(2026, 4, 14, tzinfo=timezone.utc),
            mode="reflective",
            source_kind="journal",
            source_ref="src-2",
            title="Null-safe Entry",
            body="Body text",
        )
    ]
    out = svc.rebuild_journals()
    md = Path(out.markdown_export_path).read_text(encoding="utf-8")
    assert "- trigger_kind: " in md
    assert "- trigger_summary: " in md
    assert "- conversation_frame: " in md
    assert "- active_identity_facets: " in md
    assert "- response_hazards: " in md


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


def test_chat_episodes_rebuild_and_query(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    from app import service as service_module

    service_module.settings.PAGEINDEX_REPO_PATH = str(repo)
    service_module.settings.PAGEINDEX_DATA_DIR = str(tmp_path / "data")
    chat_md = Path(tmp_path / "data" / "chat_episodes" / "chat_episode_corpus.md")
    chat_md.parent.mkdir(parents=True, exist_ok=True)
    chat_md.write_text("# Orion Chat Corpus\n\n## 2026-04-25\n\n### Episode: sample\n", encoding="utf-8")
    service_module.settings.CHAT_EPISODES_MARKDOWN_PATH = str(chat_md)

    svc = service_module.JournalPageIndexService()
    build = svc.rebuild_chat_episodes()
    assert build.build_success is True
    assert build.corpus_key == "chat_episodes"

    status = svc.chat_episodes_status()
    assert status.build_success is True
    assert status.corpus_exists is True

    query = svc.query_chat_episodes("sample")
    assert query.query_result_count == 1


def test_chat_episodes_rebuild_fails_without_markdown(monkeypatch, tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    from app import service as service_module

    service_module.settings.PAGEINDEX_REPO_PATH = str(repo)
    service_module.settings.PAGEINDEX_DATA_DIR = str(tmp_path / "data")
    service_module.settings.CHAT_EPISODES_MARKDOWN_PATH = str(tmp_path / "data" / "chat_episodes" / "missing.md")

    svc = service_module.JournalPageIndexService()
    build = svc.rebuild_chat_episodes()
    assert build.build_success is False
    assert "not found" in (build.build_error or "").lower()
