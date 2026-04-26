from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .journal_repository import JournalRepository
from .models import (
    BuildResponse,
    ChatEpisodeBuildResponse,
    ChatEpisodeStatusResponse,
    JournalEntry,
    QueryResponse,
    QueryResult,
    StatusResponse,
)
from .pageindex_cli import PageIndexCli, PageIndexCliError
from .settings import settings


@dataclass
class BuildState:
    pageindex_impl: str = settings.PAGEINDEX_IMPL
    pageindex_installation_mode: str = settings.PAGEINDEX_INSTALLATION_MODE
    journal_corpus_row_count: int = 0
    markdown_export_path: str | None = None
    pageindex_tree_artifact_path: str | None = None
    last_build_started_at: str | None = None
    last_build_completed_at: str | None = None
    build_success: bool = False
    build_error: str | None = None


@dataclass
class ChatEpisodeBuildState:
    pageindex_impl: str = settings.PAGEINDEX_IMPL
    pageindex_installation_mode: str = settings.PAGEINDEX_INSTALLATION_MODE
    corpus_key: str = "chat_episodes"
    markdown_export_path: str | None = None
    pageindex_tree_artifact_path: str | None = None
    last_build_started_at: str | None = None
    last_build_completed_at: str | None = None
    build_success: bool = False
    build_error: str | None = None


class JournalPageIndexService:
    def __init__(self) -> None:
        self._repo = JournalRepository(settings.JOURNAL_PG_DSN, settings.JOURNAL_INDEX_TABLE)
        self._cli = PageIndexCli(
            python_bin=settings.PAGEINDEX_PYTHON_BIN,
            repo_path=settings.PAGEINDEX_REPO_PATH,
            run_script=settings.PAGEINDEX_RUN_SCRIPT,
            build_args=settings.PAGEINDEX_BUILD_ARGS,
            query_args=settings.PAGEINDEX_QUERY_ARGS,
            timeout_sec=settings.PAGEINDEX_TIMEOUT_SEC,
        )
        self._base_dir = Path(settings.PAGEINDEX_DATA_DIR)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._status_path = self._base_dir / "journals_status.json"
        self._chat_episodes_status_path = self._base_dir / "chat_episodes_status.json"

    def health(self) -> dict[str, Any]:
        proof = self._cli.installation_proof()
        return {
            "ok": True,
            "service": settings.SERVICE_NAME,
            "pageindex_impl": settings.PAGEINDEX_IMPL,
            "pageindex_installation_mode": settings.PAGEINDEX_INSTALLATION_MODE,
            "pageindex": proof,
        }

    def rebuild_journals(self) -> BuildResponse:
        started = datetime.now(timezone.utc)
        entries = self._repo.fetch_entries()
        markdown_path = self._base_dir / "journals" / "journal_corpus.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir = self._base_dir / "journals" / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(self._render_markdown(entries), encoding="utf-8")

        state = BuildState(
            journal_corpus_row_count=len(entries),
            markdown_export_path=str(markdown_path),
            pageindex_tree_artifact_path=str(artifact_dir),
            last_build_started_at=started.isoformat(),
        )

        try:
            self._cli.build(md_path=markdown_path, artifact_dir=artifact_dir)
            state.build_success = True
        except Exception as exc:
            state.build_success = False
            state.build_error = str(exc)
        finally:
            state.last_build_completed_at = datetime.now(timezone.utc).isoformat()
            self._write_status(state)

        return BuildResponse(
            pageindex_impl=state.pageindex_impl,
            pageindex_installation_mode=state.pageindex_installation_mode,
            journal_corpus_row_count=state.journal_corpus_row_count,
            markdown_export_path=state.markdown_export_path or "",
            pageindex_tree_artifact_path=state.pageindex_tree_artifact_path or "",
            last_build_started_at=datetime.fromisoformat(state.last_build_started_at),
            last_build_completed_at=datetime.fromisoformat(state.last_build_completed_at),
            build_success=state.build_success,
            build_error=state.build_error,
        )

    def status(self) -> StatusResponse:
        if not self._status_path.exists():
            return StatusResponse(
                pageindex_impl=settings.PAGEINDEX_IMPL,
                pageindex_installation_mode=settings.PAGEINDEX_INSTALLATION_MODE,
                corpus_exists=False,
                journal_corpus_row_count=0,
                markdown_export_path=None,
                pageindex_tree_artifact_path=None,
                last_build_started_at=None,
                last_build_completed_at=None,
                build_success=False,
                build_error=None,
            )
        data = json.loads(self._status_path.read_text(encoding="utf-8"))
        return StatusResponse(
            pageindex_impl=data["pageindex_impl"],
            pageindex_installation_mode=data["pageindex_installation_mode"],
            corpus_exists=bool(data.get("markdown_export_path")),
            journal_corpus_row_count=int(data.get("journal_corpus_row_count", 0)),
            markdown_export_path=data.get("markdown_export_path"),
            pageindex_tree_artifact_path=data.get("pageindex_tree_artifact_path"),
            last_build_started_at=datetime.fromisoformat(data["last_build_started_at"])
            if data.get("last_build_started_at")
            else None,
            last_build_completed_at=datetime.fromisoformat(data["last_build_completed_at"])
            if data.get("last_build_completed_at")
            else None,
            build_success=bool(data.get("build_success", False)),
            build_error=data.get("build_error"),
        )

    def query_journals(self, query: str, allow_fallback: bool = False, top_k: int = 8) -> QueryResponse:
        state = self.status()
        if not state.build_success or not state.markdown_export_path or not state.pageindex_tree_artifact_path:
            raise PageIndexCliError("journal corpus is not built")
        fallback_invoked = False
        try:
            raw = self._cli.query(
                md_path=Path(state.markdown_export_path),
                artifact_dir=Path(state.pageindex_tree_artifact_path),
                query=query,
            )
            parsed = raw.get("json") or {}
            results: list[QueryResult] = []
            for item in (parsed.get("results") or [])[:top_k]:
                results.append(
                    QueryResult(
                        node_id=item.get("node_id"),
                        heading=item.get("heading"),
                        excerpt=item.get("excerpt") or "",
                        entry_id=item.get("entry_id"),
                        created_at=item.get("created_at"),
                        source_kind=item.get("source_kind"),
                        provenance=item.get("provenance") or {},
                    )
                )
            return QueryResponse(
                pageindex_impl=settings.PAGEINDEX_IMPL,
                query_invoked=True,
                query_result_count=len(results),
                fallback_invoked=False,
                results=results,
                metadata={"pageindex_raw": parsed},
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            fallback_invoked = True
            return QueryResponse(
                pageindex_impl=settings.PAGEINDEX_IMPL,
                query_invoked=True,
                query_result_count=0,
                fallback_invoked=fallback_invoked,
                results=[],
                metadata={"error": str(exc)},
            )

    def rebuild_chat_episodes(self) -> ChatEpisodeBuildResponse:
        started = datetime.now(timezone.utc)
        markdown_path = Path(settings.CHAT_EPISODES_MARKDOWN_PATH)
        artifact_dir = self._base_dir / "chat_episodes" / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        state = ChatEpisodeBuildState(
            markdown_export_path=str(markdown_path),
            pageindex_tree_artifact_path=str(artifact_dir),
            last_build_started_at=started.isoformat(),
        )
        try:
            if not markdown_path.exists():
                raise FileNotFoundError(f"chat episodes markdown not found: {markdown_path}")
            self._cli.build(md_path=markdown_path, artifact_dir=artifact_dir)
            state.build_success = True
        except Exception as exc:
            state.build_success = False
            state.build_error = str(exc)
        finally:
            state.last_build_completed_at = datetime.now(timezone.utc).isoformat()
            self._write_chat_episodes_status(state)
        return ChatEpisodeBuildResponse(
            pageindex_impl=state.pageindex_impl,
            pageindex_installation_mode=state.pageindex_installation_mode,
            corpus_key=state.corpus_key,
            markdown_export_path=state.markdown_export_path or "",
            pageindex_tree_artifact_path=state.pageindex_tree_artifact_path or "",
            last_build_started_at=datetime.fromisoformat(state.last_build_started_at),
            last_build_completed_at=datetime.fromisoformat(state.last_build_completed_at),
            build_success=state.build_success,
            build_error=state.build_error,
        )

    def chat_episodes_status(self) -> ChatEpisodeStatusResponse:
        if not self._chat_episodes_status_path.exists():
            return ChatEpisodeStatusResponse(
                pageindex_impl=settings.PAGEINDEX_IMPL,
                pageindex_installation_mode=settings.PAGEINDEX_INSTALLATION_MODE,
                corpus_key="chat_episodes",
                corpus_exists=False,
                markdown_export_path=None,
                pageindex_tree_artifact_path=None,
                last_build_started_at=None,
                last_build_completed_at=None,
                build_success=False,
                build_error=None,
            )
        data = json.loads(self._chat_episodes_status_path.read_text(encoding="utf-8"))
        return ChatEpisodeStatusResponse(
            pageindex_impl=data["pageindex_impl"],
            pageindex_installation_mode=data["pageindex_installation_mode"],
            corpus_key=data.get("corpus_key", "chat_episodes"),
            corpus_exists=bool(data.get("markdown_export_path")),
            markdown_export_path=data.get("markdown_export_path"),
            pageindex_tree_artifact_path=data.get("pageindex_tree_artifact_path"),
            last_build_started_at=datetime.fromisoformat(data["last_build_started_at"])
            if data.get("last_build_started_at")
            else None,
            last_build_completed_at=datetime.fromisoformat(data["last_build_completed_at"])
            if data.get("last_build_completed_at")
            else None,
            build_success=bool(data.get("build_success", False)),
            build_error=data.get("build_error"),
        )

    def query_chat_episodes(self, query: str, allow_fallback: bool = False, top_k: int = 8) -> QueryResponse:
        state = self.chat_episodes_status()
        if not state.build_success or not state.markdown_export_path or not state.pageindex_tree_artifact_path:
            raise PageIndexCliError("chat_episodes corpus is not built")
        try:
            raw = self._cli.query(
                md_path=Path(state.markdown_export_path),
                artifact_dir=Path(state.pageindex_tree_artifact_path),
                query=query,
            )
            parsed = raw.get("json") or {}
            results = [
                QueryResult(
                    node_id=item.get("node_id"),
                    heading=item.get("heading"),
                    excerpt=item.get("excerpt") or "",
                    entry_id=item.get("entry_id") or item.get("episode_id"),
                    created_at=item.get("created_at"),
                    source_kind=item.get("source_kind") or "chat_episode",
                    provenance=item.get("provenance") or {},
                )
                for item in (parsed.get("results") or [])[:top_k]
            ]
            return QueryResponse(
                pageindex_impl=settings.PAGEINDEX_IMPL,
                query_invoked=True,
                query_result_count=len(results),
                fallback_invoked=False,
                results=results,
                metadata={"pageindex_raw": parsed, "corpus_key": "chat_episodes"},
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            return QueryResponse(
                pageindex_impl=settings.PAGEINDEX_IMPL,
                query_invoked=True,
                query_result_count=0,
                fallback_invoked=True,
                results=[],
                metadata={"error": str(exc), "corpus_key": "chat_episodes"},
            )

    def _write_status(self, state: BuildState) -> None:
        self._status_path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")

    def _write_chat_episodes_status(self, state: ChatEpisodeBuildState) -> None:
        self._chat_episodes_status_path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")

    @staticmethod
    def _fmt_list(items: list[str] | None) -> str:
        if not isinstance(items, list):
            return ""
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return ", ".join(cleaned)

    @staticmethod
    def _render_markdown(entries: list[JournalEntry]) -> str:
        lines = ["# Orion Journal Corpus", ""]
        for row in entries:
            date_part = row.created_at.date().isoformat()
            heading_title = (row.title or "Untitled").strip()
            lines.extend(
                [
                    f"## {date_part} — {heading_title}",
                    f"- entry_id: {row.entry_id}",
                    f"- created_at: {row.created_at.isoformat()}",
                    f"- mode: {row.mode or ''}",
                    f"- source_kind: {row.source_kind or ''}",
                    f"- source_ref: {row.source_ref or ''}",
                    f"- trigger_kind: {row.trigger_kind or ''}",
                    f"- trigger_summary: {row.trigger_summary or ''}",
                    f"- conversation_frame: {row.conversation_frame or ''}",
                    f"- task_mode: {row.task_mode or ''}",
                    f"- identity_salience: {row.identity_salience or ''}",
                    f"- answer_strategy: {row.answer_strategy or ''}",
                    f"- stance_summary: {row.stance_summary or ''}",
                    f"- active_identity_facets: {JournalPageIndexService._fmt_list(row.active_identity_facets)}",
                    f"- active_growth_axes: {JournalPageIndexService._fmt_list(row.active_growth_axes)}",
                    f"- active_relationship_facets: {JournalPageIndexService._fmt_list(row.active_relationship_facets)}",
                    f"- social_posture: {JournalPageIndexService._fmt_list(row.social_posture)}",
                    f"- reflective_themes: {JournalPageIndexService._fmt_list(row.reflective_themes)}",
                    f"- active_tensions: {JournalPageIndexService._fmt_list(row.active_tensions)}",
                    f"- dream_motifs: {JournalPageIndexService._fmt_list(row.dream_motifs)}",
                    f"- response_hazards: {JournalPageIndexService._fmt_list(row.response_hazards)}",
                    "",
                    row.body.strip(),
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"
