from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

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
    db_url_present: bool = False
    db_env_key: str | None = None


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
        db_dsn, db_env_key = self._resolve_db_dsn()
        db_present = bool(db_dsn)
        ok = bool(proof.get("repo_exists")) and bool(proof.get("run_script_exists")) and db_present
        return {
            "ok": ok,
            "service": settings.SERVICE_NAME,
            "pageindex_impl": settings.PAGEINDEX_IMPL,
            "pageindex_installation_mode": settings.PAGEINDEX_INSTALLATION_MODE,
            "db_url_present": db_present,
            "db_env_key": db_env_key,
            "resolved_paths": {
                "pageindex_repo_path": str(settings.PAGEINDEX_REPO_PATH),
                "pageindex_run_script": str(settings.PAGEINDEX_RUN_SCRIPT),
            },
            "pageindex": proof,
        }

    def rebuild_journals(self) -> BuildResponse:
        started = datetime.now(timezone.utc)
        db_dsn, db_env_key = self._resolve_db_dsn()
        state = BuildState(
            last_build_started_at=started.isoformat(),
            db_url_present=bool(db_dsn),
            db_env_key=db_env_key,
        )
        proof = self._cli.installation_proof()
        state.journal_corpus_row_count = self._safe_row_count(db_dsn)
        if not db_dsn:
            state.build_error = "database URL missing: set PAGEINDEX_SQL_DATABASE_URL or ENDOGENOUS_RUNTIME_SQL_DATABASE_URL or SQL_DATABASE_URL"
            state.last_build_completed_at = datetime.now(timezone.utc).isoformat()
            self._write_status(state)
            return self._build_response_from_state(state)
        if not proof.get("run_script_exists"):
            state.build_error = f"PageIndex run script missing: {proof.get('run_script')}"
            state.last_build_completed_at = datetime.now(timezone.utc).isoformat()
            self._write_status(state)
            return self._build_response_from_state(state)
        repo = self._repo_for_dsn(db_dsn)
        entries = repo.fetch_entries()
        markdown_path = self._base_dir / "journals" / "journal_corpus.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir = self._base_dir / "journals" / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        state.journal_corpus_row_count = len(entries)
        state.markdown_export_path = str(markdown_path)
        state.pageindex_tree_artifact_path = str(artifact_dir)
        if not entries and not settings.PAGEINDEX_ALLOW_EMPTY_REBUILD:
            state.build_error = "journal_entry_index has 0 rows; refusing rebuild (set PAGEINDEX_ALLOW_EMPTY_REBUILD=true to override)"
            state.last_build_completed_at = datetime.now(timezone.utc).isoformat()
            self._write_status(state)
            return self._build_response_from_state(state)
        markdown_path.write_text(self._render_markdown(entries), encoding="utf-8")

        try:
            self._cli.build(md_path=markdown_path, artifact_dir=artifact_dir)
            state.build_success = True
        except Exception as exc:
            state.build_success = False
            state.build_error = str(exc)
        finally:
            state.last_build_completed_at = datetime.now(timezone.utc).isoformat()
            self._write_status(state)
        return self._build_response_from_state(state)

    def status(self) -> StatusResponse:
        db_dsn, db_env_key = self._resolve_db_dsn()
        db_present = bool(db_dsn)
        db_error = None
        row_count = 0
        if db_present:
            try:
                row_count = int(self._repo_for_dsn(db_dsn).stats().get("count", 0))
            except Exception as exc:
                db_error = f"database status query failed: {exc}"
        else:
            db_error = "database URL missing: set PAGEINDEX_SQL_DATABASE_URL or ENDOGENOUS_RUNTIME_SQL_DATABASE_URL or SQL_DATABASE_URL"

        if not self._status_path.exists():
            return StatusResponse(
                pageindex_impl=settings.PAGEINDEX_IMPL,
                pageindex_installation_mode=settings.PAGEINDEX_INSTALLATION_MODE,
                corpus_exists=False,
                journal_corpus_row_count=row_count,
                markdown_export_path=None,
                pageindex_tree_artifact_path=None,
                last_build_started_at=None,
                last_build_completed_at=None,
                build_success=False,
                build_error=db_error,
                db_url_present=db_present,
                db_env_key=db_env_key,
            )
        data = json.loads(self._status_path.read_text(encoding="utf-8"))
        build_error = data.get("build_error")
        if db_error:
            build_error = "; ".join(item for item in [build_error, db_error] if item)
        return StatusResponse(
            pageindex_impl=data["pageindex_impl"],
            pageindex_installation_mode=data["pageindex_installation_mode"],
            corpus_exists=bool(data.get("markdown_export_path")),
            journal_corpus_row_count=row_count,
            markdown_export_path=data.get("markdown_export_path"),
            pageindex_tree_artifact_path=data.get("pageindex_tree_artifact_path"),
            last_build_started_at=datetime.fromisoformat(data["last_build_started_at"])
            if data.get("last_build_started_at")
            else None,
            last_build_completed_at=datetime.fromisoformat(data["last_build_completed_at"])
            if data.get("last_build_completed_at")
            else None,
            build_success=bool(data.get("build_success", False)) and not db_error,
            build_error=build_error,
            db_url_present=db_present,
            db_env_key=db_env_key,
        )

    def query_journals(self, query: str, allow_fallback: bool = False, top_k: int = 8) -> QueryResponse:
        state = self.status()
        if not state.build_success or not state.markdown_export_path or not state.pageindex_tree_artifact_path:
            raise PageIndexCliError("journal corpus is not built")
        db_dsn, _ = self._resolve_db_dsn()
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
            unsupported_query = "not supported by current runner configuration" in str(exc).lower()
            if unsupported_query:
                try:
                    local = self._cli.query_local_tree(
                        md_path=Path(state.markdown_export_path),
                        query=query,
                        top_k=top_k,
                    )
                    local_results = [
                        QueryResult(
                            node_id=item.get("node_id"),
                            heading=item.get("heading"),
                            excerpt=item.get("excerpt") or "",
                            entry_id=item.get("entry_id"),
                            created_at=item.get("created_at"),
                            source_kind=item.get("source_kind"),
                            provenance=item.get("provenance") or {},
                        )
                        for item in (local.get("results") or [])[:top_k]
                    ]
                    return QueryResponse(
                        pageindex_impl=settings.PAGEINDEX_IMPL,
                        query_invoked=True,
                        query_result_count=len(local_results),
                        fallback_invoked=False,
                        results=local_results,
                        metadata={
                            "engine": "pageindex_local_tree",
                            "artifact_path": local.get("artifact_path"),
                            "upstream_query_unavailable": True,
                        },
                    )
                except Exception:
                    try:
                        md_results = self._query_markdown_local(Path(state.markdown_export_path), query=query, top_k=top_k)
                        return QueryResponse(
                            pageindex_impl=settings.PAGEINDEX_IMPL,
                            query_invoked=True,
                            query_result_count=len(md_results),
                            fallback_invoked=False,
                            results=md_results,
                            metadata={
                                "engine": "markdown_local_search",
                                "upstream_query_unavailable": True,
                            },
                        )
                    except Exception:
                        pass

            if not allow_fallback and not unsupported_query:
                raise
            fallback_invoked = True
            fallback_results: list[QueryResult] = []
            if db_dsn:
                try:
                    entries = self._repo_for_dsn(db_dsn).search_entries(query, limit=top_k)
                    fallback_results = [
                        QueryResult(
                            node_id=None,
                            heading=entry.title or "",
                            excerpt=(entry.body or "")[:500],
                            entry_id=entry.entry_id,
                            created_at=entry.created_at.isoformat(),
                            source_kind=entry.source_kind or "journal",
                            provenance={
                                "engine": "journal_db_fallback",
                                "table": settings.JOURNAL_INDEX_TABLE,
                            },
                        )
                        for entry in entries
                    ]
                except Exception as fallback_exc:
                    return QueryResponse(
                        pageindex_impl=settings.PAGEINDEX_IMPL,
                        query_invoked=True,
                        query_result_count=0,
                        fallback_invoked=fallback_invoked,
                        results=[],
                        metadata={"error": str(exc), "fallback_error": str(fallback_exc)},
                    )
            return QueryResponse(
                pageindex_impl=settings.PAGEINDEX_IMPL,
                query_invoked=True,
                query_result_count=len(fallback_results),
                fallback_invoked=fallback_invoked,
                results=fallback_results,
                metadata={"error": str(exc), "fallback_engine": "journal_db_fallback"},
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

    @staticmethod
    def _resolve_db_dsn() -> tuple[str | None, str | None]:
        for key in ("PAGEINDEX_SQL_DATABASE_URL", "ENDOGENOUS_RUNTIME_SQL_DATABASE_URL", "SQL_DATABASE_URL"):
            value = (os.environ.get(key) or "").strip()
            if value:
                return value, key
        return None, None

    @staticmethod
    def _repo_for_dsn(dsn: str) -> JournalRepository:
        return JournalRepository(dsn, settings.JOURNAL_INDEX_TABLE)

    @staticmethod
    def _build_response_from_state(state: BuildState) -> BuildResponse:
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

    @staticmethod
    def _safe_row_count(dsn: str | None) -> int:
        if not dsn:
            return 0
        try:
            return int(JournalPageIndexService._repo_for_dsn(dsn).stats().get("count", 0))
        except Exception:
            return 0

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

    @staticmethod
    def _query_markdown_local(md_path: Path, query: str, top_k: int) -> list[QueryResult]:
        if not md_path.exists():
            return []
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        terms = [token for token in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(token) > 2]
        if not terms:
            return []
        lines = text.splitlines()
        chunks: list[tuple[str, str]] = []
        current_heading = ""
        current_lines: list[str] = []
        for line in lines:
            if line.startswith("## "):
                if current_lines:
                    chunks.append((current_heading, "\n".join(current_lines)))
                current_heading = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            chunks.append((current_heading, "\n".join(current_lines)))

        scored: list[tuple[int, str, str]] = []
        for heading, body in chunks:
            hay = f"{heading}\n{body}".lower()
            score = sum(1 for term in terms if term in hay)
            if score > 0:
                scored.append((score, heading, body.strip()))
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[QueryResult] = []
        for score, heading, body in scored[:top_k]:
            parsed = JournalPageIndexService._parse_markdown_entry_fields(body)
            excerpt = (parsed.get("body_text") or body or "")[:500]
            results.append(
                QueryResult(
                    node_id=None,
                    heading=heading or None,
                    excerpt=excerpt,
                    entry_id=parsed.get("entry_id"),
                    created_at=parsed.get("created_at"),
                    source_kind=parsed.get("source_kind") or "journal",
                    provenance={
                        "engine": "markdown_local_search",
                        "score": score,
                    },
                )
            )
        return results

    @staticmethod
    def _parse_markdown_entry_fields(body: str) -> dict[str, str | None]:
        entry: dict[str, str | None] = {
            "entry_id": None,
            "created_at": None,
            "source_kind": None,
            "body_text": "",
        }
        lines = body.splitlines()
        idx = 0
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1

        while idx < len(lines):
            stripped = lines[idx].strip()
            if stripped.startswith("- entry_id:"):
                entry["entry_id"] = stripped.split(":", 1)[1].strip() or None
            elif stripped.startswith("- created_at:"):
                entry["created_at"] = stripped.split(":", 1)[1].strip() or None
            elif stripped.startswith("- source_kind:"):
                entry["source_kind"] = stripped.split(":", 1)[1].strip() or None
            elif not stripped.startswith("- "):
                break
            idx += 1

        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1

        body_start = idx
        body_text = "\n".join(lines[body_start:]).strip()
        entry["body_text"] = body_text
        return entry
