from __future__ import annotations

import logging
import sys

from fastapi import FastAPI, HTTPException

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .models import (
    BuildResponse,
    ChatEpisodeBuildResponse,
    ChatEpisodeStatusResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)
from .pageindex_cli import PageIndexCliError
from .service import JournalPageIndexService
from .settings import settings

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(settings.SERVICE_NAME)

app = FastAPI(title="Orion PageIndex", version=settings.SERVICE_VERSION)
svc = JournalPageIndexService()
heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every HEARTBEAT_INTERVAL_SEC. PageIndex had no bus connection of its own before this
    patch (it talks to Postgres directly and shells out to the PageIndex CLI) -- this
    chassis is the first bus-native wiring in this service. See
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


@app.on_event("startup")
async def _start_heartbeat_chassis() -> None:
    global heartbeat_chassis
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.SERVICE_NAME,
            settings.HEARTBEAT_INTERVAL_SEC,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat_chassis = None


@app.on_event("shutdown")
async def _stop_heartbeat_chassis() -> None:
    global heartbeat_chassis
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning("system_health_heartbeat_stop_error error=%s", exc)
        heartbeat_chassis = None


@app.on_event("startup")
def _log_startup_config() -> None:
    health = svc.health()
    pageindex_proof = health.get("pageindex") if isinstance(health, dict) else {}
    logger.info(
        "startup service=%s pageindex_repo_path=%s installation_mode=%s health_ok=%s db_env_key=%s db_url_present=%s pageindex_proof_keys=%s",
        settings.SERVICE_NAME,
        settings.PAGEINDEX_REPO_PATH,
        settings.PAGEINDEX_INSTALLATION_MODE,
        bool(health.get("ok")) if isinstance(health, dict) else False,
        health.get("db_env_key") if isinstance(health, dict) else None,
        bool(health.get("db_url_present")) if isinstance(health, dict) else False,
        sorted(pageindex_proof.keys()) if isinstance(pageindex_proof, dict) else [],
    )


@app.get("/healthz")
def healthz() -> dict:
    return svc.health()


@app.post("/corpora/journals/rebuild", response_model=BuildResponse)
def rebuild_journals() -> BuildResponse:
    out = svc.rebuild_journals()
    logger.info(
        "pageindex_impl=%s pageindex_installation_mode=%s journal_corpus_row_count=%s markdown_export_path=%s pageindex_tree_artifact_path=%s last_build_started_at=%s last_build_completed_at=%s build_success=%s",
        out.pageindex_impl,
        out.pageindex_installation_mode,
        out.journal_corpus_row_count,
        out.markdown_export_path,
        out.pageindex_tree_artifact_path,
        out.last_build_started_at,
        out.last_build_completed_at,
        out.build_success,
    )
    return out


@app.get("/corpora/journals/status", response_model=StatusResponse)
def journals_status() -> StatusResponse:
    return svc.status()


@app.post("/corpora/journals/query", response_model=QueryResponse)
def journals_query(body: QueryRequest) -> QueryResponse:
    try:
        out = svc.query_journals(query=body.query, allow_fallback=body.allow_fallback, top_k=body.top_k)
        logger.info(
            "query_invoked=true query_result_count=%s fallback_invoked=%s",
            out.query_result_count,
            out.fallback_invoked,
        )
        return out
    except PageIndexCliError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/corpora/chat_episodes/rebuild", response_model=ChatEpisodeBuildResponse)
def rebuild_chat_episodes() -> ChatEpisodeBuildResponse:
    out = svc.rebuild_chat_episodes()
    logger.info(
        "chat_episode_rebuild corpus_key=%s markdown_export_path=%s pageindex_tree_artifact_path=%s build_success=%s",
        out.corpus_key,
        out.markdown_export_path,
        out.pageindex_tree_artifact_path,
        out.build_success,
    )
    return out


@app.get("/corpora/chat_episodes/status", response_model=ChatEpisodeStatusResponse)
def chat_episodes_status() -> ChatEpisodeStatusResponse:
    return svc.chat_episodes_status()


@app.post("/corpora/chat_episodes/query", response_model=QueryResponse)
def chat_episodes_query(body: QueryRequest) -> QueryResponse:
    try:
        out = svc.query_chat_episodes(query=body.query, allow_fallback=body.allow_fallback, top_k=body.top_k)
        logger.info(
            "chat_episode_query_invoked=true query_result_count=%s fallback_invoked=%s",
            out.query_result_count,
            out.fallback_invoked,
        )
        return out
    except PageIndexCliError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
