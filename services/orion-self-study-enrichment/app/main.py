from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .cache import content_hash, read_cached, write_cached
from .claude_runner import run_claude_once
from .evidence import build_evidence_bundle, render_evidence_prompt
from .rate_limit import allow_and_record
from .settings import Settings, get_settings

logger = logging.getLogger("orion-self-study-enrichment")


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[SELF_STUDY_ENRICHMENT] %(asctime)s %(levelname)s - %(name)s - %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def build_heartbeat_chassis(settings: Settings) -> HeartbeatOnly:
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.SELF_STUDY_ENRICHMENT_NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


def handle_request_payload(settings: Settings, payload: dict) -> None:
    """Pure-ish orchestration of one enrichment run -- the one place all the
    testable pieces (evidence, cache, rate limit, claude_runner) are wired
    together. Never raises: every real failure is logged and swallowed so a
    bad event can't crash the consumer loop (this is a best-effort dev-
    tooling capability, not a critical live path)."""
    try:
        touched_paths = tuple(payload.get("touched_paths") or ())
        if not touched_paths:
            logger.info("self_study_enrichment_skip_empty_touched_paths")
            return

        delta_summary = {
            "prev_sha": payload.get("prev_sha"),
            "head_sha": payload.get("head_sha"),
            "commit_count": payload.get("commit_count"),
            "files_changed": payload.get("files_changed"),
            "lines_changed": payload.get("lines_changed"),
        }

        bundle = build_evidence_bundle(
            repo_root=settings.SELF_STUDY_ENRICHMENT_REPO_PATH,
            graph_json_path=settings.SELF_STUDY_ENRICHMENT_GRAPH_JSON_PATH,
            touched_paths=touched_paths,
            delta_summary=delta_summary,
        )
        if bundle.is_empty():
            logger.info("self_study_enrichment_skip_empty_evidence_bundle")
            return

        prompt = render_evidence_prompt(bundle)
        key = content_hash(prompt)
        cached = read_cached(settings.SELF_STUDY_ENRICHMENT_CACHE_DIR, key)
        if cached is not None:
            logger.info("self_study_enrichment_cache_hit key=%s", key)
            return

        if not allow_and_record(
            settings.SELF_STUDY_ENRICHMENT_RATE_LIMIT_STATE_PATH,
            max_per_day=settings.SELF_STUDY_ENRICHMENT_MAX_PER_DAY,
        ):
            logger.warning("self_study_enrichment_rate_limit_hit skip_key=%s", key)
            return

        env = dict(os.environ)
        if settings.ANTHROPIC_API_KEY:
            env["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY

        result = run_claude_once(
            prompt,
            claude_bin=settings.SELF_STUDY_ENRICHMENT_CLAUDE_BIN,
            model=settings.SELF_STUDY_ENRICHMENT_MODEL,
            effort=settings.SELF_STUDY_ENRICHMENT_EFFORT,
            setting_sources_env_key=settings.SELF_STUDY_ENRICHMENT_SETTING_SOURCES_ENV_KEY,
            timeout_sec=settings.SELF_STUDY_ENRICHMENT_TIMEOUT_SEC,
            env=env,
        )
        if not result.ok:
            logger.error("self_study_enrichment_claude_run_failed error=%s", result.error)
            return

        write_cached(
            settings.SELF_STUDY_ENRICHMENT_CACHE_DIR,
            key,
            {
                "key": key,
                "touched_paths": list(touched_paths),
                "delta_summary": delta_summary,
                "model": settings.SELF_STUDY_ENRICHMENT_MODEL,
                "summary": result.text,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("self_study_enrichment_run_complete key=%s clusters=%d", key, len(bundle.graph_nodes))
    except Exception:
        logger.exception("self_study_enrichment_handle_request_failed")


async def run_consumer(settings: Settings, bus: OrionBusAsync, stop: asyncio.Event) -> None:
    channel = settings.CHANNEL_SELF_STUDY_ENRICHMENT_REQUESTED
    async with bus.subscribe(channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            if stop.is_set():
                break
            try:
                raw = msg.get("data")
                payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            except Exception:
                logger.exception("self_study_enrichment_bad_payload")
                continue
            handle_request_payload(settings, payload)


async def main_async() -> None:
    setup_logging()
    settings = get_settings()
    stop = asyncio.Event()

    def _handle_signal() -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    import signal

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass

    heartbeat = build_heartbeat_chassis(settings)
    if settings.ORION_BUS_ENABLED:
        await heartbeat.start_background()

    bus = OrionBusAsync(
        settings.ORION_BUS_URL,
        enabled=settings.ORION_BUS_ENABLED,
        enforce_catalog=settings.ORION_BUS_ENFORCE_CATALOG,
    )
    try:
        if settings.ORION_BUS_ENABLED:
            await bus.connect()
            await run_consumer(settings, bus, stop)
        else:
            logger.warning("orion_bus_disabled_idle")
            await stop.wait()
    finally:
        if settings.ORION_BUS_ENABLED:
            await heartbeat.stop()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
