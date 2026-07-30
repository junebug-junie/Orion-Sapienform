from __future__ import annotations

import asyncio
import logging
import signal
import sys

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.core.bus.codec import OrionCodec

from .producers.git_delta import git_delta_loop
from .producers.graph_delta import graph_delta_loop
from .producers.pr_lifecycle import pr_lifecycle_loop
from .settings import get_settings

logger = logging.getLogger("orion-cocreation-signals")


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[COCREATION_SIGNALS] %(asctime)s %(levelname)s - %(name)s - %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def _source(settings) -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=settings.COCREATION_SIGNALS_NODE_NAME,
    )


def build_heartbeat_chassis(settings=None) -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to
    orion:system:health -- same pattern orion-power-guard/orion-mesh-guardian
    use, deliberately separate from the producer bus connection below."""
    s = settings if settings is not None else get_settings()
    return HeartbeatOnly(
        ChassisConfig(
            service_name=s.SERVICE_NAME,
            service_version=s.SERVICE_VERSION,
            node_name=s.COCREATION_SIGNALS_NODE_NAME,
            bus_url=s.ORION_BUS_URL,
            bus_enabled=s.ORION_BUS_ENABLED,
            heartbeat_interval_sec=s.HEARTBEAT_INTERVAL_SEC,
        )
    )


async def run_producers(settings, bus: OrionBusAsync, stop: asyncio.Event) -> None:
    """Each producer is its own independent async loop -- no shared tick, own
    timeout/error boundary (each loop already catches+logs its own
    exceptions per-tick, per CLAUDE.md fail-open convention), own enable
    flag, so a GitHub API hiccup in pr_lifecycle can never block git_delta or
    graph_delta."""
    source = _source(settings)
    tasks: list[asyncio.Task] = []

    if settings.COCREATION_SIGNALS_GIT_DELTA_ENABLED:
        tasks.append(
            asyncio.create_task(
                git_delta_loop(
                    bus=bus,
                    channel=settings.CHANNEL_CODEBASE_DELTA,
                    source=source,
                    repo_path=settings.COCREATION_SIGNALS_REPO_PATH,
                    poll_interval_sec=settings.COCREATION_SIGNALS_GIT_DELTA_POLL_INTERVAL_SEC,
                    stop=stop,
                ),
                name="git_delta_loop",
            )
        )
    else:
        logger.info("cocreation_git_delta_disabled")

    if settings.COCREATION_SIGNALS_PR_LIFECYCLE_ENABLED:
        tasks.append(
            asyncio.create_task(
                pr_lifecycle_loop(
                    bus=bus,
                    channel=settings.CHANNEL_CODEBASE_DELTA,
                    source=source,
                    owner=settings.COCREATION_SIGNALS_GITHUB_OWNER,
                    repo=settings.COCREATION_SIGNALS_GITHUB_REPO,
                    fetch_limit=settings.COCREATION_SIGNALS_PR_FETCH_LIMIT,
                    poll_interval_sec=settings.COCREATION_SIGNALS_PR_LIFECYCLE_POLL_INTERVAL_SEC,
                    cold_start_lookback_sec=settings.COCREATION_SIGNALS_PR_LIFECYCLE_COLD_START_LOOKBACK_SEC,
                    stop=stop,
                ),
                name="pr_lifecycle_loop",
            )
        )
    else:
        logger.info("cocreation_pr_lifecycle_disabled")

    if settings.COCREATION_SIGNALS_GRAPH_DELTA_ENABLED:
        tasks.append(
            asyncio.create_task(
                graph_delta_loop(
                    bus=bus,
                    channel=settings.CHANNEL_CODEBASE_DELTA,
                    source=source,
                    repo_path=settings.COCREATION_SIGNALS_REPO_PATH,
                    poll_interval_sec=settings.COCREATION_SIGNALS_GRAPH_DELTA_POLL_INTERVAL_SEC,
                    stop=stop,
                ),
                name="graph_delta_loop",
            )
        )
    else:
        logger.info("cocreation_graph_delta_disabled")

    if not tasks:
        logger.warning("cocreation_signals_no_producers_enabled")
        await stop.wait()
        return

    await asyncio.gather(*tasks)


async def _main_async() -> None:
    settings = get_settings()
    logger.info(
        "starting_orion_cocreation_signals service=%s version=%s node=%s repo_path=%s",
        settings.SERVICE_NAME, settings.SERVICE_VERSION, settings.COCREATION_SIGNALS_NODE_NAME,
        settings.COCREATION_SIGNALS_REPO_PATH,
    )

    bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED, codec=OrionCodec())
    if bus.enabled:
        await bus.connect()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows/some sandboxed environments don't support add_signal_handler.
            pass

    heartbeat_chassis = build_heartbeat_chassis(settings)
    await heartbeat_chassis.start_background()
    try:
        await run_producers(settings, bus, stop)
    finally:
        await heartbeat_chassis.stop()


def main() -> None:
    setup_logging()
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
