# services/orion-ollama-host/app/main.py
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
import requests
from typing import List, Dict, Any, Optional

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .settings import settings

logger = logging.getLogger("orion-ollama-host")


def wait_for_ollama(port: int = 11434, timeout: int = 30):
    """Wait for Ollama server to be ready."""
    start_time = time.time()
    url = f"http://localhost:{port}/api/version"
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                logger.info("Ollama is ready: %s", resp.text)
                return
        except Exception:
            pass
        time.sleep(1)
    logger.warning("Ollama did not report ready within %s seconds", timeout)


def pull_model(model_id: str, port: int = 11434):
    """Pull the specified model using Ollama API."""
    logger.info("Triggering pull for model: %s", model_id)
    url = f"http://localhost:{port}/api/pull"
    try:
        # Stream=False to wait for completion (or we could stream logging)
        # Using subprocess might be easier to see progress in logs,
        # but API is cleaner for automation.
        # Let's use subprocess for better log visibility if possible,
        # but 'ollama pull' command requires the binary.
        # The base image has the binary.
        subprocess.run(["ollama", "pull", model_id], check=True)
        logger.info("Successfully pulled model: %s", model_id)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to pull model %s: %s", model_id, e)
    except Exception as e:
        logger.error("Error pulling model %s: %s", model_id, e)


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Independent of the `ollama serve` subprocess this service
    launches -- see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=settings.orion_bus_url,
            bus_enabled=settings.orion_bus_enabled,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )


def run_ollama_server_blocking() -> None:
    """The service's original (pre-heartbeat) synchronous launch flow, unchanged. Run inside
    asyncio.to_thread() by _main_async() so the heartbeat chassis's own event loop stays
    responsive while this blocks on subprocess I/O."""
    # Start Ollama server in background
    # The official image entrypoint often does `ollama serve`.
    # We are overriding entrypoint, so we must run it.

    # Check if we need to set OLLAMA_HOST env var for the subprocess
    # defaults to 127.0.0.1:11434 usually, but we want 0.0.0.0
    os.environ["OLLAMA_HOST"] = "0.0.0.0:11434"

    logger.info("Launching ollama serve...")
    server_process = subprocess.Popen(["ollama", "serve"])

    # Wait for it to be up
    wait_for_ollama()

    # Determine which model to ensure
    model_id = settings.resolve_model()
    if model_id:
        logger.info("Ensuring model %s is available...", model_id)
        pull_model(model_id)
    else:
        logger.info("No specific OLLAMA_MODEL_ID or profile configured. Skipping auto-pull.")

    # Wait for server process
    server_process.wait()


async def _main_async() -> None:
    logger.info("Starting %s v%s", settings.service_name, settings.service_version)

    heartbeat_chassis: Optional[HeartbeatOnly] = None
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.service_name,
            settings.heartbeat_interval_sec,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat_chassis = None

    try:
        await asyncio.to_thread(run_ollama_server_blocking)
    finally:
        if heartbeat_chassis is not None:
            try:
                await heartbeat_chassis.stop()
            except Exception as exc:
                logger.warning("system_health_heartbeat_stop_error error=%s", exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[OLLAMA] %(levelname)s - %(name)s - %(message)s",
    )
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
