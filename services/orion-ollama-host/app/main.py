# services/orion-ollama-host/app/main.py
from __future__ import annotations

import logging
import os
import subprocess
import time
import requests
from typing import List, Dict, Any, Optional

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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[OLLAMA] %(levelname)s - %(name)s - %(message)s",
    )

    logger.info("Starting %s v%s", settings.service_name, settings.service_version)

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


if __name__ == "__main__":
    main()
