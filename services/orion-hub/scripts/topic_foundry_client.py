"""Small HTTP client for topic-foundry's read-only run/topic/keyword endpoints.

Phase 9 of ``docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md``.

This module performs the actual network calls that
``orion.substrate.adapters.topic_foundry.map_topic_foundry_run_to_substrate``
expects its caller to have already made (that adapter is a pure conversion
function with no HTTP/bus I/O of its own -- see its docstring). It is
deliberately thin: a handful of small functions, capped work, short
timeouts, and a single exception type callers can catch. No retries, no
connection pooling beyond ``requests``'s defaults, no general-purpose
external-source plugin framework.

Every function here either raises ``TopicFoundryClientError`` (for failures
that make the overall result unusable) or degrades to an empty/default value
(for failures scoped to a single topic's keywords, where the adapter already
tolerates missing keywords). Nothing here ever lets a raw ``requests``
exception escape -- callers (route handlers) should only ever need to catch
``TopicFoundryClientError``.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger("orion-hub.topic_foundry_client")

DEFAULT_TIMEOUT_SEC = 5.0

# Bounds worst-case fan-out of the per-topic keywords calls -- mirrors the
# "cap all collections" instruction and the adapter's own
# MAX_TOPICS_PER_RUN-style caps (orion/substrate/adapters/topic_foundry.py).
MAX_TOPICS_FOR_KEYWORDS = 50

# HDBSCAN's noise/outlier bucket. The adapter already excludes it, but there
# is no reason to spend an HTTP call fetching keywords for it either.
OUTLIER_TOPIC_ID = -1


class TopicFoundryClientError(Exception):
    """Raised for a topic-foundry HTTP/parsing failure that makes the result unusable.

    Callers (route handlers) should catch this and degrade to an honest
    error response -- never let it propagate to a raw 500.
    """


def fetch_latest_completed_run(base_url: str, *, timeout: float = DEFAULT_TIMEOUT_SEC) -> dict[str, Any]:
    """Return the most recently completed run's summary dict (``RunListItem`` shape).

    Raises ``TopicFoundryClientError`` on any network/HTTP/parse failure, or
    if no completed run exists yet.
    """
    url = f"{base_url.rstrip('/')}/runs"
    try:
        resp = requests.get(
            url,
            params={"format": "wrapped", "status": "complete", "limit": 1},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise TopicFoundryClientError(f"topic_foundry_runs_request_failed: {exc}") from exc
    except ValueError as exc:
        raise TopicFoundryClientError(f"topic_foundry_runs_invalid_json: {exc}") from exc

    items = payload.get("items") if isinstance(payload, dict) else None
    if not items:
        raise TopicFoundryClientError("topic_foundry_no_completed_run")
    run = items[0]
    if not isinstance(run, dict) or not run.get("run_id"):
        raise TopicFoundryClientError("topic_foundry_run_missing_run_id")
    return run


def fetch_topics_for_run(
    base_url: str,
    run_id: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return the raw ``TopicSummaryItem`` dicts for ``run_id``.

    Raises ``TopicFoundryClientError`` on any network/HTTP/parse failure or a
    response missing the expected ``items`` list. An empty ``items`` list
    (a real run with zero topics) is not an error -- returns ``[]``.
    """
    url = f"{base_url.rstrip('/')}/topics"
    try:
        resp = requests.get(url, params={"run_id": run_id, "limit": limit}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise TopicFoundryClientError(f"topic_foundry_topics_request_failed: {exc}") from exc
    except ValueError as exc:
        raise TopicFoundryClientError(f"topic_foundry_topics_invalid_json: {exc}") from exc

    items = payload.get("items") if isinstance(payload, dict) else None
    if items is None:
        raise TopicFoundryClientError("topic_foundry_topics_malformed_response")
    return [item for item in items if isinstance(item, dict)]


def fetch_keywords_for_topic(
    base_url: str,
    run_id: str,
    topic_id: int,
    *,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> list[str]:
    """Best-effort per-topic keyword fetch.

    Returns ``[]`` on any failure rather than raising -- one topic's
    keywords being unavailable should not abort the whole ingestion; the
    adapter already tolerates an empty keyword list for a topic (it falls
    back to a synthetic ``topic_{id}`` label).
    """
    url = f"{base_url.rstrip('/')}/topics/{topic_id}/keywords"
    try:
        resp = requests.get(url, params={"run_id": run_id}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.warning("topic_foundry_keywords_fetch_failed topic_id=%s error=%s", topic_id, exc)
        return []
    except ValueError as exc:
        logger.warning("topic_foundry_keywords_invalid_json topic_id=%s error=%s", topic_id, exc)
        return []

    keywords = payload.get("keywords") if isinstance(payload, dict) else None
    if not isinstance(keywords, list):
        return []
    return [str(k) for k in keywords if isinstance(k, (str, int, float))]


def list_datasets(base_url: str, *, timeout: float = DEFAULT_TIMEOUT_SEC) -> list[dict[str, Any]]:
    """Return every ``DatasetSpec`` dict topic-foundry currently has.

    Raises ``TopicFoundryClientError`` on any network/HTTP/parse failure.
    An empty list (no datasets yet) is not an error.
    """
    url = f"{base_url.rstrip('/')}/datasets"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise TopicFoundryClientError(f"topic_foundry_datasets_request_failed: {exc}") from exc
    except ValueError as exc:
        raise TopicFoundryClientError(f"topic_foundry_datasets_invalid_json: {exc}") from exc

    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if datasets is None:
        raise TopicFoundryClientError("topic_foundry_datasets_malformed_response")
    return [d for d in datasets if isinstance(d, dict)]


def list_models(base_url: str, *, timeout: float = DEFAULT_TIMEOUT_SEC) -> list[dict[str, Any]]:
    """Return every ``ModelSummary`` dict topic-foundry currently has.

    Raises ``TopicFoundryClientError`` on any network/HTTP/parse failure.
    An empty list (no models yet) is not an error.
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise TopicFoundryClientError(f"topic_foundry_models_request_failed: {exc}") from exc
    except ValueError as exc:
        raise TopicFoundryClientError(f"topic_foundry_models_invalid_json: {exc}") from exc

    models = payload.get("models") if isinstance(payload, dict) else None
    if models is None:
        raise TopicFoundryClientError("topic_foundry_models_malformed_response")
    return [m for m in models if isinstance(m, dict)]


def create_dataset(
    base_url: str, payload: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT_SEC
) -> dict[str, Any]:
    """Create a topic-foundry dataset (``DatasetCreateRequest`` shape).

    Raises ``TopicFoundryClientError`` on any network/HTTP/parse failure.
    Returns the raw ``DatasetCreateResponse`` dict (``dataset_id``, ``created_at``).
    """
    url = f"{base_url.rstrip('/')}/datasets"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise TopicFoundryClientError(f"topic_foundry_create_dataset_failed: {exc}") from exc
    except ValueError as exc:
        raise TopicFoundryClientError(f"topic_foundry_create_dataset_invalid_json: {exc}") from exc


def create_model(
    base_url: str, payload: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT_SEC
) -> dict[str, Any]:
    """Create a topic-foundry model (``ModelCreateRequest`` shape).

    Raises ``TopicFoundryClientError`` on any network/HTTP/parse failure.
    Returns the raw ``ModelCreateResponse`` dict (``model_id``, ``created_at``).
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise TopicFoundryClientError(f"topic_foundry_create_model_failed: {exc}") from exc
    except ValueError as exc:
        raise TopicFoundryClientError(f"topic_foundry_create_model_invalid_json: {exc}") from exc


def trigger_training_run(
    base_url: str,
    *,
    model_id: str,
    dataset_id: str,
    start_at: str,
    end_at: str,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    """``POST /runs/train`` (``RunTrainRequest`` shape) -- starts training as a
    background task on topic-foundry's side and returns immediately with
    ``{"run_id": ..., "status": "queued"}`` (or an existing run's id/status
    if topic-foundry's own ``spec_hash`` dedup finds an identical
    dataset+model+window run already exists -- see
    ``services/orion-topic-foundry/app/routers/runs.py::train_run_endpoint``,
    not duplicated here).

    Raises ``TopicFoundryClientError`` on any network/HTTP/parse failure.
    """
    url = f"{base_url.rstrip('/')}/runs/train"
    payload = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "start_at": start_at,
        "end_at": end_at,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise TopicFoundryClientError(f"topic_foundry_train_trigger_failed: {exc}") from exc
    except ValueError as exc:
        raise TopicFoundryClientError(f"topic_foundry_train_trigger_invalid_json: {exc}") from exc


def fetch_run_topics_and_keywords(
    base_url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    max_topics_for_keywords: int = MAX_TOPICS_FOR_KEYWORDS,
) -> dict[str, Any]:
    """Orchestrate the 3-call sequence: latest run -> topics -> per-topic keywords.

    Raises ``TopicFoundryClientError`` only for the two calls that make the
    whole result unusable (no completed run, or the topics list itself is
    unreachable/malformed). Individual keyword-fetch failures degrade to an
    empty keyword list for that topic (see ``fetch_keywords_for_topic``)
    rather than aborting the run.

    Returns:
        ``{"run_id": str, "run": dict, "topics": list[dict], "keywords_by_topic": {int: [str]}}``
    """
    run = fetch_latest_completed_run(base_url, timeout=timeout)
    run_id = str(run["run_id"])
    topics = fetch_topics_for_run(base_url, run_id, timeout=timeout)

    real_topic_ids: list[int] = []
    for item in topics:
        raw_topic_id = item.get("topic_id")
        try:
            topic_id = int(raw_topic_id)
        except (TypeError, ValueError):
            continue
        if topic_id == OUTLIER_TOPIC_ID:
            continue
        real_topic_ids.append(topic_id)

    keywords_by_topic: dict[int, list[str]] = {}
    for topic_id in real_topic_ids[:max_topics_for_keywords]:
        keywords_by_topic[topic_id] = fetch_keywords_for_topic(base_url, run_id, topic_id, timeout=timeout)

    return {
        "run_id": run_id,
        "run": run,
        "topics": topics,
        "keywords_by_topic": keywords_by_topic,
    }
