import os
import sys
from typing import List, Dict

import psycopg2


def _set_env_defaults() -> None:
    os.environ.setdefault("TOPIC_RAIL_MODEL_VERSION", "topic-rail-v1-smoke")
    os.environ.setdefault("TOPIC_RAIL_MODEL_DIR", "/mnt/telemetry/models/topic-rail")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _seed_chat_history(dsn: str) -> None:
    rows: List[Dict[str, str]] = [
        {
            "id": "smoke-topic-rail-1",
            "correlation_id": "smoke-topic-rail-1",
            "prompt": "I need help with my Redis cache evictions.",
            "response": "We can tune maxmemory-policy and monitor evictions.",
        },
        {
            "id": "smoke-topic-rail-2",
            "correlation_id": "smoke-topic-rail-2",
            "prompt": "My Postgres query is slow on a large table.",
            "response": "Add indexes and check the query plan with EXPLAIN.",
        },
        {
            "id": "smoke-topic-rail-3",
            "correlation_id": "smoke-topic-rail-3",
            "prompt": "We need a data pipeline for logs.",
            "response": "Consider batching, partitioning, and schema evolution.",
        },
        {
            "id": "smoke-topic-rail-4",
            "correlation_id": "smoke-topic-rail-4",
            "prompt": "The app UI feels cluttered.",
            "response": "Reduce visual noise and increase spacing for clarity.",
        },
        {
            "id": "smoke-topic-rail-5",
            "correlation_id": "smoke-topic-rail-5",
            "prompt": "We keep timing out in the API gateway.",
            "response": "Check upstream latencies and set appropriate timeouts.",
        },
        {
            "id": "smoke-topic-rail-6",
            "correlation_id": "smoke-topic-rail-6",
            "prompt": "Could we auto-tag user feedback?",
            "response": "Use topic modeling to group feedback into themes.",
        },
        {
            "id": "smoke-topic-rail-7",
            "correlation_id": "smoke-topic-rail-7",
            "prompt": "Memory usage spikes when batch jobs run.",
            "response": "Profile allocations and consider streaming processing.",
        },
        {
            "id": "smoke-topic-rail-8",
            "correlation_id": "smoke-topic-rail-8",
            "prompt": "We should improve onboarding documentation.",
            "response": "Add clear steps, screenshots, and a quick-start guide.",
        },
    ]

    conn = psycopg2.connect(dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO chat_history_log
                            (id, correlation_id, source, prompt, response, session_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            row["id"],
                            row["correlation_id"],
                            "smoke",
                            row["prompt"],
                            row["response"],
                            "smoke_topic_rail",
                        ),
                    )
    finally:
        conn.close()


if __name__ == "__main__":
    _set_env_defaults()

    dsn = _require_env("TOPIC_RAIL_PG_DSN")
    _require_env("TOPIC_RAIL_EMBEDDING_URL")

    sys.path.append(os.path.join(os.getcwd(), "services", "orion-topic-rail"))

    from app.main import TopicRailService
    from app.settings import settings

    _seed_chat_history(dsn)

    service = TopicRailService()
    service.writer.ensure_tables_exist()

    if not service.model_store.exists(settings.topic_rail_model_version):
        service._train_and_assign()
    service._assign_only()

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chat_id, topic_id, topic_label, topic_confidence
                FROM chat_topic
                WHERE model_version = %s
                ORDER BY created_at DESC
                LIMIT 5
                """,
                (settings.topic_rail_model_version,),
            )
            sample = cur.fetchall()

            cur.execute(
                """
                SELECT COUNT(*)
                FROM chat_topic
                WHERE model_version = %s
                """,
                (settings.topic_rail_model_version,),
            )
            total_rows = cur.fetchone()[0]

            cur.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT chat_id, model_version, COUNT(*) AS cnt
                    FROM chat_topic
                    WHERE model_version = %s
                    GROUP BY chat_id, model_version
                    HAVING COUNT(*) > 1
                ) dupes
                """,
                (settings.topic_rail_model_version,),
            )
            dupes = cur.fetchone()[0]
    finally:
        conn.close()

    print(f"chat_topic rows for model_version={settings.topic_rail_model_version}: {total_rows}")
    for row in sample:
        print("sample:", row)

    if total_rows == 0:
        raise SystemExit("No chat_topic rows were written in smoke test.")
    if dupes:
        raise SystemExit("Duplicate rows detected for (chat_id, model_version).")
