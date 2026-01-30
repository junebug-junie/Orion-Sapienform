from __future__ import annotations

from typing import Any, Dict, Iterable, List

import psycopg2
from psycopg2.extras import Json, execute_values

from app.topic_rail.db.ddl import CHAT_TOPIC_DDL, CHAT_TOPIC_SUMMARY_DDL, CHAT_TOPIC_DRIFT_DDL


class TopicRailWriter:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def ensure_tables_exist(self) -> None:
        conn = psycopg2.connect(self.dsn)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(CHAT_TOPIC_DDL)
                cur.execute(CHAT_TOPIC_SUMMARY_DDL)
                cur.execute(CHAT_TOPIC_DRIFT_DDL)
        finally:
            conn.close()

    def upsert_assignments(self, assignments: Iterable[Dict[str, Any]]) -> int:
        rows: List[tuple] = []
        for item in assignments:
            rows.append(
                (
                    item.get("chat_id"),
                    item.get("correlation_id"),
                    item.get("trace_id"),
                    item.get("session_id"),
                    item.get("topic_id"),
                    item.get("topic_label"),
                    Json(item.get("topic_keywords")),
                    item.get("topic_confidence"),
                    item.get("model_version"),
                )
            )

        if not rows:
            return 0

        query = """
        INSERT INTO chat_topic (
            chat_id,
            correlation_id,
            trace_id,
            session_id,
            topic_id,
            topic_label,
            topic_keywords,
            topic_confidence,
            model_version
        ) VALUES %s
        ON CONFLICT (chat_id, model_version)
        DO UPDATE SET
            correlation_id=EXCLUDED.correlation_id,
            trace_id=EXCLUDED.trace_id,
            session_id=EXCLUDED.session_id,
            topic_id=EXCLUDED.topic_id,
            topic_label=EXCLUDED.topic_label,
            topic_keywords=EXCLUDED.topic_keywords,
            topic_confidence=EXCLUDED.topic_confidence,
            created_at=NOW();
        """

        conn = psycopg2.connect(self.dsn)
        try:
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, query, rows)
            return len(rows)
        finally:
            conn.close()

    def upsert_summary_rows(self, rows: Iterable[Dict[str, Any]]) -> int:
        payload: List[tuple] = []
        for item in rows:
            payload.append(
                (
                    item.get("model_version"),
                    item.get("window_start"),
                    item.get("window_end"),
                    item.get("topic_id"),
                    item.get("topic_label"),
                    Json(item.get("topic_keywords")),
                    item.get("doc_count"),
                    item.get("pct_of_window"),
                    item.get("outlier_count"),
                    item.get("outlier_pct"),
                )
            )

        if not payload:
            return 0

        query = """
        INSERT INTO chat_topic_summary (
            model_version,
            window_start,
            window_end,
            topic_id,
            topic_label,
            topic_keywords,
            doc_count,
            pct_of_window,
            outlier_count,
            outlier_pct
        ) VALUES %s
        ON CONFLICT (model_version, window_start, window_end, topic_id)
        DO UPDATE SET
            topic_label=EXCLUDED.topic_label,
            topic_keywords=EXCLUDED.topic_keywords,
            doc_count=EXCLUDED.doc_count,
            pct_of_window=EXCLUDED.pct_of_window,
            outlier_count=EXCLUDED.outlier_count,
            outlier_pct=EXCLUDED.outlier_pct,
            created_at=NOW();
        """

        conn = psycopg2.connect(self.dsn)
        try:
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, query, payload)
            return len(payload)
        finally:
            conn.close()

    def upsert_drift_rows(self, rows: Iterable[Dict[str, Any]]) -> int:
        payload: List[tuple] = []
        for item in rows:
            payload.append(
                (
                    item.get("model_version"),
                    item.get("window_start"),
                    item.get("window_end"),
                    item.get("session_id"),
                    item.get("turns"),
                    item.get("unique_topics"),
                    item.get("entropy"),
                    item.get("switch_rate"),
                    item.get("dominant_topic_id"),
                    item.get("dominant_pct"),
                )
            )

        if not payload:
            return 0

        query = """
        INSERT INTO chat_topic_session_drift (
            model_version,
            window_start,
            window_end,
            session_id,
            turns,
            unique_topics,
            entropy,
            switch_rate,
            dominant_topic_id,
            dominant_pct
        ) VALUES %s
        ON CONFLICT (model_version, window_start, window_end, session_id)
        DO UPDATE SET
            turns=EXCLUDED.turns,
            unique_topics=EXCLUDED.unique_topics,
            entropy=EXCLUDED.entropy,
            switch_rate=EXCLUDED.switch_rate,
            dominant_topic_id=EXCLUDED.dominant_topic_id,
            dominant_pct=EXCLUDED.dominant_pct,
            created_at=NOW();
        """

        conn = psycopg2.connect(self.dsn)
        try:
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, query, payload)
            return len(payload)
        finally:
            conn.close()
