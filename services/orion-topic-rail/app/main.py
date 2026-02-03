from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from app.settings import settings
from app.topic_rail.db.lock import acquire_lock, release_lock
from app.topic_rail.db.reader import TopicRailReader
from app.topic_rail.db.writer import TopicRailWriter
from app.topic_rail.doc_builders.chat_history import ChatHistoryDocBuilder
from app.topic_rail.embeddings.vector_host import VectorHostEmbeddingProvider
from app.topic_rail.factories.topic_model_factory import build_topic_model
from app.topic_rail.analytics import compute_summary_rows, compute_drift_rows
from app.topic_rail.http_server import start_http_server
from app.topic_rail.lifecycle import decide_refit
from app.topic_rail.publish.bus_publisher import TopicRailBusPublisher
from app.topic_rail.persistence.model_store import ModelStore


logger = logging.getLogger("topic-rail")


class TopicRailService:
    def __init__(self) -> None:
        self._stop = False
        self.reader = TopicRailReader(settings.topic_rail_pg_dsn)
        self.writer = TopicRailWriter(settings.topic_rail_pg_dsn)
        self.doc_builder = ChatHistoryDocBuilder(settings.topic_rail_doc_mode)
        self.embedder = VectorHostEmbeddingProvider(
            settings.topic_rail_embedding_url,
            retries=settings.topic_rail_embedding_retries,
            backoff_sec=settings.topic_rail_embedding_backoff_sec,
        )
        self.model_store = ModelStore(settings.topic_rail_model_dir)
        self.publisher = None
        if settings.topic_rail_bus_publish_enabled:
            self.publisher = TopicRailBusPublisher(
                bus_url=settings.topic_rail_bus_url,
                service_name=settings.service_name,
                node_name=settings.node_name,
                service_version=settings.service_version,
            )
        self.model_loaded = False
        self.last_fit_at: Optional[str] = None
        self.last_assign_at: Optional[str] = None
        self.last_summary_at: Optional[str] = None
        self.last_drift_at: Optional[str] = None
        self.last_error: Optional[str] = None
        self.model_version = settings.topic_rail_model_version

    def run(self) -> None:
        mode = (settings.topic_rail_mode or "daemon").lower().strip()
        model_exists = self.model_store.exists(settings.topic_rail_model_version)
        logger.info(
            "Starting Topic Rail model_version=%s mode=%s run_once=%s model_exists=%s train_limit=%s assign_limit=%s",
            settings.topic_rail_model_version,
            mode,
            settings.topic_rail_run_once,
            model_exists,
            settings.topic_rail_train_limit,
            settings.topic_rail_assign_limit,
        )
        self.writer.ensure_tables_exist()
        self._start_http_server()

        if mode == "train":
            self._run_train_mode()
            return
        if mode == "backfill":
            self._run_backfill_mode()
            return

        while not self._stop:
            try:
                self._run_single_iteration()
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                logger.exception("Topic Rail loop error: %s", exc)
            if settings.topic_rail_run_once:
                logger.info("Run-once enabled; exiting after single iteration.")
                return
            self._sleep()

    def stop(self) -> None:
        self._stop = True

    def _sleep(self) -> None:
        for _ in range(settings.topic_rail_poll_seconds):
            if self._stop:
                return
            time.sleep(1)

    def _run_single_iteration(self) -> int:
        if not self.model_store.exists(settings.topic_rail_model_version):
            return self._train_and_assign()
        if self._should_refit():
            if settings.topic_rail_allow_refit_in_daemon:
                return self._train_and_assign()
            logger.warning("Refit required but not allowed in daemon mode.")
        return self._assign_only()

    def _run_train_mode(self) -> None:
        if settings.topic_rail_force_refit or not self.model_store.exists(settings.topic_rail_model_version):
            logger.info("Training mode: fitting model (force_refit=%s)", settings.topic_rail_force_refit)
            self._train_and_assign()
        else:
            logger.info("Training mode: model exists and force_refit=false; skipping training.")

    def _run_backfill_mode(self) -> None:
        if settings.topic_rail_force_refit or not self.model_store.exists(settings.topic_rail_model_version):
            logger.info("Backfill mode: ensuring model is trained (force_refit=%s)", settings.topic_rail_force_refit)
            self._train_and_assign()
        while not self._stop:
            written = self._assign_only()
            if written <= 0:
                logger.info("Backfill complete; no unassigned rows remain.")
                return

    def _train_and_assign(self) -> int:
        batch_start = time.monotonic()
        rows = self.reader.fetch_training_rows(
            limit=settings.topic_rail_train_limit,
            time_window_days=settings.topic_rail_train_time_window_days,
        )
        docs, row_map = self._build_docs(rows)
        if not docs:
            logger.info("No training docs found.")
            return 0

        embed_start = time.monotonic()
        embeddings = np.array(self.embedder.embed_texts(docs), dtype=np.float32)
        embed_secs = time.monotonic() - embed_start
        topic_model = build_topic_model(settings)
        transform_start = time.monotonic()
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        transform_secs = time.monotonic() - transform_start
        self.model_store.save(settings.topic_rail_model_version, topic_model, settings.settings_snapshot())
        self.model_loaded = True
        self.last_fit_at = datetime.now(timezone.utc).isoformat()
        self._write_manifest(topic_model, len(docs))

        assignments = self._build_assignments(topic_model, topics, probs, row_map)
        outlier_count, outlier_pct = self._outlier_stats(topics)
        self._warn_outliers(outlier_count, outlier_pct)
        written = self.writer.upsert_assignments(assignments)
        self._log_topic_summary(topic_model)
        self._maybe_run_summary(topic_model, outlier_count, outlier_pct)
        self._maybe_run_drift(topic_model)
        self.last_assign_at = datetime.now(timezone.utc).isoformat()
        self._log_metrics(
            assign_batch_docs=len(docs),
            assign_batch_secs=time.monotonic() - batch_start,
            embed_secs=embed_secs,
            model_transform_secs=transform_secs,
            outlier_pct=outlier_pct,
            topics=topics,
        )

        if settings.topic_rail_bus_publish_enabled and self.publisher:
            counts = Counter(topics)
            top_topic_ids = [topic_id for topic_id, _ in counts.most_common(3)]
            payload = {
                "model_version": settings.topic_rail_model_version,
                "node_name": settings.node_name,
                "doc_count": written,
                "outlier_pct": outlier_pct,
                "top_topic_ids": top_topic_ids,
                "created_at": datetime.now(timezone.utc),
            }
            asyncio.run(
                self.publisher.publish_assignment_batch(
                    settings.topic_rail_bus_topic_assigned_channel,
                    payload,
                )
            )

        logger.info("Training assignments written=%s", written)
        return written

    def _assign_only(self) -> int:
        batch_start = time.monotonic()
        rows = self.reader.fetch_unassigned_rows(
            model_version=settings.topic_rail_model_version,
            limit=settings.topic_rail_assign_limit,
        )
        docs, row_map = self._build_docs(rows)
        if not docs:
            logger.info("No unassigned docs found.")
            return 0

        topic_model, _, manifest = self.model_store.load(settings.topic_rail_model_version)
        self.model_loaded = True
        embed_start = time.monotonic()
        embeddings = np.array(self.embedder.embed_texts(docs), dtype=np.float32)
        embed_secs = time.monotonic() - embed_start
        self._validate_manifest(manifest)
        transform_start = time.monotonic()
        topics, probs = topic_model.transform(docs, embeddings=embeddings)
        transform_secs = time.monotonic() - transform_start
        assignments = self._build_assignments(topic_model, topics, probs, row_map)
        outlier_count, outlier_pct = self._outlier_stats(topics)
        self._warn_outliers(outlier_count, outlier_pct)
        written = self.writer.upsert_assignments(assignments)
        self._log_topic_summary(topic_model)
        self._maybe_run_summary(topic_model, outlier_count, outlier_pct)
        self._maybe_run_drift(topic_model)
        self.last_assign_at = datetime.now(timezone.utc).isoformat()
        self._log_metrics(
            assign_batch_docs=len(docs),
            assign_batch_secs=time.monotonic() - batch_start,
            embed_secs=embed_secs,
            model_transform_secs=transform_secs,
            outlier_pct=outlier_pct,
            topics=topics,
        )

        if settings.topic_rail_bus_publish_enabled and self.publisher:
            counts = Counter(topics)
            top_topic_ids = [topic_id for topic_id, _ in counts.most_common(3)]
            payload = {
                "model_version": settings.topic_rail_model_version,
                "node_name": settings.node_name,
                "doc_count": written,
                "outlier_pct": outlier_pct,
                "top_topic_ids": top_topic_ids,
                "created_at": datetime.now(timezone.utc),
            }
            asyncio.run(
                self.publisher.publish_assignment_batch(
                    settings.topic_rail_bus_topic_assigned_channel,
                    payload,
                )
            )

        logger.info("Assignment batch written=%s", written)
        return written

    def _build_docs(self, rows: Iterable[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        docs: List[str] = []
        row_map: List[Dict[str, Any]] = []
        for row in rows:
            text = self.doc_builder.build(row)
            if not text.strip():
                continue
            docs.append(text)
            row_map.append(row)
        return docs, row_map

    def _build_assignments(
        self,
        topic_model,
        topics: Iterable[int],
        probs: Optional[Any],
        row_map: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        assignments: List[Dict[str, Any]] = []
        for idx, topic_id in enumerate(topics):
            row = row_map[idx]
            keywords = self._topic_keywords(topic_model, topic_id)
            label = self._topic_label(topic_model, topic_id, keywords)
            confidence = self._topic_confidence(probs, idx)
            correlation_id = row.get("correlation_id")
            trace_id = row.get("trace_id") or correlation_id
            session_id = row.get("session_id") if settings.topic_rail_include_session else None

            assignments.append(
                {
                    "chat_id": row.get("id"),
                    "correlation_id": correlation_id,
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "topic_id": int(topic_id) if topic_id is not None else None,
                    "topic_label": label,
                    "topic_keywords": keywords,
                    "topic_confidence": confidence,
                    "model_version": settings.topic_rail_model_version,
                }
            )
        return assignments

    def _topic_keywords(self, topic_model, topic_id: int, top_n: int = 10) -> List[str]:
        words = topic_model.get_topic(topic_id) or []
        return [word for word, _ in words[:top_n]]

    def _topic_label(
        self,
        topic_model,
        topic_id: int,
        keywords: List[str],
    ) -> str:
        try:
            info = topic_model.get_topic_info()
            if "Topic" in info and "Name" in info:
                match = info[info["Topic"] == topic_id]
                if not match.empty:
                    return str(match.iloc[0]["Name"]) or ", ".join(keywords[:3])
        except Exception:  # noqa: BLE001
            pass
        return ", ".join(keywords[:3]) if keywords else f"topic-{topic_id}"

    def _topic_confidence(self, probs: Optional[Any], idx: int) -> Optional[float]:
        if probs is None:
            return None
        try:
            doc_probs = probs[idx]
            return float(max(doc_probs)) if doc_probs is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _log_topic_summary(self, topic_model) -> None:
        try:
            info = topic_model.get_topic_info()
            top = info.head(5).to_dict(orient="records")
            logger.info("Top topics: %s", top)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to summarize topics: %s", exc)

    def _maybe_run_summary(self, topic_model, outlier_count: int, outlier_pct: float) -> None:
        if not settings.topic_rail_summary_enabled:
            return
        window_end = datetime.now(timezone.utc)
        window_start = window_end - timedelta(minutes=settings.topic_rail_summary_window_minutes)
        counts_rows = self.reader.fetch_summary_counts(
            model_version=settings.topic_rail_model_version,
            window_start=window_start,
            window_end=window_end,
        )
        counts = [(row.get("topic_id"), row.get("doc_count")) for row in counts_rows]
        total_docs = sum(count for _, count in counts if count is not None)
        summary_rows = compute_summary_rows(
            model_version=settings.topic_rail_model_version,
            window_start=window_start,
            window_end=window_end,
            counts=counts,
            total_docs=total_docs,
            outlier_count=outlier_count,
            outlier_pct=outlier_pct,
            topic_labeler=self._topic_label,
            topic_keywords_fn=lambda topic_id: self._topic_keywords(topic_model, topic_id),
            max_topics=settings.topic_rail_summary_max_topics,
            min_docs=settings.topic_rail_summary_min_docs,
        )
        written = self.writer.upsert_summary_rows(summary_rows)
        logger.info("Summary rows written=%s window_start=%s", written, window_start.isoformat())
        self.last_summary_at = datetime.now(timezone.utc).isoformat()

        if settings.topic_rail_bus_publish_enabled and self.publisher and summary_rows:
            asyncio.run(
                self.publisher.publish_summary_rows(
                    settings.topic_rail_bus_topic_summary_channel,
                    summary_rows,
                )
            )

    def _maybe_run_drift(self, topic_model) -> None:
        if not settings.topic_rail_drift_enabled:
            return
        window_end = datetime.now(timezone.utc)
        window_start = window_end - timedelta(minutes=settings.topic_rail_drift_window_minutes)
        drift_rows = self.reader.fetch_drift_rows(
            model_version=settings.topic_rail_model_version,
            window_start=window_start,
            window_end=window_end,
        )
        rows = compute_drift_rows(
            model_version=settings.topic_rail_model_version,
            window_start=window_start,
            window_end=window_end,
            rows=drift_rows,
            min_turns=settings.topic_rail_drift_min_turns,
        )
        written = self.writer.upsert_drift_rows(rows)
        logger.info("Drift rows written=%s window_start=%s", written, window_start.isoformat())
        self.last_drift_at = datetime.now(timezone.utc).isoformat()

        if settings.topic_rail_bus_publish_enabled and self.publisher and rows:
            shift_rows = [
                row for row in rows
                if row.get("switch_rate", 0.0) >= settings.topic_rail_shift_switch_rate_threshold
            ]
            if shift_rows:
                asyncio.run(
                    self.publisher.publish_shift_rows(
                        settings.topic_rail_bus_topic_shift_channel,
                        shift_rows,
                    )
                )

    def _log_metrics(
        self,
        *,
        assign_batch_docs: int,
        assign_batch_secs: float,
        embed_secs: Optional[float],
        model_transform_secs: float,
        outlier_pct: float,
        topics: Iterable[int],
    ) -> None:
        counts = Counter(topics)
        top_topic_ids = [topic_id for topic_id, _ in counts.most_common(3)]
        embed_display = f"{embed_secs:.3f}" if embed_secs is not None else "n/a"
        logger.info(
            "metrics assign_batch_docs=%s assign_batch_secs=%.3f embed_secs=%s model_transform_secs=%.3f "
            "outlier_pct=%.3f top_topic_ids=%s",
            assign_batch_docs,
            assign_batch_secs,
            embed_display,
            model_transform_secs,
            outlier_pct,
            top_topic_ids,
        )

    def _outlier_stats(self, topics: Iterable[int]) -> Tuple[int, float]:
        if not settings.topic_rail_outlier_enabled:
            return 0, 0.0
        topic_list = list(topics)
        if not topic_list:
            return 0, 0.0
        outlier_count = sum(1 for topic in topic_list if int(topic) == -1)
        outlier_pct = outlier_count / max(1, len(topic_list))
        return outlier_count, outlier_pct

    def _warn_outliers(self, outlier_count: int, outlier_pct: float) -> None:
        if not settings.topic_rail_outlier_enabled:
            return
        if outlier_pct <= settings.topic_rail_outlier_max_pct:
            return
        logger.warning(
            "Outlier rate high outlier_count=%s outlier_pct=%.3f threshold=%.2f",
            outlier_count,
            outlier_pct,
            settings.topic_rail_outlier_max_pct,
        )
        if settings.topic_rail_bus_publish_enabled and self.publisher:
            asyncio.run(
                self.publisher.publish_warning(
                    f"Outlier rate high ({outlier_pct:.2%})",
                    details={
                        "outlier_count": outlier_count,
                        "outlier_pct": outlier_pct,
                        "model_version": settings.topic_rail_model_version,
                    },
                )
            )

    def _validate_manifest(self, manifest: Optional[Dict[str, Any]]) -> None:
        if not manifest:
            return
        embed_dim = manifest.get("embedding_dim")
        embed_model = manifest.get("embedding_model_name")
        embed_url = manifest.get("embedding_endpoint_url")
        if embed_dim is not None and self.embedder.embedding_dim is not None:
            if int(embed_dim) != int(self.embedder.embedding_dim):
                raise RuntimeError("Embedding dimension mismatch with manifest")
        if embed_model and self.embedder.embedding_model and embed_model != self.embedder.embedding_model:
            if not settings.topic_rail_allow_embed_model_mismatch:
                raise RuntimeError("Embedding model mismatch with manifest")
            logger.warning(
                "Embedding model mismatch manifest=%s current=%s",
                embed_model,
                self.embedder.embedding_model,
            )
        if embed_url and embed_url.rstrip("/") != settings.topic_rail_embedding_url.rstrip("/"):
            logger.warning(
                "Embedding endpoint mismatch manifest=%s current=%s",
                embed_url,
                settings.topic_rail_embedding_url,
            )

    def _write_manifest(self, topic_model, train_doc_count: int) -> None:
        bertopic_params = {}
        params = getattr(topic_model, "get_params", None)
        if callable(params):
            try:
                bertopic_params = params()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read BERTopic params for manifest: %s", exc)

        manifest = {
            "model_version": settings.topic_rail_model_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "train_doc_count": train_doc_count,
            "embedding_endpoint_url": settings.topic_rail_embedding_url.rstrip("/"),
            "embedding_model_name": self.embedder.embedding_model,
            "embedding_dim": self.embedder.embedding_dim,
            "bertopic_params": bertopic_params,
            "service_version": settings.service_version,
        }
        self.model_store.write_manifest(settings.topic_rail_model_version, manifest)

    def _should_refit(self) -> bool:
        manifest = self.model_store.load_manifest(settings.topic_rail_model_version)
        created_at = manifest.get("created_at") if manifest else None
        new_docs = 0
        if settings.topic_rail_refit_policy == "count" and created_at:
            new_docs = self.reader.count_rows_since(created_at)
        decision = decide_refit(
            policy=settings.topic_rail_refit_policy,
            force_refit=settings.topic_rail_force_refit,
            manifest_created_at=created_at,
            refit_ttl_hours=settings.topic_rail_refit_ttl_hours,
            refit_doc_threshold=settings.topic_rail_refit_doc_threshold,
            new_doc_count=new_docs,
        )
        if decision.should_refit:
            logger.info("Refit triggered reason=%s", decision.reason)
        return decision.should_refit

    def _start_http_server(self) -> None:
        if not settings.topic_rail_http_enabled:
            return
        thread = Thread(
            target=start_http_server,
            kwargs={"service": self, "host": "0.0.0.0", "port": settings.topic_rail_http_port},
            daemon=True,
        )
        thread.start()

def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    _configure_logging()
    lock_conn = acquire_lock(settings.topic_rail_pg_dsn, settings.topic_rail_model_version)
    if lock_conn is None:
        logger.warning("Lock not acquired; another instance active.")
        return

    service = TopicRailService()

    def _handle_signal(*_args: Any) -> None:
        service.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        service.run()
    finally:
        release_lock(lock_conn, settings.topic_rail_model_version)


if __name__ == "__main__":
    main()
