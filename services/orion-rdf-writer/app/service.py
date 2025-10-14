import httpx, json, logging, threading, time
from typing import List
from app.settings import settings
from app.rdf_builder import build_triples
from orion.core.bus import OrionBus

logger = logging.getLogger(settings.SERVICE_NAME)

import httpx, json, logging, threading, time, os
from typing import List
from datetime import datetime
from app.settings import settings
from app.rdf_builder import build_triples
from orion.core.bus.service import OrionBus

logger = logging.getLogger(settings.SERVICE_NAME)


class OrionRDFWriterService:
    """
    Orion RDF Writer:
    Listens to tagged, triaged, and RDF enqueue events,
    builds RDF triples, and pushes them into GraphDB.
    """

    def __init__(self):
        self.bus = OrionBus(settings.ORION_BUS_URL)
        self.queue: List[dict] = []
        self.running = True

    def start(self):
        logger.info(f"🟢 Starting RDF Writer → bus {settings.ORION_BUS_URL}")

        # Define all inbound channels this service listens to
        channels = [
            settings.CHANNEL_EVENTS_TRIAGE,     # from collapse-mirror
            settings.CHANNEL_EVENTS_TAGGED,     # from tag-service
            settings.CHANNEL_RDF_ENQUEUE,       # from enrichment pipelines
            settings.ORION_CORE_EVENTS,         # system-level events
        ]

        logger.info(f"👂 Subscribing to channels: {', '.join(channels)}")

        # Spin up listener threads per channel
        for ch in channels:
            t = threading.Thread(target=self._subscribe_loop, args=(ch,), daemon=True)
            t.start()

        # Start batch flusher
        threading.Thread(target=self._batch_flush_loop, daemon=True).start()
        logger.info(f"🚀 [{settings.SERVICE_NAME}] ready and listening")

    def _subscribe_loop(self, channel: str):
        logger.info(f"👂 Subscribing to {channel}")
        for event in self.bus.subscribe(channel):
            logger.debug(f"📥 {channel}: {event}")

            # Optional routing filter: only keep RDF-targeted events from core bus
            if channel == settings.ORION_CORE_EVENTS:
                if "targets" in event and "rdf" not in event["targets"]:
                    continue

            self.queue.append(event)

    def _batch_flush_loop(self):
        while self.running:
            if len(self.queue) >= settings.BATCH_SIZE:
                batch = [self.queue.pop(0) for _ in range(min(len(self.queue), settings.BATCH_SIZE))]
                self._process_batch(batch)
            time.sleep(1)

    def _process_batch(self, batch):
        logger.info(f"📦 Processing batch of {len(batch)} RDF events")

        for event in batch:
            try:
                nt_data, graph_name = build_triples(event)
                self._push_to_graphdb(nt_data, graph_name, event)
            except Exception as e:
                logger.exception("process_batch failed")
                self._publish_error(event, str(e))

    def _push_to_graphdb(self, nt_data: str, graph_name: str, event: dict):
        url = f"{settings.GRAPHDB_URL}/repositories/{settings.GRAPHDB_REPO}/statements?context=<{graph_name}>"
        headers = {"Content-Type": "application/n-triples"}
        for attempt in range(settings.RETRY_LIMIT):
            try:
                with httpx.Client(timeout=10) as client:
                    res = client.post(url, content=nt_data, headers=headers)
                if res.status_code in (200, 204):
                    logger.info(f"✅ RDF inserted ({event.get('id')}) → {graph_name}")
                    self._publish_confirm(event, graph_name)
                    return
                else:
                    logger.warning(f"⚠️ Insert failed ({res.status_code}): {res.text}")
            except Exception as e:
                logger.error(f"❌ GraphDB connection error: {e}")
            time.sleep(settings.RETRY_INTERVAL)
        self._publish_error(event, f"Failed after {settings.RETRY_LIMIT} attempts")

    def _publish_confirm(self, event: dict, graph_name: str):
        self.bus.publish(settings.CHANNEL_RDF_CONFIRM, {
            "event_id": event.get("id"),
            "graph": graph_name,
            "status": "success",
        })

    def _publish_error(self, event: dict, error_msg: str):
        log_file = "/app/logs/errors.txt"
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "service": settings.SERVICE_NAME,
                "error": error_msg,
                "failed_event": event,
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to error log file: {e}")

        self.bus.publish(settings.CHANNEL_RDF_ERROR, {
            "event_id": event.get("id"),
            "status": "error",
            "error": error_msg,
        })
