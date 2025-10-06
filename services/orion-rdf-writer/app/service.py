import httpx, json, logging, threading, time
from typing import List
from app.settings import settings
from app.rdf_builder import build_triples
from orion.core.bus import OrionBus

logger = logging.getLogger(settings.SERVICE_NAME)

class OrionRDFWriterService:
    def __init__(self):
        self.bus = OrionBus(settings.ORION_BUS_URL)
        self.queue: List[dict] = []
        self.running = True

    def start(self):
        logger.info(f"🟢 start() → bus {settings.ORION_BUS_URL}")
        channels = [
            settings.CHANNEL_EVENTS_TAGGED,
            settings.CHANNEL_RDF_ENQUEUE,
            settings.CHANNEL_CORE_EVENTS,
        ]
        logger.info(f"🟢 subscribing on: {', '.join(channels)}")

        for ch in channels:
            t = threading.Thread(target=self._subscribe_loop, args=(ch,), daemon=True)
            t.start()

        threading.Thread(target=self._batch_flush_loop, daemon=True).start()
        logger.info(f"🚀 [{settings.SERVICE_NAME}] ready")

    def _subscribe_loop(self, channel: str):
        logger.info(f"👂 Subscribing to {channel}")
        for event in self.bus.subscribe(channel):
            logger.debug(f"📥 {channel}: {event}")
            if channel == settings.CHANNEL_CORE_EVENTS:
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

        print(f"!!! PROCESSING BATCH of {len(batch)} items: {batch}", flush=True)

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

        LOG_FILE_PATH = "/app/logs/errors.txt"
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

            # Create a detailed log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "service": settings.SERVICE_NAME,
                "error": error_msg,
                "failed_event": event
            }

            # Append the entry as a JSON line to the log file
            with open(LOG_FILE_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Failed to write to error log file: {e}")

        # The original Redis publishing logic remains the same
        self.bus.publish(settings.CHANNEL_RDF_ERROR, {
            "event_id": event.get("id"),
            "status": "error",
            "error": error_msg,
        })
