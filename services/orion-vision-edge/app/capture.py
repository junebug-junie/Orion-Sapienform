import cv2, threading, time
import logging
from typing import Optional
from .settings import get_settings

settings = get_settings()
logger = logging.getLogger("orion-vision-edge.capture_source")

class CameraSource:
    def __init__(self, source: str):
        self.source = source
        self.cap = None
        self.last_frame = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.width = settings.WIDTH
        self.height = settings.HEIGHT
        self.fps = settings.FPS

    def start(self):
        logger.info(f"Opening camera source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera source: {self.source}")
        else:
            logger.info("Camera source opened successfully.")
            
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        interval = 1.0 / max(self.fps, 1)
        retry_count = 0
        while not self._stop.is_set():
            if not self.cap or not self.cap.isOpened():
                time.sleep(1)
                continue
                
            ok, frame = self.cap.read()
            if ok:
                self.last_frame = frame
                if retry_count > 0:
                    logger.info("Camera stream recovered.")
                    retry_count = 0
            else:
                retry_count += 1
                if retry_count % 20 == 1: # Log occasionally
                    logger.warning(f"Failed to read frame from camera (attempt {retry_count}).")
                time.sleep(0.25)
            
            # regulate loop speed roughly
            time.sleep(interval * 0.5)

    def get_frame(self):
        return self.last_frame

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            try:
                self.cap.release()
                logger.info("Camera released.")
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")
