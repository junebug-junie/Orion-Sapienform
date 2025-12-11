
import cv2, threading, time
from typing import Optional
from .settings import get_settings

settings = get_settings()


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
        self.cap = cv2.VideoCapture(self.source)
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
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                self.last_frame = frame
            else:
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
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
