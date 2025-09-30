
from typing import List, Tuple, Optional, Set
import numpy as np

from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path: str = "yolov8n.pt", classes: Optional[Set[str]] = None, conf: float = 0.25, device: Optional[str] = None):
        # device: "0" or "cuda:0" for GPU, "cpu" otherwise
        self.model = YOLO(model_path)
        self.classes = set(c.lower() for c in classes) if classes else None
        self.conf = conf
        self.device = device

        # Warmup (helps allocate on the right device)
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, conf=self.conf, device=self.device, verbose=False)

        # Cache class name mapping
        # Depending on ultralytics version, names may be at model.names or model.model.names
        self.names = getattr(self.model, 'names', None) or getattr(self.model.model, 'names', {})
        if isinstance(self.names, list):
            self.names = {i:n for i,n in enumerate(self.names)}

    def detect(self, frame) -> List[Tuple[int,int,int,int,str,float]]:
        # Returns list of (x,y,w,h, name, conf)
        res = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)
        out = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None:
            return out

        for b in r.boxes:
            cls_id = int(b.cls.item())
            name = self.names.get(cls_id, str(cls_id))
            if self.classes and name.lower() not in self.classes:
                continue
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            w, h = x2 - x1, y2 - y1
            conf = float(b.conf.item())
            out.append((x1, y1, w, h, name, conf))
        return out
