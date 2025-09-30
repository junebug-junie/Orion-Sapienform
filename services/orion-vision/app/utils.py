
import cv2, time
from typing import List, Tuple
from .settings import settings

def draw_boxes(frame, boxes: List[Tuple[int,int,int,int]], label: str):
    for (x,y,w,h) in boxes:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

def encode_jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY])
    return buf.tobytes() if ok else None

def now_ts():
    return int(time.time()*1000)
