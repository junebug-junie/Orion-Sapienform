
import cv2
from typing import List, Tuple
import numpy as np

class MotionDetector:
    def __init__(self, min_area: int = 2000):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.min_area = min_area

    def detect(self, frame) -> List[Tuple[int, int, int, int]]:
        # Expect BGR frame
        fgmask = self.bg.apply(frame)
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgmask = cv2.dilate(fgmask, None, iterations=2)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x,y,w,h = cv2.boundingRect(c)
            boxes.append((x,y,w,h))
        return boxes
