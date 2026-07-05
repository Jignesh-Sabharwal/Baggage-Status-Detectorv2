import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

CASES = {
    "D02_f9574": ((0, 0, 0), (179, 90, 90)),
    "N01_f4526": ((0, 3, 76), (179, 60, 216)),
    "N04_f5086": ((0, 0, 75), (179, 56, 205)),
}

for name, (lo, hi) in CASES.items():
    img = cv2.imread(f"debug_frames/{name}.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    overlay = img.copy()
    overlay[mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    cv2.imwrite(f"debug_frames/MASK_{name}.jpg", blended)
    print(name, "done")
