"""Frame iterator; yields (frame_idx, time_base_result, frame)."""
from __future__ import annotations

from collections.abc import Iterator

import cv2

from time_base import TimeBase, TimeBaseResult


class VideoSource:
    def __init__(self, path: str, fps_nominal: float):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        self.time_base = TimeBase(fps_nominal)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[int, TimeBaseResult, "cv2.Mat"]]:
        frame_idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            tb = self.time_base.next(frame_idx, pos_msec)
            yield frame_idx, tb, frame
            frame_idx += 1

    def release(self) -> None:
        self.cap.release()
