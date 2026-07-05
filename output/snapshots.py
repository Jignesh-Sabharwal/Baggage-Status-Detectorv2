"""Snapshot policy (throttled) + annotated frame writer (README §5)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class SnapshotWriter:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir) / "snapshots"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def save(self, annotated_bgr: np.ndarray, frame_idx: int, tag: str) -> None:
        self._count += 1
        path = self.dir / f"{frame_idx:07d}_{tag}.jpg"
        cv2.imwrite(str(path), annotated_bgr)

    @property
    def count(self) -> int:
        return self._count
