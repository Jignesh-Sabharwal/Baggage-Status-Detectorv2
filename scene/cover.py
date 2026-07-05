"""CoverClassifier — covered vs open belt (REQ-20). Texture, not area, not background
subtraction: a static canopy is invisible to MOG2, so this must be an appearance test.

Runs at ROI lock (exempt from gating — no tracks can exist yet) and every cover.recheck_s,
gated on Activity FSM = IDLE and zero confirmed tracks (bags on the belt would inflate the
edge-pixel ratio and make the classifier flap mid-session).
"""
from __future__ import annotations

import cv2
import numpy as np

from config import CoverConfig


class CoverClassifier:
    def __init__(self, cfg: CoverConfig, fps_nominal: float):
        self.cfg = cfg
        self.fps_nominal = fps_nominal
        self._recheck_frames = max(1, round(cfg.recheck_s * fps_nominal))
        self._frames_since_check = 0
        self._covered: bool | None = None
        self._checked_once = False

    def _edge_density(self, gray: np.ndarray, polygon: np.ndarray) -> float:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        edges = cv2.Canny(gray, 50, 150)
        area = mask.sum()
        if area == 0:
            return 0.0
        return float((edges.astype(bool) & mask.astype(bool)).sum() / area)

    def update(self, gray: np.ndarray, polygon: np.ndarray, activity_is_idle: bool,
               n_confirmed_tracks: int) -> str:
        due = not self._checked_once or (
            activity_is_idle and n_confirmed_tracks == 0 and
            self._frames_since_check >= self._recheck_frames)
        self._frames_since_check += 1
        if not due:
            return ""

        self._checked_once = True
        self._frames_since_check = 0
        covered = self._edge_density(gray, polygon) < self.cfg.edge_density_threshold

        if covered != self._covered:
            self._covered = covered
            return "BELT_COVERED" if covered else "BELT_UNCOVERED"
        return ""
