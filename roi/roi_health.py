"""ROI health watchdog — zero-detection watchdog forces a re-settle on a wrong lock (REQ-19).

motion_refine() needs tracks to exist; a wrong lock produces zero tracks; zero tracks means
the wrong lock is never corrected. This breaks that loop: if there's clearly foreground
activity just outside the ROI while the ROI itself sees nothing, the lock is presumed wrong.
"""
from __future__ import annotations

import collections

import cv2
import numpy as np

from config import HealthConfig


class RoiHealthWatchdog:
    def __init__(self, cfg: HealthConfig, fps_nominal: float, px_scale: float = 1.0):
        self.cfg = cfg
        self.px_scale = px_scale
        maxlen = max(1, round(cfg.sustain_s * fps_nominal))
        self._history: collections.deque[bool] = collections.deque(maxlen=maxlen)
        self._maxlen = maxlen

    def _annulus_mask(self, polygon: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        roi_mask = np.zeros(frame_shape, dtype=np.uint8)
        cv2.fillPoly(roi_mask, [polygon.astype(np.int32)], 1)
        annulus_px = max(1, int(round(self.cfg.annulus_px * self.px_scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * annulus_px + 1, 2 * annulus_px + 1))
        dilated = cv2.dilate(roi_mask, kernel)
        return (dilated.astype(bool) & ~roi_mask.astype(bool))

    def update(self, fg_mask: np.ndarray, polygon: np.ndarray, activity_is_idle: bool,
               n_confirmed_tracks: int) -> bool:
        """Returns True exactly once when ROI_HEALTH_RESET should fire."""
        if not activity_is_idle:
            self._history.clear()
            return False

        annulus = self._annulus_mask(polygon, fg_mask.shape)
        annulus_area = annulus.sum()
        motion_frac = (fg_mask.astype(bool) & annulus).sum() / annulus_area if annulus_area else 0.0

        suspicious = (motion_frac > self.cfg.motion_frac) and (n_confirmed_tracks == 0)
        self._history.append(suspicious)

        if len(self._history) == self._maxlen and all(self._history):
            self._history.clear()
            return True
        return False
