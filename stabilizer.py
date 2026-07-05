"""Global-motion guard against camera shake / PTZ nudges (REQ-09/10).

Not video stabilization — no frame warping. Maintains a few small high-texture reference
patches outside the loader blob/ROI and estimates per-frame global displacement via
cv2.phaseCorrelate (median across patches). Consumers (ROI drift monitor, dock stationarity
test) subtract this before deciding whether *they* saw motion.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import cv2
import numpy as np

from config import StabConfig

PATCH_SIZE = 48


@dataclass
class GlobalMotion:
    dx: float
    dy: float
    magnitude: float
    is_shaking: bool
    sustained: bool  # shaking has held for >= stab.shake_s


class Stabilizer:
    def __init__(self, cfg: StabConfig, fps_nominal: float):
        self.cfg = cfg
        self.fps_nominal = fps_nominal
        self._patches: list[tuple[int, int]] = []  # (x, y) top-left corners
        self._prev_crops: list[np.ndarray] | None = None
        self._window = cv2.createHanningWindow((PATCH_SIZE, PATCH_SIZE), cv2.CV_64F)
        maxlen = max(1, round(cfg.shake_s * fps_nominal))
        self._shake_history: collections.deque[bool] = collections.deque(maxlen=maxlen)
        self._cum_dx = 0.0
        self._cum_dy = 0.0

    def select_patches(self, gray: np.ndarray, exclude_mask: np.ndarray | None = None) -> None:
        """Auto-select high-texture patches outside exclude_mask (loader blob / ROI)."""
        h, w = gray.shape
        step = PATCH_SIZE // 2
        candidates: list[tuple[float, int, int]] = []
        for y in range(0, h - PATCH_SIZE, step):
            for x in range(0, w - PATCH_SIZE, step):
                if exclude_mask is not None:
                    region = exclude_mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                    if region.size and region.mean() > 0.05:
                        continue
                patch = gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                score = cv2.Laplacian(patch, cv2.CV_64F).var()
                candidates.append((score, x, y))
        candidates.sort(key=lambda c: -c[0])

        selected: list[tuple[int, int]] = []
        min_sep = PATCH_SIZE * 1.5
        for score, x, y in candidates:
            if all((x - sx) ** 2 + (y - sy) ** 2 > min_sep ** 2 for sx, sy in selected):
                selected.append((x, y))
            if len(selected) >= self.cfg.n_patches:
                break

        self._patches = selected
        self._prev_crops = [self._crop(gray, x, y) for x, y in selected]
        self._cum_dx = 0.0
        self._cum_dy = 0.0

    def _crop(self, gray: np.ndarray, x: int, y: int) -> np.ndarray:
        return gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE].astype(np.float64)

    def update(self, gray: np.ndarray) -> GlobalMotion:
        if not self._patches:
            # No patches selected yet (e.g. before BELT_PRESENT) -> assume no motion.
            self._shake_history.append(False)
            return GlobalMotion(0.0, 0.0, 0.0, False, False)

        dxs, dys = [], []
        new_crops = []
        for (x, y), prev_crop in zip(self._patches, self._prev_crops):
            curr_crop = self._crop(gray, x, y)
            (dx, dy), _response = cv2.phaseCorrelate(prev_crop, curr_crop, self._window)
            dxs.append(dx)
            dys.append(dy)
            new_crops.append(curr_crop)
        self._prev_crops = new_crops

        dx_med = float(np.median(dxs))
        dy_med = float(np.median(dys))
        magnitude = float(np.hypot(dx_med, dy_med))
        is_shaking = magnitude > self.cfg.shake_px

        self._cum_dx += dx_med
        self._cum_dy += dy_med

        self._shake_history.append(is_shaking)
        sustained = len(self._shake_history) == self._shake_history.maxlen and all(self._shake_history)

        return GlobalMotion(dx_med, dy_med, magnitude, is_shaking, sustained)

    def consume_cumulative(self) -> tuple[float, float]:
        """Return and reset accumulated displacement since patches were selected or last consumed.

        Used by the ROI drift monitor to subtract camera motion from its own periodic re-fit
        deltas (REQ-10a) before deciding a fresh fit represents genuine ROI motion.
        """
        dx, dy = self._cum_dx, self._cum_dy
        self._cum_dx = 0.0
        self._cum_dy = 0.0
        return dx, dy

    @property
    def has_patches(self) -> bool:
        return bool(self._patches)
