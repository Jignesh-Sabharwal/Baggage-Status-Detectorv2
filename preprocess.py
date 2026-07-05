"""Brightness-triggered CLAHE for low-light clips.

Applied globally to every frame before any detector sees it. Day clips (mean brightness at or
above night_threshold) pass through untouched, so this only pays the contrast-enhancement cost
on genuinely dark footage (e.g. N01/N04, which failed presence/dock detection in RESULTS.md).
"""
from __future__ import annotations

import cv2
import numpy as np

from config import PreprocessConfig


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(cfg.clahe_clip_limit, (cfg.clahe_tile_grid, cfg.clahe_tile_grid))

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.cfg.enabled:
            return frame_bgr
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) >= self.cfg.night_threshold:
            return frame_bgr
        l_chan, a_chan, b_chan = cv2.split(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB))
        l_chan = self._clahe.apply(l_chan)
        return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
