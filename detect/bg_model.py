"""MOG2 background model wrapper: shadows, per-state learning rate, burst guard, belt-pause
freeze + ghost filter (REQ-03, REQ-21/22/23).

The *model* is instantiated at BELT_PRESENT entry and runs continuously from there, so it is
fully warmed up by the time MONITORING begins (REQ-03) — its output is simply ignored by
consumers outside MONITORING. Callers are responsible for creating this once at BELT_PRESENT
entry and calling `apply()` every frame from then on, regardless of Scene FSM state.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import cv2
import numpy as np

from config import BgConfig

SHADOW_VALUE = 127


def _cuda_available(cfg: BgConfig) -> bool:
    try:
        return cfg.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


@dataclass
class ForegroundResult:
    mask: np.ndarray  # binary, shadows already dropped
    burst: bool
    learning_rate: float


class BgModel:
    def __init__(self, cfg: BgConfig, fps_nominal: float):
        self.cfg = cfg
        self.fps_nominal = fps_nominal
        self._gpu = _cuda_available(cfg)
        if self._gpu:
            self.mog2 = cv2.cuda.createBackgroundSubtractorMOG2(history=cfg.history, detectShadows=True)
            self._stream = cv2.cuda.Stream()
        else:
            self.mog2 = cv2.createBackgroundSubtractorMOG2(history=cfg.history, detectShadows=True)
        self._burst_frames_remaining = 0
        self._burst_guard_duration = max(1, round(1.0 * fps_nominal))
        pause_frames = max(1, round(cfg.pause_s * fps_nominal))
        self._speed_history: collections.deque[float] = collections.deque(maxlen=pause_frames)

    def _learning_rate(self, roi_mean_speed: float | None, frozen: bool) -> float:
        if frozen:
            # ROI_SETTLING re-entry: don't burn ghost trails from a repositioned belt.
            return 0.0
        if self._burst_frames_remaining > 0:
            return 0.2  # raised learning rate to recover quickly from the burst
        if roi_mean_speed is not None:
            self._speed_history.append(roi_mean_speed)
            if (len(self._speed_history) == self._speed_history.maxlen and
                    all(s < self.cfg.pause_speed for s in self._speed_history)):
                return 0.0  # REQ-21 belt-pause freeze
        return -1.0  # auto

    def apply(self, frame_bgr: np.ndarray, roi_mask: np.ndarray | None = None,
              roi_mean_speed: float | None = None, frozen: bool = False) -> ForegroundResult:
        lr = self._learning_rate(roi_mean_speed, frozen)
        if self._gpu:
            g = cv2.cuda_GpuMat()
            g.upload(frame_bgr, self._stream)
            fg = self.mog2.apply(g, lr, self._stream).download(stream=self._stream)
            self._stream.waitForCompletion()
        else:
            fg = self.mog2.apply(frame_bgr, learningRate=lr)
        fg = np.where(fg == SHADOW_VALUE, 0, fg).astype(np.uint8)

        burst = False
        if roi_mask is not None:
            roi_area = roi_mask.sum()
            if roi_area > 0:
                fg_ratio = (fg.astype(bool) & roi_mask.astype(bool)).sum() / roi_area
                if fg_ratio > self.cfg.burst_frac:
                    burst = True
                    self._burst_frames_remaining = self._burst_guard_duration

        if self._burst_frames_remaining > 0:
            self._burst_frames_remaining -= 1

        return ForegroundResult(mask=fg, burst=burst, learning_rate=lr)

    def get_background_image(self) -> np.ndarray:
        if self._gpu:
            g = self.mog2.getBackgroundImage(self._stream)
            self._stream.waitForCompletion()
            return g.download()
        return self.mog2.getBackgroundImage()

    def trigger_burst_guard(self) -> None:
        """External trigger (e.g. REQ-10's sustained camera-motion path shares this guard)."""
        self._burst_frames_remaining = self._burst_guard_duration
