"""PTS-first timestamping with frame_idx/fps fallback (REQ-07/08).

frame_idx alone assumes constant frame rate; DVR exports are frequently variable-frame-rate
with dropped frames, and OpenCV's CAP_PROP_POS_MSEC is itself unreliable on VFR files through
the ffmpeg backend. So: read PTS every frame, validate monotonicity + plausible inter-frame
delta, and permanently fall back to frame_idx/fps on the first violation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimeBaseResult:
    t_video: float
    used_fallback: bool
    fallback_just_triggered: bool


class TimeBase:
    def __init__(self, fps_nominal: float):
        self.fps_nominal = fps_nominal
        self._lo = 0.2 / fps_nominal
        self._hi = 5.0 / fps_nominal
        self._fallback_active = False
        self._last_t: float | None = None

    def next(self, frame_idx: int, pos_msec: float) -> TimeBaseResult:
        """Call once per frame, immediately after the frame grab."""
        fallback_just_triggered = False

        if not self._fallback_active:
            t_candidate = pos_msec / 1000.0
            if self._last_t is None:
                # First frame: accept whatever PTS reports as the origin.
                valid = True
            else:
                delta = t_candidate - self._last_t
                valid = (delta > 0) and (self._lo <= delta <= self._hi)

            if valid:
                self._last_t = t_candidate
                return TimeBaseResult(t_candidate, used_fallback=False, fallback_just_triggered=False)
            else:
                self._fallback_active = True
                fallback_just_triggered = True

        t_video = frame_idx / self.fps_nominal
        self._last_t = t_video
        return TimeBaseResult(t_video, used_fallback=True, fallback_just_triggered=fallback_just_triggered)
