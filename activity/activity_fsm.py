"""Activity FSM (fast, operational) — IDLE/LOADING/UNLOADING, M-of-N with defined semantics
(REQ-29). Ticks only while Scene FSM is in MONITORING (enforced by the caller, not here).
"""
from __future__ import annotations

import collections
from enum import Enum, auto

from config import FsmConfig


class ActivityState(Enum):
    IDLE = auto()
    LOADING = auto()
    UNLOADING = auto()


class ActivityFSM:
    def __init__(self, cfg: FsmConfig, fps_nominal: float):
        self.cfg = cfg
        self.state = ActivityState.IDLE
        # REQ-29: the debounce window is the last N *eligible* frames, not the last N frames
        # outright — a frames-based window would flush itself empty between bags during
        # sparse traffic and never accumulate M agreements.
        self._window: collections.deque[int] = collections.deque(maxlen=cfg.m_of_n_n)
        self._idle_timeout_frames = max(1, round(cfg.idle_timeout_s * fps_nominal))
        self._no_track_frames = 0

    def update(self, vote: int | None, n_confirmed_tracks: int) -> str | None:
        if n_confirmed_tracks == 0:
            self._no_track_frames += 1
            if self._no_track_frames >= self._idle_timeout_frames and self.state != ActivityState.IDLE:
                self.state = ActivityState.IDLE
                self._window.clear()
                return "IDLE_STARTED"
            return None
        self._no_track_frames = 0

        if vote is None:
            return None
        self._window.append(vote)

        n_loading = sum(1 for v in self._window if v > 0)
        n_unloading = sum(1 for v in self._window if v < 0)
        m = self.cfg.m_of_n_m

        if self.state == ActivityState.IDLE:
            if n_loading >= m:
                self.state = ActivityState.LOADING
                return "LOADING_STARTED"
            if n_unloading >= m:
                self.state = ActivityState.UNLOADING
                return "UNLOADING_STARTED"
        elif self.state == ActivityState.LOADING:
            if n_unloading >= m:
                self.state = ActivityState.UNLOADING
                return "UNLOADING_STARTED"
        elif self.state == ActivityState.UNLOADING:
            if n_loading >= m:
                self.state = ActivityState.LOADING
                return "LOADING_STARTED"
        return None
