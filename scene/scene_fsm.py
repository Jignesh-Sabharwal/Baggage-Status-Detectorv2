"""Scene FSM (slow, structural): INIT -> NO_BELT -> BELT_PRESENT -> DOCKED -> ROI_SETTLING ->
MONITORING, with teardown transitions back through earlier states (README §1, §3.5).

REQ-02: bag detector/tracker/Activity FSM exist only in MONITORING (enforced by callers
checking `state == MONITORING`, not by this module).
REQ-03: background model is instantiated at BELT_PRESENT entry, independent of this FSM's
state — callers are responsible for that split (see detect/bg_model.py docstring).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from config import Config
from roi.roi_health import RoiHealthWatchdog
from roi.roi_manager import RoiManager
from scene.cover import CoverClassifier
from scene.dock import DockDetector
from scene.presence import PresenceResult, ScenePresence


class SceneState(Enum):
    INIT = auto()
    NO_BELT = auto()
    BELT_PRESENT = auto()
    DOCKED = auto()
    ROI_SETTLING = auto()
    MONITORING = auto()


@dataclass
class SceneUpdate:
    state: SceneState
    presence: PresenceResult
    dock_docked: bool
    events: list[str] = field(default_factory=list)


class SceneFSM:
    def __init__(self, cfg: Config, frame_shape: tuple[int, int],
                 camera_motion_fn: Callable[[], tuple[float, float]] | None = None):
        self.cfg = cfg
        px_scale = cfg.px_scale(frame_shape[0])
        self.presence = ScenePresence(cfg.presence, cfg.fps_nominal, frame_shape)
        self.dock = DockDetector(cfg.dock, cfg.fps_nominal, frame_shape, px_scale, cfg.camera.aircraft_end)
        self.roi = RoiManager(cfg.roi, cfg.fps_nominal, cfg.camera.aircraft_end, px_scale,
                               camera_motion_fn=camera_motion_fn)
        self.health = RoiHealthWatchdog(cfg.health, cfg.fps_nominal, px_scale)
        self.cover = CoverClassifier(cfg.cover, cfg.fps_nominal)
        self.state = SceneState.NO_BELT

        # Teardown hysteresis: require the "gone" condition to hold for dock.window_s before
        # actually tearing down, matching the spec's own "sustained" language for undock/
        # departure triggers (§3.5) rather than flipping on a single noisy frame.
        self._undock_frames = max(1, round(cfg.dock.window_s * cfg.fps_nominal))
        self._departed_frames = max(1, round(cfg.presence.debounce_s * cfg.fps_nominal))
        self._not_docked_streak = 0
        self._not_present_streak = 0
        self.t_last_track: float | None = None

    def init_classify(self, frames: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """REQ-04/05: classify the opening window into whichever state already holds, using
        the same tests as the normal path over init.window_s, then jump directly there."""
        last_pres = None
        last_dock = None
        for frame_bgr, gray in frames:
            last_pres = self.presence.update(frame_bgr)
            last_dock = self.dock.update(last_pres.candidate, gray)
        if last_dock is not None and last_dock.docked:
            self.state = SceneState.DOCKED
        elif last_pres is not None and last_pres.present:
            self.state = SceneState.BELT_PRESENT
        else:
            self.state = SceneState.NO_BELT

    def update(self, frame_bgr: np.ndarray, gray: np.ndarray, global_motion=None,
               n_confirmed_tracks: int = 0, activity_is_idle: bool = True,
               fg_mask: np.ndarray | None = None, track_centroids: list[tuple[float, float]] | None = None
               ) -> SceneUpdate:
        events: list[str] = []
        pres = self.presence.update(frame_bgr)
        dr = self.dock.update(pres.candidate, gray, global_motion)

        if self.state == SceneState.NO_BELT:
            if pres.present:
                self.state = SceneState.BELT_PRESENT
                events.append("BELT_ARRIVED")
                self._not_present_streak = 0

        elif self.state == SceneState.BELT_PRESENT:
            if not pres.present:
                self._not_present_streak += 1
                if self._not_present_streak >= self._departed_frames:
                    self.state = SceneState.NO_BELT
                    events.append("BELT_DEPARTED")
            else:
                self._not_present_streak = 0
            if dr.docked:
                self.state = SceneState.DOCKED
                events.append("BELT_DOCKED")
                self._not_docked_streak = 0

        elif self.state == SceneState.DOCKED:
            # REQ-05: enters ROI_SETTLING immediately — this state is a pass-through, not a
            # waiting period. Settling/locking itself happens in ROI_SETTLING below.
            self.state = SceneState.ROI_SETTLING
            self._not_docked_streak = 0

        elif self.state == SceneState.ROI_SETTLING:
            # Presence/dock monitoring keep running during SETTLING (REQ-17).
            if not dr.docked:
                self._not_docked_streak += 1
                if self._not_docked_streak >= self._undock_frames:
                    self.state = SceneState.BELT_PRESENT
                    events.append("BELT_UNDOCKED")
                    if self.t_last_track is not None:
                        events.append("SESSION_CLOSE_AT_LAST_TRACK")
                    self.roi.locked = None
                    self.roi.reset_settling()
            else:
                self._not_docked_streak = 0
                fit = self.roi.structural_fit(frame_bgr, pres.candidate.bbox) if pres.candidate else None
                loader_mask = pres.candidate.mask if pres.candidate else None
                ev = self.roi.update(fit, loader_mask, dr.boom_angle_deg)
                if ev == "ROI_LOCKED":
                    self.state = SceneState.MONITORING
                    events.append(ev)
                    events.append("MONITORING_ENTERED")

        elif self.state == SceneState.MONITORING:
            if not dr.docked:
                self._not_docked_streak += 1
                if self._not_docked_streak >= self._undock_frames:
                    self.state = SceneState.BELT_PRESENT
                    events.append("BELT_UNDOCKED")
                    if self.t_last_track is not None:
                        events.append("SESSION_CLOSE_AT_LAST_TRACK")
                    self.roi.locked = None
                    self.roi.reset_settling()
            else:
                self._not_docked_streak = 0
                fit = self.roi.structural_fit(frame_bgr, pres.candidate.bbox) if pres.candidate else None
                ev = self.roi.update(fit)
                if ev == "ROI_REPOSITIONED":
                    self.state = SceneState.ROI_SETTLING
                    events.append(ev)
                elif self.roi.locked is not None:
                    if track_centroids:
                        self.roi.motion_refine(track_centroids)
                    if fg_mask is not None:
                        reset = self.health.update(fg_mask, self.roi.locked.polygon, activity_is_idle, n_confirmed_tracks)
                        if reset:
                            events.append("ROI_HEALTH_RESET")
                            self.roi.reset_settling(blacklist_current=True)
                            self.roi.locked = None
                            self.state = SceneState.ROI_SETTLING
                    if self.state == SceneState.MONITORING:
                        cover_ev = self.cover.update(gray, self.roi.locked.polygon, activity_is_idle, n_confirmed_tracks)
                        if cover_ev:
                            events.append(cover_ev)

        return SceneUpdate(state=self.state, presence=pres, dock_docked=dr.docked, events=events)
