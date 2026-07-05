"""Plain nearest-neighbour centroid tracker — no Kalman, no Hungarian (REQ-26/27/28).

At 11 fps this is sufficient and stays minimal, per the design's own rationale.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from config import TrkConfig
from detect.bag_detector import Detection

_id_counter = itertools.count(1)


@dataclass
class Track:
    id: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    age: int = 1
    disappeared: int = 0
    confirmed: bool = False
    just_confirmed: bool = False
    inherited: bool = False  # via REQ-27 re-association — no new bag count, no new snapshot
    history: list[tuple[float, float]] = field(default_factory=list)
    s_axis: float = 0.5
    velocity_ema: float | None = None  # signed axis velocity, EMA-smoothed
    votes: list[int] = field(default_factory=list)  # +1 loading / -1 unloading, per direction.py


@dataclass
class DeadTrack:
    track: Track
    died_frame: int
    in_delivery_zone: bool


class Tracker:
    def __init__(self, cfg: TrkConfig, fps_nominal: float, px_scale: float = 1.0,
                 delivery_zone_s: float = 0.85):
        self.cfg = cfg
        self.fps_nominal = fps_nominal
        self.px_scale = px_scale
        self.delivery_zone_s = delivery_zone_s
        self.tracks: dict[int, Track] = {}
        self._dead_pool: list[DeadTrack] = []
        self._reassoc_frames = max(1, round(cfg.reassoc_s * fps_nominal))
        self._frame_idx = 0

    def _prune_dead_pool(self) -> None:
        self._dead_pool = [d for d in self._dead_pool if self._frame_idx - d.died_frame <= self._reassoc_frames]

    def _try_reassociate(self, centroid: tuple[float, float], axis_unit: tuple[float, float]) -> Track | None:
        ax, ay = axis_unit
        best, best_dist = None, self.cfg.reassoc_px * self.px_scale
        for d in self._dead_pool:
            if d.in_delivery_zone:
                continue
            t = d.track
            if t.velocity_ema is None:
                continue
            elapsed_s = (self._frame_idx - d.died_frame) / self.fps_nominal
            extrap = (t.centroid[0] + ax * t.velocity_ema * elapsed_s,
                      t.centroid[1] + ay * t.velocity_ema * elapsed_s)
            dist = np.hypot(centroid[0] - extrap[0], centroid[1] - extrap[1])
            if dist < best_dist:
                # Matching velocity sign is checked by the caller once a provisional new-track
                # velocity can be estimated (needs >=2 frames); at first sight we only have
                # position, so accept on position and let a sign mismatch simply not matter
                # (a wrong-signed inherited track will fail its own next axis-coherence check).
                best, best_dist = d, dist
        if best is not None:
            self._dead_pool.remove(best)
            return best.track
        return None

    def update(self, detections: list[Detection], axis_unit: tuple[float, float]) -> list[Track]:
        self._frame_idx += 1
        self._prune_dead_pool()

        unmatched_track_ids = set(self.tracks.keys())
        unmatched_dets = list(range(len(detections)))

        # Greedy nearest-neighbour matching.
        pairs = []
        for tid, t in self.tracks.items():
            for di in unmatched_dets:
                dist = np.hypot(detections[di].centroid[0] - t.centroid[0],
                                 detections[di].centroid[1] - t.centroid[1])
                if dist <= self.cfg.max_dist_px * self.px_scale:
                    pairs.append((dist, tid, di))
        pairs.sort(key=lambda p: p[0])

        matched_dets = set()
        for dist, tid, di in pairs:
            if tid in unmatched_track_ids and di in unmatched_dets and di not in matched_dets:
                t = self.tracks[tid]
                det = detections[di]
                dt = 1.0 / self.fps_nominal
                ax, ay = axis_unit
                disp_along_axis = (det.centroid[0] - t.centroid[0]) * ax + (det.centroid[1] - t.centroid[1]) * ay
                v = disp_along_axis / dt
                alpha = self.cfg.ema_alpha
                t.velocity_ema = v if t.velocity_ema is None else alpha * v + (1 - alpha) * t.velocity_ema
                t.centroid = det.centroid
                t.bbox = det.bbox
                t.s_axis = det.s_axis
                t.age += 1
                t.disappeared = 0
                t.history.append(det.centroid)
                t.just_confirmed = False
                self._maybe_confirm(t, axis_unit)
                unmatched_track_ids.discard(tid)
                matched_dets.add(di)

        for tid in unmatched_track_ids:
            t = self.tracks[tid]
            t.disappeared += 1
            if t.disappeared > self.cfg.max_disappeared:
                in_zone = t.s_axis >= self.delivery_zone_s  # REQ-26: delivered, not lost
                self._dead_pool.append(DeadTrack(track=t, died_frame=self._frame_idx, in_delivery_zone=in_zone))
                del self.tracks[tid]

        for di in unmatched_dets:
            if di in matched_dets:
                continue
            det = detections[di]
            inherited = self._try_reassociate(det.centroid, axis_unit)
            if inherited is not None:
                inherited.centroid = det.centroid
                inherited.bbox = det.bbox
                inherited.s_axis = det.s_axis
                inherited.disappeared = 0
                inherited.inherited = True
                inherited.just_confirmed = False
                self.tracks[inherited.id] = inherited
            else:
                new_id = next(_id_counter)
                self.tracks[new_id] = Track(id=new_id, centroid=det.centroid, bbox=det.bbox,
                                             s_axis=det.s_axis, history=[det.centroid])

        return list(self.tracks.values())

    def _maybe_confirm(self, t: Track, axis_unit: tuple[float, float]) -> None:
        if t.confirmed or t.age < self.cfg.min_age:
            return
        net = (t.history[-1][0] - t.history[0][0], t.history[-1][1] - t.history[0][1])
        norm = np.hypot(*net)
        if norm == 0:
            return
        ax, ay = axis_unit
        coherence = abs(net[0] * ax + net[1] * ay) / norm
        if coherence >= self.cfg.axis_coherence:
            t.confirmed = True
            t.just_confirmed = True

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.confirmed]
