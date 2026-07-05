"""ROIManager — structural_fit(), motion_refine(), lock logic, drift monitor (REQ-14/15/16/18).

Two-phase design: Phase A (structural_fit) runs at dock time from edges/Hough on the loader
blob; Phase B (motion_refine) runs during MONITORING once confirmed tracks exist, and may
only shrink/translate within the Phase A plausibility bands (REQ-18), never grow or rotate
beyond +/-3 deg.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, replace
from typing import Callable

import cv2
import numpy as np

from config import RoiConfig
from roi.belt_detector import BeltDetector, BeltROI


@dataclass
class RoiFit:
    center: tuple[float, float]
    angle_deg: float  # long-axis angle from horizontal, 0-90
    length: float
    width: float


@dataclass
class LockedRoi:
    fit: RoiFit
    axis_unit: tuple[float, float]  # unit vector along the belt axis, pointing toward the aircraft
    aircraft_anchor: tuple[float, float]  # resolved aircraft-end anchor point (REQ-35 marker)
    polygon: np.ndarray  # 4x2 rotated-rect corners
    ground_anchor: tuple[float, float]  # the other axis endpoint (opposite aircraft_anchor)
    generation: int  # bumped on every new lock/reposition — invalidates any cached strip mapping

    def strip_transform(self, strip_h: int) -> tuple[np.ndarray, tuple[int, int]]:
        """Affine mapping this ROI band to an axis-aligned strip: ground end at x=0, aircraft
        end at x=strip_w, so +x in strip space always means "toward the aircraft" regardless of
        camera orientation (mirrors the reference implementation's BeltROI.strip_warp)."""
        strip_w = max(1, int(round(self.fit.length)))
        ax, ay = self.axis_unit
        n = np.array([-ay, ax], dtype=np.float32)  # unit normal to the axis
        halfwidth = self.fit.width / 2.0
        g = np.array(self.ground_anchor, dtype=np.float32)
        a = np.array(self.aircraft_anchor, dtype=np.float32)
        src = np.float32([g - n * halfwidth, a - n * halfwidth, g + n * halfwidth])
        dst = np.float32([[0, 0], [strip_w, 0], [0, strip_h]])
        return cv2.getAffineTransform(src, dst), (strip_w, strip_h)


class RoiManager:
    def __init__(self, cfg: RoiConfig, fps_nominal: float, aircraft_end: str, px_scale: float = 1.0,
                 camera_motion_fn: Callable[[], tuple[float, float]] | None = None):
        self.cfg = cfg
        self.fps_nominal = fps_nominal
        self.aircraft_end = aircraft_end
        self.px_scale = px_scale
        # REQ-10a: drains Stabilizer's accumulated (dx, dy) since the last drift check, so
        # camera shake between checks isn't mistaken for genuine ROI motion.
        self._camera_motion_fn = camera_motion_fn
        self._min_settle_frames = max(1, round(cfg.min_settle_s * fps_nominal))
        self._window_frames = max(1, round(cfg.window_s * fps_nominal))
        self._history: collections.deque[RoiFit] = collections.deque(maxlen=self._window_frames)
        # "Minimum settle time elapsed" (REQ) is elapsed time since settling began, tracked
        # independently of the windowed-stability deque above: window_s (used for the std
        # check) is deliberately shorter than min_settle_s, so the deque's fill level alone
        # can never reach min_settle_frames.
        self._frames_in_settling = 0
        self._rejected_fits: list[RoiFit] = []  # REQ-19: blacklisted hypotheses after health reset
        # EMA-smooth the raw per-frame Hough fit before it enters the stability window: a
        # single-frame rail-pair fit is as noisy as the boom-angle line fit (scene/dock.py),
        # for the same underlying reason (edge detection on a small, compressed frame).
        self._fit_ema: RoiFit | None = None
        self.locked: LockedRoi | None = None
        self._drift_check_frames = max(1, round(cfg.drift_check_s * fps_nominal))
        self._frames_since_check = 0
        self._drift_violations = 0
        self._belt_detector = BeltDetector(cfg)
        self._belt_prev: BeltROI | None = None
        self._lock_generation = 0

    def reset_settling(self, blacklist_current: bool = False) -> None:
        """Start a fresh settling attempt (new dock cycle, health reset, or reposition).

        blacklist_current is for REQ-19 specifically: when a *locked* ROI turns out wrong
        (health watchdog fires), exclude its parameters as a hypothesis on the next attempt.
        This is not applied to ordinary REQ-15 plausibility rejections during initial
        settling — those happen routinely (a rail-pair fit briefly clips a nearby edge) and
        the correct fit usually lands close to a rejected one, so blacklisting there would
        just keep excluding the right answer.
        """
        if blacklist_current and self.locked is not None:
            self._rejected_fits.append(self.locked.fit)
        self._history.clear()
        self._frames_in_settling = 0
        self._fit_ema = None
        self._belt_prev = None

    # ---- Phase A -----------------------------------------------------------------------

    def structural_fit(self, frame_bgr: np.ndarray, loader_bbox: tuple[int, int, int, int]) -> RoiFit | None:
        """Hypothesis-driven belt-axis fit (roi/belt_detector.py) cropped to a padded region
        around the presence blob's bbox, then converted to this module's RoiFit shape."""
        x, y, w, h = loader_bbox
        pad = int(0.3 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(frame_bgr.shape[1], x + w + pad), min(frame_bgr.shape[0], y + h + pad)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        prev_belt_roi = self._belt_prev
        belt_roi = self._belt_detector.detect_single(crop, prev_belt_roi)
        if belt_roi is None:
            return None
        self._belt_prev = belt_roi

        offset = np.array([x0, y0], dtype=np.float32)
        cx, cy = belt_roi.center + offset
        angle_deg = abs(belt_roi.angle_deg) % 180.0
        if angle_deg > 90.0:
            angle_deg = 180.0 - angle_deg

        pad_px = self.cfg.pad_px * self.px_scale
        return RoiFit(center=(float(cx), float(cy)), angle_deg=float(angle_deg),
                      length=float(belt_roi.length + 2 * pad_px),
                      width=float(2 * belt_roi.halfwidth + 2 * pad_px))

    # ---- Lock ---------------------------------------------------------------------------

    def _is_blacklisted(self, fit: RoiFit) -> bool:
        for rej in self._rejected_fits:
            if (abs(fit.center[0] - rej.center[0]) < self.cfg.center_std_px * self.px_scale * 2
                    and abs(fit.center[1] - rej.center[1]) < self.cfg.center_std_px * self.px_scale * 2
                    and abs(fit.angle_deg - rej.angle_deg) < self.cfg.angle_std_deg * 2):
                return True
        return False

    def _plausible(self, fit: RoiFit, loader_mask: np.ndarray, boom_angle_deg: float | None) -> bool:
        """REQ-15: reject a stable-but-implausible fit (ground marking, loader body edge)."""
        aspect = fit.length / fit.width if fit.width else 0.0
        lo, hi = self.cfg.aspect_band
        if not (lo <= aspect <= hi):
            return False
        wlo, whi = self.cfg.width_band_px
        w_scaled = fit.width / self.px_scale
        if not (wlo <= w_scaled <= whi):
            return False
        if boom_angle_deg is not None and abs(fit.angle_deg - boom_angle_deg) > self.cfg.angle_band_deg:
            return False

        # Overlap against the loader mask's *bounding box*, not the raw mask pixels: the color
        # mask only captures the saturated top surface of the canopy (REQ-11's single-band
        # cue), not the full mechanical structure, so it systematically undershoots the
        # physical footprint the rail fit is measured against. The bbox is a closer proxy for
        # "does this ROI sit on the loader" than the irregular color blob is.
        ys, xs = np.nonzero(loader_mask)
        if len(xs) == 0:
            return False
        bbox_mask = np.zeros(loader_mask.shape, dtype=np.uint8)
        bbox_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1

        poly = self._make_polygon(fit)
        roi_mask = np.zeros(loader_mask.shape, dtype=np.uint8)
        cv2.fillPoly(roi_mask, [poly.astype(np.int32)], 1)
        inter = np.logical_and(roi_mask.astype(bool), bbox_mask.astype(bool)).sum()
        roi_area = roi_mask.sum()
        if roi_area == 0 or inter / roi_area < self.cfg.overlap_min:
            return False
        return True

    def update(self, fit: RoiFit | None, loader_mask: np.ndarray | None = None,
               boom_angle_deg: float | None = None) -> str:
        """Feed a fresh structural_fit into the settling window. Returns an event name or ""."""
        if self.locked is not None:
            return self._update_drift(fit)

        if fit is None or self._is_blacklisted(fit):
            return ""

        if self._fit_ema is None:
            self._fit_ema = fit
        else:
            a = 0.15
            self._fit_ema = RoiFit(
                center=(a * fit.center[0] + (1 - a) * self._fit_ema.center[0],
                        a * fit.center[1] + (1 - a) * self._fit_ema.center[1]),
                angle_deg=a * fit.angle_deg + (1 - a) * self._fit_ema.angle_deg,
                length=a * fit.length + (1 - a) * self._fit_ema.length,
                width=a * fit.width + (1 - a) * self._fit_ema.width,
            )
        self._history.append(self._fit_ema)
        self._frames_in_settling += 1
        if self._frames_in_settling < self._min_settle_frames or len(self._history) < self._window_frames:
            return ""

        centers_x = np.array([f.center[0] for f in self._history])
        centers_y = np.array([f.center[1] for f in self._history])
        angles = np.array([f.angle_deg for f in self._history])
        lengths = np.array([f.length for f in self._history])

        stable = (centers_x.std() < self.cfg.center_std_px * self.px_scale and
                  centers_y.std() < self.cfg.center_std_px * self.px_scale and
                  angles.std() < self.cfg.angle_std_deg and
                  lengths.std() < self.cfg.length_std_px * self.px_scale)
        if not stable:
            return ""

        avg_fit = RoiFit(
            center=(float(centers_x.mean()), float(centers_y.mean())),
            angle_deg=float(angles.mean()),
            length=float(lengths.mean()),
            width=float(np.mean([f.width for f in self._history])),
        )

        if loader_mask is not None and not self._plausible(avg_fit, loader_mask, boom_angle_deg):
            # Stable but implausible: reject this hypothesis and keep settling (REQ-15). Not
            # blacklisted (see reset_settling docstring) — just clear the window and retry.
            self._history.clear()
            return ""

        self.locked = self._build_locked(avg_fit)
        self._frames_since_check = 0
        self._drift_violations = 0
        return "ROI_LOCKED"

    def _build_locked(self, fit: RoiFit) -> LockedRoi:
        theta = np.radians(fit.angle_deg)
        axis = np.array([np.cos(theta), -np.sin(theta)])  # image y grows downward
        # Orient axis to point toward the aircraft per the camera's convention.
        cx, cy = fit.center
        half_len = fit.length / 2
        end_a = (cx + axis[0] * half_len, cy + axis[1] * half_len)
        end_b = (cx - axis[0] * half_len, cy - axis[1] * half_len)
        anchor = self._pick_aircraft_end(end_a, end_b)
        ground = end_a if anchor is end_b else end_b
        if anchor is end_b:
            axis = -axis
        self._lock_generation += 1
        return LockedRoi(fit=fit, axis_unit=(float(axis[0]), float(axis[1])),
                          aircraft_anchor=anchor, polygon=self._make_polygon(fit),
                          ground_anchor=ground, generation=self._lock_generation)

    def _pick_aircraft_end(self, end_a: tuple[float, float], end_b: tuple[float, float]) -> tuple[float, float]:
        if self.aircraft_end == "max_x":
            return end_a if end_a[0] >= end_b[0] else end_b
        elif self.aircraft_end == "min_x":
            return end_a if end_a[0] <= end_b[0] else end_b
        elif self.aircraft_end == "max_y":
            return end_a if end_a[1] >= end_b[1] else end_b
        else:  # min_y
            return end_a if end_a[1] <= end_b[1] else end_b

    @staticmethod
    def _make_polygon(fit: RoiFit) -> np.ndarray:
        rect = (fit.center, (fit.length, fit.width), fit.angle_deg)
        return cv2.boxPoints(rect)

    # ---- Drift monitor (post-lock) -------------------------------------------------------

    def _update_drift(self, fit: RoiFit | None) -> str:
        self._frames_since_check += 1
        if self._frames_since_check < self._drift_check_frames:
            return ""
        self._frames_since_check = 0
        cam_dx, cam_dy = self._camera_motion_fn() if self._camera_motion_fn else (0.0, 0.0)
        if fit is None:
            return ""

        assert self.locked is not None
        locked_fit = self.locked.fit
        center_delta = np.hypot(fit.center[0] - locked_fit.center[0] - cam_dx,
                                 fit.center[1] - locked_fit.center[1] - cam_dy)
        angle_delta = abs(fit.angle_deg - locked_fit.angle_deg)
        length_delta = abs(fit.length - locked_fit.length)

        mult = self.cfg.drift_tolerance_mult
        deviated = (center_delta > mult * self.cfg.center_std_px * self.px_scale or
                    angle_delta > mult * self.cfg.angle_std_deg or
                    length_delta > mult * self.cfg.length_std_px * self.px_scale)
        if deviated:
            self._drift_violations += 1
        else:
            self._drift_violations = 0

        if self._drift_violations >= self.cfg.drift_consecutive:
            self.locked = None
            self.reset_settling()
            self._drift_violations = 0
            return "ROI_REPOSITIONED"
        return ""

    # ---- Phase B --------------------------------------------------------------------------

    def motion_refine(self, track_centroids: list[tuple[float, float]]) -> None:
        """REQ-18: may only shrink/translate within the plausibility bands, never grow/rotate
        beyond +/-3 deg from the locked fit."""
        if self.locked is None or len(track_centroids) < 10:
            return
        pts = np.array(track_centroids, dtype=np.float32)
        heat_center = pts.mean(axis=0)
        fit = self.locked.fit
        # Translate only toward the observed track centroid, clamped to the plausibility bands.
        max_shift = self.cfg.center_std_px * self.px_scale * 3
        dx = np.clip(heat_center[0] - fit.center[0], -max_shift, max_shift)
        dy = np.clip(heat_center[1] - fit.center[1], -max_shift, max_shift)
        new_length = min(fit.length, max(pts[:, 0].ptp(), pts[:, 1].ptp()) + 2 * self.cfg.pad_px * self.px_scale)
        new_fit = replace(fit, center=(fit.center[0] + dx, fit.center[1] + dy),
                           length=min(fit.length, new_length))
        # Center/length shifted, so the axis endpoints (ground/aircraft anchors) move with them —
        # generation bump invalidates any cached strip_transform() built from the stale anchors.
        ax, ay = self.locked.axis_unit
        half_len = new_fit.length / 2
        new_a = (new_fit.center[0] + ax * half_len, new_fit.center[1] + ay * half_len)
        new_g = (new_fit.center[0] - ax * half_len, new_fit.center[1] - ay * half_len)
        self.locked = replace(self.locked, fit=new_fit, polygon=self._make_polygon(new_fit),
                               aircraft_anchor=new_a, ground_anchor=new_g,
                               generation=self.locked.generation + 1)
