"""DockDetector — BELT_PRESENT -> DOCKED (REQ-13).

Three signatures must all hold: (1) stationary blob, (2) boom raised (dominant long-line
angle from Hough), (3) aircraft-adjacency gate (bright, low-edge-density region near the
boom's elevated end). At low resolution the fuselage test is coarse by design (REQ-13) —
it is a generous gate, not a precision measurement.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import cv2
import numpy as np

from config import DockConfig
from scene.presence import PresenceCandidate


@dataclass
class DockResult:
    stationary: bool
    boom_angle_deg: float | None
    boom_raised: bool
    fuselage_gate_pass: bool
    docked: bool
    elevated_point: tuple[float, float] | None  # approx boom elevated endpoint, image coords


class DockDetector:
    def __init__(self, cfg: DockConfig, fps_nominal: float, frame_shape: tuple[int, int],
                 px_scale: float = 1.0, aircraft_end: str = "min_y"):
        self.cfg = cfg
        self._frame_h, self._frame_w = frame_shape
        self._px_scale = px_scale
        self._aircraft_end = aircraft_end
        maxlen = max(1, round(cfg.window_s * fps_nominal))
        self._centroid_hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._centroid_ema: tuple[float, float] | None = None
        # Per-frame Hough angle estimates are noisy at this resolution (README REQ-13 already
        # flags the fuselage test as coarse; the same is true of the boom-angle line fit) —
        # smooth over a short window before thresholding, same window used for the
        # stationarity test.
        self._angle_hist: collections.deque[float] = collections.deque(maxlen=maxlen)
        # The instantaneous fuselage edge-density test flickers frame to frame from
        # compression/lighting noise; require a short-window majority rather than a single
        # frame, same rationale as the boom-angle smoothing above.
        short_len = max(1, round(1.0 * fps_nominal))
        self._fuselage_hist: collections.deque[bool] = collections.deque(maxlen=short_len)
        # The boom endpoint used by the fuselage-distance gate is even noisier frame to frame
        # than the angle scalar (single Hough segment pick) — smooth its position too.
        self._elevated_point_ema: tuple[float, float] | None = None

    def _stationary(self, candidate: PresenceCandidate | None, global_motion) -> bool:
        if candidate is None:
            self._centroid_hist.clear()
            self._centroid_ema = None
            return False
        gx, gy = 0.0, 0.0
        if global_motion is not None:
            gx, gy = global_motion.dx, global_motion.dy
        # Subtract global (camera) motion so we track object-relative displacement (REQ-09).
        corrected = (candidate.centroid[0] - gx, candidate.centroid[1] - gy)
        # EMA-smooth the raw color-blob centroid: the blob boundary itself flickers a few
        # percent frame to frame (glare/compression), which otherwise dominates a 2-3px drift
        # tolerance well before the loader has genuinely moved.
        if self._centroid_ema is None:
            self._centroid_ema = corrected
        else:
            a = 0.3
            self._centroid_ema = (a * corrected[0] + (1 - a) * self._centroid_ema[0],
                                   a * corrected[1] + (1 - a) * self._centroid_ema[1])
        self._centroid_hist.append(self._centroid_ema)
        if len(self._centroid_hist) < self._maxlen:
            return False
        xs = np.array([c[0] for c in self._centroid_hist])
        ys = np.array([c[1] for c in self._centroid_hist])
        # Robust M-of-N variant of the drift test rather than max-spread: at this resolution
        # the color-blob boundary itself shifts with passing workers/shadow even when the
        # loader is genuinely parked, producing occasional outlier frames within an otherwise
        # steady window. Require most (not all) frames to sit within tolerance of the window's
        # median, matching the M-of-N tolerance-for-noise pattern used elsewhere (presence
        # debounce, Activity FSM).
        mx, my = np.median(xs), np.median(ys)
        within = (np.abs(xs - mx) < self.cfg.max_drift_px * self._px_scale) & \
                 (np.abs(ys - my) < self.cfg.max_drift_px * self._px_scale)
        return within.mean() >= 0.8

    def _boom_angle(self, gray: np.ndarray, candidate: PresenceCandidate) -> tuple[float | None, tuple[float, float] | None]:
        x, y, w, h = candidate.bbox
        pad = int(0.15 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            return None, None

        edges = cv2.Canny(roi, 50, 150)
        min_len = 0.3 * max(w, h)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                 minLineLength=max(5, min_len), maxLineGap=5)
        if lines is None:
            return None, None
        lines = lines.reshape(-1, 4)

        # Dominant long-line angle, not simply the single longest line: the chassis sits
        # near-horizontal on the ground even when the boom is raised, and is often the
        # longest edge in frame, so picking "longest wins" would systematically undercount
        # boom elevation. Instead cluster candidate lines >= 5 deg from horizontal (i.e. not
        # the chassis/ground) into 5-degree bins weighted by length, and take the bin with
        # the most total length as the boom's angle — mirroring the angle-clustering approach
        # ROI Phase A uses for the rail pair.
        entries = []  # (angle, length, endpoints)
        for line in lines:
            lx0, ly0, lx1, ly1 = line
            length = np.hypot(lx1 - lx0, ly1 - ly0)
            angle = np.degrees(np.arctan2(abs(ly1 - ly0), abs(lx1 - lx0)))  # 0=horizontal, 90=vertical
            if angle < 5.0:
                continue
            entries.append((angle, length, ((lx0 + x0, ly0 + y0), (lx1 + x0, ly1 + y0))))

        if not entries:
            return None, None

        bin_width = 5.0
        bins: dict[int, float] = {}
        for angle, length, _ in entries:
            b = int(angle // bin_width)
            bins[b] = bins.get(b, 0.0) + length
        best_bin = max(bins, key=bins.get)

        in_bin = [(angle, length, ep) for angle, length, ep in entries if int(angle // bin_width) == best_bin]
        total_len = sum(length for _, length, _ in in_bin)
        dominant_angle = sum(angle * length for angle, length, _ in in_bin) / total_len
        # Elevated endpoint: from the single longest line within the dominant cluster, the
        # endpoint with smaller image y (higher up in frame).
        _, _, endpoints = max(in_bin, key=lambda e: e[1])
        p0, p1 = endpoints
        elevated = p0 if p0[1] < p1[1] else p1
        return dominant_angle, elevated

    def _blob_elevated_point(self, candidate: PresenceCandidate) -> tuple[float, float]:
        """Aircraft-adjacent extreme point of the presence blob, not a Hough line endpoint.

        The single-longest-Hough-segment endpoint jumps tens of pixels frame to frame at this
        resolution (a different edge wins "longest" almost every frame). An extreme point of
        the blob mask is a much more stable proxy since it's an aggregate property of the
        whole blob, not a single noisy line fit. Which extreme point (top/bottom/left/right)
        corresponds to "toward the aircraft" is a camera-framing fact — provisionally read
        from config.camera.aircraft_end (formalized properly once ROI Phase A's axis fit
        exists in M2; min_y here doubles as "this camera's aircraft side is up/right" for the
        cameras surveyed so far, all of which frame the aircraft in the upper-right).
        """
        ys, xs = np.nonzero(candidate.mask)
        convention = self._aircraft_end
        if convention == "max_x":
            edge = xs.max()
            sel = ys[xs >= edge - 2]
            return float(edge), float(sel.mean())
        elif convention == "min_x":
            edge = xs.min()
            sel = ys[xs <= edge + 2]
            return float(edge), float(sel.mean())
        elif convention == "max_y":
            edge = ys.max()
            sel = xs[ys >= edge - 2]
            return float(sel.mean()), float(edge)
        else:  # min_y
            edge = ys.min()
            sel = xs[ys <= edge + 2]
            return float(sel.mean()), float(edge)

    @staticmethod
    def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        """Axis-aligned gap between two (x, y, w, h) boxes; 0 if they overlap."""
        ax0, ay0, aw, ah = a
        ax1, ay1 = ax0 + aw, ay0 + ah
        bx0, by0, bw, bh = b
        bx1, by1 = bx0 + bw, by0 + bh
        dx = max(ax0 - bx1, bx0 - ax1, 0)
        dy = max(ay0 - by1, by0 - ay1, 0)
        return float(np.hypot(dx, dy))

    def _fuselage_gate(self, gray: np.ndarray, loader_bbox: tuple[int, int, int, int] | None) -> bool:
        if loader_bbox is None:
            return False

        _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.boxFilter((edges > 0).astype(np.float32), -1, (15, 15))
        smooth_bright = ((bright > 0) & (edge_density < 0.15)).astype(np.uint8)
        smooth_bright = cv2.morphologyEx(smooth_bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(smooth_bright, connectivity=8)
        frame_area = self._frame_h * self._frame_w
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.cfg.min_fuselage_frac * frame_area:
                continue
            region_bbox = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                           stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            # Bounding-box gap rather than a single point-to-contour distance: at this
            # resolution any single "elevated point" derived from the loader blob is noisy
            # (see _blob_elevated_point docstring), whereas the loader's overall bbox position
            # relative to the fuselage-proxy region is stable frame to frame. This is the
            # generous gate REQ-13 calls for, not a precision measurement.
            gap = self._bbox_gap(loader_bbox, region_bbox)
            if gap <= self.cfg.fuselage_dist_px * self._px_scale:
                return True
            if self.cfg.fuselage_gate == "weak":
                # 3-weak: fuselage-proxy region exists anywhere in the boom-side half of frame.
                return True
        return False

    def update(self, candidate: PresenceCandidate | None, gray: np.ndarray, global_motion=None) -> DockResult:
        stationary = self._stationary(candidate, global_motion)

        raw_angle = None
        elevated_point = None
        if candidate is not None:
            raw_angle, _ = self._boom_angle(gray, candidate)
            elevated_point = self._blob_elevated_point(candidate)
        if raw_angle is not None:
            self._angle_hist.append(raw_angle)
        elif candidate is None:
            self._angle_hist.clear()

        if elevated_point is not None:
            if self._elevated_point_ema is None:
                self._elevated_point_ema = elevated_point
            else:
                a = 0.3
                self._elevated_point_ema = (a * elevated_point[0] + (1 - a) * self._elevated_point_ema[0],
                                             a * elevated_point[1] + (1 - a) * self._elevated_point_ema[1])
        elif candidate is None:
            self._elevated_point_ema = None

        boom_angle = float(np.median(self._angle_hist)) if self._angle_hist else None
        boom_raised = boom_angle is not None and boom_angle >= self.cfg.min_boom_angle_deg

        if boom_raised:
            self._fuselage_hist.append(self._fuselage_gate(gray, candidate.bbox if candidate else None))
        else:
            self._fuselage_hist.clear()
        fuselage_pass = bool(self._fuselage_hist) and (sum(self._fuselage_hist) / len(self._fuselage_hist)) >= 0.5

        docked = stationary and boom_raised and fuselage_pass
        return DockResult(stationary, boom_angle, boom_raised, fuselage_pass, docked, self._elevated_point_ema)
