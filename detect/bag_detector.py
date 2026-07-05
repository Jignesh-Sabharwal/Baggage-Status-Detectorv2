"""Bag detector — operates on the rectified belt strip (ground/loader end at x=0, aircraft end
at x=strip_w), not the raw rotated frame, so every geometric filter measures along/across the
belt axis directly instead of being skewed by the belt's incline angle (REQ-24/25).

Order matters: margin zero first, then morphology, then contours. Each rejection is counted for
debug stats (REQ-23's ghost count plus the rest of this chain). Because the caller (main.py) warps
only the locked ROI band into the strip, every strip pixel is already inside the ROI by
construction — no polygon mask / point-in-polygon test is needed here (that's the one thing the
previous full-frame version had to do that this version doesn't).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import DetConfig


@dataclass
class Detection:
    centroid: tuple[float, float]  # strip coordinates
    bbox: tuple[int, int, int, int]  # strip coordinates (x, y, w, h)
    area: float
    solidity: float
    s_axis: float  # normalized axis coordinate, 0 = loader end, 1 = aircraft end


@dataclass
class RejectionStats:
    ghost: int = 0
    area: int = 0
    solidity: int = 0
    aspect: int = 0
    vest: int = 0
    person_rot: int = 0
    total_contours: int = 0


class _GhostTracker:
    """Lightweight position-only tracker used solely to time how long a raw blob has sat still
    (REQ-22), independent of the confirmed bag tracker in track/tracker.py."""

    def __init__(self, fps_nominal: float, ghost_s: float):
        self._fps = fps_nominal
        self._ghost_frames = max(1, round(ghost_s * fps_nominal))
        self._match_dist_px = 6.0
        self._blobs: list[dict] = []  # {centroid, still_frames}

    def update(self, centroids: list[tuple[float, float]]) -> list[int]:
        """Returns, per input centroid, the number of consecutive frames it's stayed still."""
        still_counts = [0] * len(centroids)
        used_prev = set()
        for i, c in enumerate(centroids):
            best_j, best_d = None, self._match_dist_px
            for j, blob in enumerate(self._blobs):
                if j in used_prev:
                    continue
                d = np.hypot(c[0] - blob["centroid"][0], c[1] - blob["centroid"][1])
                if d < best_d:
                    best_j, best_d = j, d
            if best_j is not None:
                used_prev.add(best_j)
                self._blobs[best_j]["still_frames"] += 1
                self._blobs[best_j]["centroid"] = c
                still_counts[i] = self._blobs[best_j]["still_frames"]
            else:
                self._blobs.append({"centroid": c, "still_frames": 1})
                still_counts[i] = 1
        # Drop unmatched blobs (they've moved on / disappeared).
        new_blobs = [b for i, b in enumerate(self._blobs) if i in used_prev]
        self._blobs = new_blobs
        return still_counts

    @property
    def ghost_frame_threshold(self) -> int:
        return self._ghost_frames


class BagDetector:
    def __init__(self, cfg: DetConfig, fps_nominal: float, bg_ghost_s: float, bg_ghost_edge_ratio: float,
                 px_scale: float = 1.0):
        self.cfg = cfg
        self.px_scale = px_scale
        self.bg_ghost_edge_ratio = bg_ghost_edge_ratio
        self._ghost_tracker = _GhostTracker(fps_nominal, bg_ghost_s)

    def _area_band(self, s: float) -> tuple[float, float]:
        far_scale = self.cfg.far_scale
        min_a = self.cfg.min_area_near * (1 + (far_scale - 1) * s) * (self.px_scale ** 2)
        max_a = self.cfg.max_area_near * (1 + (far_scale - 1) * s) * (self.px_scale ** 2)
        return min_a, max_a

    def detect(self, fg_mask: np.ndarray, strip_bgr: np.ndarray,
               gray: np.ndarray | None = None, bg_gray: np.ndarray | None = None
               ) -> tuple[list[Detection], RejectionStats]:
        stats = RejectionStats()
        strip_h, strip_w = fg_mask.shape[:2]
        margin = int(self.cfg.end_margin_frac * strip_w)

        masked = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        masked = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        masked[:, :margin] = 0                            # loader zones at both ends
        masked[:, strip_w - margin:] = 0

        contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stats.total_contours = len(contours)

        # First pass: compute raw centroids for every contour so the ghost tracker sees a
        # consistent population (including ones that will fail later geometric filters —
        # REQ-22 doesn't say ghosts only apply to plausible bag candidates).
        raw_centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] == 0:
                raw_centroids.append((0.0, 0.0))
                continue
            raw_centroids.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        still_counts = self._ghost_tracker.update(raw_centroids)

        hsv = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)
        vest_mask = cv2.inRange(hsv, np.array(self.cfg.vest_hsv_lo), np.array(self.cfg.vest_hsv_hi))

        detections = []
        for c, centroid, still in zip(contours, raw_centroids, still_counts):
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)

            # Ghost filter (REQ-22): zero displacement over ghost_s AND low current-frame edge
            # density vs the same region in the background image.
            if (still >= self._ghost_tracker.ghost_frame_threshold and gray is not None and bg_gray is not None):
                pad = 3
                y0, y1 = max(0, y - pad), min(gray.shape[0], y + h + pad)
                x0, x1 = max(0, x - pad), min(gray.shape[1], x + w + pad)
                cur_edges = cv2.Canny(gray[y0:y1, x0:x1], 50, 150)
                bg_edges = cv2.Canny(bg_gray[y0:y1, x0:x1], 50, 150)
                cur_density = cur_edges.mean() / 255.0
                bg_density = bg_edges.mean() / 255.0
                if bg_density > 0 and cur_density < self.bg_ghost_edge_ratio * bg_density:
                    stats.ghost += 1
                    continue

            s_axis = float(np.clip(centroid[0] / strip_w, 0.0, 1.0))

            min_a, max_a = self._area_band(s_axis)
            if not (min_a <= area <= max_a):
                stats.area += 1
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < self.cfg.solidity_min:
                stats.solidity += 1
                continue

            aspect = max(w, h) / max(1, min(w, h))
            if aspect > self.cfg.aspect_hi or aspect < self.cfg.aspect_lo:
                stats.aspect += 1
                continue

            # Full-strip-height blob spanning most of the belt's cross-axis width, taller than
            # wide, is a standing person, not luggage riding flat on the belt.
            if h > self.cfg.bag_max_height_frac * strip_h and h > w:
                stats.aspect += 1
                continue

            # Rotated-rect angle test: a markedly elongated blob oriented near-vertical (across
            # the belt, not along it) is a leaning/standing person, not luggage in transit.
            (_, _), (rw, rh), rangle = cv2.minAreaRect(c)
            if rw > 0 and rh > 0:
                r_aspect = max(rw, rh) / min(rw, rh)
                if rw < rh:
                    rangle += 90
                if r_aspect > self.cfg.person_rot_aspect and 60 < (rangle % 180) < 120:
                    stats.person_rot += 1
                    continue

            # Hi-vis vest color: a blob whose foreground is mostly vest-colored is a worker
            # leaning over the rails, not a bag.
            roi_vest = vest_mask[y:y + h, x:x + w].astype(bool)
            roi_fg = masked[y:y + h, x:x + w] > 0
            blob_px = max(1, int(np.count_nonzero(roi_fg)))
            if np.count_nonzero(roi_vest & roi_fg) / blob_px > self.cfg.vest_reject_frac:
                stats.vest += 1
                continue

            detections.append(Detection(centroid=centroid, bbox=(x, y, w, h), area=area,
                                         solidity=solidity, s_axis=s_axis))

        return detections, stats
