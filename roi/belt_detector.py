"""Hypothesis-driven belt-axis detector (replaces the minAreaRect-over-all-Hough-points
approach in roi_manager.structural_fit, which was pulling in canopy/chassis edges and
producing an oversized ROI).

Strategy: yellow rail-color evidence + long inclined Hough segments, fused with a Huber line
fit. Critically, only *inclined* segments may propose a candidate axis (ground paint and the
chassis are near-horizontal and structurally cannot win), while yellow evidence only verifies
an already-inclined proposal — so however much ground paint or horizontal clutter is in frame,
it can neither create nor promote a false candidate. This is the "hypothesis-driven belt
detection" the design doc's REQ-15 gestures at, made properly hypothesis-vs-verification
instead of best-single-fit-then-plausibility-check.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import RoiConfig


@dataclass
class BeltROI:
    p_ground: np.ndarray      # axis endpoint at ground/feed end (x, y)
    p_hold: np.ndarray        # axis endpoint at aircraft-hold end (x, y) — smaller image y
    halfwidth: float          # band half-width in px

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p_hold - self.p_ground))

    @property
    def angle_deg(self) -> float:
        d = self.p_hold - self.p_ground
        return float(np.degrees(np.arctan2(d[1], d[0])))

    @property
    def center(self) -> np.ndarray:
        return (self.p_ground + self.p_hold) / 2.0


class BeltDetector:
    def __init__(self, cfg: RoiConfig):
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(2.0, (8, 8))
        self._prev_gray: np.ndarray | None = None

    def _rail_color_evidence(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array(self.cfg.rail_hsv_lo), np.array(self.cfg.rail_hsv_hi))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 25:
                continue
            elong = max(w, h) / max(1, min(w, h))
            if elong >= 2.0 and w >= 0.8 * h:
                keep[lbl == i] = 255
        return keep

    def _line_segments(self, bgr: np.ndarray) -> list[tuple]:
        """Long, moderately inclined straight segments (rail/canopy geometry).

        Parallel-line filter: conveyor belts and canopies always show up as a *pair* of
        parallel edges (top/bottom rail, or canopy top/underside), so an isolated segment
        without an angle-matched partner elsewhere in frame is more likely a lone fuselage/
        engine-cowling curve that happened to pass the incline gate. Such segments are
        dropped unless they're exceptionally long, in which case they're kept as a
        hypothesis seed on their own merit.
        """
        g = self._clahe.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        edges = cv2.Canny(g, 60, 150)
        h, w = g.shape
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                                 minLineLength=int(0.22 * w), maxLineGap=8)
        if lines is None:
            return []
        raw_segs = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            a = abs(ang)
            a = min(a, 180.0 - a)
            if 3.0 < a < 55.0:
                raw_segs.append((x1, y1, x2, y2, float(np.hypot(x2 - x1, y2 - y1)), ang))

        segs = []
        for i, s1 in enumerate(raw_segs):
            has_parallel = any(
                min(abs(s1[5] - s2[5]), 180.0 - abs(s1[5] - s2[5])) <= 7.0
                for j, s2 in enumerate(raw_segs) if j != i
            )
            if has_parallel or s1[4] > 0.35 * w:
                segs.append(s1)
        return segs

    def _motion_evidence(self, bgr: np.ndarray) -> np.ndarray | None:
        """Moving pixels vs. the previous call's frame — only meaningful while bags move, so
        this is a booster on top of color/geometry evidence, never load-bearing on its own."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        motion_pts = None
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            pts = cv2.findNonZero(thresh)
            if pts is not None:
                motion_pts = pts.reshape(-1, 2).astype(np.float32)
                if len(motion_pts) > 500:
                    # Deterministic stride subsample, not random.choice: this pipeline's whole
                    # validation methodology (RESULTS.md, tools/m6_report.py) assumes re-running
                    # the same clip/config reproduces the same lock/session timings.
                    stride = len(motion_pts) // 500
                    motion_pts = motion_pts[::stride][:500]
        self._prev_gray = gray
        return motion_pts

    def _fit_axis(self, pts: np.ndarray, frame_w: int) -> BeltROI | None:
        if pts is None or len(pts) < self.cfg.roi_min_points:
            return None
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        u = np.array([vx, vy], dtype=np.float32)
        c = np.array([x0, y0], dtype=np.float32)
        t = (pts - c) @ u
        lo, hi = np.percentile(t, 3), np.percentile(t, 97)
        if hi - lo < 0.15 * frame_w:
            return None
        e1, e2 = c + lo * u, c + hi * u
        p_hold, p_ground = (e1, e2) if e1[1] < e2[1] else (e2, e1)
        halfwidth = self.cfg.roi_halfwidth_frac * float(hi - lo)
        roi = BeltROI(p_ground=p_ground, p_hold=p_hold, halfwidth=halfwidth)
        a = abs(roi.angle_deg)
        a = min(a, 180.0 - a)
        if not (3.0 <= a <= 55.0):
            return None
        return roi

    @staticmethod
    def _near_line(pts: np.ndarray | None, p1, p2, max_dist: float) -> np.ndarray:
        if pts is None or len(pts) == 0:
            return np.zeros(0, dtype=bool)
        d = np.array(p2, dtype=np.float32) - np.array(p1, dtype=np.float32)
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        s = (pts - np.array(p1, dtype=np.float32)) @ n
        return np.abs(s) <= max_dist

    def detect_single(self, bgr: np.ndarray, previous: BeltROI | None = None) -> BeltROI | None:
        yellow = self._rail_color_evidence(bgr)
        py = cv2.findNonZero(yellow)
        py = py.reshape(-1, 2).astype(np.float32) if py is not None else None
        motion_pts = self._motion_evidence(bgr)
        segs = self._line_segments(bgr)
        h, w = bgr.shape[:2]
        band = self.cfg.roi_hypo_band_frac * w

        best, best_score = None, -1e9
        for sx1, sy1, sx2, sy2, slen, sang in segs:
            p1, p2 = (sx1, sy1), (sx2, sy2)
            support_px, member_pts = 0.0, []
            for x1, y1, x2, y2, L, ang in segs:
                d_ang = abs(ang - sang)
                d_ang = min(d_ang, 180.0 - d_ang)
                mid = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], np.float32)
                if d_ang <= 6.0 and self._near_line(mid, p1, p2, band)[0]:
                    support_px += L
                    n_samples = max(2, int(L / 4))
                    ts = np.linspace(0, 1, n_samples, dtype=np.float32)[:, None]
                    member_pts.append(np.array([x1, y1], np.float32)
                                       + ts * np.array([x2 - x1, y2 - y1], np.float32))
            yellow_px = 0
            if py is not None:
                near = self._near_line(py, p1, p2, band)
                yellow_px = int(np.count_nonzero(near))
            motion_px = 0
            if motion_pts is not None:
                near_m = self._near_line(motion_pts, p1, p2, 1.5 * band)
                motion_px = int(np.count_nonzero(near_m))
            score = (support_px + 0.8 * min(yellow_px, 2 * w) + 0.6 * min(motion_px, 2 * w)
                     - 0.15 * ((sy1 + sy2) / 2.0 / h) * w)
            if previous is not None:
                d_prev = abs(sang - previous.angle_deg)
                d_prev = min(d_prev, 180.0 - d_prev)
                mid_prev = previous.center[None, :].astype(np.float32)
                if d_prev <= 8.0 and self._near_line(mid_prev, p1, p2, 2.0 * band)[0]:
                    score += 0.5 * w
            if score <= best_score:
                continue
            pts = member_pts
            if py is not None and yellow_px > 0:
                pts = member_pts + [py[self._near_line(py, p1, p2, band)]]
            if motion_pts is not None and motion_px > 0:
                pts = pts + [motion_pts[self._near_line(motion_pts, p1, p2, 1.5 * band)]]
            if not pts:
                continue
            roi = self._fit_axis(np.concatenate(pts), w)
            if roi is None:
                continue
            best, best_score = roi, score

        if best is None:
            return None

        seg_samples = []
        for x1, y1, x2, y2, L, ang in segs:
            n_samples = max(2, int(L / 4))
            ts = np.linspace(0, 1, n_samples, dtype=np.float32)[:, None]
            seg_samples.append(np.array([x1, y1], np.float32)
                                + ts * np.array([x2 - x1, y2 - y1], np.float32))
        all_pts = [np.concatenate(seg_samples)] if seg_samples else []
        if py is not None:
            all_pts.append(py)
        if all_pts:
            ap = np.concatenate(all_pts)
            near = self._near_line(ap, best.p_ground, best.p_hold, 2.5 * band)
            ap = ap[near]
            if len(ap) >= self.cfg.roi_min_points:
                d = best.p_hold - best.p_ground
                u = d / (np.linalg.norm(d) + 1e-9)
                t = (ap - best.p_ground) @ u
                lo, hi = np.percentile(t, 2), np.percentile(t, 98)
                p1 = best.p_ground + lo * u
                p2 = best.p_ground + hi * u
                p_hold, p_ground = (p1, p2) if p1[1] < p2[1] else (p2, p1)
                best = BeltROI(p_ground=p_ground, p_hold=p_hold,
                                halfwidth=self.cfg.roi_halfwidth_frac * float(hi - lo))
        return best
