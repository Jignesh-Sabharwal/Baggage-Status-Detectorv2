"""ScenePresence — is a loader in frame? Appearance test, not a motion test (REQ-11/12).

A parked loader becomes MOG2 background within seconds, so presence must be answered by
color/appearance, never by motion. This is a single-color-band detector calibrated per
installation (config.presence.hsv_lo/hi are [PER-CAMERA]) — see REQ-11's stated limitation:
it is calibrated to one loader's livery under one lighting condition and will not generalize
across liveries or lighting without recalibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import PresenceConfig


@dataclass
class PresenceCandidate:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[float, float]
    area: int
    aspect: float


@dataclass
class PresenceResult:
    present_raw: bool          # this frame's instantaneous test
    present: bool               # debounced state
    candidate: PresenceCandidate | None
    ambiguous: bool = False     # REQ-12: more than one blob passed the area band


class ScenePresence:
    # How much to pad the previous accepted bbox when restricting the search region, as a
    # fraction of frame size (not blob size — the loader blob itself often spans a large
    # fraction of the frame, so padding proportional to *its* size defeats the purpose).
    _SEARCH_PAD_FRAC_OF_FRAME = 0.08
    # Consecutive missed frames before giving up on the tracked search window and going
    # back to a full-frame search (re-acquisition after a real occlusion/departure).
    _MAX_MISSED_FRAMES = 15

    def __init__(self, cfg: PresenceConfig, fps_nominal: float, frame_shape: tuple[int, int]):
        self.cfg = cfg
        self._debounce_frames = max(1, round(cfg.debounce_s * fps_nominal))
        self._frame_h, self._frame_w = frame_shape
        self._frame_area = frame_shape[0] * frame_shape[1]
        self._raw_history: list[bool] = []
        self._debounced = False
        self._prev_accepted: PresenceCandidate | None = None
        self._search_bbox: tuple[int, int, int, int] | None = None
        self._missed_frames = 0
        # EMA of the *trusted* blob area/bbox. A padded search window that's recomputed from
        # whatever candidate won this frame will grow together with a spurious merge (chasing
        # its own contamination frame over frame) — so growth of the trusted state is gated on
        # the new candidate's area being plausible relative to recent history, and an anomalous
        # frame doesn't move the trusted state at all.
        self._trusted_area_ema: float | None = None
        self._trusted_bbox: tuple[int, int, int, int] | None = None

    def _find_candidates(self, hsv: np.ndarray) -> list[PresenceCandidate]:
        # Once a blob has been found, restrict the search to a padded region around it. This
        # is not required for correctness by the spec, but at this resolution the raw HSV
        # mask intermittently bridges into a same-hued region elsewhere in frame (e.g. a
        # fuselage-shadow patch) via the morphological close, doubling the accepted blob's
        # area for a few frames and corrupting its centroid — exactly the kind of spurious
        # merge REQ-12 is about, except it happens *inside* one connected component rather
        # than across two, so REQ-12's overlap tie-break can't see it. Constraining the
        # search region prevents the merge from happening in the first place.
        if self._search_bbox is not None:
            sx0, sy0, sx1, sy1 = self._search_bbox
            region = hsv[sy0:sy1, sx0:sx1]
            offset = (sx0, sy0)
        else:
            region = hsv
            offset = (0, 0)

        mask = cv2.inRange(region, np.array(self.cfg.hsv_lo), np.array(self.cfg.hsv_hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        min_area = self.cfg.min_area_frac * self._frame_area
        ox, oy = offset
        candidates = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT] + ox, stats[i, cv2.CC_STAT_TOP] + oy,
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            aspect = w / h if h else 0.0
            local_mask = (labels == i)
            if offset != (0, 0):
                full_mask = np.zeros((self._frame_h, self._frame_w), dtype=bool)
                full_mask[sy0:sy1, sx0:sx1] = local_mask
            else:
                full_mask = local_mask
            candidates.append(PresenceCandidate(
                mask=full_mask, bbox=(x, y, w, h),
                centroid=(centroids[i][0] + ox, centroids[i][1] + oy), area=int(area), aspect=aspect,
            ))
        return candidates

    def _is_plausible_size(self, candidate: PresenceCandidate) -> bool:
        if self.cfg.expected_area_frac is not None:
            expected = self.cfg.expected_area_frac * self._frame_area
            tol = self.cfg.expected_area_tol
            return (1 - tol) * expected <= candidate.area <= (1 + tol) * expected
        if self._trusted_area_ema is None:
            return True
        ratio = candidate.area / self._trusted_area_ema
        return 0.6 <= ratio <= 1.6

    def _update_search_window(self, candidate: PresenceCandidate | None) -> None:
        plausible = candidate is not None and self._is_plausible_size(candidate)

        if plausible:
            self._missed_frames = 0
            if self._trusted_area_ema is None:
                self._trusted_area_ema = float(candidate.area)
            else:
                a = 0.2
                self._trusted_area_ema = a * candidate.area + (1 - a) * self._trusted_area_ema
            self._trusted_bbox = candidate.bbox
        elif candidate is None:
            self._missed_frames += 1
            if self._missed_frames > self._MAX_MISSED_FRAMES:
                self._search_bbox = None
                self._trusted_area_ema = None
                self._trusted_bbox = None
                return
        # else: candidate exists but looks like a contaminated/merged blob — hold the trusted
        # state and search window steady rather than growing to chase it.

        if self._trusted_bbox is not None:
            x, y, w, h = self._trusted_bbox
            pad_x = int(self._frame_w * self._SEARCH_PAD_FRAC_OF_FRAME)
            pad_y = int(self._frame_h * self._SEARCH_PAD_FRAC_OF_FRAME)
            self._search_bbox = (
                max(0, x - pad_x), max(0, y - pad_y),
                min(self._frame_w, x + w + pad_x), min(self._frame_h, y + h + pad_y),
            )

    def _pick(self, candidates: list[PresenceCandidate]) -> tuple[PresenceCandidate | None, bool]:
        """REQ-12: never silently 'largest wins' when multiple candidates pass the area band."""
        if not candidates:
            return None, False
        if len(candidates) == 1:
            return candidates[0], False

        ambiguous = True
        if self._prev_accepted is not None:
            def overlaps(c: PresenceCandidate) -> bool:
                px, py, pw, ph = self._prev_accepted.bbox
                x, y, w, h = c.bbox
                ix0, iy0 = max(px, x), max(py, y)
                ix1, iy1 = min(px + pw, x + w), min(py + ph, y + h)
                return ix1 > ix0 and iy1 > iy0

            overlapping = [c for c in candidates if overlaps(c)]
            if overlapping:
                return max(overlapping, key=lambda c: c.area), ambiguous

        best = min(candidates, key=lambda c: abs(c.aspect - self.cfg.aspect_prior))
        return best, ambiguous

    def _reacquire_within_trusted_bbox(self, hsv: np.ndarray) -> PresenceCandidate | None:
        """Tight re-detection strictly inside the last trusted bbox (no padding).

        Used when the padded-window candidate looks contaminated: the contaminating region
        is, by construction, outside the last known-good bbox, so re-thresholding within just
        that tight box gives a clean reading unaffected by it.
        """
        x, y, w, h = self._trusted_bbox
        region = hsv[y:y + h, x:x + w]
        mask = cv2.inRange(region, np.array(self.cfg.hsv_lo), np.array(self.cfg.hsv_hi))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bw, bh = stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]
        full_mask = np.zeros((self._frame_h, self._frame_w), dtype=bool)
        full_mask[y:y + h, x:x + w] = (labels == idx)
        return PresenceCandidate(
            mask=full_mask,
            bbox=(x + stats[idx, cv2.CC_STAT_LEFT], y + stats[idx, cv2.CC_STAT_TOP], bw, bh),
            centroid=(x + centroids[idx][0], y + centroids[idx][1]),
            area=area, aspect=(bw / bh if bh else 0.0),
        )

    def update(self, frame_bgr: np.ndarray) -> PresenceResult:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        candidates = self._find_candidates(hsv)
        candidate, ambiguous = self._pick(candidates)

        if candidate is not None and not self._is_plausible_size(candidate) and self._trusted_bbox is not None:
            candidate = self._reacquire_within_trusted_bbox(hsv) or candidate

        present_raw = candidate is not None
        if candidate is not None:
            self._prev_accepted = candidate
        self._update_search_window(candidate)

        self._raw_history.append(present_raw)
        if len(self._raw_history) > self._debounce_frames:
            self._raw_history.pop(0)

        if len(self._raw_history) == self._debounce_frames:
            if all(self._raw_history):
                self._debounced = True
            elif not any(self._raw_history):
                self._debounced = False
            # mixed window: state holds (debounce means flips only after the score holds)

        return PresenceResult(present_raw, self._debounced, candidate, ambiguous)
