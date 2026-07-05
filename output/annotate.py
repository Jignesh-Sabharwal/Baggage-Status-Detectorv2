"""Overlay renderer (REQ-32 to REQ-38). Layout follows the two reference screenshots
described in the design spec; no reference screenshots were supplied to this build, so this
implements the written spec exactly but pixel-for-pixel matching against reference frames
could not be verified (flagged as a limitation, not silently assumed correct).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from activity.activity_fsm import ActivityState
from config import OverlayConfig

WHITE = (255, 255, 255)
AMBER = (0, 165, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)


class Annotator:
    def __init__(self, cfg: OverlayConfig, fps_nominal: float):
        self.cfg = cfg
        self.fps_nominal = fps_nominal
        self._flash: tuple[str, int] | None = None  # (text, frames_remaining)
        self._flash_frames = max(1, round(cfg.event_flash_s * fps_nominal))

    def trigger_flash(self, event_id: int, event_type: str) -> None:
        text = f"EVENT #{event_id}: {event_type.replace('_', ' ')}"
        self._flash = (text, self._flash_frames)

    def render(self, frame_bgr: np.ndarray, t_video: float, frame_idx: int,
               activity_state: ActivityState, n_confirmed_tracks: int, mean_speed: float,
               roi_polygon: np.ndarray | None, aircraft_anchor: tuple[float, float] | None,
               tracks: list, strip_to_frame: np.ndarray | None = None) -> np.ndarray:
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        # REQ-33: ROI polygon color-coded by activity state.
        if roi_polygon is not None:
            roi_color = YELLOW if activity_state == ActivityState.IDLE else BLUE
            cv2.polylines(vis, [roi_polygon.astype(int)], True, roi_color, 2)

        # REQ-35: aircraft-contact marker, debug-toggleable.
        if aircraft_anchor is not None and self.cfg.show_debug_markers:
            cv2.circle(vis, (int(aircraft_anchor[0]), int(aircraft_anchor[1])), 4, MAGENTA, -1)

        # REQ-34: track boxes, green, label "ID <id> (<age_s>s)". Tracks live in strip
        # coordinates (detect/bag_detector.py operates on the rectified belt strip), so an
        # axis-aligned box is only correct in the original frame once mapped back through the
        # inverse strip transform — it is no longer axis-aligned there in general.
        for t in tracks:
            if not t.confirmed:
                continue
            x, y, bw, bh = t.bbox
            age_s = t.age / self.fps_nominal
            label_pos = (x, max(0, y - 4))
            if strip_to_frame is not None:
                corners = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
                quad = cv2.transform(corners.reshape(-1, 1, 2), strip_to_frame).reshape(-1, 2)
                cv2.polylines(vis, [quad.astype(int)], True, GREEN, 1)
                label_pos = (int(quad[:, 0].min()), max(0, int(quad[:, 1].min()) - 4))
            else:
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), GREEN, 1)
            cv2.putText(vis, f"ID {t.id} ({age_s:.1f}s)", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 1, cv2.LINE_AA)

        # REQ-32: status block, top-left, black background bar.
        bar_h = 34
        cv2.rectangle(vis, (0, 0), (220, bar_h), BLACK, -1)
        if activity_state == ActivityState.IDLE:
            cv2.putText(vis, "STATUS: Idle", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, f"STATUS: {activity_state.name}", (4, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, AMBER, 1, cv2.LINE_AA)
        sign = "+" if mean_speed >= 0 else "-"
        cv2.putText(vis, f"Objects: {n_confirmed_tracks}   v={sign}{abs(mean_speed):.2f}",
                    (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)

        # REQ-36: top-right compact clock.
        clock_text = f"t= {t_video:.1f}s"
        (tw, _), _ = cv2.getTextSize(clock_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(vis, clock_text, (w - tw - 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

        # REQ-38: event flash banner, above the bottom timestamp bar, cleared after event_flash_s.
        bottom_bar_h = 20
        if self._flash is not None:
            text, remaining = self._flash
            cv2.putText(vis, text, (6, h - bottom_bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2, cv2.LINE_AA)
            remaining -= 1
            self._flash = (text, remaining) if remaining > 0 else None

        # REQ-37: bottom timestamp bar.
        cv2.rectangle(vis, (0, h - bottom_bar_h), (w, h), BLACK, -1)
        ts = time.strftime("%H:%M:%S", time.gmtime(t_video)) + f".{int((t_video % 1) * 100):02d}"
        cv2.putText(vis, f"Timestamp: {ts}   Frame: {frame_idx}", (4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)

        return vis
