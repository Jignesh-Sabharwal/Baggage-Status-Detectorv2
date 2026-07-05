import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config import load_config
from scene.dock import DockDetector
from scene.presence import ScenePresence
from stabilizer import Stabilizer
from video_source import VideoSource

clip = sys.argv[1]
target_frame = int(sys.argv[2])

cfg = load_config("configs/default.yaml", f"configs/conv_full_{clip}.yaml")
src = VideoSource(cfg.video_path, cfg.fps_nominal)
px_scale = cfg.px_scale(src.height)
presence = ScenePresence(cfg.presence, cfg.fps_nominal, (src.height, src.width))
stab = Stabilizer(cfg.stab, cfg.fps_nominal)
dock = DockDetector(cfg.dock, cfg.fps_nominal, (src.height, src.width), px_scale)

for frame_idx, tb, frame in src:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not stab.has_patches:
        stab.select_patches(gray)
    gm = stab.update(gray)
    pres = presence.update(frame)
    dr = dock.update(pres.candidate, gray, gm)
    if frame_idx == target_frame:
        print("candidate:", pres.candidate.bbox if pres.candidate else None, "area", pres.candidate.area if pres.candidate else None)
        print("stationary:", dr.stationary)
        print("boom_angle:", dr.boom_angle_deg, "boom_raised:", dr.boom_raised)
        print("fuselage_gate_pass:", dr.fuselage_gate_pass)
        print("elevated_point:", dr.elevated_point)
        print("docked:", dr.docked)
        # dump crop + edges for inspection
        if pres.candidate:
            x, y, w, h = pres.candidate.bbox
            pad = int(0.15 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            roi = gray[y0:y1, x0:x1]
            edges = cv2.Canny(roi, 50, 150)
            cv2.imwrite(f"debug_frames/DOCK_roi_{clip}_{target_frame}.jpg", roi)
            cv2.imwrite(f"debug_frames/DOCK_edges_{clip}_{target_frame}.jpg", edges)
            _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            cv2.imwrite(f"debug_frames/DOCK_bright_{clip}_{target_frame}.jpg", bright)
        break
src.release()
