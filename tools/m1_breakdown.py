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
cfg = load_config("configs/default.yaml", f"configs/conv_full_{clip}.yaml")
src = VideoSource(cfg.video_path, cfg.fps_nominal)
px_scale = cfg.px_scale(src.height)
presence = ScenePresence(cfg.presence, cfg.fps_nominal, (src.height, src.width))
stab = Stabilizer(cfg.stab, cfg.fps_nominal)
dock = DockDetector(cfg.dock, cfg.fps_nominal, (src.height, src.width), px_scale, cfg.camera.aircraft_end)

n = n_pres = n_stat = n_boom = n_fus = n_dock = 0
for frame_idx, tb, frame in src:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not stab.has_patches:
        stab.select_patches(gray)
    gm = stab.update(gray)
    pres = presence.update(frame)
    dr = dock.update(pres.candidate, gray, gm)
    n += 1
    if pres.present:
        n_pres += 1
    if dr.stationary:
        n_stat += 1
    if dr.boom_raised:
        n_boom += 1
    if dr.fuselage_gate_pass:
        n_fus += 1
    if dr.docked:
        n_dock += 1

src.release()
print(f"{clip}: n={n} present={n_pres/n:.3f} stationary={n_stat/n:.3f} boom_raised={n_boom/n:.3f} "
      f"fuselage={n_fus/n:.3f} docked={n_dock/n:.3f}")
