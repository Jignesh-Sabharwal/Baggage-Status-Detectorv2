"""Capture a rendered overlay snapshot at every ROI_LOCKED event for visual comparison."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from activity.activity_fsm import ActivityState
from config import load_config
from output.annotate import Annotator
from scene.scene_fsm import SceneFSM, SceneState
from stabilizer import Stabilizer
from video_source import VideoSource

clip = sys.argv[1]
max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None

cfg = load_config("configs/default.yaml", f"configs/conv_full_{clip}.yaml")
src = VideoSource(cfg.video_path, cfg.fps_nominal)
fsm = SceneFSM(cfg, (src.height, src.width))
stab = Stabilizer(cfg.stab, cfg.fps_nominal)
annotator = Annotator(cfg.overlay, cfg.fps_nominal)

lock_count = 0
for frame_idx, tb, frame in src:
    if max_frames and frame_idx >= max_frames:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not stab.has_patches:
        stab.select_patches(gray)
    gm = stab.update(gray)
    su = fsm.update(frame, gray, gm)
    if "ROI_LOCKED" in su.events and su.state == SceneState.MONITORING:
        lock_count += 1
        locked = fsm.roi.locked
        vis = annotator.render(frame, tb.t_video, frame_idx, ActivityState.IDLE, 0, 0.0,
                                locked.polygon, locked.aircraft_anchor, [])
        path = f"debug_frames/LOCK_{clip}_{lock_count}_f{frame_idx}.jpg"
        cv2.imwrite(path, vis)
        print(f"lock #{lock_count} at frame {frame_idx} t={tb.t_video:.1f}s -> {path}")

src.release()
