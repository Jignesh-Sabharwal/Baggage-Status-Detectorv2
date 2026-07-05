"""M2 verification: Scene FSM through ROI lock.

Usage: .venv/bin/python tools/m2_check.py D01 [--frames N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config import load_config
from scene.scene_fsm import SceneFSM
from stabilizer import Stabilizer
from video_source import VideoSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config("configs/default.yaml", f"configs/conv_full_{args.clip}.yaml")
    src = VideoSource(cfg.video_path, cfg.fps_nominal)
    fsm = SceneFSM(cfg, (src.height, src.width))
    stab = Stabilizer(cfg.stab, cfg.fps_nominal)

    last_frame = None
    prev_state = fsm.state
    for frame_idx, tb, frame in src:
        if args.frames and frame_idx >= args.frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not stab.has_patches:
            stab.select_patches(gray)
        gm = stab.update(gray)
        su = fsm.update(frame, gray, gm)
        if su.state != prev_state:
            print(f"{frame_idx} t={tb.t_video:.1f} STATE {prev_state.name} -> {su.state.name} events={su.events}")
        elif su.events:
            print(f"{frame_idx} t={tb.t_video:.1f} events={su.events}")
        prev_state = su.state
        last_frame = (frame_idx, frame.copy())

    src.release()
    print("final state:", fsm.state.name)
    if fsm.roi.locked is not None:
        fit = fsm.roi.locked.fit
        print(f"locked fit: center={fit.center} angle={fit.angle_deg:.1f} length={fit.length:.1f} width={fit.width:.1f}")
        frame_idx, frame = last_frame
        vis = frame.copy()
        cv2.polylines(vis, [fsm.roi.locked.polygon.astype(int)], True, (0, 255, 255), 2)
        cv2.circle(vis, tuple(map(int, fsm.roi.locked.aircraft_anchor)), 5, (255, 0, 255), -1)
        cv2.imwrite(f"debug_frames/M2_{args.clip}_locked.jpg", vis)


if __name__ == "__main__":
    main()
