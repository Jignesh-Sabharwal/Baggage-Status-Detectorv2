"""M1 verification: presence + dock detection full-clip run.

Usage: .venv/bin/python tools/m1_check.py D01 [--frames N] [--save-overlay]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config import load_config
from scene.dock import DockDetector
from scene.presence import ScenePresence
from stabilizer import Stabilizer
from video_source import VideoSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--save-overlay", action="store_true")
    args = ap.parse_args()

    cfg = load_config("configs/default.yaml", f"configs/conv_full_{args.clip}.yaml")
    src = VideoSource(cfg.video_path, cfg.fps_nominal)
    px_scale = cfg.px_scale(src.height)

    presence = ScenePresence(cfg.presence, cfg.fps_nominal, (src.height, src.width))
    stab = Stabilizer(cfg.stab, cfg.fps_nominal)
    dock = DockDetector(cfg.dock, cfg.fps_nominal, (src.height, src.width), px_scale, cfg.camera.aircraft_end)

    n_present = 0
    n_docked = 0
    n_ambiguous = 0
    n_total = 0
    first_docked = None
    docked_prev = False
    transitions = []

    out_writer = None
    if args.save_overlay:
        out_path = f"debug_frames/M1_{args.clip}.mp4"
        out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps_nominal, (src.width, src.height))

    for frame_idx, tb, frame in src:
        if args.frames and frame_idx >= args.frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not stab.has_patches:
            stab.select_patches(gray)
        gm = stab.update(gray)

        pres = presence.update(frame)
        n_total += 1
        if pres.present:
            n_present += 1
        if pres.ambiguous:
            n_ambiguous += 1

        dr = dock.update(pres.candidate, gray, gm)
        if dr.docked:
            n_docked += 1
            if first_docked is None:
                first_docked = frame_idx
        if dr.docked != docked_prev:
            transitions.append((frame_idx, tb.t_video, dr.docked))
        docked_prev = dr.docked

        if out_writer is not None:
            vis = frame.copy()
            if pres.candidate is not None:
                x, y, w, h = pres.candidate.bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if dr.elevated_point is not None:
                cv2.circle(vis, tuple(map(int, dr.elevated_point)), 4, (255, 0, 255), -1)
            label = f"pres={pres.present} stat={dr.stationary} boom={dr.boom_raised}({dr.boom_angle_deg and round(dr.boom_angle_deg,1)}) fus={dr.fuselage_gate_pass} DOCKED={dr.docked}"
            cv2.putText(vis, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            out_writer.write(vis)

    src.release()
    if out_writer is not None:
        out_writer.release()

    print(f"{args.clip}: frames={n_total} present_frac={n_present/n_total:.3f} "
          f"docked_frac={n_docked/n_total:.3f} ambiguous={n_ambiguous} first_docked_frame={first_docked}")
    print("transitions (frame, t_video, docked):")
    for t in transitions[:40]:
        print(" ", t)
    if len(transitions) > 40:
        print(f"  ... and {len(transitions) - 40} more")


if __name__ == "__main__":
    main()
