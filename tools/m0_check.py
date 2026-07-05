"""M0 verification: time base sanity + global-motion trace on every clip.

Usage: .venv/bin/python tools/m0_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import load_config
from stabilizer import Stabilizer
from video_source import VideoSource

CLIPS = ["D01", "D02", "D03", "D04", "N01", "N02", "N03", "N04"]


def main():
    print(f"{'clip':10s} {'frames':>7s} {'fallback@':>10s} {'motion_mean':>12s} {'motion_p95':>11s} {'shake_frac':>11s}")
    for clip in CLIPS:
        cfg = load_config("configs/default.yaml", f"configs/conv_full_{clip}.yaml")
        src = VideoSource(cfg.video_path, cfg.fps_nominal)
        stab = Stabilizer(cfg.stab, cfg.fps_nominal)

        magnitudes = []
        fallback_frame = None
        n = 0
        for frame_idx, tb, frame in src:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not stab.has_patches:
                stab.select_patches(gray)
            gm = stab.update(gray)
            magnitudes.append(gm.magnitude)
            if tb.fallback_just_triggered and fallback_frame is None:
                fallback_frame = frame_idx
            n += 1
            if n >= 400:  # sample first ~36s per clip; full-clip run happens at M1+
                break
        src.release()

        mags = np.array(magnitudes)
        shake_frac = float((mags > cfg.stab.shake_px).mean()) if len(mags) else 0.0
        print(f"{clip:10s} {n:7d} {str(fallback_frame):>10s} {mags.mean():12.3f} {np.percentile(mags, 95):11.3f} {shake_frac:11.3f}")


if __name__ == "__main__":
    main()
