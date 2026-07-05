"""Semi-automatic HSV band calibration per clip.

For each clip: try a broad, livery-appropriate candidate HSV band, find the largest connected
component across several sample frames, and derive a tight band from the percentile spread of
hue/sat/val *inside* that component. Prints suggested presence.hsv_lo/hi plus the observed area
fraction (to sanity check against presence.min_area_frac).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

CLIPS = {
    "D01": "blue", "D03": "blue", "D04": "blue", "N02": "blue", "N03": "blue",
    "D02": "dark",
    "N01": "light", "N04": "light",
}

BROAD = {
    "blue": ((90, 40, 20), (140, 255, 255)),
    "dark": ((0, 0, 0), (179, 90, 90)),
    "light": ((0, 0, 90), (179, 60, 255)),
}


def largest_component_mask(hsv, lo, hi):
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, 0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[idx, cv2.CC_STAT_AREA]
    return (labels == idx), area


def main():
    for clip, kind in CLIPS.items():
        frames = sorted(Path("debug_frames").glob(f"{clip}_f*.jpg"))
        lo, hi = BROAD[kind]
        h_vals, s_vals, v_vals = [], [], []
        area_fracs = []
        for fp in frames:
            img = cv2.imread(str(fp))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask, area = largest_component_mask(hsv, lo, hi)
            if mask is None or area < 0.005 * img.shape[0] * img.shape[1]:
                continue
            pix = hsv[mask]
            h_vals.append(pix[:, 0])
            s_vals.append(pix[:, 1])
            v_vals.append(pix[:, 2])
            area_fracs.append(area / (img.shape[0] * img.shape[1]))

        if not h_vals:
            print(f"{clip}: NO component found with broad '{kind}' band — needs manual look")
            continue

        H = np.concatenate(h_vals)
        S = np.concatenate(s_vals)
        V = np.concatenate(v_vals)
        p = lambda a, q: int(np.percentile(a, q))
        h_lo, h_hi = p(H, 2), p(H, 98)
        s_lo, s_hi = p(S, 2), p(S, 98)
        v_lo, v_hi = p(V, 2), p(V, 98)
        print(f"{clip} ({kind}): hsv_lo=[{h_lo},{s_lo},{v_lo}] hsv_hi=[{h_hi},{s_hi},{v_hi}] "
              f"area_frac~{np.mean(area_fracs):.3f} n_frames_hit={len(h_vals)}/{len(frames)}")


if __name__ == "__main__":
    main()
