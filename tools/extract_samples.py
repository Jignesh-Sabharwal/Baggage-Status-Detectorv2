"""Extract a handful of sample frames per clip for visual calibration inspection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

CLIPS = ["D01", "D02", "D03", "D04", "N01", "N02", "N03", "N04"]
OUT = Path("debug_frames")
OUT.mkdir(exist_ok=True)


def main():
    for clip in CLIPS:
        path = f"videos/conv_full_{clip}.mp4"
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # sample 5 frames spread across the clip
        idxs = [int(n * f) for f in (0.02, 0.15, 0.35, 0.55, 0.75)]
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            out_path = OUT / f"{clip}_f{idx}.jpg"
            cv2.imwrite(str(out_path), frame)
        cap.release()
        print(f"{clip}: extracted frames at {idxs} (of {n})")


if __name__ == "__main__":
    main()
