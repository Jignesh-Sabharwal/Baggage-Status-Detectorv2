"""Full-clip time-base scan (no stabilizer) — confirms no VFR/timestamp fallback anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from video_source import VideoSource

CLIPS = ["D01", "D02", "D03", "D04", "N01", "N02", "N03", "N04"]


def main():
    for clip in CLIPS:
        cfg = load_config("configs/default.yaml", f"configs/conv_full_{clip}.yaml")
        src = VideoSource(cfg.video_path, cfg.fps_nominal)
        n = 0
        fallback_at = None
        last_t = None
        max_jump = 0.0
        for frame_idx, tb, frame in src:
            if tb.fallback_just_triggered and fallback_at is None:
                fallback_at = frame_idx
            if last_t is not None:
                max_jump = max(max_jump, tb.t_video - last_t)
            last_t = tb.t_video
            n += 1
        src.release()
        print(f"{clip}: frames={n} fallback_at={fallback_at} used_fallback={tb.used_fallback} "
              f"final_t={last_t:.2f}s max_frame_delta={max_jump:.3f}s")


if __name__ == "__main__":
    main()
