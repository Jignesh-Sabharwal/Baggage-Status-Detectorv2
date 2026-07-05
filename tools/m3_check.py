"""M3 verification: MOG2 + bag detector + tracker, running on top of the M1/M2 Scene FSM.

Usage: .venv/bin/python tools/m3_check.py D01 [--frames N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import load_config
from detect.bag_detector import BagDetector
from detect.bg_model import BgModel
from scene.scene_fsm import SceneFSM, SceneState
from stabilizer import Stabilizer
from track.tracker import Tracker
from video_source import VideoSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config("configs/default.yaml", f"configs/conv_full_{args.clip}.yaml")
    src = VideoSource(cfg.video_path, cfg.fps_nominal)
    px_scale = cfg.px_scale(src.height)

    fsm = SceneFSM(cfg, (src.height, src.width))
    stab = Stabilizer(cfg.stab, cfg.fps_nominal)
    bg_model = None
    bag_detector = BagDetector(cfg.det, cfg.fps_nominal, cfg.bg.ghost_s, cfg.bg.ghost_edge_ratio, px_scale)
    tracker = Tracker(cfg.trk, cfg.fps_nominal, px_scale)
    strip_axis_unit = (1.0, 0.0)

    n_monitoring = 0
    n_detections = 0
    n_confirmed_events = 0
    rejection_totals = {}
    last_vis = None
    last_bag_gen = -1
    bag_bg_model = None
    strip_M = strip_Minv = None
    strip_dims = (0, 0)
    warmup_frames = max(1, round(cfg.bg.warmup_s * cfg.fps_nominal))
    frames_since_strip_lock = 0

    for frame_idx, tb, frame in src:
        if args.frames and frame_idx >= args.frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not stab.has_patches:
            stab.select_patches(gray)
        gm = stab.update(gray)

        prev_state = fsm.state
        if prev_state == SceneState.NO_BELT and bg_model is not None:
            bg_model = None  # torn all the way down; re-instantiate on next BELT_PRESENT

        n_confirmed = len(tracker.confirmed_tracks) if fsm.state == SceneState.MONITORING else 0
        mean_speed = None
        if tracker.confirmed_tracks:
            speeds = [abs(t.velocity_ema) for t in tracker.confirmed_tracks if t.velocity_ema is not None]
            if speeds:
                mean_speed = float(np.mean(speeds))

        su = fsm.update(frame, gray, gm, n_confirmed_tracks=n_confirmed,
                         activity_is_idle=(n_confirmed == 0), fg_mask=None, track_centroids=None)

        if su.state in (SceneState.BELT_PRESENT, SceneState.DOCKED, SceneState.ROI_SETTLING, SceneState.MONITORING):
            if bg_model is None:
                bg_model = BgModel(cfg.bg, cfg.fps_nominal)
            roi_mask = None
            if fsm.roi.locked is not None:
                roi_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(roi_mask, [fsm.roi.locked.polygon.astype(np.int32)], 1)
            frozen = su.state == SceneState.ROI_SETTLING
            fgr = bg_model.apply(frame, roi_mask, mean_speed, frozen)

            if su.state == SceneState.MONITORING and fsm.roi.locked is not None:
                n_monitoring += 1
                locked = fsm.roi.locked
                if locked.generation != last_bag_gen:
                    strip_h = max(1, int(round(locked.fit.width)))
                    strip_M, strip_dims = locked.strip_transform(strip_h)
                    strip_Minv = cv2.invertAffineTransform(strip_M)
                    bag_bg_model = BgModel(cfg.bg, cfg.fps_nominal)
                    bag_detector = BagDetector(cfg.det, cfg.fps_nominal, cfg.bg.ghost_s, cfg.bg.ghost_edge_ratio, px_scale)
                    tracker = Tracker(cfg.trk, cfg.fps_nominal, px_scale)
                    last_bag_gen = locked.generation
                    frames_since_strip_lock = 0

                strip_frame = cv2.warpAffine(frame, strip_M, strip_dims)
                strip_gray = cv2.warpAffine(gray, strip_M, strip_dims)
                strip_fgr = bag_bg_model.apply(strip_frame, None, mean_speed)
                strip_bg_gray = cv2.cvtColor(bag_bg_model.get_background_image(), cv2.COLOR_BGR2GRAY)

                dets, tracks = [], []
                if frames_since_strip_lock >= warmup_frames:
                    dets, stats = bag_detector.detect(strip_fgr.mask, strip_frame, strip_gray, strip_bg_gray)
                    n_detections += len(dets)
                    for k in ("ghost", "area", "solidity", "aspect", "vest", "person_rot"):
                        rejection_totals[k] = rejection_totals.get(k, 0) + getattr(stats, k)
                frames_since_strip_lock += 1

                tracks = tracker.update(dets, strip_axis_unit)
                for t in tracks:
                    if t.just_confirmed and not t.inherited:
                        n_confirmed_events += 1

                if frame_idx % 200 == 0:
                    vis = strip_frame.copy()
                    for t in tracks:
                        x, y, w, h = t.bbox
                        color = (0, 255, 0) if t.confirmed else (0, 128, 255)
                        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
                    last_vis = (frame_idx, vis)

    src.release()

    print(f"{args.clip}: monitoring_frames={n_monitoring} detections={n_detections} "
          f"confirmed_track_events={n_confirmed_events}")
    print("rejection totals:", rejection_totals)
    if last_vis is not None:
        cv2.imwrite(f"debug_frames/M3_{args.clip}_sample.jpg", last_vis[1])


if __name__ == "__main__":
    main()
