"""Belt Loader Activity Detection — full pipeline driver.

Usage:
    .venv/bin/python main.py <clip_name> [--max-frames N] [--debug] [--no-snapshots]
    .venv/bin/python main.py <clip_name> --live          # live preview window, press q to quit
    .venv/bin/python main.py <clip_name> --save-video    # write runs/<run>/annotated.mp4

<clip_name> is the stem used to find configs/conv_full_<clip_name>.yaml and
videos/conv_full_<clip_name>.mp4.
"""
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import cv2
import numpy as np

from activity.activity_fsm import ActivityFSM, ActivityState
from activity.direction import frame_vote
from config import Config, dump_config, load_config
from detect.bag_detector import BagDetector
from detect.bg_model import BgModel
from output.annotate import Annotator
from output.events import EventLogger, SessionManager
from output.snapshots import SnapshotWriter
from preprocess import Preprocessor
from scene.scene_fsm import SceneFSM, SceneState
from stabilizer import Stabilizer
from track.tracker import Tracker
from video_source import VideoSource

SCENE_STATES_WITH_BG = {SceneState.BELT_PRESENT, SceneState.DOCKED, SceneState.ROI_SETTLING, SceneState.MONITORING}


def run(cfg: Config, clip_name: str, max_frames: int | None, save_snapshots: bool, save_debug: bool,
        run_dir: Path | None = None, save_video: bool = False, live: bool = False,
        live_speed: float = 1.0) -> Path:
    src = VideoSource(cfg.video_path, cfg.fps_nominal)
    px_scale = cfg.px_scale(src.height)

    if run_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"{clip_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")

    preprocessor = Preprocessor(cfg.preprocess)
    stab = Stabilizer(cfg.stab, cfg.fps_nominal)
    fsm = SceneFSM(cfg, (src.height, src.width), camera_motion_fn=stab.consume_cumulative)
    bag_detector = BagDetector(cfg.det, cfg.fps_nominal, cfg.bg.ghost_s, cfg.bg.ghost_edge_ratio, px_scale)
    tracker = Tracker(cfg.trk, cfg.fps_nominal, px_scale)
    activity = ActivityFSM(cfg.fsm, cfg.fps_nominal)
    logger = EventLogger(run_dir)
    sessions = SessionManager(cfg.fsm.reposition_grace_s)
    annotator = Annotator(cfg.overlay, cfg.fps_nominal)
    snapshots = SnapshotWriter(run_dir) if save_snapshots else None

    debug_dir = None
    rejection_rows: list[dict] = []
    if save_debug:
        debug_dir = run_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if save_video:
        video_writer = cv2.VideoWriter(str(run_dir / "annotated.mp4"),
                                        cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps_nominal,
                                        (src.width, src.height))

    bg_model: BgModel | None = None
    roi_locked_frame: int | None = None
    seen_frames_since_lock = 0
    seeded_confirmed_ids: set[int] = set()

    # Bag detection/tracking run on a rectified strip (ground end at x=0, aircraft end at
    # x=strip_w) rather than the raw rotated frame, so blob geometry is measured along/across
    # the belt axis directly instead of being skewed by the belt's incline angle. The strip
    # mapping and its dedicated background model are only valid for one lock_generation — both
    # are rebuilt whenever RoiManager reports a new lock or a reposition-relock (bag ID
    # numbering continues across the rebuild via track/tracker.py's module-level counter).
    last_bag_gen = -1
    bag_bg_model: BgModel | None = None
    strip_M: np.ndarray | None = None
    strip_Minv: np.ndarray | None = None
    strip_dims: tuple[int, int] = (0, 0)
    warmup_frames = max(1, round(cfg.bg.warmup_s * cfg.fps_nominal))
    frames_since_strip_lock = 0

    def emit(frame_idx, t_video, event_type, extra_n_tracks=0):
        eid = logger.log_event(frame_idx, t_video, event_type, fsm.state.name, activity.state.name, extra_n_tracks)
        annotator.trigger_flash(eid, event_type)

    target_frame_s = 1.0 / (cfg.fps_nominal * live_speed) if live else None

    # REQ-04/05: classify the opening init.window_s into whichever state already holds (e.g.
    # a clip that starts already docked) instead of always cold-starting in NO_BELT and
    # waiting for the normal per-frame transitions to catch up.
    init_frames_n = max(1, round(cfg.init_window_s * cfg.fps_nominal))
    init_buffer: list[tuple[np.ndarray, np.ndarray]] = []
    init_done = False

    for frame_idx, tb, frame in src:
        frame_start = time.monotonic() if live else None
        if max_frames and frame_idx >= max_frames:
            break
        frame = preprocessor(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not stab.has_patches:
            stab.select_patches(gray)
        gm = stab.update(gray)

        if not init_done:
            if tb.fallback_just_triggered:
                emit(frame_idx, tb.t_video, "TIMEBASE_FALLBACK")
            init_buffer.append((frame, gray))
            if len(init_buffer) >= init_frames_n:
                fsm.init_classify(init_buffer)
                init_done = True
            if video_writer is not None or live:
                vis = annotator.render(frame, tb.t_video, frame_idx, activity.state, 0, 0.0, None, None, [])
                if video_writer is not None:
                    video_writer.write(vis)
                if live:
                    cv2.imshow(f"belt-loader-detection: {clip_name}", vis)
                    elapsed = time.monotonic() - frame_start
                    wait_ms = max(1, int((target_frame_s - elapsed) * 1000))
                    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                        break
            continue

        n_confirmed = len(tracker.confirmed_tracks) if fsm.state == SceneState.MONITORING else 0
        mean_speed = 0.0
        speeds = [t.velocity_ema for t in tracker.confirmed_tracks if t.velocity_ema is not None]
        if speeds:
            mean_speed = float(np.mean(speeds))

        su = fsm.update(frame, gray, gm, n_confirmed_tracks=n_confirmed, activity_is_idle=(n_confirmed == 0))

        if tb.fallback_just_triggered:
            emit(frame_idx, tb.t_video, "TIMEBASE_FALLBACK")

        if su.state in SCENE_STATES_WITH_BG:
            if bg_model is None:
                bg_model = BgModel(cfg.bg, cfg.fps_nominal)  # REQ-03: instantiated at BELT_PRESENT entry
            roi_mask = None
            if fsm.roi.locked is not None:
                roi_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(roi_mask, [fsm.roi.locked.polygon.astype(np.int32)], 1)
            frozen = su.state == SceneState.ROI_SETTLING
            # REQ-10b: sustained camera motion shares the burst-guard path (skip detection,
            # raised learning rate), not just the ratio-based ghost/motion-blowout trigger.
            camera_burst = bool(gm and gm.sustained)
            if camera_burst:
                bg_model.trigger_burst_guard()
            fgr = bg_model.apply(frame, roi_mask, mean_speed if speeds else None, frozen)
            if fgr.burst or camera_burst:
                emit(frame_idx, tb.t_video, "CAMERA_MOTION" if camera_burst else "BURST_GUARD")
        else:
            bg_model = None

        for ev in su.events:
            if ev in ("BELT_ARRIVED", "BELT_DOCKED", "ROI_LOCKED", "ROI_REPOSITIONED",
                      "ROI_HEALTH_RESET", "BELT_COVERED", "BELT_UNCOVERED", "BELT_UNDOCKED",
                      "BELT_DEPARTED"):
                emit(frame_idx, tb.t_video, ev, n_confirmed)
            if ev == "ROI_LOCKED" and su.state == SceneState.MONITORING:
                roi_locked_frame = frame_idx
            if ev == "ROI_REPOSITIONED":
                sessions.on_reposition(tb.t_video)
            if ev in ("BELT_UNDOCKED", "BELT_DEPARTED"):
                sessions.close_at_teardown(tb.t_video, logger)

        tracks = []
        locked = fsm.roi.locked
        strip_axis_unit = (1.0, 0.0)  # strip x-axis is the belt axis by construction

        if su.state == SceneState.MONITORING and fsm.roi.locked is not None:
            if locked.generation != last_bag_gen:
                # New lock or reposition-relock: the strip mapping and any bg-model history from
                # the previous geometry are meaningless now — rebuild all three from scratch.
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
            strip_fgr = bag_bg_model.apply(strip_frame, None, mean_speed if speeds else None)
            strip_bg_gray = cv2.cvtColor(bag_bg_model.get_background_image(), cv2.COLOR_BGR2GRAY)

            dets: list = []
            rstats = None
            if frames_since_strip_lock >= warmup_frames:
                dets, rstats = bag_detector.detect(strip_fgr.mask, strip_frame, strip_gray, strip_bg_gray)
            frames_since_strip_lock += 1
            tracks = tracker.update(dets, strip_axis_unit)
            if save_debug and rstats is not None:
                rejection_rows.append({
                    "frame": frame_idx, "t_video": round(tb.t_video, 3),
                    "n_contours": rstats.total_contours, "n_detections": len(dets),
                    "rej_ghost": rstats.ghost, "rej_area": rstats.area,
                    "rej_solidity": rstats.solidity, "rej_aspect": rstats.aspect,
                    "rej_vest": rstats.vest, "rej_person_rot": rstats.person_rot,
                })

            for t in tracks:
                if t.confirmed:
                    sessions.on_confirmed_track_sighting(tb.t_video, t.id,
                                                          1 if (t.velocity_ema or 0) > 0 else -1)
                if t.just_confirmed and not t.inherited and t.id not in seeded_confirmed_ids:
                    seeded_confirmed_ids.add(t.id)
                    if snapshots is not None:
                        vis = annotator.render(frame, tb.t_video, frame_idx, activity.state,
                                                len(tracker.confirmed_tracks), mean_speed,
                                                locked.polygon, locked.aircraft_anchor, tracks,
                                                strip_Minv)
                        snapshots.save(vis, frame_idx, f"track{t.id}")

            sessions.check_grace_and_maybe_close(tb.t_video, logger)

            vote = frame_vote(tracker.confirmed_tracks, cfg.dir, strip_axis_unit, px_scale)
            n_conf = len(tracker.confirmed_tracks)
            ev = activity.update(vote, n_conf)

            if roi_locked_frame is not None:
                seen_frames_since_lock = frame_idx - roi_locked_frame
            if ev is not None:
                truncated = (activity.state != ActivityState.IDLE and
                             seen_frames_since_lock < 2 * cfg.fsm.idle_timeout_s * cfg.fps_nominal and
                             sessions.current is None)
                if sessions.on_resume_same_direction(activity.state):
                    pass  # seamless continuation, no new session, no event needed beyond FSM's own
                else:
                    sessions.on_activity_transition(activity.state, tb.t_video, frame_idx, logger,
                                                     fsm.state.name, n_conf, truncated_start=truncated)
                emit(frame_idx, tb.t_video, ev, n_conf)

        if video_writer is not None or live:
            render_strip_minv = strip_Minv if (su.state == SceneState.MONITORING and locked is not None) else None
            vis = annotator.render(frame, tb.t_video, frame_idx, activity.state,
                                    len(tracker.confirmed_tracks), mean_speed,
                                    locked.polygon if locked else None,
                                    locked.aircraft_anchor if locked else None, tracks,
                                    render_strip_minv)
            if video_writer is not None:
                video_writer.write(vis)
            if live:
                cv2.imshow(f"belt-loader-detection: {clip_name}", vis)
                elapsed = time.monotonic() - frame_start
                wait_ms = max(1, int((target_frame_s - elapsed) * 1000))
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    break

    # Any still-open session at end of run closes at its last-track bookmark (never "now").
    if sessions.current is not None:
        logger.log_session_close(sessions.current, sessions.current.t_last_track)

    src.release()
    if video_writer is not None:
        video_writer.release()
    if live:
        cv2.destroyAllWindows()
    logger.write()

    if save_debug and debug_dir is not None and rejection_rows:
        import csv
        with open(debug_dir / "rejection_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rejection_rows[0].keys()))
            w.writeheader()
            w.writerows(rejection_rows)

    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="clip stem, e.g. D01 (looks for configs/conv_full_D01.yaml)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-snapshots", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save-video", action="store_true", help="write a full annotated video (annotated.mp4) to the run folder")
    ap.add_argument("--live", action="store_true", help="show the annotated overlay in a live window as it processes (press q to quit)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="--live playback speed multiplier (1.0 = real-time at the clip's native fps, 4.0 = 4x faster)")
    args = ap.parse_args()

    cfg = load_config("configs/default.yaml", f"configs/conv_full_{args.clip}.yaml")
    run_dir = run(cfg, args.clip, args.max_frames, not args.no_snapshots, args.debug,
                  save_video=args.save_video, live=args.live, live_speed=args.speed)
    print(f"run written to {run_dir}")


if __name__ == "__main__":
    main()
