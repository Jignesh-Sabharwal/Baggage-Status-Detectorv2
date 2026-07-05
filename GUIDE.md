# Command guide

All commands assume you're in the project root (`cd ~/Desktop/belt-loader-detection`) with
the virtualenv already created (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
if you haven't yet). `<clip>` below is one of: `D01`, `D02`, `D03`, `D04`, `N01`, `N02`, `N03`,
`N04`. Only `D01` and `D04` actually reach a working ROI lock with the current calibration —
see [RESULTS.md](RESULTS.md) before spending time on the others.

## Run the full pipeline

```bash
# Standard full run: writes runs/<clip>_<timestamp>/{config.yaml,events.csv,sessions.csv,snapshots/}
.venv/bin/python main.py <clip>

# Watch it live in a window at real-time (native fps), press q to quit
.venv/bin/python main.py <clip> --live --no-snapshots

# Watch live at 4x speed — useful since D01's first bag activity doesn't start until
# ~85s in, and D04's not until ~1020s in. Real-time (1x) means literally waiting that long.
.venv/bin/python main.py <clip> --live --no-snapshots --speed 4

# Write a full annotated video instead of/alongside snapshots
.venv/bin/python main.py <clip> --save-video

# Quick partial run — first N frames only, skip snapshot writing for speed
.venv/bin/python main.py <clip> --max-frames 2000 --no-snapshots

# Also dump per-frame rejection stats to runs/<run>/debug/rejection_stats.csv
.venv/bin/python main.py <clip> --debug

# Combine flags freely, e.g. live preview at 8x of just the first 3000 frames:
.venv/bin/python main.py <clip> --max-frames 3000 --live --speed 8 --no-snapshots
```

| Flag | Effect |
|---|---|
| `--max-frames N` | stop after frame N (omit for full clip) |
| `--no-snapshots` | skip writing `snapshots/` (faster) |
| `--save-video` | write `runs/<run>/annotated.mp4` |
| `--live` | show the overlay in a live `cv2` window, paced to real time; press `q` to quit early |
| `--speed X` | multiply `--live` playback speed (default 1.0 = real-time at the clip's native ~11fps) |
| `--debug` | also write `runs/<run>/debug/rejection_stats.csv` |

**Auto-CLAHE for low light:** every frame is brightness-checked before anything else touches it
(`preprocess.py`); frames with mean brightness below `preprocess.night_threshold` (default 70,
tune per camera in `configs/conv_full_<clip>.yaml`) get CLAHE contrast enhancement, everything
else passes through untouched. No flag needed — set `preprocess.enabled: false` in a clip's
config to disable it entirely.

**Note on `--live` timing:** D01's first bag activity starts at ~85s in (loader has to
arrive, dock, and the ROI has to settle for ~21s first); D04 doesn't lock until ~1020s in.
Without `--speed`, `--live` paces to the clip's real ~11fps, so at 1x you'll wait that long
before seeing any green track boxes — that's expected, not a bug. Use `--speed 4` or higher
to skip ahead faster, or just check `runs/<clip>_*/sessions.csv` instead of watching live.

## Check results after a run

```bash
# One-line summary across every clip's latest run (belt/dock/lock rates, sessions, bags)
.venv/bin/python tools/m6_report.py

# Inspect a specific run directly
cat runs/<clip>_*/sessions.csv
cat runs/<clip>_*/events.csv
open runs/<clip>_*/annotated.mp4        # if run with --save-video
open runs/<clip>_*/snapshots/           # if run without --no-snapshots
```

## Per-milestone diagnostics (isolate one stage without running the full pipeline)

```bash
# M0 — time base + stabilizer sanity, first ~36s of every clip
.venv/bin/python tools/m0_check.py

# M0 — full-clip time base scan (checks for VFR/timestamp fallback across whole clip)
.venv/bin/python tools/m0_fullscan.py

# M1 — presence + dock detection, full clip, prints DOCKED transitions
.venv/bin/python tools/m1_check.py <clip>
.venv/bin/python tools/m1_check.py <clip> --frames 2000   # limit frame count

# M1 — breakdown of presence/stationary/boom/fuselage/docked pass rates for one clip
.venv/bin/python tools/m1_breakdown.py <clip>

# M2 — Scene FSM through ROI lock; saves debug_frames/M2_<clip>_locked.jpg on success
.venv/bin/python tools/m2_check.py <clip>
.venv/bin/python tools/m2_check.py <clip> --frames 3000

# M2 — snapshot every ROI_LOCKED event across a full clip, for visually comparing lock quality
# (e.g. after a BeltDetector change). Writes debug_frames/LOCK_<clip>_<n>_f<frame>.jpg per lock.
.venv/bin/python tools/m2_lock_snapshots.py <clip>
.venv/bin/python tools/m2_lock_snapshots.py <clip> 5000    # limit to first 5000 frames

# M3 — bg model + bag detector + tracker on top of M1/M2; prints rejection totals
.venv/bin/python tools/m3_check.py <clip>
.venv/bin/python tools/m3_check.py <clip> --frames 5000
```

## Calibration / exploration scripts

These were used to derive the per-camera config values in `configs/conv_full_*.yaml` and are
not meant for ad-hoc reuse — they loop over a hardcoded clip list rather than taking a `<clip>`
argument. Re-run them only if you're recalibrating a camera from scratch.

```bash
# Extract 5 sample frames per clip into debug_frames/ for visual inspection
.venv/bin/python tools/extract_samples.py

# Semi-automatic HSV band suggestion per clip (prints hsv_lo/hi + area fraction)
.venv/bin/python tools/calibrate_presence.py

# Visualize a specific HSV mask overlay (edit the CASES dict at the top of the file first)
.venv/bin/python tools/inspect_mask.py

# Dump dock-detector internals (stationary/boom/fuselage) for one clip+frame
.venv/bin/python tools/debug_dock_frame.py <clip> <frame_number>
```

## Notes

- Every `tools/*.py` script and `main.py` must be run with `.venv/bin/python`, not bare
  `python`/`python3` — the venv has `opencv-python`/`numpy`/`scipy`/`pyyaml` installed, the
  system interpreter doesn't.
- `runs/` and `debug_frames/` accumulate over time; delete old runs freely, nothing reads them
  back except `tools/m6_report.py` (which always picks the *latest* run per clip).
