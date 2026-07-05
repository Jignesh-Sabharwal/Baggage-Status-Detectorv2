# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

M0-M6 are implemented (see `RESULTS.md` for what was actually validated vs. not). There is no
automated test suite — validation is empirical, via the `tools/m*_check.py` scripts and full
pipeline runs, not unit tests.

Setup and running (always use `.venv/bin/python`, not bare `python`/`python3` — the system
interpreter doesn't have the deps):
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py D01                          # full run, writes runs/D01_<timestamp>/
.venv/bin/python main.py D01 --max-frames 2000 --no-snapshots --debug
.venv/bin/python main.py D01 --live --speed 4          # live cv2 preview window at 4x, press q to quit
.venv/bin/python main.py D01 --save-video              # also write runs/<run>/annotated.mp4
```
`<clip>` is a stem matching `configs/conv_full_<clip>.yaml` / `videos/conv_full_<clip>.mp4`
(one of D01-D04, N01-N04). **D01 (in-sample), D04, N01, N02, and N04 (zero-shot) currently
reach a working ROI lock** — N01/N02/N04 needed per-camera dock (`min_boom_angle_deg`, etc.)
and ROI-stability (`width_band_px`, etc.) overrides beyond presence/HSV, same category of
tuning D01 itself required. D02/D03/N03 still fail at the dock stage (fuselage-gate and/or
stationarity); see `RESULTS.md` for the per-clip breakdown before spending time on them.

Per-milestone checks (each isolates one pipeline stage without running the full thing):
`tools/m0_check.py` / `m0_fullscan.py` (time base + stabilizer), `m1_check.py` /
`m1_breakdown.py` (presence + dock), `m2_check.py` / `m2_lock_snapshots.py` (ROI lock),
`m3_check.py` (bg model + detector + tracker) each take a clip name (`<clip> [--frames N]`).
`tools/m6_report.py` (no args) aggregates the latest run per clip into the zero-shot summary
table. Calibration scripts (`extract_samples.py`, `calibrate_presence.py`, `inspect_mask.py`,
`debug_dock_frame.py`) loop over a hardcoded clip list to derive per-camera config values —
not for ad-hoc reuse, only re-run when recalibrating a camera from scratch. Full flag/command
reference: `GUIDE.md`.

`runs/` and `debug_frames/` accumulate across runs (both gitignored) and nothing reads old
ones back except `m6_report.py`, which always picks the latest run per clip — delete freely.

Allowed dependencies (per the spec): `opencv-python` (`cv2`), `numpy`, `scipy` (generic signal
ops only), `pyyaml`, standard library. No other third-party packages.

## Non-negotiable constraint: classical CV only

This is the single most important rule in the spec (REQ-00) and overrides convenience in every
other requirement: **no deep learning anywhere** — no neural networks, no pretrained models, no
learned embeddings, no ONNX/TensorRT inference, no external ML API calls, for detection,
tracking, or classification.

Banned unconditionally: `torch`, `tensorflow`, `keras`, `onnxruntime`, `ultralytics`/YOLO (any
version), any Hugging Face model or `transformers` import, any pretrained person/object
detector, any Re-ID/embedding network, any GAN/CNN-based background modeling, any "small CNN"
quick fix. The one borderline exception is `cv2.HOGDescriptor_getDefaultPeopleDetector()`
(classical HOG+SVM), permitted only as an explicit, separately-toggleable fallback if the
geometric person-rejection stack proves inadequate — never silently substituted in.

If a classical approach underperforms during calibration, the correct response is to retune
thresholds or add another classical geometric cue — not to introduce a learned model. If a
classical approach seems fundamentally insufficient somewhere, stop and say so explicitly rather
than quietly adding a DL dependency.

## Requirements are mandatory, not suggestions

Every `REQ-nn` in `README.md` is mandatory and testable. Do not silently drop, weaken, or
"simplify" a REQ. If a REQ conflicts with something else, stop and flag it rather than resolving
it unilaterally. Sections marked *Known limitation* must be reproduced verbatim in the generated
project README — they are honesty commitments, not TODOs.

**[PER-CAMERA]**-tagged values (HSV color band, dock boom angle, ROI angle/aspect bands,
axis-end sign convention, `far_scale`) do not transfer across cameras and must live in
per-installation config files (`configs/conv_full_D01.yaml`, `configs/conv_full_N01.yaml`,
...), never as shared defaults or hard-coded constants.

## Reference data

`README.md`'s original spec was written against exactly two clips (`conv_D01.mp4` /
`conv_N01.mp4`, ~300s/240s). The clips actually present in this repo are eight longer,
higher-variety recordings (`configs/conv_full_<clip>.yaml`, `videos/conv_full_<clip>.mp4`);
per user decision the scope was extended to treat all 8 as separate installations rather than
reverting to the spec's original two. See `RESULTS.md` for full per-clip specs and results.

- **D01** is the sole tuning clip (all structural threshold tuning happens here, per the
  spec's calibration protocol).
- **D02-D04, N01-N04** are held out: non-PER-CAMERA thresholds are applied unchanged from D01
  as a zero-shot stress test (only PER-CAMERA values like the HSV band are recalibrated per
  clip). This is a materially larger generalization test than the spec's original one-clip
  design — never describe it as "the pipeline supports 8 cameras," and never claim
  generalization beyond what `RESULTS.md` actually measured (currently: D04 is the only
  zero-shot clip that reaches a working ROI lock; the rest fail at the dock stage).
- Four of the eight clips (N01-N04 by naming, but verify per `RESULTS.md` — the spec's
  assumption that "N" means "stand identifier, not night clip" does **not** hold for this
  dataset) are genuine night footage, which uses a value-only (brightness) presence cue
  instead of the daytime hue band — a narrower, known-weaker cue (see REQ-11 in `RESULTS.md`'s
  limitations section).

## Architecture: two stacked state machines

**Scene FSM** (slow, structural): `INIT → NO_BELT → BELT_PRESENT → DOCKED → ROI_SETTLING →
MONITORING`, with teardown transitions back through earlier states.

**Activity FSM** (fast, operational, ticks only while Scene FSM = MONITORING):
`IDLE ⇄ LOADING`, `IDLE ⇄ UNLOADING`, with a direct LOADING⇄UNLOADING flip allowed.

The bag detector, tracker, and Activity FSM exist only during MONITORING, but the background
model (MOG2) is instantiated earlier, at `BELT_PRESENT` entry, so it's fully warmed up by the
time MONITORING starts — its output is simply ignored before that. This split (model runs early,
consumers don't) must stay explicit in `bg_model.py`'s docstring.

## Module layout (as specified)

```
main.py
├── config.py            # dataclass tree from YAML; verbatim dump into each run folder
├── time_base.py         # PTS-first timestamping with frame_idx/fps fallback
├── video_source.py      # frame iterator; yields (frame_idx, t_video, frame)
├── stabilizer.py        # global-motion estimate from reference patches
├── preprocess.py        # brightness-triggered CLAHE, applied before any detector
├── scene/
│   ├── presence.py      # ScenePresence — is a loader in frame? [PER-CAMERA color band]
│   ├── dock.py          # DockDetector — stationary + boom raised + aircraft gate
│   └── cover.py         # CoverClassifier — texture-based canopy test
├── roi/
│   ├── belt_detector.py # BeltDetector — hypothesis-driven belt-axis fit (color+geometry+motion)
│   ├── roi_manager.py   # structural_fit(), motion_refine(), lock logic, drift monitor
│   └── roi_health.py    # zero-detection watchdog → forced re-settle
├── detect/
│   ├── bg_model.py      # MOG2 wrapper: shadows, per-state learning rate, burst guard,
│   │                    #   belt-pause freeze + ghost filter
│   └── bag_detector.py  # contours + position-dependent geometric filters
├── track/
│   └── tracker.py       # centroid tracker + dead-track re-association
├── activity/
│   ├── direction.py     # axis projection, signed velocity, per-track vote
│   └── activity_fsm.py  # IDLE/LOADING/UNLOADING, M-of-N
├── output/
│   ├── events.py        # event log (CSV) + session ledger
│   ├── snapshots.py     # snapshot policy + annotated frame writer
│   └── annotate.py      # overlay renderer
└── runs/<video>_<ts>/   # config.yaml, events.csv, sessions.csv, snapshots/, debug/
```

Every module consumes and returns plain data (numpy arrays, dataclasses) — no module reaches
into another's state. Thresholds are always expressed in config as seconds and converted to
frames at load time; output timestamps always come from `time_base.py`, never a private
`idx / fps` computation in a module.

## Key semantics to preserve

- **Session `t_end`** is always the last confirmed-track sighting timestamp, regardless of how
  the session closes (idle timeout, ROI reposition, teardown) — never the transition time. This
  is what makes the boundary-accuracy target achievable; `sessions.csv` also records `t_closed`
  (when the FSM actually transitioned) so both are auditable.
- **ROI reposition** holds any open session open with a `t_last_track` bookmark; if activity of
  the same direction resumes within the grace window, the session continues seamlessly,
  otherwise it closes with `t_end = t_last_track`.
- **Dead-track re-association** (extrapolated position + velocity match) exists to control ID
  churn without a learned Re-ID model — inherited tracks take no new snapshot and count no new
  bag.
- **Ghost filter**: a foreground blob is discarded as a ghost (absorbed-object imprint) by
  comparing its live-frame edge density against the background model's own image at the same
  region — not by a learned ghost-detection network.
- Bag counts have known bidirectional error (merged blobs undercount, ID churn overcounts) —
  this must be reported with both caveats, never hidden.

## Build order

Milestones M0–M6 in `README.md` §8 define the required build sequence (time base →  Scene FSM →
ROI manager → background/detection/tracking → activity FSM → overlay → calibration/validation).
Each milestone must run end-to-end on `conv_D01.mp4` and produce a viewable artifact before the
next milestone is started — never build more than one untested layer deep.

## Overlay format

The annotated output overlay (§5.1) must match two supplied reference screenshots
pixel-for-pixel in layout — this is the acceptance criterion for milestone M5, not a stylistic
suggestion. See `README.md` REQ-32 through REQ-38 for the exact status block, ROI color-coding,
track box, contact marker, clock, timestamp bar, and event-flash specifications.
