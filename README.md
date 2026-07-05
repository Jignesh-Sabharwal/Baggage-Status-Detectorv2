# Belt Loader Activity Detection — Pipeline Design (v2, corrected)

Classical CV pipeline (Python + OpenCV, **no deep learning, no pretrained or learned
detectors of any kind** — see REQ-00) that watches ramp CCTV of an aircraft belt loader
and produces a complete turnaround timeline: belt arrival, docking, loading/unloading
sessions with start/end times, per-bag snapshots, and departure.

Reference input: `conv_D01.mp4` — 324×216, 11 fps, 300 s, 3301 frames.
All defaults below are calibrated for this clip and must be treated as
**config values, not constants**. Where a value cannot transfer across cameras
(color band, axis sign), it is explicitly marked **[PER-CAMERA]** and belongs in a
per-installation calibration file, not the shared defaults.

> **Instruction to the implementing model:** every requirement tagged `REQ-nn` is
> mandatory and testable. Do not silently drop, weaken, or "simplify" a REQ. If a REQ
> conflicts with something else you infer, stop and flag it rather than resolving it
> yourself. Sections marked *Known limitation* must be reproduced verbatim in the
> generated project README — they are honesty commitments, not TODOs.

---

## 0. Reference assets (verified)

Two independent clips are available — not one. Specs below are ffprobe-verified, not
estimated:

| Clip | Resolution | FPS | Duration | Frames | Role |
|---|---|---|---|---|---|
| `conv_D01.mp4` | 324×216 | 11 | 300.09 s | 3301 | **Primary** — all threshold tuning, milestones M1–M5 |
| `conv_N01.mp4` | 438×208 | 11 | 241.09 s | 2652 | **Secondary / held-out** — validation only |

Both are daytime footage of an IndiGo-liveried belt loader (confirmed by direct frame
inspection — "N" is a stand/camera identifier, not a night clip). The two clips differ
in resolution and framing, which means every **[PER-CAMERA]** value in this document
(HSV band, dock boom angle, ROI angle/aspect bands, axis-end sign convention, area-band
`far_scale`) is genuinely different between them and must be calibrated twice, into two
separate config files (`configs/conv_D01.yaml`, `configs/conv_N01.yaml`).

- **REQ-01 (revised — held-out data now exists):** the "no generalization claims" rule
  from the original scope is now partially liftable. All threshold tuning (§7) happens
  against `conv_D01.mp4` only. `conv_N01.mp4` is *never* used to tune any
  **non-PER-CAMERA** threshold (solidity, axis coherence, idle-timeout profile, M-of-N,
  health-watchdog constants, ghost-filter ratios) — those are evaluated on N01 exactly
  as tuned on D01, with zero retuning, and the resulting boundary/count error against
  N01's own manually-scrubbed ground truth is reported as genuine **out-of-sample**
  performance for the structural logic. PER-CAMERA values are recalibrated for N01
  through the normal §7 procedure (that recalibration is expected and is not a
  generalization claim in itself). A one-clip-tuned / one-clip-validated result is still
  a small sample — report it as "validated on one held-out clip," not "generalizes."
- **REQ-01b (stress test, cheap and informative):** additionally run N01 once through
  the **D01-tuned PER-CAMERA config unchanged** (no recalibration at all) and log what
  breaks. This "zero-shot" failure report (which stage fails first — presence, dock,
  ROI lock, or detection — and why) is more diagnostic than the properly-recalibrated
  run and must be included in the generated README as a short table, not discarded.

---

## 0.1 Hard constraint: classical CV only — no deep learning, anywhere

- **REQ-00 (non-negotiable, overrides convenience in every other REQ):** the entire
  pipeline uses **classical computer vision only**: color thresholding, background
  subtraction (MOG2), edge/line detection (Canny, Hough), contour geometry, morphology,
  optical-flow/phase-correlation, centroid tracking, and hand-written heuristics/FSMs.
  **No neural networks, no pretrained models, no learned embeddings, no ONNX/TensorRT
  inference, and no calls to an external ML API, at any stage** — not for detection, not
  for tracking, not for classification, not "just for the hard part."
  - **Banned, unconditionally:** `torch`, `tensorflow`, `keras`, `onnxruntime`,
    `ultralytics`/YOLO (any version), any Hugging Face model or `transformers` import,
    any pretrained person/object detector (HOG+SVM is the one classical exception, see
    below), any Re-ID / feature-embedding network, any GAN- or CNN-based background
    modeling (e.g. the BGAN-style ghost-removal networks that show up in the
    literature), any "small CNN" proposed as a quick fix.
  - **Allowed dependencies:** `opencv-python` (`cv2`), `numpy`, `scipy` (only for
    generic signal ops such as filtering — not `scipy` ML submodules), `pyyaml`,
    standard library. `cv2`'s classical HOG+SVM pedestrian detector
    (`cv2.HOGDescriptor_getDefaultPeopleDetector()`) is the **only** borderline
    exception permitted if the person-rejection stack (REQ-25) proves inadequate, and
    only if added as an explicit, documented, separately-toggleable fallback — never
    silently substituted for the geometric filters.
  - **Why this matters here specifically:** several of this document's own failure
    modes are exactly the kind of problem a coding assistant instinctively "fixes" by
    reaching for a pretrained model — person detection (REQ-25), perspective-robust
    object detection (REQ-24), ghost/sleeping-object removal (REQ-22), and re-appearance
    matching for tracking (REQ-27). Every one of those must be solved with the
    classical technique already specified in this document (geometric filters,
    position-scaled thresholds, background-image edge comparison, and
    extrapolated-position matching, respectively). If a REQ's classical approach
    underperforms during calibration, the correct response is to retune thresholds or
    add another classical geometric cue — **not** to introduce a learned model. If the
    implementing model believes a classical approach is fundamentally insufficient
    somewhere, it must say so explicitly and stop, rather than quietly adding a DL
    dependency.
  - This constraint is also why the project exists: the design deliberately avoids GPU
    dependency (shared/no GPU environment) and favors auditable, debuggable heuristics
    over black-box inference. A DL component anywhere breaks that premise even if it
    "only" improves one metric.

---

## 0.2 Scope and honesty statement

- **REQ-01** All quantitative claims produced by this system on `conv_D01.mp4` are
  **in-sample** (thresholds were tuned on the same clip). The generated README and all
  metric reports must label them as such. Out-of-sample validation requires at least one
  held-out clip (different bay / time of day / loader if possible); until one exists,
  no claim of generalization is permitted anywhere in code comments, docs, or output.
- The system detects "**loader stationary with elevated belt, adjacent to a large
  aircraft-like region**" — not "docked" in an operational sense. Naming in code and
  events keeps the DOCKED label for brevity, but the docstring of `DockDetector` must
  state this definition.

---

## 1. Core architecture: two stacked state machines

**Scene FSM (slow, structural):**

```
INIT → NO_BELT → BELT_PRESENT → DOCKED → ROI_SETTLING → MONITORING
                     ↑______________|_________|______________|
                            (teardown transitions)
```

**Activity FSM (fast, operational)** — ticks only while Scene FSM is in `MONITORING`:

```
IDLE ⇄ LOADING
IDLE ⇄ UNLOADING
LOADING ⇄ UNLOADING (direct flip allowed, same debounce)
```

- **REQ-02** The bag *detector*, *tracker*, and *Activity FSM* exist only in MONITORING.
- **REQ-03** The *background model* is instantiated earlier — at entry to
  `BELT_PRESENT` — and runs continuously from there, so it is fully warmed up by the
  time MONITORING begins. Its output is simply ignored outside MONITORING. This closes
  the "MOG2 warm-up blind window" without violating REQ-02. State this split explicitly
  in `bg_model.py`'s docstring: the *model* runs early; the *consumers* do not.

### 1.1 INIT — consistent with its own tests

`conv_D01.mp4` starts with the loader already docked. INIT classifies the opening frames
into whichever state is already true and jumps directly to the deepest state that holds.

- **REQ-04** INIT's observation window is
  `init.window_s = max(dock.window_s, presence.debounce_s)` (≈ 5 s with defaults) —
  **not** a shorter ad-hoc window. The dock stationarity test needs its full window; a
  3 s INIT cannot evaluate a 5 s test. INIT runs the same presence and dock tests as the
  normal path, on the same windows.
- **REQ-05** If INIT lands in DOCKED, the run enters ROI_SETTLING immediately. The
  stability-based lock criterion (§3.3) may fire as soon as its minimum settle time and
  variance conditions are met; since the belt is already stationary, lock typically
  happens near `roi.min_settle_s`.
- **REQ-06 (startup blind window is real and must be reported):** any activity occurring
  before ROI lock is unobservable. If the Activity FSM's *first* session begins within
  `2 × fsm.idle_timeout_s` of ROI lock, that session must be written to `sessions.csv`
  with `truncated_start = true` and `t_start` set to lock time. The pipeline never
  fabricates a pre-lock start time. This limitation goes in the generated README.

---

## 2. Module decomposition

```
main.py
├── config.py            # dataclass tree, loaded from YAML; verbatim dump into each run folder
├── time_base.py         # PTS-first timestamping with frame_idx/fps fallback (REQ-07/08)
├── video_source.py      # frame iterator; yields (frame_idx, t_video, frame)
├── stabilizer.py        # global-motion estimate from reference patches (REQ-09)
├── scene/
│   ├── presence.py      # ScenePresence  — is a loader in frame? [PER-CAMERA color band]
│   ├── dock.py          # DockDetector   — stationary + boom raised + aircraft gate
│   └── cover.py         # CoverClassifier — texture-based canopy test
├── roi/
│   ├── roi_manager.py   # structural_fit(), motion_refine(), lock logic, drift monitor
│   └── roi_health.py    # zero-detection watchdog → forced re-settle (REQ-19)
├── detect/
│   ├── bg_model.py      # MOG2 wrapper: shadows, per-state learning rate, burst guard,
│   │                    #   belt-pause freeze + ghost filter (REQ-21/22/23)
│   └── bag_detector.py  # contours + position-dependent geometric filters (REQ-24)
├── track/
│   └── tracker.py       # centroid tracker + dead-track re-association (REQ-27)
├── activity/
│   ├── direction.py     # axis projection, signed velocity, per-track vote
│   └── activity_fsm.py  # IDLE/LOADING/UNLOADING, M-of-N with defined semantics (REQ-29)
├── output/
│   ├── events.py        # event log (CSV) + session ledger with defined t_end (REQ-31)
│   ├── snapshots.py     # snapshot policy + annotated frame writer
│   └── annotate.py      # overlay renderer
└── runs/<video>_<ts>/   # config.yaml, events.csv, sessions.csv, snapshots/, debug/
```

Every module consumes and returns plain data (numpy arrays, dataclasses). No module
reaches into another's state.

### 2.1 Time base — `time_base.py`

`frame_idx` alone assumes constant frame rate; DVR exports are frequently variable-frame-rate
with dropped frames, and OpenCV's `CAP_PROP_POS_MSEC` is itself unreliable on VFR files
through the ffmpeg backend (documented OpenCV issues). So:

- **REQ-07** `t_video` for each frame = `CAP_PROP_POS_MSEC / 1000` read immediately after
  the frame grab, **validated** by a monotonicity-and-plausibility check: timestamps must
  be strictly increasing and each inter-frame delta must lie in
  `[0.2/fps_nominal, 5/fps_nominal]`. On the first violation, the source permanently
  falls back to `frame_idx / fps_nominal` for the remainder of the run and logs
  `TIMEBASE_FALLBACK` once in events.csv.
- **REQ-08** All thresholds in config are expressed in **seconds** and converted to
  frames at load time (`frames = round(seconds * fps_nominal)`). All *output* timestamps
  come from `time_base`, never from a private `idx / fps` computation in a module.

### 2.2 Global-motion guard — `stabilizer.py`

Pole-mounted CCTV shakes; PTZ presets get nudged. Untreated, this floods MOG2 and makes
the ROI drift monitor fire on camera motion instead of loader motion.

- **REQ-09** Maintain 2–3 small reference patches (config: `stab.patches`, default
  auto-selected as high-texture regions **outside** the loader blob and ROI at
  BELT_PRESENT entry). Each frame, estimate global displacement via
  `cv2.phaseCorrelate` on the patches (median of patch displacements).
- **REQ-10** If |global displacement| > `stab.shake_px` (≈ 1.5 px): (a) the ROI drift
  monitor treats this frame as **camera motion, not ROI motion** and does not count it
  toward reposition detection; (b) if sustained > `stab.shake_s` (≈ 2 s), fire the
  burst-guard path (skip detection, raised learning rate) and log `CAMERA_MOTION`.
- This is a guard, not video stabilization. No warping of frames.

---

## 3. Scene FSM — stage by stage

### 3.1 NO_BELT → BELT_PRESENT (ScenePresence)

Cue: the loader is a large saturated-color mass on grey tarmac.

- HSV threshold on `presence.hsv_lo / presence.hsv_hi` **[PER-CAMERA]** → binary mask →
  morphological close (5×5) → largest connected component.
- Presence score = blob passes area band (`presence.min_area_frac` ≈ 3–10 % of frame)
  and coarse aspect check.
- Debounce: state flips only after the score holds for `presence.debounce_s` (≈ 2 s).

Why not MOG2: a parked loader becomes background within seconds. Presence must be an
**appearance** test, not a motion test.

- **REQ-11 (stated limitation, verbatim in README):** *Presence detection is a
  single-color-band detector calibrated to one operator's blue loaders under daytime
  lighting. It will fail on night/IR camera modes, sodium-vapor lighting, other
  operators' liveries, and can lock onto other large same-colored GSE (e.g. a catering
  truck). Supporting other cameras requires re-calibrating `presence.hsv_*` per
  installation; supporting night operation requires replacing the cue entirely.*
- **REQ-12** If two blobs pass the area band, prefer the one that (a) overlaps the
  previous accepted blob, else (b) has aspect ratio nearer the loader prior
  (`presence.aspect_prior`). Never silently "largest wins" when multiple candidates exist —
  log `PRESENCE_AMBIGUOUS` to debug.

Optional arrival refinement (dual-rate trick) retained as before; not required for
correctness.

### 3.2 BELT_PRESENT → DOCKED (DockDetector)

Three signatures:

1. **Stationary** (primary): blue-blob centroid displacement < `dock.max_drift_px`
   (≈ 2 px, *after subtracting global displacement from stabilizer*, REQ-09) over
   `dock.window_s` (≈ 5 s).
2. **Boom raised** (primary): Canny + probabilistic Hough inside the loader blob's
   bounding region; dominant long-line angle ≥ `dock.min_boom_angle_deg`
   (≈ 15°, calibrate on this clip — driving posture is near-horizontal).
3. **Aircraft adjacency** (gate, upgraded from "bonus"): fuselage proxy = large bright
   low-edge-density connected region occupying ≥ `dock.min_fuselage_frac` (≈ 8 %) of the
   frame; require the boom's elevated endpoint within `dock.fuselage_dist_px`
   (10–20 px) of that region's contour.

- **REQ-13** DOCKED fires when **(1) AND (2) AND (3)** hold. Rationale: a loader parked
  in queue with boom partially raised passes (1)+(2); only aircraft adjacency separates
  "docked" from "parked with boom up". At 324×216 the fuselage test is coarse, so it is
  a **gate with a generous threshold**, not a precision measurement: it need only answer
  "is there a large aircraft-like region on the boom's elevated side?" If it is genuinely
  unreliable on the calibration clip, it may be relaxed to *(1) AND (2) AND (3-weak)*
  where 3-weak only requires the fuselage-proxy region to exist anywhere in the boom-side
  half of the frame — but it may not be dropped, and the relaxation must be a config flag
  (`dock.fuselage_gate: strict|weak`), defaulting to `strict`.

### 3.3 DOCKED → ROI_SETTLING → ROI locked (ROIManager)

Two-phase design, explicit in the API:

**Phase A — `structural_fit()` at dock time.** Within the loader blob: edges → Hough →
cluster long lines by angle → dominant parallel pair (belt rails) → construct rotated
rect (`cv2.minAreaRect` on rail support points, padded by `roi.pad_px` ≈ 4 px).
Output: center (x, y), angle θ, length L, width W — plus the belt axis unit vector û and
the aircraft-end sign convention.

- **REQ-14 [PER-CAMERA]** The aircraft-end sign convention ("elevated end = smaller
  image y = aircraft") is a **per-installation config value**
  (`camera.aircraft_end: min_y | max_y | min_x | max_x`), verified during calibration
  step 5 (§7). It is true for this clip and false for a camera on the other side of the
  stand. It must never be hard-coded.

**Lock criterion — stability-driven, not a fixed timer.** Re-fit every frame; keep a
sliding window of `roi.window_s` (≈ 10 s) of the four parameters. Lock when BOTH:

- minimum settle time elapsed (`roi.min_settle_s` ≈ 15 s), and
- windowed stds under tolerance: center std < 3 px, angle std < 2°, length std < 5 px.

Locked ROI = EMA over the window, not the last raw fit.

- **REQ-15 (lock plausibility check):** before accepting a lock, verify the candidate ROI
  against structural priors: aspect ratio L/W ∈ `roi.aspect_band` (≈ [2.5, 8]),
  W ∈ `roi.width_band_px`, ROI ∩ loader-blob overlap ≥ 60 %, and θ within
  `roi.angle_band_deg` of the boom angle from DockDetector. A fit that is stable but
  implausible (locked onto a ground marking or the loader's body edge) is rejected and
  settling continues. This is the "hypothesis-driven belt detection" made concrete.

**Drift monitor after lock.** Re-fit every `roi.drift_check_s` (≈ 5 s). If the fresh fit
deviates beyond 2× lock tolerance for 3 consecutive checks (**after global-motion
subtraction**, REQ-10) → emit `ROI_REPOSITIONED`, fall back to ROI_SETTLING, pause the
Activity FSM.

- **REQ-16 (session semantics through reposition):** on `ROI_REPOSITIONED`, any open
  session is **held open** with a bookmark `t_last_track` = last confirmed-track
  sighting. If MONITORING resumes and activity of the same direction restarts within
  `fsm.reposition_grace_s` (≈ 60 s), the session continues seamlessly. Otherwise it is
  closed with `t_end = t_last_track` (never the reposition time, never the resume time).
- **REQ-17 (teardown from SETTLING is defined, not implied):** presence and dock
  monitoring keep running during ROI_SETTLING. If undock/teardown fires from SETTLING,
  the held-open session closes with `t_end = t_last_track`, then normal teardown events
  are emitted. The FSM diagram allows this path; this REQ gives it semantics.

**Phase B — `motion_refine()` during MONITORING.** Accumulate confirmed-track centroids
into a heat image; after the first ~10 confirmed tracks, shrink/re-center the ROI to the
band where bags actually travel.

- **REQ-18** `motion_refine()` may only *shrink or translate within* the plausibility
  bands of REQ-15, never grow the ROI or rotate it beyond ±3°.

### 3.3b ROI health watchdog — `roi_health.py`

`motion_refine()` needs tracks to exist; a wrong lock produces zero tracks; zero tracks
means the wrong lock is never corrected. Break the loop:

- **REQ-19** While in MONITORING with Activity FSM = IDLE: compute the foreground ratio
  in a dilated annulus around the ROI (ROI expanded by `health.annulus_px` ≈ 15 px,
  minus the ROI itself). If (annulus foreground ratio > `health.motion_frac` ≈ 0.10,
  sustained ≥ `health.sustain_s` ≈ 20 s) AND (zero confirmed tracks in the same period),
  the lock is presumed wrong: emit `ROI_HEALTH_RESET`, discard the lock, return to
  ROI_SETTLING with the previous lock excluded as a hypothesis (its parameters are
  blacklisted within tolerance for the next fit).

### 3.4 Covered vs open belt (CoverClassifier)

Runs at ROI lock and every `cover.recheck_s` (≈ 30 s). **Texture, not area, not
background subtraction.** Inside the ROI: edge-pixel ratio; gradient-orientation
histogram (open belt peaks along û); mean HSV as supporting cue.

- **REQ-20** Rechecks are **gated on Activity FSM = IDLE and zero confirmed tracks**.
  Bags on the belt inflate the edge-pixel ratio and would make the classifier flap
  mid-session. The at-lock check is exempt (no tracks can exist yet). If COVERED: stay
  in MONITORING, suppress the Activity FSM, log `BELT_COVERED` / `BELT_UNCOVERED`.

### 3.5 Teardown transitions

- Boom lowers or blob centroid moves (> `dock.max_drift_px` sustained, global motion
  subtracted) → `DOCKED → BELT_PRESENT`: close any open session with
  `t_end = t_last_track`, emit `BELT_UNDOCKED`.
- Presence test fails for `presence.debounce_s` → `NO_BELT`, emit `BELT_DEPARTED`.

---

## 4. MONITORING — the bag pipeline

### 4.1 Background model (bg_model.py)

- `cv2.createBackgroundSubtractorMOG2(history=400, detectShadows=True)`; drop shadow
  pixels (value 127) before any morphology. Instantiated at BELT_PRESENT entry (REQ-03).
- **Burst guard:** foreground ratio inside ROI > `bg.burst_frac` (≈ 0.4) → skip
  detection this frame, apply raised learning rate for 1 s. Also triggered by REQ-10.
- **Per-state learning rate:** normal (`lr = -1`, auto) in MONITORING; frozen (`lr = 0`)
  during ROI_SETTLING re-entry so a repositioned belt doesn't burn ghost trails.

Two additions close the stopped-belt failure (bags absorbed into background; ghosts on
restart):

- **REQ-21 (belt-pause freeze):** while MONITORING and ≥ 1 confirmed track exists but
  the mean |axis velocity| across confirmed tracks < `bg.pause_speed` for
  ≥ `bg.pause_s` (≈ 3 s) — i.e. the belt has stopped with bags on it — call
  `apply(frame, learningRate=0)` so stationary bags are **not** absorbed into the
  background. Restore automatic learning when mean speed recovers or all tracks end.
  (MOG2's `apply` with `learningRate=0` performs subtraction without updating the model —
  this is the documented mechanism, and mirrors the standard "sleeping object" strategy
  of slowing learning where confirmed foreground sits.)
- **REQ-22 (ghost filter):** a foreground blob is flagged as a ghost and discarded if it
  (a) has zero centroid displacement over `bg.ghost_s` (≈ 2 s) AND (b) its interior
  Canny edge density in the *current frame* is < `bg.ghost_edge_ratio` (≈ 0.5) × the
  edge density of the same region in the model's background image
  (`getBackgroundImage()`). A ghost is the imprint of an absorbed object — it has edges
  in the background model but not in the live frame. This edge-density comparison is
  the classical solution to ghost removal (per REQ-00); do not substitute a learned
  ghost-detection network even though such models exist in the literature.
- **REQ-23** Ghost discards are counted in the per-frame rejection stats (see 4.2).

### 4.2 Bag detector (bag_detector.py)

Order matters — mask first, then morphology, then contours:

1. AND foreground with the ROI polygon mask.
2. Morphology: open 3×3 → close 5×5. Larger kernels erase bags at this resolution.
3. Contours → filter chain, each rejection counted for debug stats:

- **REQ-24 (position-dependent area band):** the belt recedes from the camera, so
  apparent bag area varies along the axis. The area band is a **linear function of the
  normalized axis coordinate** `s ∈ [0, 1]` (0 = loader end, 1 = aircraft end):
  `min_area(s) = det.min_area_near · lerp(1, det.far_scale, s)` and likewise for
  `max_area(s)`. Calibrate `det.far_scale` in calibration step 4 by measuring one
  real bag's blob area at both ends of the belt on this clip. Defaults:
  `min_area_near = 80 px²`, `max_area_near = 400 px²`, `far_scale = 0.5`
  (placeholder — **must** be measured, not trusted).
- solidity ≥ 0.45 — **provisional**: this value was carried from a different pipeline
  with different preprocessing; it is downstream of mask quality. Re-derive it in
  calibration step 4 from the solidity histogram of confirmed real-bag blobs vs
  rejected person blobs on this clip. Log both histograms in `debug/`.
- aspect ratio ∈ [0.3, 3.5];
- centroid strictly inside the ROI polygon;
- ghost filter (REQ-22).

Output: `Detection(centroid, bbox, area, solidity, s_axis)` per frame.

- **REQ-25 (operator limitation, verbatim in README):** *The rejection stack (ROI mask →
  position-scaled area → solidity → centroid-in-polygon → min track age → axis
  coherence) is designed against people crossing or leaning over the belt. It is NOT
  reliable against the loader operator working at the belt end: an operator handling a
  bag merges with it into one blob, and an operator walking alongside the belt moves
  axis-coherently. Expect occasional operator-inflated blob sizes and rare spurious
  tracks at the belt ends. The position-dependent `max_area(s)` ceiling (REQ-24) is the
  main mitigation; residual error is accepted and reported, not hidden.* Per REQ-00,
  do not resolve this by adding a person/pose detector; the classical HOG+SVM fallback
  described in REQ-00 may be evaluated as an explicit, toggleable, separately-reported
  addition, but geometric filtering remains the primary and default mechanism.

### 4.3 Tracker (tracker.py)

Plain nearest-neighbour centroid tracker — no Kalman, no Hungarian. At 11 fps this is
sufficient and stays minimal.

- match gate: `trk.max_dist_px` ≈ 25 px;
- `trk.max_disappeared` ≈ 8 frames (0.7 s);
- confirmation at age ≥ `trk.min_age` = 5 frames;
- **axis-coherence filter** at confirmation:
  |net displacement · û| / |net displacement| ≥ `trk.axis_coherence` (≈ 0.7).
- Per confirmed track: EMA of signed axis velocity `v = d(centroid·û)/dt`, α ≈ 0.3.

- **REQ-26 (occlusion is signal):** tracks terminating within the aircraft-end zone
  (last 15 % of ROI length, i.e. `s > 0.85` with REQ-14's sign convention) count as
  *delivered into the hold*, not lost.
- **REQ-27 (dead-track re-association — ID churn control):** when a new track reaches
  confirmation, search tracks that died within the last `trk.reassoc_s` (≈ 2 s) whose
  *extrapolated* position (last centroid + last EMA velocity × elapsed time, along û) is
  within `trk.reassoc_px` (≈ 20 px) of the new track's first centroid, with matching
  velocity sign. On match: the new track **inherits** the dead track's ID, age, and
  vote history; no new bag is counted; no new snapshot is taken. Tracks that died in
  the aircraft-end delivery zone (REQ-26) are excluded from re-association. This
  extrapolated-position-and-velocity match is the classical solution (per REQ-00); do
  not substitute a learned Re-ID / appearance-embedding matcher.
- **REQ-28 (bidirectional count error, verbatim in README):** *Bag counts have
  bidirectional error: merged adjacent bags undercount (splitting merged blobs
  classically via distance transform + watershed is too noisy at 324×216 and is
  deliberately not attempted); residual ID churn past REQ-27 overcounts. Counts are
  reported with both caveats.*

### 4.4 Direction and the Activity FSM

Per-track vote — a confirmed track votes only if:

- net axis displacement ≥ `dir.min_disp_px` (≈ 8 px), and |v| ≥ `dir.min_speed`;
- sign(v) > 0 (toward aircraft end, per REQ-14's convention) → LOADING vote;
  sign(v) < 0 → UNLOADING vote.

Frame vote = majority of eligible tracks.

- **REQ-29 (M-of-N semantics, defined):** the debounce window is the last N **eligible
  frames** — frames in which at least one track cast a vote. Frames with zero eligible
  votes do not advance the window (during sparse traffic, a frames-based window would
  flush itself empty between bags and never accumulate M agreements). Defaults:
  M = 8, N = 12 eligible frames.

| Transition | Condition |
|---|---|
| IDLE → LOADING / UNLOADING | ≥ M of last N eligible-frame votes agree |
| LOADING ⇄ UNLOADING | same M-of-N with the opposite sign |
| any → IDLE | zero confirmed tracks for `fsm.idle_timeout_s` |

- **REQ-30 (idle timeout — honest defaults, two profiles):** the idle timeout trades
  session fragmentation against end-boundary noise, and its right value depends on
  operational cadence, not on this clip. Ship two config profiles:
  `profile: reference_clip` → `fsm.idle_timeout_s = 12` (matches conv_D01's cadence);
  `profile: operational` → `fsm.idle_timeout_s = 60` (real inter-bag gaps at a stand —
  cart swaps, hold repositioning — run to minutes). Default = `reference_clip`, and the
  generated README must state that operational deployment requires the second profile
  or per-site tuning. Do not present 12 s as a general value.
- **REQ-31 (session t_end is always last-track time):** every session, regardless of
  how it closes (idle timeout, reposition per REQ-16, teardown per §3.5), closes with
  `t_end` = timestamp of the **last confirmed-track sighting**, never the transition
  time. Without this, timeout-closed sessions would carry a built-in
  `+idle_timeout_s` end bias and the ±3 s boundary target (§7) would be unreachable
  by construction. `sessions.csv` additionally records `t_closed` (when the FSM actually
  transitioned) so the two are auditable.

Session ledger: IDLE→ACTIVE opens `(type, t_start, frame_start)`; close records
`(t_end, t_closed, frame_end, bag_count, truncated_start)`. `bag_count` = confirmed
tracks whose vote matched the session direction, after REQ-27 de-duplication.

---

## 5. Outputs

Per-run folder `runs/<video_stem>_<YYYYmmdd_HHMMSS>/`:

| File | Contents |
|---|---|
| `config.yaml` | verbatim dump of every threshold used — reproducibility |
| `events.csv` | `event_id, event_type, video_time, frame, scene_state, activity_state, n_tracks` |
| `sessions.csv` | `session_id, type, t_start, t_end, t_closed, duration_s, bag_count, truncated_start` |
| `snapshots/` | annotated JPEGs |
| `debug/` (flag-gated) | fg masks, rejection stats (incl. ghost discards), ROI fit history, solidity histograms |

Event types: `BELT_ARRIVED, BELT_DOCKED, ROI_LOCKED, ROI_REPOSITIONED, ROI_HEALTH_RESET,
BELT_COVERED, BELT_UNCOVERED, LOADING_STARTED, LOADING_ENDED, UNLOADING_STARTED,
UNLOADING_ENDED, IDLE_STARTED, BELT_UNDOCKED, BELT_DEPARTED, CAMERA_MOTION,
TIMEBASE_FALLBACK, PRESENCE_AMBIGUOUS` (last three: debug-severity, still logged).

**Snapshot policy** — throttled:

- one snapshot per newly confirmed track ID (**post** REQ-27 re-association — an
  inherited ID takes no new snapshot);
- one snapshot per FSM transition (scene or activity);
- overlay: per §5.1 below, matching the two provided reference screenshots exactly.

### 5.1 Overlay format (`annotate.py`) — must match provided reference frames pixel-for-pixel in layout

Two reference screenshots were supplied (an idle frame and an unloading frame from the
project's earlier reference implementation). Every element below is taken directly from
them; this is not a stylistic suggestion, it is the acceptance criterion for M5.

- **REQ-32 (status block, top-left, black background bar):**
  - Line 1: `STATUS: <state>`. When Activity FSM = IDLE: state rendered as plain
    white title case (`Idle`). When LOADING or UNLOADING: state rendered **uppercase,
    in amber/orange** (`UNLOADING`) — the color change is the primary at-a-glance
    activity cue.
  - Line 2: `Objects: <n_confirmed_tracks>   v=<sign><speed:.2f>` where speed is the
    mean signed axis EMA velocity across confirmed tracks (0.00 when idle). Always
    rendered, both states.
- **REQ-33 (ROI polygon color-coding):** the ROI outline is **amber/yellow** while
  Activity FSM = IDLE, and **blue** while LOADING or UNLOADING. This is a genuine
  operational signal (confirmed identically in both reference frames — yellow ROI on
  the idle frame, blue ROI on the unloading frame), not decoration: it lets a reviewer
  scrub the video and spot activity transitions without reading text.
- **REQ-34 (track boxes):** each confirmed track gets a **green** bounding box with
  label `ID <id> (<age_s:.1f>s)` rendered above the box, where `age_s = (current_frame -
  confirm_frame) / fps`.
- **REQ-35 (aircraft-contact marker, debug-toggleable):** a small filled **magenta**
  circle at the ROI's resolved aircraft-end anchor point — the same point used by
  REQ-13's fuselage-adjacency gate and REQ-14's axis sign convention. Rendered whenever
  the ROI is locked. Purely diagnostic (`overlay.show_debug_markers`, default on) — it
  is what makes a wrong sign convention or a bad fuselage gate visible during
  calibration instead of only showing up as a silent logic error.
- **REQ-36 (top-right compact clock):** `t= <video_time:.1f>s`, every frame, using the
  validated `t_video` from `time_base.py` (REQ-07) — never a private frame/fps
  computation.
- **REQ-37 (bottom timestamp bar):** a black bar across the bottom of the frame reading
  `Timestamp: HH:MM:SS.ss   Frame: <frame_idx>`, every frame, same `t_video` source as
  REQ-36 (the two clocks must never disagree — if they can, that's a bug in
  `time_base.py`, not two independent implementations).
- **REQ-38 (event flash banner):** when any row is appended to `events.csv`, overlay a
  bold, large, yellow `EVENT #<event_id>: <EVENT_TYPE with underscores as spaces>` for
  `overlay.event_flash_s` (≈ 2 s) above the bottom timestamp bar, then clear it. This
  is also the built-in visual audit for REQ-30: the reference frame supplied shows
  `EVENT #18: IDLE STARTED` at t=104.9s on a 300 s clip — exactly the idle-timeout
  spam the original 12 s-only design produced, and exactly what the `operational`
  profile (60 s) and eligible-frame M-of-N (REQ-29) exist to fix. Rendering every event
  this way is how M4/M5 catch a regression back to that spam pattern by eye.

---

## 6. Failure modes designed against

| Failure | Mitigation |
|---|---|
| Video starts mid-turnaround | INIT with full-length test windows (REQ-04); pre-lock activity flagged `truncated_start`, never fabricated (REQ-06) |
| MOG2 cold at MONITORING entry | model instantiated at BELT_PRESENT, warm before consumers exist (REQ-03) |
| Auto-exposure / lighting burst | ROI foreground-ratio burst guard + temporary high learning rate |
| Camera shake / PTZ nudge | reference-patch global-motion estimate; drift monitor and dock stationarity operate on residual motion (REQ-09/10) |
| Loader parked near stand, boom up, not docked | aircraft-adjacency gate on DOCKED (REQ-13) |
| ROI locks on ground marking / body edge | lock plausibility priors (REQ-15) + zero-detection health watchdog (REQ-19) |
| Operator repositions loader | drift monitor → ROI_SETTLING; session held open with defined resume/close semantics (REQ-16/17) |
| Belt pauses with bags aboard | learning freeze while tracks are stationary (REQ-21); ghost filter on restart (REQ-22) |
| People crossing / leaning | ROI mask → position-scaled area → solidity → centroid-in-polygon → min age → axis coherence |
| Operator working at belt end | position-dependent max-area ceiling; residual error stated, not hidden (REQ-25) |
| Perspective size change along belt | linear area band in axis coordinate (REQ-24) |
| Track fragmentation inflating counts | dead-track re-association with ID inheritance (REQ-27) |
| Gaps between bags | eligible-frame M-of-N (REQ-29) + idle hysteresis profiles (REQ-30) |
| Timeout-biased session end times | t_end = last-track sighting on every close path (REQ-31) |
| Bags vanishing under fuselage | aircraft-end termination zone = delivery (REQ-26) |
| Static canopy invisible to MOG2 | texture-based cover test, rechecks gated on IDLE (REQ-20) |
| VFR / dropped-frame DVR exports | PTS-first time base with validated fallback (REQ-07) |
| fps ≠ 11 on another camera | all config in seconds, converted at load (REQ-08) |

**Known limitations (reproduce verbatim in generated README):** REQ-01 (in-sample
metrics), REQ-06 (startup blind window), REQ-11 (single-livery daytime presence cue),
REQ-25 (operator interference), REQ-28 (bidirectional count error), and: *the DOCKED
state means "stationary loader with elevated belt adjacent to an aircraft-like region",
not operationally confirmed docking.*

---

## 7. Calibration & validation protocol

1. **Ground truth**: manually scrub **both** `conv_D01.mp4` and `conv_N01.mp4`; record
   true session boundaries (±1 s) and bag counts for each. ~20 minutes per clip.
2. **Two real clips, not a hypothetical (REQ-01):** `conv_D01.mp4` is the tuning clip;
   `conv_N01.mp4` is held out per REQ-01/REQ-01b. Every metric reported must state which
   clip it came from and whether it's in-sample (D01) or out-of-sample (N01, structural
   thresholds only — its own PER-CAMERA values are separately calibrated, see table in
   §0.1). A metrics CSV header row must carry this label; it is not optional formatting.
3. **Metrics** (computed per clip):
   - session boundary error |t_detected − t_true|, target ≤ 3 s on D01 (achievable only
     because of REQ-31 — verify t_end uses last-track time); report, don't gate on,
     the N01 number;
   - session-level precision/recall (no phantom sessions, none missed);
   - bag count error vs ground truth, reported with both REQ-28 caveats;
   - REQ-01b zero-shot failure table for N01 under D01's unmodified PER-CAMERA config.
4. **Calibration order** (each stage visually verified before the next, run once fully
   for D01, then repeated for N01's PER-CAMERA fields only):
   1. presence HSV band **[PER-CAMERA]**;
   2. dock boom angle + fuselage-gate mode (strict vs weak, REQ-13);
   3. ROI lock tolerances + plausibility bands (REQ-15);
   4. area band at **both belt ends** → fit `det.far_scale` (REQ-24); re-derive
      solidity threshold from the logged histograms;
   5. direction sign convention: watch 3 bags, confirm loading = positive, set
      `camera.aircraft_end` **[PER-CAMERA]** (REQ-14) — cross-check visually against the
      REQ-35 magenta contact-point marker;
   6. M-of-N and idle timeout last, against the ground-truth timeline, per profile
      (REQ-30) — tuned on D01 only, then left untouched for N01's out-of-sample run.

---

## 8. Build order (milestones)

| # | Deliverable | Verifies |
|---|---|---|
| M0 | `time_base.py` + `stabilizer.py` + config loader; timestamp validation and global-motion trace plotted for the full clip, run on **both** D01 and N01 | time base sane on both exports; shake baseline known before anything depends on it |
| M1 | Scene FSM states 1–2 + INIT (full-window, REQ-04), aircraft-adjacency gate, state name on every frame, full-clip run on D01 | presence + dock, no false undocks, no dock on parked-only posture |
| M2 | ROIManager structural fit + plausibility check + lock + drift monitor + health watchdog stub, ROI drawn live with REQ-33 color-coding | stable *plausible* lock, lock time logged, reposition path exercised |
| M3 | MOG2 (early instantiation) + pause-freeze + ghost filter + bag detector (position-scaled band) + tracker + re-association; boxes/IDs rendered per REQ-34, rejection stats printed | bags detected, people/ghosts rejected, ID churn measured before/after REQ-27 |
| M4 | Direction + Activity FSM (eligible-frame M-of-N) + events.csv + sessions.csv with t_end semantics | timeline matches manual scrub; t_end vs t_closed audited; REQ-30 idle-spam regression check via REQ-38 event flashes |
| M5 | Full overlay per §5.1 (REQ-32–38) + snapshot writer, run on D01 | output visually matches the two provided reference frames element-for-element |
| M6 | D01 calibration pass + validation vs D01 ground truth; **then** N01 PER-CAMERA recalibration + REQ-01/REQ-01b out-of-sample and zero-shot runs; README with in-sample vs out-of-sample metrics clearly separated and the verbatim limitations block | honest, reproducible numbers on two independent clips |

Each milestone runs end-to-end on `conv_D01.mp4` (N01 joins from M0 for the time-base
check and again at M6 for validation) and produces a viewable artifact — never more
than one untested layer deep.
