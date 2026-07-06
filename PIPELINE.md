# Belt ROI Detection & Baggage Detection — Pipeline Walkthrough

This document explains **how the belt ROI is found** and **how bags are detected and tracked** in the codebase. It does not cover the Scene FSM or Activity FSM state-machine logic — only the computer-vision pipelines that feed them.

---

## Table of Contents

1. [High-Level Flow](#high-level-flow)
2. [Camera Stabilization](#camera-stabilization)
3. [Belt Loader Presence Detection](#belt-loader-presence-detection)
4. [Dock Detection](#dock-detection)
5. [Belt ROI Detection (Phase A — Structural Fit)](#belt-roi-detection-phase-a--structural-fit)
6. [ROI Lock & Settling](#roi-lock--settling)
7. [ROI Drift Monitor & Health Watchdog](#roi-drift-monitor--health-watchdog)
8. [ROI Motion Refinement (Phase B)](#roi-motion-refinement-phase-b)
9. [Strip-Space Warping](#strip-space-warping)
10. [Background Subtraction (MOG2)](#background-subtraction-mog2)
11. [Bag Detection](#bag-detection)
12. [Bag Tracking](#bag-tracking)
13. [Direction Voting](#direction-voting)
14. [Key Data Structures](#key-data-structures)
15. [File Map](#file-map)

---

## High-Level Flow

The pipeline processes video frame-by-frame. At a high level, the stages run in this order:

```
┌─────────────────┐
│   Video Frame   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Estimates global camera shake per frame.
│   Stabilizer    │     Consumers subtract this before making motion decisions.
└────────┬────────┘
         │
         ▼
┌─────────────────┐     "Is a belt loader in frame?" — answered by color/appearance,
│ Presence Detect │     not motion. Outputs a blob mask + bounding box.
└────────┬────────┘
         │
         ▼
┌─────────────────┐     "Is the loader parked and connected to the aircraft?"
│  Dock Detection │     Three tests: stationary blob, boom raised, fuselage nearby.
└────────┬────────┘
         │
         ▼
┌─────────────────────┐  Phase A: finds the conveyor belt axis inside the loader blob
│ Belt ROI Detection  │  using yellow rail color + Hough line segments + Huber line fit.
│  (structural_fit)   │  Runs every frame during settling until the fit stabilizes.
└────────┬────────────┘
         │
         ▼
┌─────────────────┐     The stabilized ROI rectangle is "locked". A strip-space affine
│   ROI Lock      │     warp is computed, and a dedicated background model is created.
└────────┬────────┘
         │
         ▼
┌─────────────────┐     The locked ROI band is warped into an axis-aligned rectangle
│  Strip Warp     │     ("strip"). Ground end at x=0, aircraft end at x=strip_w.
└────────┬────────┘
         │
         ▼
┌─────────────────┐     MOG2 background subtraction runs on the strip image.
│ Background Sub  │     Produces a foreground mask. Has burst guard + belt-pause freeze.
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Contour analysis on the foreground mask with multiple rejection
│  Bag Detection  │     filters: ghost, area, solidity, aspect ratio, vest color, person.
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Nearest-neighbor centroid tracker. Tracks are "confirmed" once
│  Bag Tracking   │     they show axis-coherent motion over several frames.
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Each confirmed track votes +1 (loading) or -1 (unloading)
│ Direction Vote  │     based on its signed velocity along the belt axis.
└─────────────────┘
```

---

## Camera Stabilization

**File:** [`stabilizer.py`](stabilizer.py)

**Purpose:** Detect and quantify camera shake / PTZ nudges so downstream modules can subtract it from their own motion measurements. This is **not** video stabilization — no frame warping occurs.

**How it works:**

1. **Patch selection** (`select_patches`): At startup, the stabilizer scans the grayscale frame in a grid and picks the top-N highest-texture patches (ranked by Laplacian variance). Patches overlapping the loader blob are excluded. Patches are kept spatially separated (min 1.5× patch size apart).

2. **Per-frame update** (`update`): Each frame, every reference patch is correlated against the same region in the current frame using `cv2.phaseCorrelate` (sub-pixel frequency-domain registration). The median displacement across all patches gives the global motion estimate `(dx, dy)`.

3. **Shake detection**: If the median displacement exceeds `stab.shake_px`, the frame is marked as "shaking". If shaking persists for `stab.shake_s` consecutive seconds, it's flagged as "sustained" — which triggers the burst guard in the background model.

4. **Cumulative tracking** (`consume_cumulative`): The stabilizer accumulates total `(dx, dy)` displacement over time. The ROI drift monitor drains this accumulator periodically to subtract camera motion from its own re-fit deltas.

**Key parameters (from `StabConfig`):**
- `n_patches`: Number of reference patches to track (default: 4)
- `shake_px`: Per-frame displacement threshold for "shaking" (default: 1.0 px)
- `shake_s`: Duration threshold for "sustained" shake

---

## Belt Loader Presence Detection

**File:** [`scene/presence.py`](scene/presence.py)

**Purpose:** Answer "is a belt loader in frame?" This is an **appearance test** (color-based), not a motion test — a parked loader becomes MOG2 background within seconds, so motion-based detection would fail.

**How it works:**

1. **HSV thresholding**: Convert the frame to HSV and apply `cv2.inRange` with per-camera calibrated bounds (`presence.hsv_lo/hi`). This targets the loader's distinctive livery color (e.g. blue canopy for daytime clips, brightness-only band for night clips).

2. **Morphological cleanup**: A 5×5 closing operation fills small gaps in the mask.

3. **Connected components**: `cv2.connectedComponentsWithStats` finds distinct blobs. Each blob's area is checked against a minimum fraction of the total frame area (`min_area_frac`).

4. **Search window tracking**: Once a loader blob is found, subsequent frames restrict the HSV search to a padded region around the last known position. This prevents the mask from bridging into same-hued regions elsewhere in the frame (e.g., fuselage shadows). If no blob is found for 15 consecutive frames, the search reverts to full-frame.

5. **Size plausibility check**: If a per-camera `expected_area_frac` is configured, candidates whose area deviates by more than `expected_area_tol` fraction are rejected. Otherwise, an adaptive EMA of trusted blob area is used — new candidates must be within 60–160% of the running average.

6. **Ambiguity handling**: If multiple blobs pass the area filter, the one overlapping the previously accepted blob's bounding box is preferred. If no previous blob exists, the one closest to the `aspect_prior` wins.

7. **Debouncing**: The raw per-frame `present`/`absent` signal is debounced over a sliding window of `debounce_s` seconds. State flips only when the entire window agrees.

**Output:** `PresenceResult` containing:
- `present_raw` — instantaneous per-frame test
- `present` — debounced state
- `candidate` — the winning `PresenceCandidate` (mask, bbox, centroid, area, aspect)
- `ambiguous` — whether multiple candidates existed

---

## Dock Detection

**File:** [`scene/dock.py`](scene/dock.py)

**Purpose:** Determine if the loader is **parked at the aircraft and ready for operation**. Three independent signals must all agree:

### 1. Stationary Test
- Tracks the presence blob's centroid over a sliding window (`dock.window_s` seconds)
- Centroid is EMA-smoothed (α=0.3) and corrected for global camera motion (from stabilizer)
- Uses an M-of-N approach: at least 80% of frames in the window must have centroids within `max_drift_px` of the window's median
- This is robust to occasional outlier frames from passing workers / shadow

### 2. Boom Raised Test
- Runs Canny edge detection + HoughLinesP on a padded region around the loader blob
- Filters out near-horizontal lines (< 5° from horizontal) — these are chassis/ground
- Clusters remaining lines into 5° angle bins, weighted by length
- Takes the highest-weight bin's mean angle as the "boom angle"
- Boom angle is smoothed over a sliding window (median filter)
- Boom is "raised" if the smoothed angle exceeds `min_boom_angle_deg` (default: 15°)

### 3. Fuselage Gate
- Looks for a large, bright, low-edge-density region near the loader — this is the aircraft fuselage
- Thresholds grayscale > 150 for brightness, then filters out high-edge-density regions
- Connected components analysis finds candidate fuselage regions
- Requires the region's area to be at least `min_fuselage_frac` of frame area
- Checks that the bounding box gap between loader and fuselage region is within `fuselage_dist_px`
- Result is debounced with a majority filter over ~1 second

**Output:** `DockResult` with individual test results and the composite `docked` boolean.

---

## Belt ROI Detection (Phase A — Structural Fit)

**File:** [`roi/belt_detector.py`](roi/belt_detector.py) (core detection) + [`roi/roi_manager.py`](roi/roi_manager.py) (settling/lock logic)

**Purpose:** Find the conveyor belt's axis (centerline, angle, length, width) inside the loader blob. This is the most complex single detection stage.

### Strategy: Hypothesis-Driven Belt Detection

The key insight is that the belt shows up as a pair of yellow rails at an inclined angle. The detector uses a **hypothesis-and-verification** architecture — inclined Hough line segments *propose* candidate axes, and yellow color evidence *verifies* them. This prevents horizontal ground markings or chassis edges from creating false ROI candidates.

### Step-by-Step

#### 1. Yellow Rail Color Evidence (`_rail_color_evidence`)
- Convert to HSV and threshold with per-camera `rail_hsv_lo/hi` bounds targeting the yellow rail color
- Morphological close (3×3) to fill gaps
- Connected components analysis filters out:
  - Blobs smaller than 25 pixels
  - Blobs with elongation < 2.0 (not rail-like)
  - Blobs taller than wide (elongation axis wrong for rails)

#### 2. Hough Line Segments (`_line_segments`)
- CLAHE contrast enhancement on grayscale
- Canny edge detection (thresholds 60/150)
- Probabilistic Hough Transform (`cv2.HoughLinesP`) with `minLineLength = 22% of frame width`
- Filter to keep only **inclined** segments: absolute angle between 3° and 55° from horizontal
- **Parallel-line filter**: Conveyor rails always appear as parallel pairs. A segment without an angle-matched partner (within 7°) is dropped unless it's exceptionally long (> 35% of frame width)

#### 3. Motion Evidence (`_motion_evidence`)
- Frame-to-frame absolute difference of grayscale
- Threshold at 25, morphological open to remove noise
- Moving pixels are a **booster** on top of color/geometry — never load-bearing on their own
- Subsampled to 500 points max (deterministic stride, not random, for reproducibility)

#### 4. Hypothesis Scoring (`detect_single`)
For each inclined Hough segment as a **seed hypothesis**:

- **Geometric support**: Total length of other segments within the hypothesis band that are angle-matched (within 6°)
- **Yellow support**: Count of yellow-evidence pixels near the hypothesis line
- **Motion support**: Count of motion pixels near the hypothesis line (wider band, 1.5×)
- **Vertical bias**: Slight penalty for lower-in-frame hypotheses (less likely to be the belt vs ground)
- **Continuity bonus**: If a previous detection exists with similar angle and position, add a bonus score

**Score formula:**
```
score = support_px + 0.8 × yellow_px + 0.6 × motion_px - 0.15 × (vertical_position) × width
```

#### 5. Axis Fitting (`_fit_axis`)
The winning hypothesis's supporting points (from line segments + yellow pixels + motion pixels) are fit with a **Huber-weighted line** (`cv2.fitLine` with `cv2.DIST_HUBER`). This is robust to outliers.

- Project all points onto the fitted direction vector
- Take the 3rd–97th percentile range as the belt endpoints
- Endpoint at smaller image-y is `p_hold` (aircraft end), larger image-y is `p_ground`
- Half-width is set as `roi_halfwidth_frac × belt_length`
- Final angle plausibility check: must be 3°–55° from horizontal

#### 6. Endpoint Refinement
After the best hypothesis wins, a second pass collects **all** supporting evidence within 2.5× the hypothesis band and re-fits the endpoints using a tighter 2nd–98th percentile range.

**Output:** A `BeltROI` dataclass with `p_ground`, `p_hold`, `halfwidth`.

### Conversion to `RoiFit`

`roi_manager.structural_fit()` wraps the belt detector:
1. Crops the frame to a padded region around the presence blob's bounding box (30% padding)
2. Calls `belt_detector.detect_single()` on the crop
3. Translates the crop-local coordinates back to full-frame coordinates
4. Adds `pad_px` padding to length and width
5. Returns a `RoiFit(center, angle_deg, length, width)`

---

## ROI Lock & Settling

**File:** [`roi/roi_manager.py`](roi/roi_manager.py) — `update()` method

**Purpose:** The per-frame structural fit is noisy. The settling process requires the fit to **stabilize** over a time window before committing ("locking") the ROI.

### Settling Process

1. **Blacklist check**: If the ROI health watchdog previously fired, the fit parameters from the failed lock are blacklisted. New fits matching those parameters are rejected.

2. **EMA smoothing**: Each incoming `RoiFit` is blended with a running EMA (α=0.15) to dampen frame-to-frame noise before entering the stability window.

3. **Stability window**: The smoothed fits are collected in a sliding window of `window_s` seconds. Stability requires **all four** metrics to be below their thresholds:
   - Center X standard deviation < `center_std_px` (default: 3.0 px)
   - Center Y standard deviation < `center_std_px`
   - Angle standard deviation < `angle_std_deg` (default: 2.0°)
   - Length standard deviation < `length_std_px` (default: 5.0 px)

4. **Minimum settle time**: Even if the window is stable, at least `min_settle_s` (default: 15s) must have elapsed since settling began. This prevents premature locks on transient stable fits.

5. **Plausibility check** (`_plausible`): A stable fit must also pass geometric sanity checks:
   - **Aspect ratio**: `length/width` must be within `aspect_band` (default: 2.5–8.0)
   - **Width**: Must be within `width_band_px` (default: 10–80 px, scaled by `px_scale`)
   - **Boom agreement**: If the dock detector provided a boom angle, the ROI angle must be within `angle_band_deg` (default: 20°) of it
   - **Overlap**: The ROI polygon must overlap at least `overlap_min` (60%) with the loader blob's bounding box

6. **Lock**: If stable + plausible, the averaged fit is committed as a `LockedRoi`. This involves:
   - Computing the axis unit vector (oriented toward the aircraft using `aircraft_end` config)
   - Identifying the aircraft-end and ground-end anchor points
   - Building a 4-corner rotated-rectangle polygon via `cv2.boxPoints`

---

## ROI Drift Monitor & Health Watchdog

### Drift Monitor
**File:** [`roi/roi_manager.py`](roi/roi_manager.py) — `_update_drift()` method

Once locked, the ROI manager continues receiving structural fits and checks for drift every `drift_check_s` seconds:

- Computes deltas in center, angle, and length between the fresh fit and the locked fit
- **Subtracts camera motion** (from stabilizer's cumulative accumulator) from center delta
- If any delta exceeds `drift_tolerance_mult × threshold`, it's a violation
- After `drift_consecutive` (default: 3) consecutive violations → `ROI_REPOSITIONED` event: lock is released and settling restarts

### Health Watchdog
**File:** [`roi/roi_health.py`](roi/roi_health.py)

Catches **wrong locks** (the ROI locked onto the wrong part of the loader):

1. Builds an annulus mask around the locked ROI polygon (dilate by `annulus_px`)
2. Checks if there's foreground motion in the annulus (from background subtraction) while the ROI itself has zero confirmed tracks
3. If this condition persists for `sustain_s` seconds → `ROI_HEALTH_RESET`: lock is released, the failed fit is blacklisted, and settling restarts

---

## ROI Motion Refinement (Phase B)

**File:** [`roi/roi_manager.py`](roi/roi_manager.py) — `motion_refine()` method

Once in MONITORING with confirmed tracks, Phase B allows **minor adjustments** to the locked ROI:

- Collects all confirmed track centroids
- Computes the centroid of all track positions ("heat center")
- **Translates** the ROI center toward the heat center, clamped to `3 × center_std_px`
- **Shrinks** the length (never grows) to tightly fit the observed track extent
- Axis angle is **never changed** (rotation beyond ±3° is forbidden)
- Each adjustment bumps the `generation` counter, invalidating cached strip transforms

---

## Strip-Space Warping

**File:** [`roi/roi_manager.py`](roi/roi_manager.py) — `LockedRoi.strip_transform()` + [`main.py`](main.py)

**Purpose:** Transform the tilted, rotated ROI band into an axis-aligned rectangle ("strip") where the belt axis runs horizontally. This makes all downstream geometry (blob width vs height, velocity direction, end margins) trivial — no trigonometry needed.

**How it works:**

1. The locked ROI defines three source points:
   - Ground anchor ± half-width along the normal
   - Aircraft anchor - half-width along the normal
2. These map to destination points in the strip:
   - `(0, 0)`, `(strip_w, 0)`, `(0, strip_h)`
3. `cv2.getAffineTransform` computes the 2×3 affine matrix
4. `cv2.warpAffine` warps the full frame into the strip image each frame

**Strip coordinate convention:**
- **x = 0** → ground/loader end of the belt
- **x = strip_w** → aircraft/hold end of the belt
- **+x direction** → "toward the aircraft" = LOADING direction
- **strip_h** = ROI width (cross-axis), **strip_w** = ROI length (along-axis)

An inverse transform (`cv2.invertAffineTransform`) is also computed to map detection coordinates back to the original frame for annotation/visualization.

---

## Background Subtraction (MOG2)

**File:** [`detect/bg_model.py`](detect/bg_model.py)

**Purpose:** Produce a per-pixel foreground mask that highlights moving objects (bags) against the static belt surface.

### Key Design Decisions

1. **Early instantiation**: The background model is created when the belt is first detected (`BELT_PRESENT` state), long before monitoring begins. This gives MOG2 time to warm up and build a stable background model.

2. **Two separate MOG2 instances**:
   - **Scene-level model** (in `main.py`): Applied to the full frame, used for the ROI health watchdog's annulus motion check
   - **Strip-level model** (in `main.py`): Applied to the warped strip image, used for actual bag detection. Rebuilt whenever the ROI lock/generation changes

3. **Shadow handling**: MOG2's built-in shadow detection is enabled (`detectShadows=True`). Shadow pixels (value 127) are zeroed out in the foreground mask — shadows are not bags.

4. **GPU support**: If CUDA is available and `bg.use_gpu` is set, uses `cv2.cuda.createBackgroundSubtractorMOG2` for acceleration.

### Adaptive Learning Rate

The learning rate is adjusted based on context:

| Condition | Learning Rate | Reason |
|---|---|---|
| Normal operation | `-1` (auto) | MOG2's default adaptive rate |
| Burst guard active | `0.2` (high) | Quickly recover from lighting/motion burst |
| Belt paused | `0.0` (frozen) | Prevent stationary bags from being absorbed into background |
| ROI re-settling | `0.0` (frozen) | Don't burn ghost trails from repositioned belt geometry |

### Burst Guard
- If foreground pixels exceed `burst_frac` of the ROI area → burst detected
- Learning rate is raised to 0.2 for ~1 second to quickly re-stabilize
- Also triggered externally by sustained camera shake (from stabilizer)

### Belt-Pause Freeze
- Tracks mean bag speed over `pause_s` seconds
- If all speeds are below `pause_speed` → learning rate drops to 0.0
- Prevents MOG2 from absorbing stationary bags into the background

**Output:** `ForegroundResult` with binary mask, burst flag, and learning rate used.

---

## Bag Detection

**File:** [`detect/bag_detector.py`](detect/bag_detector.py)

**Purpose:** Find individual bag blobs in the foreground mask of the **strip image** (not the raw frame). Operating in strip space means all geometric filters measure along/across the belt axis directly.

### Detection Pipeline

```
  Foreground mask (from MOG2 on strip)
           │
           ▼
  ┌──────────────────┐
  │ Morphology       │  Open (3×3) to remove noise, Close (5×5) to fill gaps
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ End Margin Zero  │  Zero out pixels within end_margin_frac of each strip end.
  │                  │  Removes loader mechanisms at ground/aircraft ends.
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Find Contours   │  cv2.findContours (RETR_EXTERNAL) on the cleaned mask
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Ghost Filter    │  Has this blob sat still for ghost_s seconds?
  │                  │  AND does it have low edge density vs background?
  │                  │  → Reject (it's a background artifact, not a bag)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Area Filter     │  Area must be within [min_area, max_area].
  │                  │  Band scales with s_axis position (perspective correction).
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Solidity Filter  │  convex_hull_area / contour_area must exceed solidity_min.
  │                  │  Rejects irregular/fragmented blobs.
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Aspect Ratio     │  max(w,h)/min(w,h) must be within [aspect_lo, aspect_hi].
  │  Filter          │  Rejects very elongated or very square blobs.
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Person Height    │  Blob spanning most of strip height AND taller than wide
  │  Filter          │  → Standing person, not luggage. Rejected.
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Rotated Rect     │  Elongated blob oriented near-vertical (60°–120°)
  │  Person Filter   │  → Leaning/standing person. Rejected.
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Hi-Vis Vest      │  Blob's foreground is mostly vest-colored (HSV filter)?
  │  Filter          │  → Worker leaning over rails, not a bag. Rejected.
  └────────┬─────────┘
           │
           ▼
      Detection ✓
```

### Ghost Filter Detail

The ghost filter uses a lightweight internal position tracker (`_GhostTracker`) separate from the main bag tracker:
- Matches each contour centroid to the nearest previous centroid (< 6px)
- Counts consecutive frames each centroid has been "still"
- If a blob has been still for ≥ `ghost_s` seconds:
  - Compare Canny edge density in the blob's region between the current frame and the background image
  - If current edge density < `ghost_edge_ratio × background edge density` → ghost (background texture misclassified as foreground)

### Perspective-Scaled Area Band

The area thresholds scale with the blob's position along the belt axis (`s_axis`, 0.0 = loader end, 1.0 = aircraft end):
```
min_area(s) = min_area_near × (1 + (far_scale - 1) × s)
max_area(s) = max_area_near × (1 + (far_scale - 1) × s)
```
This accounts for perspective: bags closer to the aircraft end appear slightly larger/smaller depending on camera geometry.

### Vest Color Rejection

- Converts the strip to HSV
- Applies `cv2.inRange` with `vest_hsv_lo/hi` targeting hi-vis vest colors (typically fluorescent yellow/orange)
- For each blob: if the fraction of vest-colored pixels within the foreground exceeds `vest_reject_frac` → rejected as a worker, not a bag

**Output:** List of `Detection` objects (centroid, bbox, area, solidity, s_axis) + `RejectionStats` for debugging.

---

## Bag Tracking

**File:** [`track/tracker.py`](track/tracker.py)

**Purpose:** Associate detections across frames into persistent tracks, and confirm tracks that show coherent belt-axis motion (i.e., actual bags moving on the belt, not noise).

### Algorithm: Greedy Nearest-Neighbor

No Kalman filter, no Hungarian algorithm. At 11 fps with relatively few objects, a simple nearest-neighbor approach is sufficient:

1. **Matching**: For each existing track, compute Euclidean distance to each new detection. Pairs within `max_dist_px` are collected and sorted by distance. Greedily assign the closest pair first, then the next closest, etc. (no double-assignment).

2. **Velocity update**: For each matched pair, compute displacement along the belt axis (dot product with `axis_unit`), divide by `dt` to get instantaneous velocity, and blend into a running EMA:
   ```
   v_ema = α × v_instantaneous + (1 - α) × v_ema_prev
   ```

3. **Disappearance handling**: Unmatched tracks increment their `disappeared` counter. If it exceeds `max_disappeared` → track dies and enters the dead pool.

4. **New track creation**: Unmatched detections that don't match any dead-pool track create new tracks with a unique incrementing ID.

### Track Confirmation

A track is **confirmed** (= "this is a real bag") when:
- It has existed for at least `min_age` frames
- Its net displacement shows **axis coherence** ≥ `axis_coherence` threshold:
  ```
  coherence = |net_displacement · axis_unit| / |net_displacement|
  ```
  This measures how much of the track's motion is along the belt axis (vs. perpendicular). A bag on a belt moves almost exclusively along the axis; noise / workers move in random directions.

### Dead Pool & Re-Association

When a confirmed track disappears (e.g., brief occlusion by a worker):
- It enters the dead pool with a timestamp and whether it was in the "delivery zone" (s_axis ≥ 0.85)
- Dead pool entries expire after `reassoc_s` seconds
- New unmatched detections try to re-associate with dead tracks by extrapolating the dead track's last known velocity:
  ```
  predicted_position = last_centroid + velocity × elapsed_time
  ```
- If the new detection is within `reassoc_px` of the predicted position → the old track resumes (marked `inherited = True`, so no duplicate bag count or snapshot)
- Tracks that died in the delivery zone are excluded from re-association (they've been delivered, not lost)

---

## Direction Voting

**File:** [`activity/direction.py`](activity/direction.py)

**Purpose:** Determine, each frame, whether bags are being **loaded** (moving toward the aircraft) or **unloaded** (moving away).

### Per-Track Vote

Each confirmed track votes independently:

1. Must have ≥ 2 history points and a valid velocity EMA
2. Net axis displacement must exceed `min_disp_px` (prevents noise votes from barely-moving tracks)
3. Absolute velocity must exceed `min_speed`
4. **Vote: +1** if `velocity_ema > 0` (positive = toward aircraft = LOADING)
5. **Vote: -1** if `velocity_ema < 0` (negative = toward ground = UNLOADING)
6. **Vote: 0** (abstain) if thresholds not met

### Frame Vote

The frame-level vote is a simple **majority** of all eligible track votes:
- Sum all non-zero votes
- If sum > 0 → frame vote = +1 (LOADING)
- If sum < 0 → frame vote = -1 (UNLOADING)
- If sum = 0 (exact tie) or no eligible tracks → frame vote = `None`

This frame vote feeds into the Activity FSM's M-of-N logic (not covered in this document).

---

## Key Data Structures

| Structure | File | Purpose |
|---|---|---|
| `BeltROI` | `roi/belt_detector.py` | Raw belt axis: two endpoints + halfwidth |
| `RoiFit` | `roi/roi_manager.py` | Normalized fit: center, angle, length, width |
| `LockedRoi` | `roi/roi_manager.py` | Committed ROI: fit + axis unit vector + anchors + polygon + strip transform |
| `PresenceCandidate` | `scene/presence.py` | Loader blob: mask, bbox, centroid, area, aspect |
| `DockResult` | `scene/dock.py` | Dock test results: stationary, boom angle, fuselage, composite |
| `ForegroundResult` | `detect/bg_model.py` | MOG2 output: binary mask, burst flag, learning rate |
| `Detection` | `detect/bag_detector.py` | Single bag detection: centroid, bbox, area, solidity, s_axis |
| `RejectionStats` | `detect/bag_detector.py` | Debug counters: ghost, area, solidity, aspect, vest, person |
| `Track` | `track/tracker.py` | Tracked bag: id, centroid, bbox, age, velocity, history, confirmed |
| `GlobalMotion` | `stabilizer.py` | Camera shake: dx, dy, magnitude, is_shaking, sustained |

---

## File Map

```
belt-loader-detection/
├── main.py                     # Pipeline driver: wires all modules together
├── stabilizer.py               # Camera shake estimation (phase correlation)
├── video_source.py             # Video file reader with timestamp extraction
├── time_base.py                # Frame timestamp normalization
├── config.py                   # Dataclass config tree (loaded from YAML)
├── preprocess.py               # Night-mode CLAHE preprocessing
│
├── scene/
│   ├── presence.py             # "Is a loader in frame?" (HSV color)
│   ├── dock.py                 # "Is it parked at the aircraft?" (stationary + boom + fuselage)
│   ├── cover.py                # "Is the belt covered?" (edge density)
│   └── scene_fsm.py            # Scene state machine (not covered here)
│
├── roi/
│   ├── belt_detector.py        # Belt axis detection (Hough + yellow rails + Huber fit)
│   ├── roi_manager.py          # ROI settling, lock, drift monitor, Phase B refinement
│   └── roi_health.py           # Health watchdog (wrong-lock detector)
│
├── detect/
│   ├── bg_model.py             # MOG2 background subtraction wrapper
│   └── bag_detector.py         # Bag blob detection with rejection filters
│
├── track/
│   └── tracker.py              # Nearest-neighbor centroid tracker
│
├── activity/
│   ├── direction.py            # Per-track direction vote (+1 loading / -1 unloading)
│   └── activity_fsm.py         # Activity state machine (not covered here)
│
├── output/
│   ├── annotate.py             # Overlay rendering for live view / video
│   ├── events.py               # Event logging + session management
│   └── snapshots.py            # Per-bag snapshot writer
│
├── configs/                    # Per-camera YAML config files
│   ├── default.yaml
│   ├── conv_full_D01.yaml
│   └── ...
│
└── tools/                      # Diagnostic / calibration scripts
    ├── m0_check.py ... m6_report.py
    └── calibrate_presence.py, extract_samples.py, etc.
```
