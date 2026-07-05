"""Dataclass config tree, loaded from YAML. Verbatim-dumped into each run folder (REQ-08)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    # [PER-CAMERA] elevated-end sign convention (REQ-14)
    aircraft_end: str = "min_y"  # one of: min_y | max_y | min_x | max_x


@dataclass
class PresenceConfig:
    hsv_lo: list = field(default_factory=lambda: [95, 60, 40])   # [PER-CAMERA]
    hsv_hi: list = field(default_factory=lambda: [130, 255, 255])  # [PER-CAMERA]
    min_area_frac: float = 0.03
    aspect_prior: float = 2.0
    debounce_s: float = 2.0
    # [PER-CAMERA] Fixed reference blob-area fraction, measured once during calibration on a
    # clean stable frame. Extension beyond the original spec: at this resolution the raw HSV
    # mask intermittently bridges into a same-hued region elsewhere in frame (e.g. a
    # fuselage-shadow patch), and an *adaptive* plausibility reference (EMA/rolling median)
    # gets dragged upward by several seconds of gradual, partially-plausible contamination
    # (each single frame's growth looks acceptable relative to the reference just before it,
    # even though the cumulative drift is not). A fixed per-camera reference has no such
    # ratchet failure mode. None disables the check (falls back to an adaptive EMA).
    expected_area_frac: float | None = None
    expected_area_tol: float = 0.4  # fractional tolerance around expected_area_frac


@dataclass
class DockConfig:
    max_drift_px: float = 2.0
    window_s: float = 5.0
    min_boom_angle_deg: float = 15.0
    min_fuselage_frac: float = 0.08
    fuselage_dist_px: float = 15.0
    fuselage_gate: str = "strict"  # strict | weak


@dataclass
class CoverConfig:
    recheck_s: float = 30.0
    edge_density_threshold: float = 0.08  # below this -> COVERED (smooth canopy, few edges)


@dataclass
class RoiConfig:
    pad_px: float = 4.0
    window_s: float = 10.0
    min_settle_s: float = 15.0
    center_std_px: float = 3.0
    angle_std_deg: float = 2.0
    length_std_px: float = 5.0
    aspect_band: list = field(default_factory=lambda: [2.5, 8.0])
    width_band_px: list = field(default_factory=lambda: [10.0, 80.0])
    overlap_min: float = 0.6
    angle_band_deg: float = 20.0
    drift_check_s: float = 5.0
    drift_tolerance_mult: float = 2.0
    drift_consecutive: int = 3
    # Belt-axis detector (roi/belt_detector.py): yellow rail color + hypothesis-driven Hough
    # line fitting, replacing the earlier minAreaRect-over-all-points fit which pulled in
    # canopy/chassis edges and produced an oversized ROI. rail_hsv_* is [PER-CAMERA] like
    # presence.hsv_*.
    rail_hsv_lo: list = field(default_factory=lambda: [8, 35, 70])
    rail_hsv_hi: list = field(default_factory=lambda: [30, 255, 255])
    roi_min_points: int = 8
    # halfwidth = roi_halfwidth_frac * belt_length, i.e. a geometric ratio, not a measured
    # rail width. 0.18 (36% of length as total width) visually extended well into the loader
    # canopy above the actual conveyor bed on D01 (measured: 325px length -> 117px width,
    # spanning canopy height). 0.09 (measured 325px -> 58px) hugs the real rail band instead.
    roi_halfwidth_frac: float = 0.09
    roi_hypo_band_frac: float = 0.05


@dataclass
class PreprocessConfig:
    # Brightness-triggered CLAHE (LAB lightness channel): only engages below night_threshold,
    # so day clips are left untouched and only genuinely dark footage (e.g. N01/N04) pays the
    # contrast-enhancement cost. [PER-CAMERA] night_threshold may need retuning per install.
    enabled: bool = True
    night_threshold: float = 70.0  # mean grayscale brightness (0-255) below which CLAHE engages
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: int = 8


@dataclass
class HealthConfig:
    annulus_px: float = 15.0
    motion_frac: float = 0.10
    sustain_s: float = 20.0


@dataclass
class BgConfig:
    history: int = 400
    burst_frac: float = 0.4
    pause_speed: float = 1.0
    pause_s: float = 3.0
    ghost_s: float = 2.0
    ghost_edge_ratio: float = 0.5
    # Routes MOG2 through cv2.cuda when a CUDA device is present, with a transparent CPU
    # fallback producing identical output otherwise (see detect/bg_model.py::_cuda_available).
    use_gpu: bool = True
    # Strip-scoped background model (main.py) suppresses detections for this long after every
    # (re)lock, since ROI_LOCKED enters MONITORING immediately with zero warm-up history —
    # unlike the full-frame bg_model, which has been running since BELT_PRESENT by then.
    warmup_s: float = 3.0


@dataclass
class DetConfig:
    min_area_near: float = 80.0
    max_area_near: float = 400.0
    far_scale: float = 0.5
    solidity_min: float = 0.45
    aspect_lo: float = 0.3
    aspect_hi: float = 3.5
    # Strip-space filters (detect/bag_detector.py operates on the rectified belt strip, not the
    # raw rotated frame, so these are all measured along/across the belt axis directly).
    end_margin_frac: float = 0.06  # fraction of strip width zeroed at each end (loader zones)
    bag_max_height_frac: float = 0.85  # full-strip-height blob (across belt) = standing person
    person_rot_aspect: float = 2.5  # minAreaRect elongation above which the near-vertical test applies
    vest_hsv_lo: list = field(default_factory=lambda: [20, 80, 120])  # [PER-CAMERA] hi-vis vest
    vest_hsv_hi: list = field(default_factory=lambda: [45, 255, 255])
    vest_reject_frac: float = 0.35  # fraction of a blob's fg pixels that must be vest-colored to reject it


@dataclass
class TrkConfig:
    max_dist_px: float = 25.0
    max_disappeared: int = 8
    min_age: int = 5
    axis_coherence: float = 0.7
    ema_alpha: float = 0.3
    reassoc_s: float = 2.0
    reassoc_px: float = 20.0


@dataclass
class DirConfig:
    min_disp_px: float = 8.0
    min_speed: float = 0.5


@dataclass
class FsmConfig:
    idle_timeout_s: float = 12.0  # profile: reference_clip default
    m_of_n_m: int = 8
    m_of_n_n: int = 12
    reposition_grace_s: float = 60.0
    profile: str = "reference_clip"  # reference_clip | operational


@dataclass
class StabConfig:
    shake_px: float = 1.5
    shake_s: float = 2.0
    n_patches: int = 3


@dataclass
class OverlayConfig:
    show_debug_markers: bool = True
    event_flash_s: float = 2.0


@dataclass
class Config:
    video_path: str = ""
    fps_nominal: float = 11.0
    init_window_s: float = 0.0  # computed at load: max(dock.window_s, presence.debounce_s)
    # Extension beyond the original spec: the 8 installations span ~162-336px frame height
    # (~2x), so absolute-pixel thresholds tuned on one resolution don't transfer to another.
    # pixel_ref_height is the height the *default* px constants were reasoned about; px_scale()
    # rescales them to a clip's actual frame height. This is applied wherever a REQ specifies
    # an absolute pixel threshold that isn't already marked [PER-CAMERA].
    pixel_ref_height: float = 216.0
    camera: CameraConfig = field(default_factory=CameraConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    dock: DockConfig = field(default_factory=DockConfig)
    cover: CoverConfig = field(default_factory=CoverConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    bg: BgConfig = field(default_factory=BgConfig)
    det: DetConfig = field(default_factory=DetConfig)
    trk: TrkConfig = field(default_factory=TrkConfig)
    dir: DirConfig = field(default_factory=DirConfig)
    fsm: FsmConfig = field(default_factory=FsmConfig)
    stab: StabConfig = field(default_factory=StabConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)

    def px_scale(self, frame_height: float) -> float:
        return frame_height / self.pixel_ref_height

    def finalize(self) -> "Config":
        self.init_window_s = max(self.dock.window_s, self.presence.debounce_s)
        return self


_SECTION_TYPES = {
    "camera": CameraConfig,
    "preprocess": PreprocessConfig,
    "presence": PresenceConfig,
    "dock": DockConfig,
    "cover": CoverConfig,
    "roi": RoiConfig,
    "health": HealthConfig,
    "bg": BgConfig,
    "det": DetConfig,
    "trk": TrkConfig,
    "dir": DirConfig,
    "fsm": FsmConfig,
    "stab": StabConfig,
    "overlay": OverlayConfig,
}


def load_config(*paths: str | Path) -> Config:
    """Load and merge one or more YAML files in order (later overrides earlier).

    Typical usage: load_config('configs/default.yaml', 'configs/conv_full_D01.yaml')
    so shared, non-PER-CAMERA defaults come first and PER-CAMERA overrides come last.
    """
    merged: dict[str, Any] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)

    cfg = Config()
    for key, value in merged.items():
        if key in _SECTION_TYPES and isinstance(value, dict):
            section_cls = _SECTION_TYPES[key]
            section = section_cls(**{**dataclasses.asdict(section_cls()), **value})
            setattr(cfg, key, section)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg.finalize()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def dump_config(cfg: Config, path: str | Path) -> None:
    """Verbatim dump of every threshold used, for reproducibility (REQ-08 / §5 outputs)."""
    with open(path, "w") as f:
        yaml.safe_dump(_to_plain(cfg), f, sort_keys=False)
